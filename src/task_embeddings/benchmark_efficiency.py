from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration

from .common import (
    PROJECT_ROOT,
    OnlineAccumulator,
    normalized_rows,
    save_json,
    seed_everything,
)
from .controlled import GatedMLP, Teachers, task_mixtures
from .language import T5NeuronProbe, sequence_ce_residual_norm_sq
from .vision import CifarResNet18, cross_entropy_residual_norm_sq


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_loops(
    baseline_step: Callable[[], None],
    profiled_step: Callable[[], None],
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        baseline_step()
        profiled_step()
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        baseline_step()
    _synchronize(device)
    baseline_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(iterations):
        profiled_step()
    _synchronize(device)
    profiled_seconds = time.perf_counter() - start
    return {
        "iterations": iterations,
        "baseline_seconds": baseline_seconds,
        "profiled_seconds": profiled_seconds,
        "overhead_fraction": (profiled_seconds - baseline_seconds)
        / max(baseline_seconds, 1e-12),
    }


def embedding_storage_bytes(n_modules: int, dimensions: Sequence[int]) -> int:
    """Float32 numerator plus one denominator vector for each representation."""
    return int(n_modules * sum(dimension + 1 for dimension in dimensions) * 4)


def _controlled(device: torch.device, root: Path) -> dict:
    checkpoint = torch.load(
        root / "01_exploratory/artifacts/controlled/controlled_seed1.pt",
        map_location=device,
        weights_only=False,
    )
    model = GatedMLP(width=512, depth=5).to(device)
    model.load_state_dict(checkpoint["model"])
    teachers = Teachers().to(device)
    alpha = task_mixtures().to(device)
    generator = torch.Generator().manual_seed(31)
    x_raw = torch.randn(512, 32, generator=generator).to(device)
    with torch.no_grad():
        y = teachers(x_raw) @ alpha[0]
    ids = torch.zeros(512, 16, device=device)
    ids[:, 0] = 1
    x = torch.cat([x_raw, ids], dim=1)
    representations = {"onehot": torch.eye(16), "latent4": normalized_rows(alpha.cpu())}
    accumulator = OnlineAccumulator(
        model.n_modules, representations, device, total_steps=220
    )
    state = {"step": 0}

    def backward() -> tuple[torch.Tensor, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            residual = model(x) - y
            loss = 0.5 * residual.square().mean()
        loss.backward()
        return residual, loss

    def baseline() -> None:
        backward()

    def profiled() -> None:
        residual, _ = backward()
        accumulator.update(
            model.block_grad_norms(),
            0,
            residual.detach().float().square().sum() / (len(x) ** 2),
            state["step"],
        )
        state["step"] += 1

    result = benchmark_loops(baseline, profiled, device, warmup=10, iterations=200)
    result["n_modules"] = model.n_modules
    result["online_state_bytes"] = 6 * embedding_storage_bytes(model.n_modules, [16, 4])
    result["batch_size"] = len(x)
    return result


def _vision(device: torch.device, root: Path) -> dict:
    checkpoint = torch.load(
        root / "01_exploratory/artifacts/vision_corrected/vision_seed1.pt",
        map_location=device,
        weights_only=False,
    )
    model = CifarResNet18().to(device)
    model.load_state_dict(checkpoint["model"])
    images = torch.randn(64, 3, 32, 32, device=device)
    targets = torch.zeros(64, dtype=torch.long, device=device)
    jl32 = normalized_rows(
        torch.randn(100, 32, generator=torch.Generator().manual_seed(717))
    )
    representations = {"onehot": torch.eye(100), "jl32": jl32}
    accumulator = OnlineAccumulator(
        model.n_modules, representations, device, total_steps=120
    )
    state = {"step": 0}

    def backward() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        return logits

    def baseline() -> None:
        backward()

    def profiled() -> None:
        logits = backward()
        accumulator.update(
            model.block_grad_norms(),
            0,
            cross_entropy_residual_norm_sq(logits.detach(), targets),
            state["step"],
        )
        state["step"] += 1

    result = benchmark_loops(baseline, profiled, device, warmup=10, iterations=100)
    result["n_modules"] = model.n_modules
    result["online_state_bytes"] = 6 * embedding_storage_bytes(
        model.n_modules, [100, 32]
    )
    result["batch_size"] = len(images)
    return result


def _language(device: torch.device, root: Path) -> dict:
    checkpoint = torch.load(
        root / "01_exploratory/artifacts/language/language_seed1.pt",
        map_location=device,
        weights_only=False,
    )
    base = T5ForConditionalGeneration.from_pretrained(
        "google-t5/t5-small", local_files_only=True
    ).to(device)
    base.load_state_dict(checkpoint["model"])
    model = T5NeuronProbe(base)
    input_ids = torch.randint(2, 1000, (16, 96), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full((16, 4), -100, device=device)
    labels[:, 0] = 1176
    labels[:, 1] = 1
    representations = {
        "onehot": torch.eye(6),
        "semantic4": checkpoint["semantic4"],
    }
    accumulator = OnlineAccumulator(
        model.n_modules, representations, device, total_steps=70
    )
    state = {"step": 0}

    def backward():
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        output.loss.backward()
        return output

    def baseline() -> None:
        backward()

    def profiled() -> None:
        output = backward()
        accumulator.update(
            model.block_grad_norms(),
            0,
            sequence_ce_residual_norm_sq(output.logits.detach(), labels),
            state["step"],
        )
        state["step"] += 1

    result = benchmark_loops(baseline, profiled, device, warmup=5, iterations=50)
    result["n_modules"] = model.n_modules
    result["online_state_bytes"] = 6 * embedding_storage_bytes(model.n_modules, [6, 4])
    result["batch_size"] = len(input_ids)
    return result


def run(args: argparse.Namespace) -> None:
    seed_everything(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.project_root)
    results = {
        "device": str(device),
        "controlled": _controlled(device, root),
        "vision": _vision(device, root),
        "language": _language(device, root),
    }
    output = root / "01_exploratory/artifacts/efficiency_benchmark.json"
    save_json(output, results)
    print(f"EFFICIENCY_RESULT={output}")
    print(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT.with_name("task_embeddings_internal")),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
