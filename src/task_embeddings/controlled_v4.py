from __future__ import annotations

import argparse
import copy
import itertools
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .common import (
    EPS,
    accelerator_peak_memory,
    arc_artifact_dir,
    reset_accelerator_peak_memory,
    save_json,
    seed_everything,
)
from .controlled_v3 import ControlledTransformer


def hard_task_mixtures() -> torch.Tensor:
    """Six pure tasks, every pair, and nine deterministic triples."""
    rows: list[torch.Tensor] = []
    for primitive in range(6):
        row = torch.zeros(6)
        row[primitive] = 1.0
        rows.append(row)
    for pair in itertools.combinations(range(6), 2):
        row = torch.zeros(6)
        row[list(pair)] = 0.5
        rows.append(row)
    for triple in list(itertools.combinations(range(6), 3))[:9]:
        row = torch.zeros(6)
        row[list(triple)] = 1.0 / 3.0
        rows.append(row)
    return torch.stack(rows)


def program_primitive_targets(
    payloads: torch.Tensor, payload_modulus: int
) -> torch.Tensor:
    """Targets for six tagged fields with distinct computational structure."""
    if payloads.ndim != 3 or payloads.shape[1] != 6 or payloads.shape[2] < 8:
        raise ValueError("payloads must have shape [batch, 6, length>=8]")
    values = payloads.long()
    denominator = max(1, payload_modulus - 1)

    # A smooth periodic checksum retains global accumulation and modular token
    # structure without requiring brittle exact integer modular addition.
    checksum = torch.sin(
        2.0 * math.pi * values[:, 0].float() / max(1, payload_modulus)
    ).mean(dim=1)

    motif = (values[:, 1, :-1].remainder(4) == 0) & (values[:, 1, 1:].remainder(4) == 1)
    motif_parity = 2.0 * motif.sum(dim=1).remainder(2).float() - 1.0

    associative = values[:, 2]
    query = associative[:, 6].remainder(3)
    retrieved = associative[:, 3:6].gather(1, query[:, None]).squeeze(1).float()
    retrieved = 2.0 * retrieved / denominator - 1.0

    pointer = values[:, 3]
    mapping = pointer[:, :4].remainder(4)
    start = pointer[:, 4].remainder(4)
    first = mapping.gather(1, start[:, None]).squeeze(1)
    second = mapping.gather(1, first[:, None]).squeeze(1).float()
    pointer_target = 2.0 * second / 3.0 - 1.0

    order_values = values[:, 4].float()
    split = order_values.shape[1] // 2
    order_target = torch.tanh(
        (order_values[:, :split].mean(dim=1) - order_values[:, split:].mean(dim=1))
        / max(1.0, payload_modulus / 4)
    )

    parentheses = 2.0 * values[:, 5].remainder(2).float() - 1.0
    depth = parentheses.cumsum(dim=1).amax(dim=1)
    stack_target = 2.0 * depth / values.shape[2] - 1.0

    return torch.stack(
        (
            checksum,
            motif_parity,
            retrieved,
            pointer_target,
            order_target,
            stack_target,
        ),
        dim=1,
    ).clamp(-1.0, 1.0)


@dataclass(frozen=True)
class HardControlledConfig:
    seed: int = 1
    payload_modulus: int = 16
    payload_length: int = 8
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    train_steps: int = 24_000
    refinement_steps: int = 6_000
    pure_task_fraction: float = 0.5
    batch_size: int = 128
    reference_per_task: int = 12
    calibration_per_task: int = 48
    test_per_task: int = 96
    learning_rate: float = 3e-4
    refinement_learning_rate: float = 1e-4
    causal_tasks: int = 6
    circuit_fractions: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
    attainable_steps: int = 80
    attainable_restarts: int = 3
    random_masks: int = 32
    pruning_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)
    cl_tasks: int = 10
    cl_steps_per_task: int = 80
    cl_learning_rate: float = 5e-4
    cl_strength: float = 8.0

    @property
    def sequence_length(self) -> int:
        return 6 * (self.payload_length + 1)

    @property
    def vocabulary_size(self) -> int:
        return self.payload_modulus + 6


def make_program_batch(
    config: HardControlledConfig,
    mixtures: torch.Tensor,
    task: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    *,
    noise: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payloads = torch.randint(
        0,
        config.payload_modulus,
        (batch_size, 6, config.payload_length),
        generator=generator,
    )
    primitives = program_primitive_targets(payloads, config.payload_modulus)
    targets = primitives @ mixtures[task].cpu()
    if noise:
        targets = targets + noise * torch.randn(batch_size, generator=generator)
    blocks = []
    for primitive in range(6):
        tag = torch.full(
            (batch_size, 1), config.payload_modulus + primitive, dtype=torch.long
        )
        blocks.append(torch.cat((tag, payloads[:, primitive]), dim=1))
    stacked = torch.stack(blocks, dim=1)
    shuffled = torch.empty_like(stacked)
    for sample in range(batch_size):
        order = torch.randperm(6, generator=generator)
        shuffled[sample] = stacked[sample, order]
    tokens = shuffled.flatten(1)
    task_ids = torch.full((batch_size,), task, dtype=torch.long)
    return tokens.to(device), targets.to(device), task_ids.to(device)


def make_stratified_task_order(
    n_tasks: int,
    n_pure_tasks: int,
    steps: int,
    pure_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Balance batches within pure and composite task strata, then shuffle."""
    if not 0 < n_pure_tasks < n_tasks:
        raise ValueError("pure and composite task strata must both be nonempty")
    if not 0.0 < pure_fraction < 1.0:
        raise ValueError("pure_fraction must lie strictly between zero and one")
    if steps < 1:
        raise ValueError("steps must be positive")

    pure_steps = round(steps * pure_fraction)
    composite_steps = steps - pure_steps

    def balanced_indices(start: int, stop: int, count: int) -> torch.Tensor:
        choices = torch.arange(start, stop)
        repeats = math.ceil(count / len(choices))
        return choices.repeat(repeats)[:count]

    pure = balanced_indices(0, n_pure_tasks, pure_steps)
    composite = balanced_indices(n_pure_tasks, n_tasks, composite_steps)
    unshuffled = torch.cat((pure, composite))
    return unshuffled[torch.randperm(steps, generator=generator)]


def training_phases(
    config: HardControlledConfig,
) -> tuple[tuple[str, int, float], ...]:
    """Return the fixed optimization phases selected on development runs."""
    return (
        ("primary", config.train_steps, config.learning_rate),
        ("refinement", config.refinement_steps, config.refinement_learning_rate),
    )


def global_topk_indices(
    scores: torch.Tensor,
    layer_sizes: Sequence[int],
    count: int,
) -> list[torch.Tensor]:
    """Select exactly ``count`` modules globally and return local indices."""
    values = scores.detach().float().cpu().flatten()
    total = sum(layer_sizes)
    if values.numel() != total:
        raise ValueError("layer sizes do not match score count")
    if count < 0 or count > total:
        raise ValueError("count must lie between zero and the module count")
    chosen = (
        torch.topk(values, count).indices if count else torch.empty(0, dtype=torch.long)
    )
    selected: list[torch.Tensor] = []
    offset = 0
    for size in layer_sizes:
        local = chosen[(chosen >= offset) & (chosen < offset + size)] - offset
        selected.append(local.sort().values)
        offset += size
    return selected


def fraction_to_count(fraction: float, n_modules: int) -> int:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    return min(n_modules, max(0, math.ceil(fraction * n_modules)))


def make_program_dataset(
    config: HardControlledConfig,
    mixtures: torch.Tensor,
    per_task: int,
    seed: int,
    *,
    noise: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pieces = [
        make_program_batch(
            config,
            mixtures,
            task,
            per_task,
            generator,
            torch.device("cpu"),
            noise=noise,
        )
        for task in range(mixtures.shape[0])
    ]
    return tuple(torch.cat([piece[index] for piece in pieces]) for index in range(3))


@torch.no_grad()
def evaluate_program_losses(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    model.eval()
    tokens, targets, task_ids = dataset
    n_tasks = int(task_ids.max()) + 1
    sums = torch.zeros(n_tasks, device=device)
    counts = torch.zeros(n_tasks, device=device)
    for start in range(0, len(tokens), batch_size):
        local_tokens = tokens[start : start + batch_size].to(device)
        local_targets = targets[start : start + batch_size].to(device)
        local_tasks = task_ids[start : start + batch_size].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            predictions = model(local_tokens, local_tasks)
        losses = 0.5 * (predictions.float() - local_targets).square()
        sums.scatter_add_(0, local_tasks, losses)
        counts.scatter_add_(0, local_tasks, torch.ones_like(losses))
    return (sums / counts.clamp_min(1)).cpu()


def train_joint_model(
    model: ControlledTransformer,
    config: HardControlledConfig,
    mixtures: torch.Tensor,
    device: torch.device,
) -> dict:
    trace = []
    task_batch_counts = torch.zeros(mixtures.shape[0], dtype=torch.long)
    total_steps = sum(steps for _, steps, _ in training_phases(config))
    global_step = 0
    started = time.perf_counter()
    model.train()
    for phase_index, (phase, phase_steps, learning_rate) in enumerate(
        training_phases(config)
    ):
        if phase_steps < 1:
            continue
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-3
        )
        generator = torch.Generator().manual_seed(config.seed + 11 + 10 * phase_index)
        task_order = make_stratified_task_order(
            n_tasks=mixtures.shape[0],
            n_pure_tasks=6,
            steps=phase_steps,
            pure_fraction=config.pure_task_fraction,
            generator=generator,
        )
        task_batch_counts += torch.bincount(task_order, minlength=mixtures.shape[0])
        for phase_step, task_tensor in enumerate(task_order):
            tokens, targets, task_ids = make_program_batch(
                config,
                mixtures,
                int(task_tensor),
                config.batch_size,
                generator,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                residual = model(tokens, task_ids) - targets
                loss = 0.5 * residual.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (
                global_step % max(1, total_steps // 20) == 0
                or global_step + 1 == total_steps
                or phase_step + 1 == phase_steps
            ):
                trace.append(
                    {
                        "step": global_step,
                        "phase": phase,
                        "half_mse": float(loss.detach()),
                    }
                )
            global_step += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "wall_seconds": elapsed,
        "examples_per_second": total_steps * config.batch_size / max(elapsed, EPS),
        "loss_trace": trace,
        "task_batch_counts": task_batch_counts,
        "sampling": {
            "scheme": "balanced within pure and composite strata",
            "pure_task_fraction": config.pure_task_fraction,
        },
        "phases": [
            {"name": name, "steps": steps, "learning_rate": rate}
            for name, steps, rate in training_phases(config)
        ],
    }


def competence_metrics(
    model_losses: torch.Tensor,
    calibration: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict:
    calibration_targets, calibration_tasks = calibration[1], calibration[2]
    test_targets, test_tasks = test[1], test[2]
    n_tasks = model_losses.numel()
    means = torch.stack(
        [
            calibration_targets[calibration_tasks == task].mean()
            for task in range(n_tasks)
        ]
    )
    null_losses = torch.stack(
        [
            0.5 * (test_targets[test_tasks == task] - means[task]).square().mean()
            for task in range(n_tasks)
        ]
    )
    recovered = 1.0 - model_losses / null_losses.clamp_min(EPS)
    return {
        "model_half_mse": model_losses,
        "task_mean_null_half_mse": null_losses,
        "null_gap_recovered": recovered,
        "minimum_gap_recovered": float(recovered.min()),
        "median_gap_recovered": float(recovered.median()),
        "tasks_passing_80_percent": int((recovered >= 0.8).sum()),
        "total_tasks": n_tasks,
        "passed": bool((recovered >= 0.8).all()),
    }


def build_model(
    config: HardControlledConfig, device: torch.device
) -> ControlledTransformer:
    return ControlledTransformer(
        token_modulus=config.vocabulary_size,
        sequence_length=config.sequence_length,
        n_tasks=hard_task_mixtures().shape[0],
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
    ).to(device)


def run_competence_pilot(
    config: HardControlledConfig,
    *,
    device: torch.device | None = None,
) -> tuple[dict, dict[str, torch.Tensor]]:
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_accelerator_peak_memory(device)
    mixtures = hard_task_mixtures()
    model = build_model(config, device)
    initial_state = copy.deepcopy(model.state_dict())
    train = train_joint_model(model, config, mixtures, device)
    calibration = make_program_dataset(
        config, mixtures, config.calibration_per_task, config.seed + 201
    )
    test = make_program_dataset(
        config, mixtures, config.test_per_task, config.seed + 301
    )
    losses = evaluate_program_losses(model, test, device)
    competence = competence_metrics(losses, calibration, test)
    result = {
        "setting": "tagged_compositional_programs",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "n_modules": model.n_modules,
            "module_owned_parameters": sum(
                parameter.numel() for parameter in model.module_owned_parameters()
            ),
        },
        "train": train,
        "competence": competence,
        "efficiency": accelerator_peak_memory(device),
    }
    states = {
        "trained": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "initial": {
            name: value.detach().cpu() for name, value in initial_state.items()
        },
    }
    return result, states


def _smoke_config(seed: int) -> HardControlledConfig:
    return HardControlledConfig(
        seed=seed,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=24,
        train_steps=2,
        refinement_steps=1,
        batch_size=4,
        reference_per_task=1,
        calibration_per_task=1,
        test_per_task=1,
        attainable_steps=1,
        attainable_restarts=1,
        random_masks=2,
        cl_tasks=2,
        cl_steps_per_task=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=arc_artifact_dir("03_exploratory", "controlled"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()
    config = (
        _smoke_config(args.seed) if args.smoke else HardControlledConfig(seed=args.seed)
    )
    if args.train_steps is not None:
        config = HardControlledConfig(
            **{**asdict(config), "train_steps": args.train_steps}
        )
    result, states = run_competence_pilot(config)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.artifact_dir / f"controlled_v4_premise_seed{args.seed}.json"
    save_json(metrics_path, result)
    torch.save(
        {
            "states": states,
            "configuration": asdict(config),
            "mixtures": hard_task_mixtures(),
        },
        args.artifact_dir / f"controlled_v4_premise_seed{args.seed}.pt",
    )
    print(f"CONTROLLED_V4_PREMISE_RESULT={metrics_path}")


if __name__ == "__main__":
    main()
