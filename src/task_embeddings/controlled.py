from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from .common import (
    EPS,
    OnlineAccumulator,
    PosthocAccumulator,
    TangentModule,
    WallTimer,
    arc_artifact_dir,
    cluster_fidelity,
    embedding_fidelity,
    layer_balanced_indices,
    normalized_rows,
    query_scores,
    random_layer_balanced_scores,
    save_json,
    seed_everything,
    select_evenly,
    tangent_kernel_metrics,
)


def task_mixtures() -> torch.Tensor:
    rows: list[list[float]] = []
    for i in range(4):
        row = [0.0] * 4
        row[i] = 1.0
        rows.append(row)
    for i in range(4):
        for j in range(i + 1, 4):
            row = [0.0] * 4
            row[i] = row[j] = 0.5
            rows.append(row)
    for omitted in range(4):
        row = [1.0 / 3.0] * 4
        row[omitted] = 0.0
        rows.append(row)
    rows.extend([[0.6, 0.2, 0.2, 0.0], [0.0, 0.2, 0.2, 0.6]])
    return torch.tensor(rows)


class Teachers(nn.Module):
    def __init__(self, input_dim: int = 32, hidden: int = 64, experts: int = 4) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(9917)
        self.w1 = nn.Parameter(
            torch.randn(experts, hidden, input_dim, generator=generator)
            / math.sqrt(input_dim),
            requires_grad=False,
        )
        self.b1 = nn.Parameter(
            torch.randn(experts, hidden, generator=generator) * 0.1, requires_grad=False
        )
        self.w2 = nn.Parameter(
            torch.randn(experts, hidden, generator=generator) / math.sqrt(hidden),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(torch.einsum("bi,ehi->beh", x, self.w1) + self.b1)
        return torch.einsum("beh,eh->be", hidden, self.w2)


class GatedMLP(nn.Module):
    def __init__(self, input_dim: int = 48, width: int = 512, depth: int = 5) -> None:
        super().__init__()
        dims = [input_dim] + [width] * depth
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in pairwise(dims)])
        self.output = nn.Linear(width, 1)
        self.masks: list[torch.Tensor | None] = [None] * depth
        self.activations: list[torch.Tensor] = []

    @property
    def layer_sizes(self) -> list[int]:
        return [layer.out_features for layer in self.layers]

    @property
    def n_modules(self) -> int:
        return sum(self.layer_sizes)

    def set_ablation(self, selected: Sequence[torch.Tensor] | None) -> None:
        if selected is None:
            self.masks = [None] * len(self.layers)
            return
        masks: list[torch.Tensor] = []
        device = next(self.parameters()).device
        for size, indices in zip(self.layer_sizes, selected):
            mask = torch.ones(size, device=device)
            mask[indices.to(device)] = 0
            masks.append(mask)
        self.masks = masks

    def forward(self, x: torch.Tensor, *, capture: bool = False) -> torch.Tensor:
        self.activations = []
        for layer, mask in zip(self.layers, self.masks):
            x = torch.nn.functional.gelu(layer(x))
            if mask is not None:
                x = x * mask
            if capture:
                x.retain_grad()
                self.activations.append(x)
        return self.output(x).squeeze(-1)

    def block_grad_norms(self) -> torch.Tensor:
        pieces = []
        for layer in self.layers:
            score = layer.weight.grad.float().square().sum(dim=1)
            if layer.bias is not None:
                score = score + layer.bias.grad.float().square()
            pieces.append(score)
        return torch.cat(pieces)

    def block_weight_norms(self) -> torch.Tensor:
        pieces = []
        for layer in self.layers:
            score = layer.weight.detach().float().square().sum(dim=1)
            if layer.bias is not None:
                score = score + layer.bias.detach().float().square()
            pieces.append(score.sqrt())
        return torch.cat(pieces).cpu()

    def block_grad_rows(self, global_indices: Sequence[int]) -> list[torch.Tensor]:
        result = []
        offsets = np.cumsum([0] + self.layer_sizes)
        for index in global_indices:
            layer_id = int(np.searchsorted(offsets[1:], index, side="right"))
            local = index - int(offsets[layer_id])
            layer = self.layers[layer_id]
            parts = [layer.weight.grad[local].detach().float()]
            if layer.bias is not None:
                parts.append(layer.bias.grad[local : local + 1].detach().float())
            result.append(torch.cat(parts))
        return result


def make_batch(
    teachers: Teachers,
    alpha: torch.Tensor,
    task: int,
    batch_size: int,
    device: torch.device,
    noise: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, 32, generator=generator, device="cpu").to(device)
    with torch.no_grad():
        experts = teachers(x)
        y = experts @ alpha[task]
        if noise:
            y = y + noise * torch.randn(
                batch_size, generator=generator, device="cpu"
            ).to(device)
    task_ids = torch.zeros(batch_size, 16, device=device)
    task_ids[:, task] = 1
    return torch.cat([x, task_ids], dim=1), y


def make_balanced_set(
    teachers: Teachers,
    alpha: torch.Tensor,
    per_task: int,
    device: torch.device,
    noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    xs, ys, tasks = [], [], []
    for task in range(16):
        x, y = make_batch(teachers, alpha, task, per_task, device, noise, generator)
        xs.append(x.cpu())
        ys.append(y.cpu())
        tasks.append(torch.full((per_task,), task))
    return torch.cat(xs), torch.cat(ys), torch.cat(tasks)


@torch.no_grad()
def evaluate_losses(
    model: GatedMLP,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    selected: Sequence[torch.Tensor] | None = None,
    batch_size: int = 2048,
) -> torch.Tensor:
    model.set_ablation(selected)
    x, y, task = data
    losses = torch.zeros(16, device=device)
    counts = torch.zeros(16, device=device)
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        tb = task[start : start + batch_size].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            pred = model(xb)
            sample_loss = 0.5 * (pred - yb).square()
        losses.scatter_add_(0, tb, sample_loss.float())
        counts.scatter_add_(0, tb, torch.ones_like(sample_loss, dtype=torch.float32))
    model.set_ablation(None)
    return (losses / counts.clamp_min(1)).cpu()


def train(
    model: GatedMLP,
    teachers: Teachers,
    alpha: torch.Tensor,
    representations: dict[str, torch.Tensor],
    device: torch.device,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[dict, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    accumulator = OnlineAccumulator(
        model.n_modules,
        representations,
        device,
        beta=0.99,
        late_fraction=0.7,
        total_steps=steps,
    )
    generator = torch.Generator().manual_seed(seed + 17)
    task_order = torch.randint(0, 16, (steps,), generator=generator)
    losses = []
    accumulation_seconds = 0.0
    timer = WallTimer()
    model.train()
    for step, task_tensor in enumerate(task_order):
        task = int(task_tensor)
        x, y = make_batch(teachers, alpha, task, batch_size, device, 0.05, generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            pred = model(x)
            residual = pred - y
            loss = 0.5 * residual.square().mean()
        loss.backward()
        t0 = time.perf_counter()
        scores = model.block_grad_norms()
        residual_norm_sq = residual.detach().float().square().sum() / (batch_size**2)
        accumulator.update(scores, task, residual_norm_sq, step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        accumulation_seconds += time.perf_counter() - t0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            losses.append({"step": step, "loss": float(loss.detach())})
    elapsed = timer.elapsed()
    return accumulator.finalize(), {
        "wall_seconds": elapsed,
        "steps": steps,
        "examples_per_second": steps * batch_size / elapsed,
        "accumulator_seconds_including_sync": accumulation_seconds,
        "accumulator_fraction_upper_bound": accumulation_seconds / elapsed,
        "loss_trace": losses,
    }


def posthoc(
    model: GatedMLP,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    representations: dict[str, torch.Tensor],
    device: torch.device,
    tangent_per_task: int = 2,
) -> tuple[dict, dict, dict, list[TangentModule], torch.Tensor]:
    accumulator = PosthocAccumulator(model.n_modules, representations, device)
    selected_modules = select_evenly(model.layer_sizes, per_layer=8)
    tangent_rows: dict[int, list[torch.Tensor]] = {idx: [] for idx in selected_modules}
    tangent_tasks: list[int] = []
    tangent_seen = [0] * 16
    x_all, y_all, task_all = reference
    timer = WallTimer()
    model.eval()
    for x_cpu, y_cpu, task_tensor in zip(x_all, y_all, task_all):
        task = int(task_tensor)
        x = x_cpu[None].to(device)
        y = y_cpu[None].to(device)
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            pred = model(x, capture=True)
            residual = pred - y
            loss = 0.5 * residual.square().sum()
        loss.backward()
        raw = model.block_grad_norms()
        residual_norm_sq = residual.detach().float().square().sum()
        ief = raw / residual_norm_sq.clamp_min(EPS)
        activation = torch.cat(
            [a.detach().float().abs().squeeze(0) for a in model.activations]
        )
        actgrad = torch.cat(
            [
                (a.detach().float() * a.grad.detach().float()).abs().squeeze(0)
                for a in model.activations
            ]
        )
        accumulator.update(
            task, ief=ief, raw=raw, activation=activation, actgrad=actgrad
        )
        if tangent_seen[task] < tangent_per_task:
            scale = residual_norm_sq.sqrt().clamp_min(EPS)
            for idx, row in zip(
                selected_modules, model.block_grad_rows(selected_modules)
            ):
                tangent_rows[idx].append((row / scale).cpu())
            tangent_tasks.append(task)
            tangent_seen[task] += 1
    elapsed = timer.elapsed()
    embeddings, amplitudes = accumulator.finalize()
    modules = [
        TangentModule(name=f"module_{idx}", gradients=torch.stack(rows))
        for idx, rows in tangent_rows.items()
    ]
    return (
        embeddings,
        amplitudes,
        {
            "wall_seconds": elapsed,
            "samples": len(x_all),
            "samples_per_second": len(x_all) / elapsed,
            "tangent_modules": len(modules),
            "tangent_samples": len(tangent_tasks),
        },
        modules,
        torch.tensor(tangent_tasks),
    )


def causal_study(
    model: GatedMLP,
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    methods: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> dict:
    baseline = evaluate_losses(model, test_data, device)
    records = []
    for method_name, score_matrix in methods.items():
        for fraction in (0.01, 0.05, 0.10):
            for task in range(16):
                selected = layer_balanced_indices(
                    score_matrix[:, task], model.layer_sizes, fraction
                )
                losses = evaluate_losses(model, test_data, device, selected)
                delta = losses - baseline
                non_target = torch.cat([delta[:task], delta[task + 1 :]])
                records.append(
                    {
                        "method": method_name,
                        "fraction": fraction,
                        "task": task,
                        "target_delta_loss": float(delta[task]),
                        "nontarget_delta_loss": float(non_target.mean()),
                        "selective_drop": float(delta[task] - non_target.mean()),
                    }
                )
    summary = {}
    for method in methods:
        summary[method] = {}
        for fraction in (0.01, 0.05, 0.10):
            values = [
                r["selective_drop"]
                for r in records
                if r["method"] == method and r["fraction"] == fraction
            ]
            targets = [
                r["target_delta_loss"]
                for r in records
                if r["method"] == method and r["fraction"] == fraction
            ]
            summary[method][str(fraction)] = {
                "selective_drop_mean": float(np.mean(values)),
                "selective_drop_std": float(np.std(values)),
                "target_delta_loss_mean": float(np.mean(targets)),
            }
    return {"baseline_task_losses": baseline, "records": records, "summary": summary}


def individual_ablation_correlation(
    model: GatedMLP,
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    score_matrix: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    baseline = evaluate_losses(model, test_data, device)
    sampled = select_evenly(model.layer_sizes, per_layer=6)
    predicted, observed = [], []
    offsets = np.cumsum([0] + model.layer_sizes)
    for index in sampled:
        layer = int(np.searchsorted(offsets[1:], index, side="right"))
        local = index - int(offsets[layer])
        selected = [torch.empty(0, dtype=torch.long) for _ in model.layer_sizes]
        selected[layer] = torch.tensor([local])
        delta = evaluate_losses(model, test_data, device, selected) - baseline
        predicted.extend(score_matrix[index].tolist())
        observed.extend(delta.tolist())
    return {
        "sampled_modules": len(sampled),
        "module_task_spearman": float(spearmanr(predicted, observed).statistic),
    }


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    alpha = task_mixtures()
    onehot = torch.eye(16)
    latent = normalized_rows(alpha)
    random_rep = normalized_rows(
        torch.randn(16, 4, generator=torch.Generator().manual_seed(73))
    )
    representations = {"onehot": onehot, "latent4": latent, "random4": random_rep}
    teachers = Teachers().to(device)
    model = GatedMLP(width=args.width, depth=args.depth).to(device)

    online, train_metrics = train(
        model,
        teachers,
        alpha.to(device),
        representations,
        device,
        args.steps,
        args.batch_size,
        args.seed,
    )
    test_data = make_balanced_set(
        teachers, alpha.to(device), args.test_per_task, device, 0.05, args.seed + 100
    )
    reference = tuple(
        t[: 16 * args.reference_per_task]
        for t in make_balanced_set(
            teachers,
            alpha.to(device),
            args.reference_per_task,
            device,
            0.05,
            args.seed + 200,
        )
    )
    post, amplitudes, post_metrics, tangent_modules, tangent_tasks = posthoc(
        model, reference, representations, device
    )

    fidelity = {}
    for variant in online:
        fidelity[variant] = {}
        for rep_name, representation in representations.items():
            fidelity[variant][rep_name] = embedding_fidelity(
                online[variant][rep_name], post["ief"][rep_name], representation
            )
            if rep_name == "onehot":
                fidelity[variant][rep_name].update(
                    cluster_fidelity(online[variant][rep_name], post["ief"][rep_name])
                )
    tangent = tangent_kernel_metrics(tangent_modules, tangent_tasks, representations)
    weight = model.block_weight_norms()
    method_scores = {
        "post_ief_onehot": query_scores(post["ief"]["onehot"], onehot),
        "post_ief_latent4": query_scores(post["ief"]["latent4"], latent),
        "post_raw_latent4": query_scores(post["raw"]["latent4"], latent),
        "train_late_raw_latent4": query_scores(online["late_raw"]["latent4"], latent),
        "train_late_ief_latent4": query_scores(online["late_ief"]["latent4"], latent),
        "activation_latent4": query_scores(post["activation"]["latent4"], latent),
        "actgrad_latent4": query_scores(post["actgrad"]["latent4"], latent),
        "weight_norm": weight[:, None].repeat(1, 16),
        "random": torch.stack(
            [
                random_layer_balanced_scores(model.layer_sizes, args.seed + task)
                for task in range(16)
            ],
            dim=1,
        ),
    }
    causal = causal_study(model, test_data, method_scores, device, args.seed)
    individual = individual_ablation_correlation(
        model, test_data, method_scores["post_ief_latent4"], device
    )

    checkpoint = artifact_dir / f"controlled_seed{args.seed}.pt"
    torch.save(
        {"model": model.state_dict(), "alpha": alpha, "args": vars(args)}, checkpoint
    )
    arrays = {}
    for statistic, reps in post.items():
        for name, value in reps.items():
            arrays[f"post_{statistic}_{name}"] = value.numpy()
    for variant, reps in online.items():
        for name, value in reps.items():
            arrays[f"online_{variant}_{name}"] = value.numpy()
    for name, value in amplitudes.items():
        arrays[f"amplitude_{name}"] = value.numpy()
    np.savez_compressed(
        artifact_dir / f"controlled_embeddings_seed{args.seed}.npz", **arrays
    )

    compression_overlap = {}
    for fraction in (0.01, 0.05, 0.10):
        overlaps = []
        for task in range(16):
            a = layer_balanced_indices(
                method_scores["post_ief_onehot"][:, task], model.layer_sizes, fraction
            )
            b = layer_balanced_indices(
                method_scores["post_ief_latent4"][:, task], model.layer_sizes, fraction
            )
            a_global, b_global, offset = set(), set(), 0
            for aa, bb, size in zip(a, b, model.layer_sizes):
                a_global.update((aa + offset).tolist())
                b_global.update((bb + offset).tolist())
                offset += size
            overlaps.append(len(a_global & b_global) / max(1, len(a_global)))
        compression_overlap[str(fraction)] = float(np.mean(overlaps))

    metrics = {
        "setting": "controlled",
        "seed": args.seed,
        "device": str(device),
        "configuration": vars(args),
        "train": train_metrics,
        "posthoc": post_metrics,
        "final_test_loss_mean": float(evaluate_losses(model, test_data, device).mean()),
        "proxy_fidelity": fidelity,
        "kernel_fidelity": tangent,
        "causal": causal,
        "individual_ablation": individual,
        "compression_topk_overlap": compression_overlap,
        "checkpoint": str(checkpoint),
    }
    save_json(artifact_dir / f"controlled_metrics_seed{args.seed}.json", metrics)
    print(
        f"CONTROLLED_RESULT={artifact_dir / f'controlled_metrics_seed{args.seed}.json'}"
    )
    print(json_summary(metrics))


def json_summary(metrics: dict) -> str:
    primary = metrics["causal"]["summary"]
    return str(
        {
            "loss": metrics["final_test_loss_mean"],
            "kernel": metrics["kernel_fidelity"],
            "sel_5pct": {
                k: round(v["0.05"]["selective_drop_mean"], 6)
                for k, v in primary.items()
            },
            "individual_spearman": metrics["individual_ablation"][
                "module_task_spearman"
            ],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default=str(arc_artifact_dir("01_exploratory", "controlled"))
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--reference-per-task", type=int, default=32)
    parser.add_argument("--test-per-task", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
