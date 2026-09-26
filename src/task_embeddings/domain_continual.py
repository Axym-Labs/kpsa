"""Sequential domain adaptation: acquisition/forgetting, not a task-atlas claim.

All methods use the same latest-model anchor. Exact diagonal and exact grouped
OPG consolidate by addition, so their deployed state is n and M, not n*T/M*T.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_applications import load_checkpoint, query_scores
from .domain_optimizer import ParameterPartition
from .domain_train import batch_loss, evaluate, logit_residual_norm


def task_sequence(total, count, seed):
    if not 1 <= count <= total:
        raise ValueError("sequence length must fit the task catalogue")
    g = torch.Generator().manual_seed(seed + 1200)
    pool = (
        torch.tensor([1, 3, 7, 10, 17, 19])
        if count == 6 and total == 20
        else torch.arange(total)
    )
    return pool[torch.randperm(len(pool), generator=g)[:count]].tolist()


def importance(model, blocks, partition, estimator, samples=16, diagonal=False):
    if len(blocks) < samples:
        raise ValueError("insufficient importance examples")
    result = (
        {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        if diagonal
        else torch.zeros(partition.n_groups, device="cuda")
    )
    model.eval()
    for block in blocks[:samples]:
        model.zero_grad(set_to_none=True)
        batch = block[None].cuda().long()
        loss, logits = batch_loss(model, batch)
        scale = (
            logit_residual_norm(logits.detach(), batch[:, 1:]).clamp_min(1e-12)
            if estimator == "normalized"
            else 1.0
        )
        loss.backward()
        if diagonal:
            for name, p in model.named_parameters():
                if p.grad is not None:
                    result[name].add_(p.grad.detach().square() / scale / samples)
        else:
            result.add_(partition.gradient_scores() / scale / samples)
    model.zero_grad(set_to_none=True)
    return result


def run_continual(
    checkpoint,
    data,
    output,
    method="none",
    partition_kind="row",
    estimator="raw",
    strength=0.0,
    steps=300,
    seed=11,
    eval_blocks=32,
    role="validation",
    samples=16,
    sequence_length=6,
    feature_file="task_features.pt",
):
    seed_everything(seed)
    torch.set_num_threads(4)
    corpus = torch.load(data / "corpus.pt", weights_only=False)
    features = torch.load(data / feature_file, weights_only=False)["features"]
    model, _ = load_checkpoint(checkpoint)
    partition = ParameterPartition(model, partition_kind)
    generator = torch.Generator().manual_seed(seed + 1200)
    # Six distinct domains; seed changes both initialization checkpoint and order.
    sequence = task_sequence(len(features), sequence_length, seed)
    # Preserve the original batch stream after drawing the ordering.
    torch.randperm(sequence_length, generator=generator)
    evaluation = [corpus[role][t] for t in sequence]
    train_data = [x.cuda().long() for x in corpus["train"]]
    initial = evaluate(model, evaluation, eval_blocks)
    matrix = []
    curves = []
    sum_importance = None
    anchor = None
    profiles = []
    protection = None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01, fused=True
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for stage, task in enumerate(sequence):
        before = evaluate(model, evaluation, eval_blocks)
        for step in range(steps):
            # Replay uses the same token budget: 1/2 current, 1/2 past domains.
            count = 8 if method == "replay" and stage else 16
            local = train_data[task]
            blocks = local[
                torch.randint(len(local), (count,), generator=generator).cuda()
            ]
            if count == 8:
                old = sequence[int(torch.randint(stage, (), generator=generator))]
                previous = train_data[old]
                blocks = torch.cat(
                    (
                        blocks,
                        previous[
                            torch.randint(
                                len(previous), (8,), generator=generator
                            ).cuda()
                        ],
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_loss(model, blocks)
            if protection is not None and strength:
                if method == "diagonal":
                    penalty = (
                        sum(
                            ((p - anchor[name]).square() * protection[name]).sum()
                            for name, p in model.named_parameters()
                        )
                        / partition.n_parameters
                    )
                else:
                    penalty = partition.regularization(anchor, protection)
                loss = loss + 0.5 * strength * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        after = evaluate(model, evaluation, eval_blocks)
        matrix.append(after.tolist())
        curves.append(
            {
                "stage": stage,
                "task": task,
                "acquisition_nll": float(before[stage] - after[stage]),
                "macro_nll": float(after.mean()),
            }
        )
        if method not in {"none", "replay"}:
            measured = importance(
                model,
                corpus["train"][task],
                partition,
                estimator,
                samples,
                method == "diagonal",
            )
            if method == "diagonal":
                if sum_importance is None:
                    sum_importance = measured
                else:
                    for name in measured:
                        sum_importance[name].add_(measured[name])
                mean = (
                    sum(v.sum() for v in sum_importance.values())
                    / partition.n_parameters
                )
                protection = {
                    name: value / mean.clamp_min(1e-20)
                    for name, value in sum_importance.items()
                }
            else:
                profiles.append(measured.cpu())
                atlas = torch.stack(profiles, 1)
                queried = query_scores(
                    atlas, features[sequence[: stage + 1]], method, seed + 800
                )
                aggregate = queried.sum(1).cuda()
                mean = (aggregate * partition.sizes).sum() / partition.n_parameters
                protection = aggregate / mean.clamp_min(1e-20)
            anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
        print(
            f"CL {method}/{partition_kind}/{estimator} lambda={strength:g} stage={stage + 1} nll={after.mean():.4f}",
            flush=True,
        )
    matrix = torch.tensor(matrix)
    # Forgetting from best observed performance after a domain was introduced.
    forgetting = [
        float(matrix[-1, t] - matrix[t:, t].min()) for t in range(len(sequence) - 1)
    ]
    result = {
        "method": method,
        "partition": partition_kind,
        "estimator": estimator,
        "strength": strength,
        "seed": seed,
        "sequence": sequence,
        "steps_per_task": steps,
        "feature_file": feature_file,
        "task_feature_dimension": features.shape[1],
        "importance_samples": samples,
        "checkpoint": str(checkpoint.resolve()),
        "evaluation_blocks": eval_blocks,
        "data_role": role,
        "initial_nll": initial.tolist(),
        "nll_matrix": matrix.tolist(),
        "curve": curves,
        "mean_acquisition_nll": sum(c["acquisition_nll"] for c in curves) / len(curves),
        "mean_forgetting_nll": sum(forgetting) / len(forgetting),
        "final_macro_nll": float(matrix[-1].mean()),
        "importance_floats_fair_baseline": partition.n_parameters
        if method == "diagonal"
        else partition.n_groups,
        "group_count": partition.n_groups,
        "parameter_count": partition.n_parameters,
        "task_sketch_floats_hypothetical": partition.n_groups * (features.shape[1] + 1)
        + features.numel()
        + features.shape[1],
        "anchor_floats": 0 if method in {"none", "replay"} else partition.n_parameters,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": time.perf_counter() - started,
        "claim_ready": False,
        "state_caveat": "screen materializes profiles; deployed group sum needs only M entries; TBE is unnecessary for this query",
    }
    save_json(output, result)
    return result


def screen_configs():
    configs = [{"method": m, "strength": 0.0} for m in ("none", "replay")]
    for strength in (1e3, 1e4, 1e5):
        for method in ("diagonal", "full", "tbe", "jl", "mean"):
            for estimator in ("raw", "normalized"):
                configs.append(
                    {"method": method, "strength": strength, "estimator": estimator}
                )
    return configs


def screen(checkpoint, data, output, **kwargs):
    output.mkdir(parents=True, exist_ok=True)
    configs = screen_configs()
    save_json(
        output / "protocol.json",
        {"configs": configs, "selection_role": "validation", "steps": 300, **kwargs},
    )
    for c in configs:
        target = (
            output / f"{c['method']}_{c.get('estimator', 'raw')}_{c['strength']:g}.json"
        )
        if target.exists():
            continue
        run_continual(checkpoint, data, target, **c, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
    records = [
        json.loads(p.read_text())
        for p in output.glob("*.json")
        if p.name not in {"protocol.json", "summary.json"}
    ]
    save_json(output / "summary.json", {"runs": records})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--screen", action="store_true")
    p.add_argument("--sequence-length", type=int, default=6)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--eval-blocks", type=int, default=32)
    p.add_argument("--feature-file", default="task_features.pt")
    p.add_argument("--partition", default="row", choices=("row", "tensor", "swiglu"))
    a = p.parse_args()
    kwargs = {
        "sequence_length": a.sequence_length,
        "steps": a.steps,
        "samples": a.samples,
        "eval_blocks": a.eval_blocks,
        "feature_file": a.feature_file,
        "partition_kind": a.partition,
    }
    if a.screen:
        screen(a.checkpoint, a.data, a.output, **kwargs)
    else:
        run_continual(a.checkpoint, a.data, a.output, **kwargs)
