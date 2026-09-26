"""Full parameter-diagonal action scores, reduced only after weighting.

For fixed pruning/quantization actions, averaging and group reduction commute.
This oracle therefore need not materialize an n-by-T parameter atlas. It is
action-specific, and is privileged when target-task gradients are withheld.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import seed_everything
from .domain_applications import load_checkpoint, quantize_weight
from .domain_optimizer import ParameterPartition
from .domain_train import batch_loss, logit_residual_norm


@torch.no_grad()
def weighted_diagonal_scores(partition, quant_error_gain):
    output = {
        action: torch.zeros_like(partition.sizes)
        for action in ("absolute", "pruning", "quantization")
    }
    for spec in partition.slices:
        if spec.parameter.grad is None:
            continue
        squared = spec.parameter.grad.float().square()
        factors = {
            "absolute": None,
            "pruning": spec.parameter.float().square(),
            "quantization": quant_error_gain[spec.name].float(),
        }
        for action, factor in factors.items():
            values = squared if factor is None else squared * factor
            dims = tuple(i for i in range(values.ndim) if i != spec.axis)
            reduced = values.sum(dim=dims) if dims else values
            output[action][spec.offset : spec.offset + spec.count] += reduced.reshape(
                -1
            )
    return {action: values / partition.sizes for action, values in output.items()}


def collect(
    checkpoint, data, output, samples=64, kind="swiglu", include_fisher=True, start=0
):
    model, config = load_checkpoint(checkpoint)
    seed_everything(config.seed + start * 100003)
    torch.set_num_threads(4)
    model.eval()
    kinds = ("row", "tensor", "swiglu") if kind == "all" else (kind,)
    partitions = {name: ParameterPartition(model, name) for name in kinds}
    corpus = torch.load(data / "corpus.pt", weights_only=False)
    errors = {}
    with torch.no_grad():
        for spec in next(iter(partitions.values())).slices:
            p = spec.parameter
            if p.ndim < 2:
                errors[spec.name] = torch.zeros_like(p)
            else:
                error = (p.float() - quantize_weight(p, 4).float()).square()
                error -= (p.float() - quantize_weight(p, 8).float()).square()
                errors[spec.name] = error.to(p.dtype)
    stats = ("raw", "normalized", "fisher") if include_fisher else ("raw", "normalized")
    result = {
        name: {
            stat: {
                action: torch.zeros(part.n_groups, len(corpus["train"]))
                for action in ("absolute", "pruning", "quantization")
            }
            for stat in stats
        }
        for name, part in partitions.items()
    }
    for task, blocks in enumerate(corpus["train"]):
        indices = torch.randperm(
            len(blocks),
            generator=torch.Generator().manual_seed(
                31415 + (0 if corpus.get("parallel_examples", False) else task)
            ),
        )[start : start + samples]
        if len(indices) != samples:
            raise ValueError("insufficient profiling samples")
        for block in blocks[indices]:
            model.zero_grad(set_to_none=True)
            batch = block[None].cuda().long()
            loss, logits = batch_loss(model, batch)
            norm = float(
                logit_residual_norm(logits.detach(), batch[:, 1:]).clamp_min(1e-12)
            )
            loss.backward(retain_graph=include_fisher)
            for name, part in partitions.items():
                scores = weighted_diagonal_scores(part, errors)
                for action, score in scores.items():
                    score = score.cpu()
                    result[name]["raw"][action][:, task] += score / samples
                    result[name]["normalized"][action][:, task] += score / (
                        samples * norm
                    )
            if include_fisher:
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    labels = torch.multinomial(
                        logits.detach().float().softmax(-1).flatten(0, 1), 1
                    ).reshape(batch[:, 1:].shape)
                    labels[batch[:, 1:] < 0] = -1
                sampled = torch.nn.functional.cross_entropy(
                    logits.float().flatten(0, 1), labels.flatten(), ignore_index=-1
                )
                sampled.backward()
                for name, part in partitions.items():
                    for action, score in weighted_diagonal_scores(part, errors).items():
                        result[name]["fisher"][action][:, task] += (
                            score.cpu() * int((labels >= 0).sum()) / samples
                        )
        print(f"DIAGONAL ACTION ORACLE {task + 1}/{len(corpus['train'])}", flush=True)
    torch.save(
        {
            "scores": result,
            "checkpoint": str(checkpoint.resolve()),
            "data": str(data.resolve()),
            "samples": samples,
            "start": start,
            "low_bits": 4,
            "quant_group_size": 128,
            "definition": "mean per-parameter squared gradient times action error, then group sum/size; normalization and bias excluded from quantization",
        },
        output,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--partition", choices=("row", "tensor", "swiglu", "all"), default="swiglu"
    )
    a = p.parse_args()
    collect(a.checkpoint, a.data, a.output, a.samples, a.partition, start=a.start)
