from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .common import (
    EPS,
    accelerator_peak_memory,
    build_task_atlas,
    conditional_importance,
    reset_accelerator_peak_memory,
    save_json,
    seed_everything,
)
from .controlled_assay_v4 import _model_config_from_checkpoint
from .controlled_v3 import ControlledTransformer
from .controlled_v4 import build_model, hard_task_mixtures, make_program_dataset
from .streaming_v5 import (
    LinearTBEIndex,
    TaskBlockedTBEAccumulator,
    fit_linear_tbe,
    query_linear_tbe,
    tbe_resource_counts,
)


def _normalized_opg_trace(
    model: ControlledTransformer,
    token_row: torch.Tensor,
    target: torch.Tensor,
    task: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    prediction = model(token_row[None].to(device), task[None].to(device), capture=True)
    residual = prediction - target.to(device)
    (0.5 * residual.square().sum()).backward()
    raw = model.block_grad_norms().detach().float().cpu()
    residual_norm_sq = residual.detach().float().square().sum().cpu()
    return raw / residual_norm_sq.clamp_min(EPS)


def collect_full_opg_atlas(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tokens, targets, task_ids = dataset
    n_tasks = int(task_ids.max()) + 1
    rows = []
    started = time.perf_counter()
    reset_accelerator_peak_memory(device)
    model.eval()
    model.set_intervention(None)
    model.set_feature_gates(None)
    for token_row, target, task in zip(tokens, targets, task_ids):
        rows.append(_normalized_opg_trace(model, token_row, target, task, device))
    atlas, amplitude = build_task_atlas(torch.stack(rows), task_ids, n_tasks=n_tasks)
    importance = conditional_importance(atlas, amplitude)
    model.clear_capture()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return importance, {
        "wall_seconds": time.perf_counter() - started,
        "sample_buffer_floats": len(rows) * model.n_modules,
        **accelerator_peak_memory(device),
    }


def collect_streaming_opg_index(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    task_features: torch.Tensor,
    device: torch.device,
    *,
    ridge: float,
) -> tuple[LinearTBEIndex, dict[str, Any]]:
    tokens, targets, task_ids = dataset
    features = task_features.detach().float().cpu()
    if features.shape[0] != int(task_ids.max()) + 1:
        raise ValueError("task features must cover every dataset task")
    accumulator = TaskBlockedTBEAccumulator(model.n_modules, features.shape[1])
    started = time.perf_counter()
    reset_accelerator_peak_memory(device)
    model.eval()
    model.set_intervention(None)
    model.set_feature_gates(None)
    for task in range(features.shape[0]):
        selected = torch.where(task_ids == task)[0]
        if not selected.numel():
            raise ValueError(f"dataset contains no samples for task {task}")
        accumulator.begin_task(features[task])
        for index in selected:
            accumulator.update(
                _normalized_opg_trace(
                    model,
                    tokens[index],
                    targets[index],
                    task_ids[index],
                    device,
                )
            )
        accumulator.end_task()
    index = accumulator.finalize(ridge=ridge)
    model.clear_capture()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return index, {
        "wall_seconds": time.perf_counter() - started,
        "sample_buffer_floats": 0,
        **accelerator_peak_memory(device),
    }


def compare_construction(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    task_features: torch.Tensor,
    device: torch.device,
    *,
    ridge: float,
) -> dict[str, Any]:
    full, posthoc_timing = collect_full_opg_atlas(model, dataset, device)
    posthoc = fit_linear_tbe(full, task_features, ridge=ridge)
    streaming, streaming_timing = collect_streaming_opg_index(
        model, dataset, task_features, device, ridge=ridge
    )
    posthoc_scores = query_linear_tbe(posthoc, task_features)
    streaming_scores = query_linear_tbe(streaming, task_features)
    denominator = posthoc_scores.flatten().norm() * streaming_scores.flatten().norm()
    cosine = (
        float(
            torch.dot(posthoc_scores.flatten(), streaming_scores.flatten())
            / denominator
        )
        if float(denominator) > EPS
        else float("nan")
    )
    if math.isfinite(cosine):
        cosine = min(1.0, max(-1.0, cosine))
    return {
        "prediction_max_abs": float((posthoc_scores - streaming_scores).abs().max()),
        "prediction_cosine": cosine,
        "posthoc": posthoc_timing,
        "streaming": streaming_timing,
        "resource_accounting": tbe_resource_counts(
            n_modules=full.shape[0],
            n_tasks=full.shape[1],
            dimension=task_features.shape[1],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-per-task", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = _model_config_from_checkpoint(payload)
    if model_config.seed != args.seed:
        raise ValueError("experiment seed must match checkpoint seed")
    model = build_model(model_config, device)
    model.load_state_dict(payload["states"]["trained"])
    features = hard_task_mixtures()
    dataset = make_program_dataset(
        model_config,
        features,
        args.reference_per_task,
        args.seed + 4001,
    )
    result = {
        "seed": args.seed,
        "device": str(device),
        "model_configuration": asdict(model_config),
        "reference_per_task": args.reference_per_task,
        "ridge": args.ridge,
        **compare_construction(model, dataset, features, device, ridge=args.ridge),
    }
    save_json(args.output, result)
    print(f"PAPER_STREAMING_V5_RESULT={args.output}")


if __name__ == "__main__":
    main()
