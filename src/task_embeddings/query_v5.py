from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .common import (
    EPS,
    _continuous_ndcg,
    _safe_corr,
    _validated_task_folds,
    cross_validated_task_scores,
    jl_task_representation,
    normalized_rows,
)


def fold_training_means(
    matrix: torch.Tensor,
    *,
    folds: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Return the training-column mean associated with every held-out task."""
    values = matrix.detach().float().cpu()
    if values.ndim != 2:
        raise ValueError("matrix must have shape [modules, tasks]")
    result = torch.full_like(values, float("nan"))
    for heldout in _validated_task_folds(values.shape[1], folds):
        train_mask = torch.ones(values.shape[1], dtype=torch.bool)
        train_mask[heldout] = False
        if not bool(train_mask.any()):
            continue
        result[:, heldout] = values[:, train_mask].mean(dim=1, keepdim=True)
    return result


def centered_linear_task_scores(
    importance: torch.Tensor,
    task_features: torch.Tensor,
    *,
    folds: Sequence[Sequence[int]],
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict held-out scores with an intercept and linear task residuals.

    Each fold fits ``importance = module_mean + coefficient @ task_feature``
    using training-task columns only. Centering makes the shared module ordering
    explicit instead of allowing it to masquerade as task-conditioned signal.
    """
    values = importance.detach().float().cpu()
    features = task_features.detach().float().cpu()
    if values.ndim != 2 or features.ndim != 2:
        raise ValueError("importance and task_features must be matrices")
    if values.shape[1] != features.shape[0]:
        raise ValueError("importance task axis must match task feature rows")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")

    predictions = torch.full_like(values, float("nan"))
    queryable = torch.zeros(values.shape[1], dtype=torch.bool)
    for heldout in _validated_task_folds(values.shape[1], folds):
        train_mask = torch.ones(values.shape[1], dtype=torch.bool)
        train_mask[heldout] = False
        if not bool(train_mask.any()):
            continue
        train_values = values[:, train_mask]
        train_features = features[train_mask]
        value_mean = train_values.mean(dim=1, keepdim=True)
        feature_mean = train_features.mean(dim=0, keepdim=True)
        centered_values = train_values - value_mean
        centered_features = train_features - feature_mean
        if ridge == 0:
            coefficients = torch.linalg.lstsq(
                centered_features, centered_values.T
            ).solution.T
        else:
            gram = centered_features.T @ centered_features
            regularized = gram + ridge * torch.eye(gram.shape[0])
            coefficients = (
                centered_values @ centered_features @ torch.linalg.pinv(regularized)
            )
        query_features = features[heldout] - feature_mean
        predictions[:, heldout] = value_mean + coefficients @ query_features.T
        queryable[heldout] = True
    return predictions, queryable


def _ranking_summary(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    tasks: list[int],
    queryable: torch.Tensor,
    top_count: int,
) -> dict[str, Any]:
    records = []
    for task in tasks:
        candidate = predictions[:, task]
        truth = targets[:, task]
        valid = bool(queryable[task]) and bool(torch.isfinite(candidate).all())
        nonconstant = valid and float(candidate.max() - candidate.min()) > EPS
        if not nonconstant:
            records.append(
                {
                    "task": task,
                    "queryable": valid,
                    "spearman": float("nan"),
                    "ndcg": float("nan"),
                    "topk_recall": float("nan"),
                }
            )
            continue
        predicted_top = set(torch.topk(candidate, top_count).indices.tolist())
        truth_top = set(torch.topk(truth, top_count).indices.tolist())
        records.append(
            {
                "task": task,
                "queryable": True,
                "spearman": _safe_corr(candidate.numpy(), truth.numpy(), "spearman"),
                "ndcg": _continuous_ndcg(candidate, truth),
                "topk_recall": len(predicted_top & truth_top) / top_count,
            }
        )

    def macro(key: str) -> float:
        finite = [record[key] for record in records if math.isfinite(record[key])]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        "spearman_mean": macro("spearman"),
        "ndcg_mean": macro("ndcg"),
        "topk_recall_mean": macro("topk_recall"),
        "per_task": records,
    }


def query_ranking_metrics(
    predictions: torch.Tensor,
    source_importance: torch.Tensor,
    target_importance: torch.Tensor,
    *,
    folds: Sequence[Sequence[int]],
    queryable: torch.Tensor,
    top_fraction: float = 0.05,
) -> dict[str, Any]:
    """Report raw and task-mean-residualized held-out ranking fidelity."""
    predicted = predictions.detach().float().cpu()
    source = source_importance.detach().float().cpu()
    target = target_importance.detach().float().cpu()
    if predicted.shape != source.shape or target.shape != source.shape:
        raise ValueError("predictions, source, and target must share one shape")
    if queryable.shape != (source.shape[1],):
        raise ValueError("queryable must have one entry per task")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0, 1]")
    validated_folds = _validated_task_folds(source.shape[1], folds)
    tasks = [task for fold in validated_folds for task in fold]
    source_means = fold_training_means(source, folds=validated_folds)
    target_means = fold_training_means(target, folds=validated_folds)
    top_count = max(1, math.ceil(source.shape[0] * top_fraction))
    raw = _ranking_summary(predicted, target, tasks, queryable, top_count)
    residual = _ranking_summary(
        predicted - source_means,
        target - target_means,
        tasks,
        queryable,
        top_count,
    )
    return {
        "raw": raw,
        "residual": residual,
        "topk_fraction": top_fraction,
        "queryable_tasks": int(queryable[tasks].sum()),
        "total_tasks": len(tasks),
    }


def _scale_label(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def build_query_methods(
    importance: torch.Tensor,
    task_features: torch.Tensor,
    *,
    folds: Sequence[Sequence[int]],
    seed: int,
    repeats: int,
    residual_scales: Sequence[float],
    ridge: float = 1e-6,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Construct the cold-start method matrix with matched compact controls."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not residual_scales or any(scale <= 0 for scale in residual_scales):
        raise ValueError("residual scales must be positive")
    values = importance.detach().float().cpu()
    features = task_features.detach().float().cpu()
    means = fold_training_means(values, folds=folds)
    heldout_queryable = torch.isfinite(means).all(dim=0)
    linear, linear_queryable = centered_linear_task_scores(
        values, features, folds=folds, ridge=ridge
    )
    kernel, kernel_queryable = cross_validated_task_scores(
        values, normalized_rows(features), folds=folds
    )
    methods: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "mean_only": (means, heldout_queryable),
        "tbe_kernel": (kernel, kernel_queryable),
    }
    for scale in residual_scales:
        label = _scale_label(float(scale))
        methods[f"tbe_linear_scale_{label}"] = (
            means + float(scale) * (linear - means),
            linear_queryable,
        )
    for repeat in range(repeats):
        jl_features = jl_task_representation(
            features.shape[0], features.shape[1], seed + 3000 + repeat
        )
        jl_linear, jl_queryable = centered_linear_task_scores(
            values, jl_features, folds=folds, ridge=ridge
        )
        permutation = torch.randperm(
            features.shape[0],
            generator=torch.Generator().manual_seed(seed + 4000 + repeat),
        )
        permuted_linear, permuted_queryable = centered_linear_task_scores(
            values, features[permutation], folds=folds, ridge=ridge
        )
        for scale in residual_scales:
            label = _scale_label(float(scale))
            methods[f"jl_linear_scale_{label}_seed{repeat}"] = (
                means + float(scale) * (jl_linear - means),
                jl_queryable,
            )
            methods[f"permuted_basis_linear_scale_{label}_seed{repeat}"] = (
                means + float(scale) * (permuted_linear - means),
                permuted_queryable,
            )
        methods[f"random_seed{repeat}"] = (
            torch.rand(
                values.shape,
                generator=torch.Generator().manual_seed(seed + 5000 + repeat),
            ),
            heldout_queryable,
        )
    methods["observed_opg_oracle"] = (values.clone(), heldout_queryable)
    return methods


def select_residual_scale(
    source_importance: torch.Tensor,
    target_importance: torch.Tensor,
    task_features: torch.Tensor,
    *,
    task_indices: Sequence[int],
    residual_scales: Sequence[float],
    top_fraction: float,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Select residual strength by inner held-out-task top-k fidelity."""
    indices = [int(task) for task in task_indices]
    if len(indices) < 3 or len(set(indices)) != len(indices):
        raise ValueError("task_indices must contain at least three unique tasks")
    source = source_importance.detach().float().cpu()
    target = target_importance.detach().float().cpu()
    features = task_features.detach().float().cpu()
    if source.shape != target.shape or source.shape[1] != features.shape[0]:
        raise ValueError("importance matrices and task features must align")
    if any(task < 0 or task >= source.shape[1] for task in indices):
        raise ValueError("task index outside the available task range")
    local_source = source[:, indices]
    local_target = target[:, indices]
    local_features = features[indices]
    folds = [[task] for task in range(len(indices))]
    means = fold_training_means(local_source, folds=folds)
    linear, queryable = centered_linear_task_scores(
        local_source, local_features, folds=folds, ridge=ridge
    )
    scale_results: dict[str, Any] = {}
    candidates = []
    for scale in residual_scales:
        value = float(scale)
        if value <= 0:
            raise ValueError("residual scales must be positive")
        metrics = query_ranking_metrics(
            means + value * (linear - means),
            local_source,
            local_target,
            folds=folds,
            queryable=queryable,
            top_fraction=top_fraction,
        )["raw"]
        label = _scale_label(value)
        scale_results[label] = metrics
        score = (
            metrics["topk_recall_mean"],
            metrics["ndcg_mean"],
            metrics["spearman_mean"],
            -value,
        )
        candidates.append((score, value))
    selected = max(candidates, key=lambda item: item[0])[1]
    return {
        "selected_scale": selected,
        "selection_metric": "raw_topk_recall",
        "tie_breakers": ["raw_ndcg", "raw_spearman", "smaller_scale"],
        "task_indices": indices,
        "scale_results": scale_results,
    }
