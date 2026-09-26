from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def selective_effect(effects: torch.Tensor, target: int) -> dict[str, float]:
    values = effects.detach().float().cpu()
    others = torch.cat((values[:target], values[target + 1 :]))
    target_effect = float(values[target])
    nontarget_effect = float(others.mean())
    return {
        "target_effect": target_effect,
        "nontarget_effect": nontarget_effect,
        "selectivity": target_effect - nontarget_effect,
    }


def continual_learning_metrics(
    history: torch.Tensor,
    *,
    higher_is_better: bool,
    pre_update_per_task: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize a square stage-by-task matrix with unseen entries as NaN."""
    values = history.detach().float().cpu()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("history must be a square stage-by-task matrix")
    n_tasks = values.shape[0]
    if not bool(torch.isfinite(torch.diagonal(values)).all()):
        raise ValueError("every task needs a finite post-acquisition observation")
    future = torch.triu(values, diagonal=1)
    future_mask = torch.triu(torch.ones_like(values, dtype=torch.bool), diagonal=1)
    if bool(torch.isfinite(future[future_mask]).any()):
        raise ValueError("future-task history entries must be NaN")
    forgetting = []
    for task in range(n_tasks):
        learned = values[task:, task]
        learned = learned[torch.isfinite(learned)]
        # Forgetting requires at least one observation after the task was
        # learned. In particular, the final task is not a zero-forgetting
        # observation: it is ineligible because no later training occurred.
        if learned.numel() < 2 or not torch.isfinite(values[-1, task]):
            forgetting.append(float("nan"))
            continue
        best = learned.max() if higher_is_better else learned.min()
        final = values[-1, task]
        change = best - final if higher_is_better else final - best
        forgetting.append(float(change))
    eligible_forgetting = [value for value in forgetting if math.isfinite(value)]
    diagonal = torch.diagonal(values)
    result: dict[str, Any] = {
        "average_forgetting": (
            float(np.mean(eligible_forgetting)) if eligible_forgetting else float("nan")
        ),
        "per_task_forgetting": forgetting,
        "forgetting_eligible_tasks": len(eligible_forgetting),
        "final_average": float(torch.nanmean(values[-1])),
        "new_task_plasticity": float(torch.nanmean(diagonal)),
    }
    if pre_update_per_task is not None:
        before = pre_update_per_task.detach().float().cpu()
        if before.shape != diagonal.shape or not bool(torch.isfinite(before).all()):
            raise ValueError("pre-update scores must be one finite value per task")
        acquisition = diagonal - before if higher_is_better else before - diagonal
        result["per_task_acquisition_gain"] = [float(value) for value in acquisition]
        result["average_acquisition_gain"] = float(acquisition.mean())
    else:
        result["per_task_acquisition_gain"] = None
        result["average_acquisition_gain"] = float("nan")
    return result


def faithfulness_scores(
    *,
    kept_divergence: torch.Tensor,
    dropped_divergence: torch.Tensor,
    null_divergence: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Normalize circuit sufficiency/necessity against an all-module null."""
    kept = kept_divergence.detach().float().cpu()
    dropped = dropped_divergence.detach().float().cpu()
    if kept.shape != dropped.shape:
        raise ValueError("keep and drop curves must share one shape")
    denominator = torch.as_tensor(null_divergence, dtype=torch.float32).cpu()
    if denominator.numel() != 1 or not torch.isfinite(denominator):
        raise ValueError("null divergence must be one finite scalar")
    if float(denominator) <= 0:
        raise ValueError("null divergence must be positive")
    return {
        "sufficiency": 1.0 - kept / denominator,
        "necessity": dropped / denominator,
    }


def causal_assay_gate(
    *,
    null_divergence: float,
    attainable_sufficiency: float,
    random_sufficiency_p95: float,
    minimum_span: float,
    minimum_gap: float,
) -> dict[str, Any]:
    """Check that a circuit assay has a measurable, independently found signal."""
    reasons: list[str] = []
    if not math.isfinite(null_divergence) or null_divergence < minimum_span:
        reasons.append("insufficient_intact_to_null_span")
    if (
        not math.isfinite(attainable_sufficiency)
        or not math.isfinite(random_sufficiency_p95)
        or attainable_sufficiency - random_sufficiency_p95 < minimum_gap
    ):
        reasons.append("attainable_reference_not_above_random")
    return {
        "passed": not reasons,
        "status_if_failed": "inconclusive",
        "reasons": reasons,
        "attainable_minus_random_p95": (
            attainable_sufficiency - random_sufficiency_p95
        ),
    }


def continual_learning_assay_gate(
    *,
    no_protection_forgetting: float,
    no_protection_acquisition_gain: float,
    minimum_forgetting: float,
    minimum_acquisition_gain: float,
) -> dict[str, Any]:
    """Reject CL comparisons with no interference signal or no acquisition."""
    reasons: list[str] = []
    if (
        not math.isfinite(no_protection_forgetting)
        or no_protection_forgetting < minimum_forgetting
    ):
        reasons.append("negligible_forgetting")
    if (
        not math.isfinite(no_protection_acquisition_gain)
        or no_protection_acquisition_gain < minimum_acquisition_gain
    ):
        reasons.append("insufficient_acquisition")
    return {
        "passed": not reasons,
        "status_if_failed": "inconclusive",
        "reasons": reasons,
    }


def layer_balanced_overlap(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
    layer_sizes: Sequence[int],
) -> float:
    if len(first) != len(second) or len(first) != len(layer_sizes):
        raise ValueError("one selection is required per layer")
    first_global: set[int] = set()
    second_global: set[int] = set()
    offset = 0
    for left, right, size in zip(first, second, layer_sizes):
        first_global.update((left.detach().cpu() + offset).tolist())
        second_global.update((right.detach().cpu() + offset).tolist())
        offset += size
    union_scale = max(1, min(len(first_global), len(second_global)))
    return len(first_global & second_global) / union_scale
