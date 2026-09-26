from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearTBEIndex:
    module_mean: torch.Tensor
    feature_mean: torch.Tensor
    coefficients: torch.Tensor


def _solve_coefficients(
    cross_covariance: torch.Tensor,
    feature_gram: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    regularized = feature_gram + ridge * torch.eye(
        feature_gram.shape[0], dtype=feature_gram.dtype
    )
    return cross_covariance @ torch.linalg.pinv(regularized)


def fit_linear_tbe(
    importance: torch.Tensor,
    task_features: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> LinearTBEIndex:
    """Fit a mean-plus-linear TBE from a materialized task atlas."""
    values = importance.detach().float().cpu()
    features = task_features.detach().float().cpu()
    if values.ndim != 2 or features.ndim != 2:
        raise ValueError("importance and task_features must be matrices")
    if values.shape[1] != features.shape[0]:
        raise ValueError("importance task axis must match task feature rows")
    module_mean = values.mean(dim=1)
    feature_mean = features.mean(dim=0)
    centered_values = values - module_mean[:, None]
    centered_features = features - feature_mean[None, :]
    coefficients = _solve_coefficients(
        centered_values @ centered_features,
        centered_features.T @ centered_features,
        ridge,
    )
    return LinearTBEIndex(module_mean, feature_mean, coefficients)


def query_linear_tbe(
    index: LinearTBEIndex,
    task_features: torch.Tensor,
    *,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    """Query module scores for one or more task-feature rows."""
    features = task_features.detach().float().cpu()
    if features.ndim != 2 or features.shape[1] != index.feature_mean.numel():
        raise ValueError("task features have the wrong shape")
    centered = features - index.feature_mean[None, :]
    return index.module_mean[:, None] + residual_scale * index.coefficients @ centered.T


class TaskBlockedTBEAccumulator:
    """Build an equal-task-weighted linear TBE without storing an atlas.

    Samples must be presented as contiguous task blocks. Only the current
    task's module sum and the cross-task sufficient statistics are retained.
    """

    def __init__(self, n_modules: int, dimension: int) -> None:
        if n_modules < 1 or dimension < 1:
            raise ValueError("n_modules and dimension must be positive")
        self.n_modules = n_modules
        self.dimension = dimension
        self.module_sum = torch.zeros(n_modules)
        self.feature_sum = torch.zeros(dimension)
        self.module_feature_sum = torch.zeros(n_modules, dimension)
        self.feature_gram = torch.zeros(dimension, dimension)
        self.n_tasks = 0
        self._current_feature: torch.Tensor | None = None
        self._current_sum = torch.zeros(n_modules)
        self._current_count = 0

    def begin_task(self, feature: torch.Tensor) -> None:
        if self._current_feature is not None:
            raise RuntimeError("finish the current task before beginning another")
        value = feature.detach().float().cpu().flatten()
        if value.shape != (self.dimension,):
            raise ValueError("task feature has the wrong dimension")
        self._current_feature = value
        self._current_sum.zero_()
        self._current_count = 0

    def update(self, module_scores: torch.Tensor) -> None:
        if self._current_feature is None:
            raise RuntimeError("begin_task must be called before update")
        scores = module_scores.detach().float().cpu().flatten()
        if scores.shape != (self.n_modules,):
            raise ValueError("module score has the wrong dimension")
        self._current_sum.add_(scores)
        self._current_count += 1

    def end_task(self) -> None:
        if self._current_feature is None or self._current_count == 0:
            raise RuntimeError("a task must contain at least one sample")
        task_mean = self._current_sum / self._current_count
        feature = self._current_feature
        self.module_sum.add_(task_mean)
        self.feature_sum.add_(feature)
        self.module_feature_sum.add_(torch.outer(task_mean, feature))
        self.feature_gram.add_(torch.outer(feature, feature))
        self.n_tasks += 1
        self._current_feature = None
        self._current_count = 0

    def finalize(self, *, ridge: float = 1e-6) -> LinearTBEIndex:
        if self._current_feature is not None:
            raise RuntimeError("end the current task before finalizing")
        if self.n_tasks < 1:
            raise RuntimeError("at least one completed task is required")
        module_mean = self.module_sum / self.n_tasks
        feature_mean = self.feature_sum / self.n_tasks
        cross_covariance = self.module_feature_sum - self.n_tasks * torch.outer(
            module_mean, feature_mean
        )
        centered_gram = self.feature_gram - self.n_tasks * torch.outer(
            feature_mean, feature_mean
        )
        coefficients = _solve_coefficients(cross_covariance, centered_gram, ridge)
        return LinearTBEIndex(module_mean, feature_mean, coefficients)


def tbe_resource_counts(
    *, n_modules: int, n_tasks: int, dimension: int
) -> dict[str, int | float]:
    """Analytical float counts for full, post-hoc, and streamed construction."""
    if min(n_modules, n_tasks, dimension) < 1:
        raise ValueError("all dimensions must be positive")
    full = n_modules * n_tasks
    compact = n_modules * (dimension + 1) + n_tasks * dimension + dimension
    streaming_state = (
        n_modules * dimension
        + dimension * dimension
        + 2 * n_modules
        + 2 * dimension
        + 2
    )
    return {
        "full_atlas_stored_floats": full,
        "tbe_stored_floats": compact,
        "storage_reduction_fraction": 1.0 - compact / full,
        "posthoc_construction_state_floats": full + compact,
        "streaming_construction_state_floats": streaming_state,
        "streaming_conservative_peak_floats": streaming_state + compact,
    }
