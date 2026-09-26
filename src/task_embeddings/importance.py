"""Canonical importance estimators and representation comparisons."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .common import EPS
from .experiment_core import Estimator, Representation, canonical_estimator_name
from .streaming_v5 import fit_linear_tbe, query_linear_tbe


def transform_opg_trace(
    raw_trace: torch.Tensor,
    estimator: str | Estimator,
    *,
    residual_norm_squared: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Apply one named estimator transform to a raw diagonal-OPG trace."""
    kind = canonical_estimator_name(estimator)
    raw = raw_trace.detach().float()
    if kind is Estimator.RAW_OPG:
        return raw
    if residual_norm_squared is None:
        raise ValueError("residual normalization requires residual_norm_squared")
    denominator = torch.as_tensor(residual_norm_squared).detach().float()
    return raw / denominator.clamp_min(EPS)


def canonicalize_profile_mapping(
    profiles: Mapping[str | Estimator, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Read legacy keys while emitting only canonical estimator names."""
    result: dict[str, torch.Tensor] = {}
    for name, values in profiles.items():
        key = canonical_estimator_name(name).value
        if key in result:
            raise ValueError(f"duplicate estimator profile for {key!r}")
        result[key] = values
    return result


def build_regular_score_grid(
    profiles: Mapping[str | Estimator, torch.Tensor],
    task_features: torch.Tensor,
    *,
    jl_features: torch.Tensor,
    ridge: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Build the common estimator-by-representation matrix for seen tasks."""
    canonical = canonicalize_profile_mapping(profiles)
    features = task_features.detach().float().cpu()
    jl = jl_features.detach().float().cpu()
    if features.shape != jl.shape:
        raise ValueError("task and JL features must have matching shapes")
    scores: dict[str, torch.Tensor] = {}
    for estimator in Estimator:
        try:
            atlas = canonical[estimator.value].detach().float().cpu()
        except KeyError as error:
            raise ValueError(f"missing profile for {estimator.value!r}") from error
        if atlas.ndim != 2 or atlas.shape[1] != features.shape[0]:
            raise ValueError("profiles must have shape [modules, tasks]")
        tbe = query_linear_tbe(
            fit_linear_tbe(atlas, features, ridge=ridge), features
        ).clamp_min(0)
        jl_scores = query_linear_tbe(
            fit_linear_tbe(atlas, jl, ridge=ridge), jl
        ).clamp_min(0)
        mean = atlas.mean(dim=1, keepdim=True).repeat(1, atlas.shape[1])
        by_representation = {
            Representation.FULL_ATLAS: atlas,
            Representation.TBE: tbe,
            Representation.JL: jl_scores,
            Representation.MEAN_ONLY: mean,
        }
        for representation, values in by_representation.items():
            scores[f"{estimator.value}/{representation.value}"] = values
    return scores
