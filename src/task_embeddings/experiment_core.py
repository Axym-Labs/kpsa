"""Shared experiment vocabulary and run contracts.

The module is deliberately small: experiment implementations own their scale
parameters, while this file standardizes comparison labels, evidence tiers,
and the distinction between queryable and algebraically reducible state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Estimator(str, Enum):
    RAW_OPG = "raw_opg"
    RESIDUAL_NORMALIZED_OPG = "residual_normalized_opg"


class Representation(str, Enum):
    FULL_ATLAS = "full_atlas"
    TBE = "tbe"
    JL = "jl"
    MEAN_ONLY = "mean_only"


@dataclass(frozen=True)
class RunProfile:
    name: str
    minimum_independent_seeds: int
    control_repeats: int
    claim_ready: bool


@dataclass(frozen=True)
class ComparisonCell:
    estimator: Estimator
    representation: Representation

    @property
    def key(self) -> str:
        return f"{self.estimator.value}/{self.representation.value}"


@dataclass(frozen=True)
class ApplicationContract:
    name: str
    requires_distinct_task_queries: bool
    primary_metrics: tuple[str, ...]


_RUN_PROFILES = {
    "explore": RunProfile("explore", 1, 1, False),
    # A requested budget is not evidence that validity/precision gates passed.
    "paper": RunProfile("paper", 3, 3, False),
}

_ESTIMATOR_ALIASES = {
    "raw": Estimator.RAW_OPG,
    "raw_ef": Estimator.RAW_OPG,
    "raw_opg": Estimator.RAW_OPG,
    "ief": Estimator.RESIDUAL_NORMALIZED_OPG,
    "normalized_opg": Estimator.RESIDUAL_NORMALIZED_OPG,
    "residual_normalized_opg": Estimator.RESIDUAL_NORMALIZED_OPG,
}

_APPLICATIONS = {
    "cold_start_retrieval": ApplicationContract(
        "cold_start_retrieval",
        True,
        ("residual_spearman", "residual_causal_spearman", "sufficiency"),
    ),
    "continual_learning": ApplicationContract(
        "continual_learning",
        False,
        ("matched_acquisition_forgetting", "acquisition_gain", "final_loss"),
    ),
    "optimizer": ApplicationContract(
        "optimizer",
        True,
        ("validation_loss", "steps_to_target", "optimizer_state_bytes"),
    ),
    "inference_control": ApplicationContract(
        "inference_control",
        True,
        ("target_effect", "off_target_spillover", "latency"),
    ),
    "structured_pruning": ApplicationContract(
        "structured_pruning",
        True,
        ("retained_quality_curve", "active_parameter_count"),
    ),
}


def run_profile(name: str) -> RunProfile:
    try:
        return _RUN_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown run profile {name!r}") from error


def canonical_estimator_name(name: str | Estimator) -> Estimator:
    if isinstance(name, Estimator):
        return name
    try:
        return _ESTIMATOR_ALIASES[name]
    except KeyError as error:
        raise ValueError(f"unknown estimator {name!r}") from error


def regular_comparison_grid() -> tuple[ComparisonCell, ...]:
    return tuple(
        ComparisonCell(estimator, representation)
        for estimator in Estimator
        for representation in Representation
    )


def application_contract(name: str) -> ApplicationContract:
    try:
        return _APPLICATIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown application {name!r}") from error
