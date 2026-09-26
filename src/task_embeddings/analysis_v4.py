from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

from .common import arc_artifact_dir, save_json

_CANONICAL_METHOD_NAMES = {
    "semantic6": "tbe6",
    "permuted_semantic6": "permuted_task_basis6",
    "posthoc_ief__semantic6": "tbe_ief",
    "posthoc_ief__onehot": "full_atlas_ief",
    "posthoc_ief__permuted_semantic6": "permuted_task_basis_ief",
    "posthoc_ief__jl6_mean": "jl6_ief_mean",
    "posthoc_raw_ef__semantic6": "tbe_raw_ef",
    "online_raw_ef__semantic6": "online_tbe_raw_ef",
    "posthoc_activation__semantic6": "tbe_activation",
    "posthoc_activation_gradient__semantic6": "tbe_activation_gradient",
    "module_iewc__semantic6": "tbe_module_iewc",
}


def canonical_method_name(raw_name: str) -> str:
    """Map archival v4 identifiers to the provisional Task-Basis name."""
    if raw_name in _CANONICAL_METHOD_NAMES:
        return _CANONICAL_METHOD_NAMES[raw_name]
    prefix = "posthoc_ief__jl6_seed"
    if raw_name.startswith(prefix):
        return f"jl6_ief_seed{raw_name.removeprefix(prefix)}"
    return raw_name


def resource_accounting(
    n_modules: int,
    n_tasks: int,
    n_features: int,
) -> dict[str, Any]:
    """Analytical float32 storage and dense-query operation counts.

    The compact representation stores both the module embedding ``E`` and the
    task-feature table ``Phi``. A multiply-accumulate (MAC) count describes the
    algebraic query cost; it is not a latency claim.
    """
    if min(n_modules, n_tasks, n_features) <= 0:
        raise ValueError("module, task, and feature counts must be positive")
    full_floats = n_modules * n_tasks
    embedding_floats = n_modules * n_features
    feature_floats = n_tasks * n_features
    compact_floats = embedding_floats + feature_floats
    return {
        "shape": {
            "modules": n_modules,
            "tasks": n_tasks,
            "task_features": n_features,
        },
        "full_atlas": {
            "stored_floats": full_floats,
            "bytes_float32": 4 * full_floats,
        },
        "task_basis": {
            "embedding_floats": embedding_floats,
            "task_feature_floats": feature_floats,
            "stored_floats": compact_floats,
            "bytes_float32": 4 * compact_floats,
        },
        "storage_compression_ratio": full_floats / compact_floats,
        "storage_reduction_fraction": 1.0 - compact_floats / full_floats,
        "seen_task_query_macs": {
            "full_atlas": 0,
            "task_basis": embedding_floats,
        },
        "dense_composition_query_macs": {
            "full_atlas": full_floats,
            "task_basis": embedding_floats,
        },
        "task_basis_build_macs": n_modules * n_tasks * n_features,
        "scope": (
            "index and task-feature payload only; excludes the model and the "
            "cost of acquiring the full importance atlas"
        ),
    }


def paired_benefit_retention(
    *,
    baseline: float,
    compact: float,
    full: float,
    higher_is_better: bool,
) -> float | None:
    """Fraction of the full reference's improvement over a baseline retained."""
    if higher_is_better:
        denominator = full - baseline
        numerator = compact - baseline
    else:
        denominator = baseline - full
        numerator = baseline - compact
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def within_primitive_spearman(
    records: list[dict[str, Any]],
    predictor: str,
    *,
    outcome: str = "necessity",
) -> dict[str, Any]:
    """Spearman correlation per primitive, avoiding pooled intercept effects."""
    primitive_ids = sorted({int(record["primitive_circuit"]) for record in records})
    correlations = []
    for primitive in primitive_ids:
        local = [
            record
            for record in records
            if int(record["primitive_circuit"]) == primitive
        ]
        x = np.asarray([record[predictor] for record in local], dtype=float)
        y = np.asarray([record[outcome] for record in local], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
            correlations.append(float("nan"))
        else:
            correlations.append(float(spearmanr(x, y).statistic))
    finite_correlations = [value for value in correlations if math.isfinite(value)]
    return {
        "per_primitive": correlations,
        "mean": (
            float(np.mean(finite_correlations)) if finite_correlations else float("nan")
        ),
        "n_primitives": len(finite_correlations),
    }


def benefit_retention_summary(
    *,
    baselines: list[float],
    compact: list[float],
    full: list[float],
    higher_is_better: bool,
) -> dict[str, Any]:
    """Summarize paired seed-level compact/full benefit ratios."""
    if not (len(baselines) == len(compact) == len(full)):
        raise ValueError("baseline, compact, and full values must be paired")
    ratios = [
        paired_benefit_retention(
            baseline=baseline,
            compact=compact_value,
            full=full_value,
            higher_is_better=higher_is_better,
        )
        for baseline, compact_value, full_value in zip(baselines, compact, full)
    ]
    finite = [value for value in ratios if value is not None and math.isfinite(value)]
    if not finite:
        return {
            "mean": None,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_independent_seeds": 0,
            "evidence_strength": "undefined",
            "per_seed": ratios,
        }
    result = mean_sd_ci(finite)
    result["per_seed"] = ratios
    return result


def structure_diagnostic_summary(
    natural: list[dict[str, Any]],
    *,
    predictors: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate exploratory circuit diagnostics at the model-seed level."""
    correlations_by_outcome: dict[str, Any] = {}
    for outcome in ("necessity", "dropped_divergence"):
        correlations: dict[str, Any] = {}
        for predictor in predictors:
            seed_values = [
                within_primitive_spearman(
                    item["cross_task_structure"]["cross_task_records"],
                    predictor,
                    outcome=outcome,
                )["mean"]
                for item in natural
            ]
            correlations[predictor] = mean_sd_ci(seed_values)
            correlations[predictor]["per_seed"] = seed_values
        correlations_by_outcome[f"{outcome}_correlations"] = correlations
    overlap_keys = (
        "mean_pairwise_overlap_fraction",
        "mean_task_agnostic_overlap_fraction",
        "union_fraction_of_modules",
    )
    overlap = {
        key: mean_sd_ci(
            [
                item["cross_task_structure"]["pure_circuit_overlap"][key]
                for item in natural
            ]
        )
        for key in overlap_keys
    }
    return {
        **correlations_by_outcome,
        "pure_circuit_overlap": overlap,
        "analysis_level": (
            "Spearman is computed separately within each primitive circuit "
            "across tasks, averaged across primitives within seed, then "
            "summarized across model seeds"
        ),
        "status": "exploratory; predictors were inspected after the weak mixture-weight result",
    }


def mean_sd_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        raise ValueError("at least one value is required")
    mean = float(array.mean())
    if array.size < 2:
        return {
            "mean": mean,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_independent_seeds": int(array.size),
            "evidence_strength": "suggestive",
        }
    sd = float(array.std(ddof=1))
    half_width = float(t.ppf(0.975, array.size - 1) * sd / math.sqrt(array.size))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "n_independent_seeds": int(array.size),
        "evidence_strength": "repeated",
    }


def _load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _planted_source(planted: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fixed = (("semantic6", "tbe6"), ("task_agnostic", "task_agnostic"))
    for raw_method, output_method in fixed:
        result[output_method] = {
            metric: mean_sd_ci(
                [
                    item["cross_validated_task_query"][raw_method][metric]
                    for item in planted
                ]
            )
            for metric in ("spearman_mean", "ndcg_mean", "topk_recall_mean")
        }
    for label, prefix in (
        ("jl6", "jl6_seed"),
        ("permuted_task_basis6", "permuted_semantic6_seed"),
    ):
        result[label] = {}
        for metric in ("spearman_mean", "ndcg_mean", "topk_recall_mean"):
            seed_means = []
            for item in planted:
                rows = item["cross_validated_task_query"]
                seed_means.append(
                    float(
                        np.mean(
                            [
                                value[metric]
                                for key, value in rows.items()
                                if key.startswith(prefix)
                            ]
                        )
                    )
                )
            result[label][metric] = mean_sd_ci(seed_means)
    result["onehot"] = {
        "queryable_tasks": [
            item["cross_validated_task_query"]["onehot"]["queryable_tasks"]
            for item in planted
        ],
        "status": "N/A for held-out query",
    }
    semantic = [
        item["cross_validated_task_query"]["semantic6"]["spearman_mean"]
        for item in planted
    ]
    jl = []
    permuted = []
    for item in planted:
        rows = item["cross_validated_task_query"]
        jl.append(
            float(
                np.mean(
                    [
                        value["spearman_mean"]
                        for key, value in rows.items()
                        if key.startswith("jl6_seed")
                    ]
                )
            )
        )
        permuted.append(
            float(
                np.mean(
                    [
                        value["spearman_mean"]
                        for key, value in rows.items()
                        if key.startswith("permuted_semantic6_seed")
                    ]
                )
            )
        )
    result["paired_differences"] = {
        "tbe_minus_jl_spearman": mean_sd_ci(
            [left - right for left, right in zip(semantic, jl)]
        ),
        "tbe_minus_permuted_task_basis_spearman": mean_sd_ci(
            [left - right for left, right in zip(semantic, permuted)]
        ),
        "tbe_minus_task_agnostic_spearman": mean_sd_ci(
            [
                item["cross_validated_task_query"]["semantic6"]["spearman_mean"]
                - item["cross_validated_task_query"]["task_agnostic"]["spearman_mean"]
                for item in planted
            ]
        ),
    }
    return result


def _curve_rows(
    planted: list[dict[str, Any]], section: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = (
        "requested_fraction"
        if section == "causal_interpretability"
        else "retained_fraction_requested"
    )
    metric = "sufficiency"
    values: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    for item in planted:
        seed = item["seed"]
        for record in item[section]["records"]:
            values[(record["method"], float(record[key]), seed)].append(record[metric])
    rows = []
    nested: dict[str, Any] = defaultdict(dict)
    method_fractions = sorted({(method, fraction) for method, fraction, _ in values})
    for method, fraction in method_fractions:
        seed_means = [
            float(np.mean(values[(method, fraction, seed)]))
            for seed in sorted({item["seed"] for item in planted})
        ]
        summary = mean_sd_ci(seed_means)
        output_method = canonical_method_name(method)
        nested[output_method][str(fraction)] = summary
        rows.append({"method": output_method, "fraction": fraction, **summary})
    return rows, dict(nested)


def natural_seed_method_mean(
    item: dict[str, Any],
    *,
    section: str,
    method: str,
    fraction: float,
    metric: str,
    require_premise: bool,
) -> float:
    key = (
        "requested_fraction"
        if section == "causal_interpretability"
        else "retained_fraction_requested"
    )
    records = item[section]["records"]
    if method == "posthoc_ief__jl6_mean":
        matches_method = lambda name: name.startswith("posthoc_ief__jl6_seed")
    else:
        matches_method = lambda name: name == method
    values = [
        float(record[metric])
        for record in records
        if matches_method(record["method"])
        and math.isclose(float(record[key]), fraction)
        and (not require_premise or record["premise_passed"])
    ]
    if not values:
        raise ValueError(f"no valid {section} records for {method} at {fraction}")
    return float(np.mean(values))


def natural_seed_method_auc(item: dict[str, Any], method: str) -> float:
    """Area under a seed's mean structured-pruning sufficiency curve."""
    records = item["pruning"]["records"]
    fractions = sorted(
        {float(record["retained_fraction_requested"]) for record in records}
    )
    values = [
        natural_seed_method_mean(
            item,
            section="pruning",
            method=method,
            fraction=fraction,
            metric="sufficiency",
            require_premise=False,
        )
        for fraction in fractions
    ]
    return float(np.trapezoid(values, fractions))


def _natural_curve_rows(
    natural: list[dict[str, Any]], section: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = (
        "requested_fraction"
        if section == "causal_interpretability"
        else "retained_fraction_requested"
    )
    raw_methods = {
        record["method"] for item in natural for record in item[section]["records"]
    }
    methods = sorted(
        name for name in raw_methods if not name.startswith("posthoc_ief__jl6_seed")
    ) + ["posthoc_ief__jl6_mean"]
    fractions = sorted(
        {float(record[key]) for item in natural for record in item[section]["records"]}
    )
    rows = []
    nested: dict[str, Any] = defaultdict(dict)
    require_premise = section == "causal_interpretability"
    for method in methods:
        for fraction in fractions:
            seed_means = [
                natural_seed_method_mean(
                    item,
                    section=section,
                    method=method,
                    fraction=fraction,
                    metric="sufficiency",
                    require_premise=require_premise,
                )
                for item in natural
            ]
            summary = mean_sd_ci(seed_means)
            if require_premise:
                valid_tasks = [
                    sum(
                        record["gate"]["passed"]
                        for record in item[section]["premise_records"]
                        if math.isclose(float(record["requested_fraction"]), fraction)
                    )
                    for item in natural
                ]
                summary["valid_tasks_per_seed"] = valid_tasks
            output_method = canonical_method_name(method)
            nested[output_method][str(fraction)] = summary
            rows.append({"method": output_method, "fraction": fraction, **summary})
    return rows, dict(nested)


def _natural_paired_differences(
    natural: list[dict[str, Any]], *, fraction: float
) -> dict[str, Any]:
    semantic = "posthoc_ief__semantic6"
    comparators = (
        "posthoc_ief__onehot",
        "posthoc_ief__permuted_semantic6",
        "posthoc_ief__jl6_mean",
        "posthoc_raw_ef__semantic6",
        "posthoc_activation_gradient__semantic6",
        "task_agnostic_mean",
        "random",
    )
    output = {}
    for comparator in comparators:
        differences = []
        for item in natural:
            left = natural_seed_method_mean(
                item,
                section="causal_interpretability",
                method=semantic,
                fraction=fraction,
                metric="sufficiency",
                require_premise=True,
            )
            right = natural_seed_method_mean(
                item,
                section="causal_interpretability",
                method=comparator,
                fraction=fraction,
                metric="sufficiency",
                require_premise=True,
            )
            differences.append(left - right)
        output[f"tbe_minus_{canonical_method_name(comparator)}__sufficiency"] = (
            mean_sd_ci(differences)
        )
    return output


def _cl_method_value(item: dict[str, Any], method: str, metric: str) -> float:
    if method != "posthoc_ief__jl6_mean":
        return float(item["methods"][method]["summary"][metric])
    return float(
        np.mean(
            [
                item["methods"][name]["summary"][metric]
                for name in (
                    "posthoc_ief__jl6",
                    "posthoc_ief__jl6_seed1",
                    "posthoc_ief__jl6_seed2",
                )
            ]
        )
    )


def _continual_summary(
    continual: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = [
        "none",
        "posthoc_ief__semantic6",
        "posthoc_ief__onehot",
        "posthoc_ief__jl6_mean",
        "posthoc_raw_ef__semantic6",
        "online_raw_ef__semantic6",
        "posthoc_activation__semantic6",
        "random",
        "ewc",
        "module_iewc__semantic6",
        "replay",
    ]
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in continual:
        by_seed[int(item["seed"])].append(item)
    if any(len(items) != 2 for items in by_seed.values()):
        raise ValueError("expected forward and reverse task orders for every seed")
    rows = []
    output: dict[str, Any] = {}
    for method in methods:
        output_method = canonical_method_name(method)
        output[output_method] = {}
        for metric in (
            "average_forgetting",
            "average_acquisition_gain",
            "final_average",
        ):
            independent_seed_means = [
                float(
                    np.mean([_cl_method_value(item, method, metric) for item in items])
                )
                for _, items in sorted(by_seed.items())
            ]
            summary = mean_sd_ci(independent_seed_means)
            output[output_method][metric] = summary
            rows.append({"method": output_method, "metric": metric, **summary})
    semantic = "posthoc_ief__semantic6"
    paired = {}
    for comparator in (
        "none",
        "posthoc_ief__onehot",
        "posthoc_ief__jl6_mean",
        "posthoc_raw_ef__semantic6",
        "replay",
    ):
        differences = []
        for _, items in sorted(by_seed.items()):
            differences.append(
                float(
                    np.mean(
                        [
                            _cl_method_value(item, semantic, "average_forgetting")
                            - _cl_method_value(item, comparator, "average_forgetting")
                            for item in items
                        ]
                    )
                )
            )
        paired[f"tbe_minus_{canonical_method_name(comparator)}__forgetting"] = (
            mean_sd_ci(differences)
        )
    output["paired_differences"] = paired
    output["premise_runs"] = [
        {
            "seed": item["seed"],
            "task_order": item["task_order"],
            "passed": item["premise_gate"]["passed"],
            "no_protection_forgetting": item["methods"]["none"]["summary"][
                "average_forgetting"
            ],
            "no_protection_acquisition_gain": item["methods"]["none"]["summary"][
                "average_acquisition_gain"
            ],
        }
        for item in continual
    ]
    return rows, output


def compact_utility_summary(
    natural: list[dict[str, Any]], continual: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare compact TBE-iEF utility with full-atlas iEF and a baseline."""
    tbe_method = "posthoc_ief__semantic6"
    full_method = "posthoc_ief__onehot"
    random_method = "random"
    causal_values = {
        label: [
            natural_seed_method_mean(
                item,
                section="causal_interpretability",
                method=method,
                fraction=0.10,
                metric="sufficiency",
                require_premise=True,
            )
            for item in natural
        ]
        for label, method in (
            ("tbe_ief", tbe_method),
            ("full_atlas_ief", full_method),
            ("random", random_method),
        )
    }
    pruning_values = {
        label: [natural_seed_method_auc(item, method) for item in natural]
        for label, method in (
            ("tbe_ief", tbe_method),
            ("full_atlas_ief", full_method),
            ("random", random_method),
        )
    }
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in continual:
        by_seed[int(item["seed"])].append(item)
    continual_values = {
        label: [
            float(
                np.mean(
                    [
                        _cl_method_value(item, method, "average_forgetting")
                        for item in items
                    ]
                )
            )
            for _, items in sorted(by_seed.items())
        ]
        for label, method in (
            ("tbe_ief", tbe_method),
            ("full_atlas_ief", full_method),
            ("no_protection", "none"),
        )
    }

    def with_retention(
        values: dict[str, list[float]],
        *,
        baseline_key: str,
        higher_is_better: bool,
    ) -> dict[str, Any]:
        return {
            **{key: mean_sd_ci(metric_values) for key, metric_values in values.items()},
            "benefit_retained_vs_full_atlas": benefit_retention_summary(
                baselines=values[baseline_key],
                compact=values["tbe_ief"],
                full=values["full_atlas_ief"],
                higher_is_better=higher_is_better,
            ),
        }

    return {
        "causal_10_percent": with_retention(
            causal_values, baseline_key="random", higher_is_better=True
        ),
        "pruning_curve_auc": with_retention(
            pruning_values, baseline_key="random", higher_is_better=True
        ),
        "continual_forgetting": with_retention(
            continual_values,
            baseline_key="no_protection",
            higher_is_better=False,
        ),
        "interpretation": (
            "Ratios are paired within model seed. Causal and pruning share the "
            "same keep/drop intervention family and are not independent breadth evidence."
        ),
    }


def aggregate(artifact_root: Path) -> dict[str, Any]:
    planted = [
        _load(path)
        for path in sorted(
            (artifact_root / "planted").glob("planted_v4_metrics_seed*.json")
        )
    ]
    continual_paths = sorted(
        list(
            (artifact_root / "continual").glob(
                "continual_v4_metrics_forward_seed*.json"
            )
        )
        + list(
            (artifact_root / "continual").glob(
                "continual_v4_metrics_reverse_seed*.json"
            )
        )
    )
    continual = [_load(path) for path in continual_paths]
    natural_paths = sorted(
        (artifact_root / "controlled").glob("controlled_v4_assay_seed*.json")
    )
    natural = [_load(path) for path in natural_paths]
    if len(planted) != 3 or len(continual) != 6 or len(natural) != 3:
        raise ValueError(
            "expected 3 planted, 6 continual, and 3 natural-assay runs; "
            f"got {len(planted)}, {len(continual)}, and {len(natural)}"
        )
    causal_rows, causal = _curve_rows(planted, "causal_interpretability")
    pruning_rows, pruning = _curve_rows(planted, "pruning")
    cl_rows, cl = _continual_summary(continual)
    natural_causal_rows, natural_causal = _natural_curve_rows(
        natural, "causal_interpretability"
    )
    natural_pruning_rows, natural_pruning = _natural_curve_rows(natural, "pruning")
    controlled_attempts = [
        _load(
            artifact_root
            / "controlled"
            / f"controlled_v4_premise_attempt{attempt}_seed1.json"
        )
        for attempt in (1, 2, 3)
    ]
    controlled_confirmatory = [
        _load(artifact_root / "controlled" / f"controlled_v4_premise_seed{seed}.json")
        for seed in (2, 3, 4)
    ]
    structure_diagnostics = structure_diagnostic_summary(
        natural,
        predictors=(
            "mixture_weight",
            "task_basis_profile_cosine",
            "full_atlas_profile_cosine",
            "task_basis_selected_score_mass",
            "full_atlas_selected_score_mass",
            "task_basis_topk_overlap",
            "task_cardinality",
            "null_divergence",
            "full_variance",
            "full_half_mse",
        ),
    )
    result = {
        "terminology": {
            "representation": "Task-Basis Embedding (TBE)",
            "main_method": "TBE-iEF",
            "status": "provisional internal name",
            "raw_schema_note": (
                "Per-run JSONs retain their original v4 method identifiers for "
                "artifact provenance; aggregate and CSV outputs use TBE names."
            ),
        },
        "run_counts": {
            "planted_model_seeds": len(planted),
            "natural_model_seeds": len(natural),
            "continual_model_seeds": 3,
            "continual_task_orders_per_seed": 2,
            "jl_projections_per_cl_run": 3,
            "jl_projections_per_source_seed": planted[0]["configuration"]["jl_repeats"],
        },
        "controlled_natural_premise": {
            "development_attempts": [
                {
                    "attempt": index,
                    "minimum_gap_recovered": item["competence"][
                        "minimum_gap_recovered"
                    ],
                    "tasks_passing_80_percent": item["competence"][
                        "tasks_passing_80_percent"
                    ],
                    "total_tasks": item["competence"]["total_tasks"],
                    "passed": item["competence"]["passed"],
                }
                for index, item in enumerate(controlled_attempts, start=1)
            ],
            "confirmatory_minimum_gap_recovered": mean_sd_ci(
                [
                    item["competence"]["minimum_gap_recovered"]
                    for item in controlled_confirmatory
                ]
            ),
            "all_confirmatory_tasks_passed": all(
                item["competence"]["passed"] for item in controlled_confirmatory
            ),
            "status": "passed after development-only sampling and optimization repair",
        },
        "planted_competence": {
            "minimum_gap_recovered": mean_sd_ci(
                [item["competence"]["minimum_null_gap_recovered"] for item in planted]
            ),
            "all_passed": all(item["competence"]["passed"] for item in planted),
        },
        "reference_reliability": {
            "mean_spearman": mean_sd_ci(
                [item["reference_reliability"]["mean_spearman"] for item in planted]
            ),
            "all_passed": all(
                item["reference_reliability"]["passed"] for item in planted
            ),
        },
        "cross_validated_task_query": _planted_source(planted),
        "natural_reference_reliability": {
            "mean_spearman": mean_sd_ci(
                [item["reference_reliability"]["mean_spearman"] for item in natural]
            ),
            "minimum_spearman": mean_sd_ci(
                [item["reference_reliability"]["minimum_spearman"] for item in natural]
            ),
            "all_passed": all(
                item["reference_reliability"]["passed"] for item in natural
            ),
        },
        "natural_cross_validated_task_query": _planted_source(natural),
        "held_out_query_definition": {
            "held_out_unit": "one triple-composition task",
            "index_training_data": (
                "reference-A iEF importance columns and task-feature vectors for "
                "the other 29 tasks"
            ),
            "query": (
                "the held-out task's six-dimensional primitive-mixture feature vector"
            ),
            "prediction": (
                "a length-M ranking of modules predicted without the held-out "
                "importance column"
            ),
            "target": (
                "the held-out task's independently measured reference-B iEF "
                "importance ranking"
            ),
            "metrics": "Spearman, NDCG, and top-5% recall over modules",
        },
        "natural_causal_interpretability": {
            "application_enabled_all_seeds": all(
                item["causal_interpretability"]["application_enabled"]
                for item in natural
            ),
            "curves": natural_causal,
            "paired_differences_at_10_percent": _natural_paired_differences(
                natural, fraction=0.10
            ),
            "premise_at_10_percent": {
                "attainable_sufficiency_all_cells": mean_sd_ci(
                    [
                        float(
                            np.mean(
                                [
                                    record["attainable"]["sufficiency"]
                                    for record in item["causal_interpretability"][
                                        "premise_records"
                                    ]
                                    if math.isclose(
                                        float(record["requested_fraction"]), 0.10
                                    )
                                ]
                            )
                        )
                        for item in natural
                    ]
                ),
                "random_sufficiency_p95_all_cells": mean_sd_ci(
                    [
                        float(
                            np.mean(
                                [
                                    record["random_sufficiency_p95"]
                                    for record in item["causal_interpretability"][
                                        "premise_records"
                                    ]
                                    if math.isclose(
                                        float(record["requested_fraction"]), 0.10
                                    )
                                ]
                            )
                        )
                        for item in natural
                    ]
                ),
                "all_tasks_passed_all_seeds": all(
                    record["gate"]["passed"]
                    for item in natural
                    for record in item["causal_interpretability"]["premise_records"]
                    if math.isclose(float(record["requested_fraction"]), 0.10)
                ),
                "valid_tasks_per_seed": [
                    sum(
                        record["gate"]["passed"]
                        for record in item["causal_interpretability"]["premise_records"]
                        if math.isclose(float(record["requested_fraction"]), 0.10)
                    )
                    for item in natural
                ],
                "attainable_sufficiency_valid_cells": mean_sd_ci(
                    [
                        float(
                            np.mean(
                                [
                                    record["attainable"]["sufficiency"]
                                    for record in item["causal_interpretability"][
                                        "premise_records"
                                    ]
                                    if math.isclose(
                                        float(record["requested_fraction"]), 0.10
                                    )
                                    and record["gate"]["passed"]
                                ]
                            )
                        )
                        for item in natural
                    ]
                ),
                "random_sufficiency_p95_valid_cells": mean_sd_ci(
                    [
                        float(
                            np.mean(
                                [
                                    record["random_sufficiency_p95"]
                                    for record in item["causal_interpretability"][
                                        "premise_records"
                                    ]
                                    if math.isclose(
                                        float(record["requested_fraction"]), 0.10
                                    )
                                    and record["gate"]["passed"]
                                ]
                            )
                        )
                        for item in natural
                    ]
                ),
            },
        },
        "natural_pruning": {
            "retention_identity_all_seeds": all(
                item["pruning"]["retention_identity_passed"] for item in natural
            ),
            "curves": natural_pruning,
        },
        "natural_cross_task_structure": {
            "mean_primitive_weight_necessity_spearman": mean_sd_ci(
                [
                    float(
                        np.mean(
                            item["cross_task_structure"][
                                "primitive_weight_necessity_spearman"
                            ]
                        )
                    )
                    for item in natural
                ]
            ),
            "per_primitive": {
                str(primitive): mean_sd_ci(
                    [
                        item["cross_task_structure"][
                            "primitive_weight_necessity_spearman"
                        ][primitive]
                        for item in natural
                    ]
                )
                for primitive in range(6)
            },
            "exploratory_diagnostics": structure_diagnostics,
            "status": (
                "descriptive; TBE-iEF circuit necessity does not consistently "
                "track known mixture weights"
            ),
        },
        "causal_interpretability": {
            "application_enabled_all_seeds": all(
                item["causal_interpretability"]["application_enabled"]
                for item in planted
            ),
            "curves": causal,
        },
        "pruning": {
            "retention_identity_all_seeds": all(
                item["pruning"]["retention_identity_passed"] for item in planted
            ),
            "curves": pruning,
        },
        "continual_learning": cl,
        "application_breadth": {
            "named_applications_enabled": 3,
            "named_applications": [
                "causal circuit ranking",
                "structured task-conditioned pruning",
                "continual-learning protection",
            ],
            "distinct_mechanism_families": 2,
            "mechanism_families": {
                "keep_drop_module_intervention": [
                    "causal circuit ranking",
                    "structured task-conditioned pruning",
                ],
                "parameter_protection": ["continual-learning protection"],
            },
            "scope": "controlled assays only; repaired real-model breadth is untested",
        },
        "compact_utility": compact_utility_summary(natural, continual),
        "resource_accounting": {
            "natural": resource_accounting(1536, 30, 6),
            "planted": resource_accounting(300, 30, 6),
            "continual": resource_accounting(256, 10, 6),
            "comparison_notes": {
                "jl": "matched-dimensional JL has the same payload and query MAC count as TBE",
                "seen_task": (
                    "full-atlas lookup uses no matrix-vector MACs, whereas TBE "
                    "reconstructs a task column in M*d MACs"
                ),
                "dense_or_unseen_task": (
                    "TBE uses M*d MACs versus M*T for a dense full-atlas "
                    "composition; one-hot cannot name a genuinely unseen task"
                ),
                "acquisition": (
                    "the current implementation first measures the full iEF "
                    "atlas, so no attribution-acquisition compute or peak-memory "
                    "saving has been demonstrated"
                ),
            },
        },
        "validity_matrix": [
            {
                "cell": "old source CKA/distance/kNN",
                "status": "invalid",
                "reason": "one-hot reference geometry structurally favors JL",
            },
            {
                "cell": "cross-fitted task query on planted positive control",
                "status": "valid within positive-control scope",
                "reason": "disjoint references, reliable target, repeated seeds/JL, no target-column leakage",
            },
            {
                "cell": "natural tagged-program causal/pruning",
                "status": "valid controlled natural assay",
                "reason": "three confirmatory seeds passed all 30 competence tasks; independent references, attainable-mask/random gates, held-out keep/drop curves, and 100% identity passed",
            },
            {
                "cell": "planted causal interpretation",
                "status": "valid positive control",
                "reason": "competence, reliability, attainable-reference/random, and held-out keep/drop gates passed",
            },
            {
                "cell": "planted structured pruning",
                "status": "valid positive control",
                "reason": "100% identity and non-saturated exact module-retention curves",
            },
            {
                "cell": "old continual learning",
                "status": "invalid",
                "reason": "joint warm start, future-task leakage, unmatched streams, omitted arms",
            },
            {
                "cell": "task-neutral modular continual learning",
                "status": "valid controlled assay",
                "reason": "premise passed in all seed/order runs; identical streams; seen-only scores; complete module parameter coverage",
            },
        ],
        "claim_ledger": [
            {
                "claim": "Task-basis features predict held-out task importance in the planted positive control better than matched JL or a permuted task basis.",
                "status": "supported within positive-control scope",
            },
            {
                "claim": "TBE-iEF uniquely enables causal circuit recovery or structured pruning.",
                "status": "not supported; full iEF, raw EF, and activation-gradient match or exceed it in the planted and natural controlled assays",
            },
            {
                "claim": "TBE identifies the known primitive organization in the natural controlled model.",
                "status": "not supported; task-agnostic and permuted-task-basis query rankings are nearly as strong, while primitive-circuit necessity does not consistently follow known mixture weights",
            },
            {
                "claim": "TBE-iEF enables useful continual-learning protection in the controlled assay.",
                "status": "supported; forgetting and final loss improve while acquisition remains nonzero",
            },
            {
                "claim": "The task basis or iEF normalization is responsible for the continual-learning gain.",
                "status": "not supported; semantic/full/JL/raw have unresolved paired differences, while plasticity-constrained replay is unstable across seeds/orders",
            },
            {
                "claim": "The repaired study supports a global paper-scale go/no-go decision.",
                "status": "not supported; the natural controlled result is valid but remains one synthetic model family, while repaired real-model evidence is absent",
            },
        ],
    }
    output = artifact_root
    pd.DataFrame(causal_rows).to_csv(output / "causal_curves.csv", index=False)
    pd.DataFrame(pruning_rows).to_csv(output / "pruning_curves.csv", index=False)
    pd.DataFrame(cl_rows).to_csv(output / "continual_summary.csv", index=False)
    pd.DataFrame(natural_causal_rows).to_csv(
        output / "natural_causal_curves.csv", index=False
    )
    pd.DataFrame(natural_pruning_rows).to_csv(
        output / "natural_pruning_curves.csv", index=False
    )
    diagnostic_rows = []
    diagnostics = structure_diagnostics
    for outcome_key in (
        "necessity_correlations",
        "dropped_divergence_correlations",
    ):
        for predictor, summary in diagnostics[outcome_key].items():
            diagnostic_rows.append(
                {
                    "outcome": outcome_key.removesuffix("_correlations"),
                    "predictor": predictor,
                    **{
                        key: value
                        for key, value in summary.items()
                        if key != "per_seed"
                    },
                    "per_seed": json.dumps(summary["per_seed"]),
                }
            )
    pd.DataFrame(diagnostic_rows).to_csv(
        output / "circuit_diagnostics.csv", index=False
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=arc_artifact_dir("03_exploratory")
    )
    args = parser.parse_args()
    result = aggregate(args.artifact_root)
    path = args.artifact_root / "aggregate_metrics.json"
    save_json(path, result)
    print(f"ANALYSIS_V4_RESULT={path}")


if __name__ == "__main__":
    main()
