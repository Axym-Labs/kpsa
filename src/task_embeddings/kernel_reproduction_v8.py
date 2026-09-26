"""Reproduce the retained vision result through the v8 kernel-mean interface."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .common import save_json
from .kernel_sensitivity import (
    LinearKernelMap,
    fit_empirical_rbf_index,
    fit_nystrom_rbf,
    fit_weighted_kernel_mean_index,
    median_squared_distance,
    rbf_kernel,
)
from .refined_analysis import paired_t_summary
from .representation_sensitivity import profile_ranking_metrics


def _selected_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in (
            "mean_spearman",
            "mean_topk_recall",
            "mean_ndcg",
            "mean_cosine",
        )
    }


def summarize_causal_records(records: list[dict]) -> dict[str, object]:
    """Summarize correct-query effects and paired contrasts by intervention budget."""
    grouped: dict[tuple[str, float], dict[int, float]] = defaultdict(dict)
    for row in records:
        if row["target_correct"]:
            grouped[(row["method"], float(row["requested_parameter_fraction"]))][
                int(row["query_column"])
            ] = float(row["degradation"])
    methods: dict[str, dict[str, object]] = defaultdict(dict)
    paired_vs_scalar: dict[str, dict[str, object]] = defaultdict(dict)
    paired_vs_exact: dict[str, dict[str, object]] = defaultdict(dict)
    exact_name = "sample_dinov2_rbf_scale_0.1"
    for (method, fraction), values in grouped.items():
        label = f"{fraction:g}"
        methods[method][label] = paired_t_summary(list(values.values()))
        for baseline, output in (
            ("scalar_mass", paired_vs_scalar),
            (exact_name, paired_vs_exact),
        ):
            reference = grouped.get((baseline, fraction), {})
            paired = [
                value - reference[query]
                for query, value in values.items()
                if query in reference
            ]
            if len(paired) >= 2 and method != baseline:
                output[method][label] = paired_t_summary(paired)
    return {
        "correct_queries": len(
            {query for values in grouped.values() for query in values}
        ),
        "methods": dict(methods),
        "paired_vs_scalar": dict(paired_vs_scalar),
        "paired_vs_exact_rbf": dict(paired_vs_exact),
    }


def reproduce(
    payload: dict[str, torch.Tensor],
    *,
    scale: float,
    nystrom_rank: int,
    seed: int,
    legacy_result: dict | None = None,
    causal_result: dict | None = None,
) -> dict[str, object]:
    source_weights = payload["source_profiles"].double()
    target_weights = payload["target_profiles"].double()
    source_features = payload["source_features"].double()
    target_features = payload["target_features"].double()
    median_distance = median_squared_distance(source_features)
    bandwidth_squared = median_distance * scale**2

    exact_index = fit_empirical_rbf_index(
        source_weights,
        source_features,
        bandwidth_squared=bandwidth_squared,
    )
    exact_mass_weighted = exact_index.query(target_features, mass_weighted=True)
    distribution_match = exact_index.query(target_features, mass_weighted=False)

    legacy_kernel = rbf_kernel(
        source_features,
        target_features,
        bandwidth_squared=bandwidth_squared,
    )
    legacy_formula = source_weights @ legacy_kernel / source_weights.shape[1]
    equality_error = (exact_mass_weighted - legacy_formula).abs()

    linear_map = LinearKernelMap(source_features.shape[1], normalize_inputs=True)
    linear_index = fit_weighted_kernel_mean_index(
        source_weights.T,
        source_features,
        linear_map,
        weight_mode="normalized",
    )
    linear_scores = linear_index.query(target_features, mass_weighted=True)

    nystrom_map = fit_nystrom_rbf(
        source_features,
        nystrom_rank,
        bandwidth_squared=bandwidth_squared,
        landmark_method="kmeans++",
        seed=seed,
        eigenvalue_floor=1e-6,
    )
    nystrom_index = fit_weighted_kernel_mean_index(
        source_weights.T,
        source_features,
        nystrom_map,
        weight_mode="normalized",
    )
    nystrom_scores = nystrom_index.query(target_features, mass_weighted=True)
    scalar_scores = exact_index.mass[:, None].expand_as(exact_mass_weighted)

    methods = {
        "exact_rbf_mass_weighted_Q": exact_mass_weighted,
        "exact_rbf_distribution_match_R": distribution_match,
        "kmeanspp_nystrom_mass_weighted_Q": nystrom_scores,
        "linear_mass_weighted_Q": linear_scores,
        "scalar_mass": scalar_scores,
    }
    metrics = {
        name: _selected_metrics(profile_ranking_metrics(scores, target_weights))
        for name, scores in methods.items()
    }
    query_fidelity = _selected_metrics(
        profile_ranking_metrics(nystrom_scores, exact_mass_weighted)
    )
    residual = nystrom_scores - exact_mass_weighted
    query_fidelity.update(
        {
            "rmse": float(residual.square().mean().sqrt()),
            "relative_frobenius_error": float(
                residual.norm() / exact_mass_weighted.norm().clamp_min(1e-30)
            ),
        }
    )

    result: dict[str, object] = {
        "setting": "v8_kernel_mean_reproduction_from_retained_vitb16_confirmation",
        "status": "reproduced" if float(equality_error.max()) < 1e-12 else "mismatch",
        "protocol": {
            "groups": source_weights.shape[0],
            "reference_samples": source_weights.shape[1],
            "heldout_queries": target_weights.shape[1],
            "representation_dimension": source_features.shape[1],
            "weight_mode": "normalized",
            "kernel": "RBF on L2-normalized DINOv2 representations",
            "legacy_scale": scale,
            "median_squared_distance": median_distance,
            "bandwidth_squared": bandwidth_squared,
            "nystrom_rank": nystrom_rank,
            "nystrom_landmarks": "kmeans++ centroids",
            "seed": seed,
        },
        "legacy_formula_equivalence": {
            "max_absolute_error": float(equality_error.max()),
            "mean_absolute_error": float(equality_error.mean()),
        },
        "cold_query": metrics,
        "nystrom_query_fidelity_to_exact_rbf": query_fidelity,
    }
    if legacy_result is not None:
        legacy_name = f"sample_dinov2_rbf_scale_{scale:g}"
        expected = {
            key: float(legacy_result["cold_query"][legacy_name][key])
            for key in (
                "mean_spearman",
                "mean_topk_recall",
                "mean_ndcg",
                "mean_cosine",
            )
        }
        observed = metrics["exact_rbf_mass_weighted_Q"]
        result["legacy_metric_comparison"] = {
            "expected": expected,
            "observed": observed,
            "absolute_differences": {
                key: abs(observed[key] - expected[key]) for key in expected
            },
        }
    if causal_result is not None:
        result["fresh_causal_reproduction"] = summarize_causal_records(
            causal_result["records"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--legacy-json", type=Path)
    parser.add_argument("--causal-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--nystrom-rank", type=int, default=200)
    parser.add_argument("--seed", type=int, default=23_041)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    legacy = json.loads(args.legacy_json.read_text()) if args.legacy_json else None
    causal = json.loads(args.causal_json.read_text()) if args.causal_json else None
    result = reproduce(
        payload,
        scale=args.scale,
        nystrom_rank=args.nystrom_rank,
        seed=args.seed,
        legacy_result=legacy,
        causal_result=causal,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
