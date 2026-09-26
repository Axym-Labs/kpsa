from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_v4 import benefit_retention_summary, mean_sd_ci
from .common import save_json


def _scale_label(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def _method_names(item: dict[str, Any], family: str) -> list[str]:
    split = item["splits"]["all_triples_out"]
    available = split["ranking"].keys()
    selections = split["scale_selections"]
    if family == "selected_tbe":
        scale = _scale_label(float(selections["tbe"]["selected_scale"]))
        wanted = f"tbe_linear_scale_{scale}"
        return [wanted] if wanted in available else []
    if family == "selected_jl":
        result = []
        for key, selection in selections.items():
            if not key.startswith("jl_seed"):
                continue
            repeat = key.removeprefix("jl_seed")
            scale = _scale_label(float(selection["selected_scale"]))
            wanted = f"jl_linear_scale_{scale}_seed{repeat}"
            if wanted in available:
                result.append(wanted)
        return sorted(result)
    if family == "selected_permuted":
        result = []
        for key, selection in selections.items():
            if not key.startswith("permuted_basis_seed"):
                continue
            repeat = key.removeprefix("permuted_basis_seed")
            scale = _scale_label(float(selection["selected_scale"]))
            wanted = f"permuted_basis_linear_scale_{scale}_seed{repeat}"
            if wanted in available:
                result.append(wanted)
        return sorted(result)
    if family == "random":
        return sorted(name for name in available if name.startswith("random_seed"))
    return [family] if family in available else []


def _finite_summary(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "mean": None,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_independent_seeds": 0,
            "evidence_strength": "undefined",
            "per_seed": values,
        }
    result = mean_sd_ci(finite)
    result["per_seed"] = values
    return result


def _ranking_seed_value(
    item: dict[str, Any], family: str, domain: str, metric: str
) -> float:
    rows = item["splits"]["all_triples_out"]["ranking"]
    names = _method_names(item, family)
    values = [rows[name][domain][metric] for name in names]
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else float("nan")


def _causal_seed_value(item: dict[str, Any], family: str, metric: str) -> float:
    names = set(_method_names(item, family))
    rows = item["splits"]["all_triples_out"]["causal_records"]
    values = [float(row[metric]) for row in rows if row["method"] in names]
    return float(np.mean(values)) if values else float("nan")


def aggregate_query_runs(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated controls within, and model checkpoints across, seeds."""
    if not items:
        raise ValueError("at least one query run is required")
    ordered = sorted(items, key=lambda item: int(item["seed"]))
    families = (
        "mean_only",
        "tbe_kernel",
        "selected_tbe",
        "selected_jl",
        "selected_permuted",
        "observed_opg_oracle",
        "random",
    )
    present = [
        family
        for family in families
        if all(_method_names(item, family) for item in ordered)
    ]
    ranking: dict[str, Any] = {}
    for family in present:
        ranking[family] = {}
        for domain in ("raw", "residual"):
            ranking[family][domain] = {}
            for metric in ("spearman_mean", "ndcg_mean", "topk_recall_mean"):
                values = [
                    _ranking_seed_value(item, family, domain, metric)
                    for item in ordered
                ]
                ranking[family][domain][metric] = _finite_summary(values)

    causal: dict[str, Any] = {}
    for family in present:
        causal[family] = {}
        for metric in (
            "sufficiency",
            "necessity",
            "kept_half_mse",
            "dropped_half_mse",
        ):
            values = [_causal_seed_value(item, family, metric) for item in ordered]
            causal[family][metric] = _finite_summary(values)

    paired: dict[str, Any] = {}
    ranking_paired: dict[str, Any] = {}
    if "selected_tbe" in present:
        for comparator in present:
            if comparator == "selected_tbe":
                continue
            comparison_key = f"selected_tbe_minus_{comparator}"
            ranking_paired[comparison_key] = {}
            for domain in ("raw", "residual"):
                for metric in (
                    "spearman_mean",
                    "ndcg_mean",
                    "topk_recall_mean",
                ):
                    differences = [
                        _ranking_seed_value(item, "selected_tbe", domain, metric)
                        - _ranking_seed_value(item, comparator, domain, metric)
                        for item in ordered
                    ]
                    ranking_paired[comparison_key][f"{domain}_{metric}"] = (
                        _finite_summary(differences)
                    )
            paired[comparison_key] = {}
            for metric in ("sufficiency", "necessity"):
                differences = [
                    _causal_seed_value(item, "selected_tbe", metric)
                    - _causal_seed_value(item, comparator, metric)
                    for item in ordered
                ]
                paired[comparison_key][metric] = _finite_summary(differences)

    retention: dict[str, Any] = {}
    if {"selected_tbe", "mean_only", "observed_opg_oracle"} <= set(present):
        for baseline in ("mean_only", "random"):
            if baseline not in present:
                continue
            retention[f"versus_{baseline}"] = {}
            for metric in ("sufficiency", "necessity"):
                retention[f"versus_{baseline}"][metric] = benefit_retention_summary(
                    baselines=[
                        _causal_seed_value(item, baseline, metric) for item in ordered
                    ],
                    compact=[
                        _causal_seed_value(item, "selected_tbe", metric)
                        for item in ordered
                    ],
                    full=[
                        _causal_seed_value(item, "observed_opg_oracle", metric)
                        for item in ordered
                    ],
                    higher_is_better=True,
                )

    resources = ordered[0].get("resource_accounting")
    return {
        "seeds": [int(item["seed"]) for item in ordered],
        "selected_scales": {
            "tbe": [
                float(
                    item["splits"]["all_triples_out"]["scale_selections"]["tbe"][
                        "selected_scale"
                    ]
                )
                for item in ordered
            ],
            "jl": [
                [
                    float(selection["selected_scale"])
                    for key, selection in sorted(
                        item["splits"]["all_triples_out"]["scale_selections"].items()
                    )
                    if key.startswith("jl_seed")
                ]
                for item in ordered
            ],
            "permuted": [
                [
                    float(selection["selected_scale"])
                    for key, selection in sorted(
                        item["splits"]["all_triples_out"]["scale_selections"].items()
                    )
                    if key.startswith("permuted_basis_seed")
                ]
                for item in ordered
            ],
        },
        "ranking": ranking,
        "causal": causal,
        "ranking_paired_differences": ranking_paired,
        "paired_differences": paired,
        "benefit_retention": retention,
        "resource_accounting": resources,
        "independence_unit": "trained model seed; random controls averaged within seed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = [
        json.loads(path.read_text())
        for path in sorted(args.input_dir.glob("query_v5_seed*.json"))
    ]
    result = aggregate_query_runs(items)
    save_json(args.output, result)
    print(f"PAPER_ANALYSIS_V5_RESULT={args.output}")


if __name__ == "__main__":
    main()
