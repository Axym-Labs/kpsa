from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_v4 import mean_sd_ci
from .common import save_json


def _family(method: str) -> str:
    if method.startswith("jl_linear_seed"):
        return "jl_linear"
    if method.startswith("permuted_basis_linear_seed"):
        return "permuted_basis_linear"
    return method


def _summary(values: list[float]) -> dict[str, Any]:
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


def _seed_value(item: dict[str, Any], family: str, metric: str) -> float:
    values = [
        record[metric]
        for record in item["records"]
        if _family(record["method"]) == family and record[metric] is not None
    ]
    return float(np.mean(values)) if values else float("nan")


def aggregate_causal_runs(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tasks and projection repeats within each trained-model seed."""
    if not items:
        raise ValueError("at least one causal-ranking run is required")
    ordered = sorted(items, key=lambda item: int(item["seed"]))
    families = sorted(
        {_family(record["method"]) for item in ordered for record in item["records"]}
    )
    methods: dict[str, Any] = {}
    for family in families:
        methods[family] = {}
        for metric in ("raw_spearman", "residual_spearman"):
            values = [_seed_value(item, family, metric) for item in ordered]
            methods[family][metric] = _summary(values)

    paired: dict[str, Any] = {}
    if "tbe_linear" in families:
        for comparator in families:
            if comparator == "tbe_linear":
                continue
            key = f"tbe_linear_minus_{comparator}"
            paired[key] = {}
            for metric in ("raw_spearman", "residual_spearman"):
                differences = [
                    _seed_value(item, "tbe_linear", metric)
                    - _seed_value(item, comparator, metric)
                    for item in ordered
                ]
                paired[key][metric] = _summary(differences)
    return {
        "seeds": [int(item["seed"]) for item in ordered],
        "module_samples": [int(item["module_sample"]) for item in ordered],
        "methods": methods,
        "paired_differences": paired,
        "resource_accounting": ordered[0].get("resource_accounting"),
        "independence_unit": (
            "trained model seed; target tasks and random-feature repeats averaged within seed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = [
        json.loads(path.read_text())
        for path in sorted(args.input_dir.glob("causal_v5_seed*.json"))
    ]
    save_json(args.output, aggregate_causal_runs(items))
    print(f"PAPER_CAUSAL_ANALYSIS_V5_RESULT={args.output}")


if __name__ == "__main__":
    main()
