from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_v4 import mean_sd_ci
from .common import save_json


def _family(method: str) -> str:
    family, separator, repeat = method.rpartition("_seed")
    if separator and repeat.isdigit() and family.endswith("/jl"):
        return family
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


def _run_family_points(
    item: dict[str, Any], family: str
) -> dict[float, dict[str, float]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in item["records"]:
        if _family(record["method"]) == family:
            grouped[float(record["strength"])].append(record)
    return {
        strength: {
            metric: float(np.mean([float(record[metric]) for record in records]))
            for metric in (
                "average_forgetting",
                "average_acquisition_gain",
                "final_average",
            )
        }
        for strength, records in grouped.items()
    }


def _matched_forgetting(
    item: dict[str, Any], family: str, acquisition_ratio: float
) -> float:
    none = _run_family_points(item, "none")[0.0]
    points = _run_family_points(item, family)
    x = [none["average_acquisition_gain"] / none["average_acquisition_gain"]]
    y = [none["average_forgetting"]]
    for point in points.values():
        x.append(point["average_acquisition_gain"] / none["average_acquisition_gain"])
        y.append(point["average_forgetting"])
    order = np.argsort(x)
    sorted_x = np.asarray(x, dtype=float)[order]
    sorted_y = np.asarray(y, dtype=float)[order]
    if acquisition_ratio < sorted_x[0] or acquisition_ratio > sorted_x[-1]:
        return float("nan")
    return float(np.interp(acquisition_ratio, sorted_x, sorted_y))


def _average_orders_by_seed(items: list[dict[str, Any]], value_fn) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for item in items:
        grouped[int(item["seed"])].append(float(value_fn(item)))
    return [
        float(np.mean(grouped[seed]))
        for seed in sorted(grouped)
        if all(math.isfinite(value) for value in grouped[seed])
    ]


def aggregate_continual_runs(
    items: list[dict[str, Any]],
    *,
    acquisition_ratios: tuple[float, ...] = (0.7, 0.8, 0.9),
) -> dict[str, Any]:
    """Aggregate CL curves with orders and controls nested within model seed."""
    if not items:
        raise ValueError("at least one continual-learning run is required")
    if any(not 0 < ratio <= 1 for ratio in acquisition_ratios):
        raise ValueError("acquisition ratios must lie in (0, 1]")
    families = sorted(
        {_family(record["method"]) for item in items for record in item["records"]}
    )
    curves: dict[str, Any] = {}
    for family in families:
        strengths = sorted(
            {
                strength
                for item in items
                for strength in _run_family_points(item, family)
            }
        )
        curves[family] = {}
        for strength in strengths:
            curves[family][f"{strength:g}"] = {}
            for metric in (
                "average_forgetting",
                "average_acquisition_gain",
                "final_average",
            ):
                per_seed = _average_orders_by_seed(
                    items,
                    lambda item, f=family, s=strength, m=metric: (
                        _run_family_points(item, f).get(s, {}).get(m, float("nan"))
                    ),
                )
                curves[family][f"{strength:g}"][metric] = _summary(per_seed)

    matched: dict[str, Any] = {}
    for family in families:
        if family == "none":
            continue
        matched[family] = {}
        for ratio in acquisition_ratios:
            per_seed = _average_orders_by_seed(
                items,
                lambda item, f=family, r=ratio: _matched_forgetting(item, f, r),
            )
            matched[family][f"{ratio:g}"] = _summary(per_seed)

    paired: dict[str, Any] = {}
    for current in (f for f in matched if f == "tbe_linear" or f.endswith("/tbe")):
        for comparator in matched:
            if comparator == current:
                continue
            if "/" in current and current.split("/")[0] != comparator.split("/")[0]:
                continue
            key = f"{current}_minus_{comparator}"
            paired[key] = {}
            for ratio in acquisition_ratios:
                differences = _average_orders_by_seed(
                    items,
                    lambda item, f=comparator, c=current, r=ratio: (
                        _matched_forgetting(item, c, r)
                        - _matched_forgetting(item, f, r)
                    ),
                )
                paired[key][f"{ratio:g}"] = _summary(differences)

    seeds = sorted({int(item["seed"]) for item in items})
    return {
        "seeds": seeds,
        "runs": len(items),
        "premise_passed_all_runs": all(
            bool(item["premise_gate"]["passed"]) for item in items
        ),
        "curves": curves,
        "matched_acquisition": matched,
        "paired_differences": paired,
        "resource_accounting": items[0].get("resource_accounting"),
        "independence_unit": (
            "trained model seed; task orders and random-feature repeats averaged within seed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = [
        item
        for path in sorted(args.input_dir.glob("continual_v5_*.json"))
        for item in [json.loads(path.read_text())]
        if "records" in item
    ]
    save_json(args.output, aggregate_continual_runs(items))
    print(f"PAPER_CONTINUAL_ANALYSIS_V5_RESULT={args.output}")


if __name__ == "__main__":
    main()
