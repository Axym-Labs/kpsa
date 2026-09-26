"""Paper-facing summaries and figures for the refined sensitivity scope."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import t


def paired_t_summary(values) -> dict[str, float | int]:
    """Summarize paired effects with a two-sided 95% Student-t interval."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("at least two finite paired effects are required")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    half_width = float(t.ppf(0.975, len(array) - 1) * standard_error)
    return {
        "n": len(array),
        "mean": mean,
        "lower_95": mean - half_width,
        "upper_95": mean + half_width,
        "standard_error": standard_error,
    }


def summarize_precision_records(records: list[dict]) -> dict[str, dict]:
    """Aggregate per-domain precision outcomes and pair each method to scalar mass."""
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in records:
        grouped[
            (row["method"], float(row["requested_high_precision_scope_fraction"]))
        ].append(row)
    methods: dict[str, dict] = defaultdict(dict)
    for (method, fraction), rows in grouped.items():
        values = [row["nll_increase"] for row in rows]
        summary = paired_t_summary(values)
        summary["mean_bits_per_parameter"] = float(
            np.mean([row["ideal_packed_bits_per_parameter"] for row in rows])
        )
        methods[method][str(fraction)] = summary
    paired: dict[str, dict] = defaultdict(dict)
    for (method, fraction), rows in grouped.items():
        if method == "scalar_mass":
            continue
        scalar_rows = grouped.get(("scalar_mass", fraction))
        if scalar_rows is None:
            continue
        scalar = {int(row["task"]): row["nll_increase"] for row in scalar_rows}
        values = [
            row["nll_increase"] - scalar[int(row["task"])]
            for row in rows
            if int(row["task"]) in scalar
        ]
        if len(values) >= 2:
            paired[method][str(fraction)] = paired_t_summary(values)
    return {"methods": dict(methods), "paired_vs_scalar": dict(paired)}
