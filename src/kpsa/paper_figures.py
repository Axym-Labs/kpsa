"""Paper-facing KPSA figures with cross-modal measurement regularity."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .paper_style import (
    CATEGORICAL,
    CONSTANT,
    DIRECT,
    SEMANTIC,
    SEMANTIC_LIGHT,
    finish_axis,
    method_style,
    outside_legend,
    panel_title,
    save_plot,
)

CANONICAL_METHODS = (
    "Semantic KPSA",
    "Categorical KPSA",
    "Semantic nearest",
    "Constant sensitivity",
    "Activation attribution",
    "Direct gradient",
)


def _read(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _query_method_values(
    records,
    *,
    budget_field,
    budget,
    metric,
    query_field,
    method_map,
    valid_field=None,
):
    grouped = defaultdict(list)
    for record in records:
        if record[budget_field] != budget:
            continue
        if valid_field and not record.get(valid_field, False):
            continue
        raw_method = record["method"]
        if raw_method not in method_map:
            continue
        grouped[(method_map[raw_method], record[query_field])].append(
            float(record[metric])
        )
    values = defaultdict(dict)
    for (method, query), current in grouped.items():
        values[method][query] = float(np.mean(current))
    return values


def _bootstrap_ratio(
    numerator,
    low_anchor,
    high_anchor,
    *,
    subtract_low=True,
    seed=91_027,
    draws=5000,
):
    numerator = np.asarray(numerator, dtype=np.float64)
    low_anchor = np.asarray(low_anchor, dtype=np.float64)
    high_anchor = np.asarray(high_anchor, dtype=np.float64)
    if not (len(numerator) == len(low_anchor) == len(high_anchor)):
        raise ValueError("paired bootstrap inputs must have equal length")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(numerator), size=(draws, len(numerator)))
    num = numerator[sampled].mean(1)
    low = low_anchor[sampled].mean(1)
    high = high_anchor[sampled].mean(1)
    denominator = high - low if subtract_low else high
    ratio = (num - low) / denominator if subtract_low else num / denominator
    point_denominator = (
        high_anchor.mean() - low_anchor.mean() if subtract_low else high_anchor.mean()
    )
    point = (
        (numerator.mean() - low_anchor.mean()) / point_denominator
        if subtract_low
        else numerator.mean() / point_denominator
    )
    low_ci, high_ci = np.quantile(ratio[np.isfinite(ratio)], (0.025, 0.975))
    return float(point), float(low_ci), float(high_ci), len(numerator)


def normalized_series(
    payload,
    *,
    budget_field,
    budgets,
    budget_percent,
    metric,
    query_field,
    method_map,
    constant="Constant sensitivity",
    reference="Direct gradient",
    valid_field=None,
    relative=False,
):
    """Bootstrap paired gap recovery or relative energy coverage by budget."""
    rows = []
    for budget in budgets:
        values = _query_method_values(
            payload["records"],
            budget_field=budget_field,
            budget=budget,
            metric=metric,
            query_field=query_field,
            method_map=method_map,
            valid_field=valid_field,
        )
        for method in CANONICAL_METHODS:
            if method not in values or reference not in values:
                continue
            required = [values[method], values[reference]]
            if not relative:
                if constant not in values:
                    continue
                required.append(values[constant])
            queries = sorted(set.intersection(*(set(item) for item in required)))
            numerator = [values[method][query] for query in queries]
            high = [values[reference][query] for query in queries]
            low = (
                [0.0 for _ in queries]
                if relative
                else [values[constant][query] for query in queries]
            )
            mean, ci_low, ci_high, count = _bootstrap_ratio(
                numerator,
                low,
                high,
                subtract_low=not relative,
                seed=91_027 + round(1e7 * budget) + CANONICAL_METHODS.index(method),
            )
            rows.append(
                {
                    "budget": float(budget_percent[budget]),
                    "method": method,
                    "mean": 100 * mean,
                    "ci95_low": 100 * ci_low,
                    "ci95_high": 100 * ci_high,
                    "queries": count,
                }
            )
    return rows


def application1_series(vision_path, protein_path):
    vision = _read(vision_path)
    protein = _read(protein_path)
    fractions = tuple(vision["protocol"]["fractions"])
    protein_fractions = tuple(protein["protocol"]["fractions"])
    return {
        "Vision": normalized_series(
            vision,
            budget_field="fraction",
            budgets=fractions,
            budget_percent={fraction: 100 * fraction for fraction in fractions},
            metric="degradation",
            query_field="query_column",
            valid_field="target_correct",
            method_map={
                "prototype_800_scale_0.025": "Semantic KPSA",
                "class_onehot": "Categorical KPSA",
                "nearest_encoder": "Semantic nearest",
                "scalar_mass": "Constant sensitivity",
                "direct_coordinate_taylor": "Activation attribution",
            },
            reference="Activation attribution",
        ),
        "Protein": normalized_series(
            protein,
            budget_field="fraction",
            budgets=protein_fractions,
            budget_percent={fraction: 100 * fraction for fraction in protein_fractions},
            metric="causal_effect",
            query_field="query",
            method_map={
                "exact_rbf": "Semantic KPSA",
                "categorical": "Categorical KPSA",
                "nearest": "Semantic nearest",
                "scalar": "Constant sensitivity",
                "activation_attribution": "Activation attribution",
            },
            reference="Activation attribution",
        ),
    }


def application2_series(vision_path, etth1_path, weather_path, protein_path):
    vision, etth1, weather, protein = map(
        _read, (vision_path, etth1_path, weather_path, protein_path)
    )
    vision_groups = tuple(vision["protocol"]["selected_groups"])
    vision_budget = {8: 0.02, 19: 0.05, 37: 0.1}
    fraction_budget = lambda payload: {
        value: 100 * value for value in payload["protocol"]["fractions"]
    }
    specifications = {
        "Vision": {
            "payload": vision,
            "budget_field": "selected_groups",
            "budgets": vision_groups,
            "budget_percent": {value: vision_budget[value] for value in vision_groups},
            "query_field": "query_column",
            "valid_field": "target_correct",
            "method_map": {
                "prototype": "Semantic KPSA",
                "class_onehot": "Categorical KPSA",
                "nearest": "Semantic nearest",
                "scalar": "Constant sensitivity",
                "direct": "Direct gradient",
            },
            "causal_metric": "target_squared_susceptibility",
            "energy_metric": "gradient_energy_coverage",
        },
        "ETTh1": {
            "payload": etth1,
            "budget_field": "fraction",
            "budgets": tuple(etth1["protocol"]["fractions"]),
            "budget_percent": fraction_budget(etth1),
            "query_field": "query",
            "method_map": {
                "exact_rbf": "Semantic KPSA",
                "temporal_categorical": "Categorical KPSA",
                "nearest": "Semantic nearest",
                "scalar": "Constant sensitivity",
                "direct_gradient": "Direct gradient",
            },
            "causal_metric": "squared_susceptibility",
            "energy_metric": "direct_gradient_energy_coverage",
        },
        "Weather": {
            "payload": weather,
            "budget_field": "fraction",
            "budgets": tuple(weather["protocol"]["fractions"]),
            "budget_percent": fraction_budget(weather),
            "query_field": "query",
            "method_map": {
                "exact_rbf": "Semantic KPSA",
                "temporal_categorical": "Categorical KPSA",
                "nearest": "Semantic nearest",
                "scalar": "Constant sensitivity",
                "direct_gradient": "Direct gradient",
            },
            "causal_metric": "squared_susceptibility",
            "energy_metric": "direct_gradient_energy_coverage",
        },
        "Protein": {
            "payload": protein,
            "budget_field": "fraction",
            "budgets": tuple(protein["protocol"]["fractions"]),
            "budget_percent": fraction_budget(protein),
            "query_field": "query",
            "method_map": {
                "exact_rbf": "Semantic KPSA",
                "categorical": "Categorical KPSA",
                "nearest": "Semantic nearest",
                "scalar": "Constant sensitivity",
                "direct_parameter": "Direct gradient",
            },
            "causal_metric": "squared_susceptibility",
            "energy_metric": "direct_gradient_energy_coverage",
        },
    }
    output = {}
    for name, spec in specifications.items():
        common = {
            key: spec[key]
            for key in (
                "payload",
                "budget_field",
                "budgets",
                "budget_percent",
                "query_field",
                "method_map",
            )
        }
        if spec.get("valid_field"):
            common["valid_field"] = spec["valid_field"]
        output[name] = {
            "causal": normalized_series(metric=spec["causal_metric"], **common),
            "energy": normalized_series(
                metric=spec["energy_metric"], relative=True, **common
            ),
        }
    return output


def _plot_budget_series(axis, rows, *, ylabel, title, letter, xlabel=True):
    budgets = sorted({row["budget"] for row in rows})
    for method in CANONICAL_METHODS:
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["budget"],
        )
        if not selected:
            continue
        means = np.asarray([row["mean"] for row in selected])
        low = means - np.asarray([row["ci95_low"] for row in selected])
        high = np.asarray([row["ci95_high"] for row in selected]) - means
        axis.errorbar(
            [row["budget"] for row in selected],
            means,
            yerr=(np.maximum(0, low), np.maximum(0, high)),
            label=method,
            **method_style(method),
        )
    axis.set_xscale("log")
    axis.set_xticks(budgets, [f"{value:g}%" for value in budgets])
    axis.xaxis.set_minor_locator(mticker.NullLocator())
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    axis.set_xlabel("Selected MLP parameters" if xlabel else "")
    axis.set_ylabel(ylabel)
    panel_title(axis, letter, title)
    finish_axis(axis, zero_line=True)


def plot_application1_primary(series, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    _plot_budget_series(
        axes[0],
        series["Vision"],
        ylabel="Oracle gap recovered (%)",
        title="Vision",
        letter="A",
    )
    _plot_budget_series(
        axes[1],
        series["Protein"],
        ylabel="",
        title="Protein",
        letter="B",
    )
    for axis in axes:
        axis.set_ylim(-8, 112)
    outside_legend(fig, axes, ncol=5, y=1.08)
    save_plot(fig, output)


def plot_application2_primary(series, output):
    names = ("Vision", "ETTh1", "Weather", "Protein")
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 4.25), sharey="row")
    for column, name in enumerate(names):
        _plot_budget_series(
            axes[0, column],
            series[name]["causal"],
            ylabel="Causal gap recovered (%)" if column == 0 else "",
            title=name,
            letter=chr(ord("A") + column),
            xlabel=False,
        )
        _plot_budget_series(
            axes[1, column],
            series[name]["energy"],
            ylabel="Oracle gradient energy (%)" if column == 0 else "",
            title=name,
            letter=chr(ord("E") + column),
        )
    for axis in axes.reshape(-1):
        axis.set_ylim(-8, 112)
    outside_legend(fig, axes, ncol=5, y=1.035)
    save_plot(fig, output)


def plot_application1_efficiency(cost_path, vision_series, atlas_rows, output):
    cost = _read(cost_path)
    online = cost["online"]
    feature = online["batched_compact_atlas"]["feature"]
    bundle = online["batched_compact_atlas"]["bundle_32"]
    direct = online["batched_activation_attribution"]
    feature_accuracy = {
        row["budget"]: row["mean"]
        for row in vision_series
        if row["method"] == "Semantic KPSA"
    }
    points = [
        {
            "label": "Activation attribution",
            "throughput": direct["mean_images_per_second"],
            "memory": direct["peak_allocated_bytes"] / 2**30,
            "accuracy": 100.0,
            "color": DIRECT,
            "marker": "P",
        }
    ]
    for budget, accuracy in sorted(feature_accuracy.items()):
        points.append(
            {
                "label": "Semantic KPSA",
                "throughput": feature["mean_images_per_second"],
                "memory": feature["peak_allocated_bytes"] / 2**30,
                "accuracy": accuracy,
                "color": SEMANTIC,
                "marker": "o",
                "budget": budget,
            }
        )
    bundle_accuracy = (
        100 * cost["accuracy"]["bundle_32_at_parameter_fraction_1_over_12"]["prototype"]
    )
    points.append(
        {
            "label": "KPSA, 32-feature groups",
            "throughput": bundle["mean_images_per_second"],
            "memory": bundle["peak_allocated_bytes"] / 2**30,
            "accuracy": bundle_accuracy,
            "color": SEMANTIC_LIGHT,
            "marker": "s",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    for point in points:
        for axis, key in ((axes[0], "throughput"), (axes[1], "memory")):
            axis.scatter(
                point[key],
                point["accuracy"],
                color=point["color"],
                marker=point["marker"],
                s=30,
                label=point["label"],
                zorder=3,
            )
            if "budget" in point:
                axis.annotate(
                    f"{point['budget']:g}%",
                    (point[key], point["accuracy"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.2,
                    color=point["color"],
                )
    axes[0].set_xlabel("Query throughput (images/s)")
    axes[1].set_xlabel("Peak GPU memory (GiB)")
    for axis in axes[:2]:
        axis.set_ylabel("Oracle gap recovered (%)")
        axis.set_ylim(0, 108)
        finish_axis(axis)
    panel_title(axes[0], "A", "Accuracy–throughput")
    panel_title(axes[1], "B", "Accuracy–memory")

    development = [
        row
        for row in atlas_rows
        if row["split"] == "development"
        and row["kind"] == "causal_gain"
        and row["comparison"] == "scalar"
        and row["method"] == "prototype_rbf"
    ]
    fractions = sorted({float(row["parameter_fraction"]) for row in development})
    for index, fraction in enumerate(fractions):
        selected = sorted(
            [
                row
                for row in development
                if float(row["parameter_fraction"]) == fraction
            ],
            key=lambda row: int(row["source_examples"]),
        )
        axes[2].plot(
            [int(row["source_examples"]) for row in selected],
            [float(row["mean"]) for row in selected],
            color=(SEMANTIC, SEMANTIC_LIGHT)[index % 2],
            marker=("o", "s")[index % 2],
            label=f"{100 * fraction:g}% budget",
        )
    counts = sorted({int(row["source_examples"]) for row in development})
    axes[2].set_xscale("log", base=2)
    labeled_counts = [value for value in counts if value <= 6400]
    axes[2].set_xticks(
        labeled_counts, [f"{value / 1000:g}k" for value in labeled_counts]
    )
    axes[2].xaxis.set_minor_locator(mticker.NullLocator())
    axes[2].set_xlabel("Source images, $N$")
    axes[2].set_ylabel("Causal gain vs. constant")
    selected_n = 1600
    selected_cost = next(
        float(row["estimated_batched_break_even_queries"])
        for row in development
        if int(row["source_examples"]) == selected_n
    )
    axes[2].axvline(selected_n, color=SEMANTIC_LIGHT, linestyle="--", linewidth=0.9)
    axes[2].text(
        0.04,
        0.94,
        f"$N={selected_n:,}$: break-even ≈ {selected_cost / 1000:.0f}k queries",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=SEMANTIC,
    )
    panel_title(axes[2], "C", "Atlas size and amortization")
    finish_axis(axes[2], zero_line=True)
    outside_legend(fig, axes, ncol=5, y=1.08)
    save_plot(fig, output)


def plot_parameter_resolution_appendix(rows, output):
    """Compact resolution ablation: fidelity, causal utility, and storage."""
    resolutions = ("feature", "bundle_4", "bundle_32", "mlp_block")
    labels = ("Feature", "4-feature", "32-feature", "MLP block")
    x = np.arange(len(resolutions))
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))

    for method, label, color, marker, linestyle in (
        ("prototype_rbf", "Semantic KPSA", SEMANTIC, "o", "-"),
        ("scalar_mass", "Constant sensitivity", CONSTANT, "x", "-."),
    ):
        selected = [
            next(
                row
                for row in rows
                if row["resolution"] == resolution
                and row["assay"] == "query_fidelity"
                and row["method"] == method
            )
            for resolution in resolutions
        ]
        means = np.asarray([float(row["mean"]) for row in selected])
        axes[0].errorbar(
            x,
            means,
            yerr=(
                means - np.asarray([float(row["ci95_low"]) for row in selected]),
                np.asarray([float(row["ci95_high"]) for row in selected]) - means,
            ),
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    axes[0].set_ylabel("Cold-profile Spearman")
    panel_title(axes[0], "A", "Query fidelity")

    assay_styles = (
        ("neuron_deactivation", "Neuron localization", SEMANTIC, "o"),
        ("parameter_influence", "Parameter influence", CATEGORICAL, "s"),
    )
    for assay, label, color, marker in assay_styles:
        selected = [
            next(
                row
                for row in rows
                if row["resolution"] == resolution
                and row["assay"] == assay
                and row["method"] == "prototype_rbf"
            )
            for resolution in resolutions
        ]
        axes[1].plot(
            x,
            [100 * float(row["oracle_gap_recovered"] or 0) for row in selected],
            color=color,
            marker=marker,
            label=label,
        )
        storage = np.asarray(
            [float(row["atlas_bytes_float32"]) / 2**20 for row in selected]
        )
        utility = np.asarray(
            [100 * float(row["oracle_gap_recovered"] or 0) for row in selected]
        )
        axes[2].plot(storage, utility, color=color, marker=marker, label=label)
        offsets = ((3, 3), (3, 3), (3, 4), (3, 3))
        if assay == "parameter_influence":
            offsets = ((3, -10), (3, -10), (3, -10), (3, -10))
        for sx, sy, short, offset in zip(
            storage,
            utility,
            ("Feat.", "4", "32", "Block"),
            offsets,
        ):
            axes[2].annotate(
                short,
                (sx, sy),
                xytext=offset,
                textcoords="offset points",
                color=color,
                fontsize=5.8,
            )
    axes[1].set_ylabel("Oracle gap recovered (%)")
    panel_title(axes[1], "B", "Causal utility")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Atlas storage (MiB)")
    axes[2].set_ylabel("Oracle gap recovered (%)")
    panel_title(axes[2], "C", "Utility–storage tradeoff")

    for axis in axes[:2]:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_xlabel("Parameter-group resolution")
    for axis in axes:
        finish_axis(axis, zero_line=axis is not axes[0])
    outside_legend(fig, axes, ncol=4, y=1.08)
    save_plot(fig, output)
