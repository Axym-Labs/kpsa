"""Package the refined positive-evidence search as tables, plots, and raw metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PRIMARY = "#3F21B6"
SECONDARY = "#8C7AD3"
GRAY = "#777777"
LIGHT = "#BDBDBD"
BLACK = "#222222"


def outside_legend(fig, axes, *, ncol=None, fontsize=7):
    """Place a deduplicated one-line legend above every panel."""
    axes = np.asarray(axes, dtype=object).reshape(-1)
    handles_by_label = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                handles_by_label.setdefault(label, handle)
    labels = list(handles_by_label)
    fig.legend(
        [handles_by_label[label] for label in labels],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=ncol or len(labels),
        frameon=False,
        fontsize=fontsize,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))


def save_plot(fig, output):
    for extension in ("png", "pdf"):
        fig.savefig(
            output.with_suffix(f".{extension}"),
            dpi=300,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def bootstrap(values, *, seed=23_041, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.choice(values, (draws, len(values)), replace=True).mean(1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(values.mean()), float(low), float(high), len(values)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def causal_index(payload, fraction, *, require_off_target=False):
    return {
        (record["method"], record["query_column"]): record
        for record in payload["records"]
        if record["requested_parameter_fraction"] == fraction
        and record.get("target_correct", True)
        and (not require_off_target or record.get("off_target_correct", False))
    }


def build_cold_table(studies):
    methods = {
        "sample_dinov2_rbf_scale_0.1": "Sensitivity RBF",
        "sample_nearest": "Nearest sample",
        "class_source_onehot_reference": "Class one-hot",
        "scalar_mass": "Scalar sensitivity",
        "sample_jl": "JL control",
    }
    rows = []
    for split, model, path in studies:
        payload = read_json(path)
        for source, method in methods.items():
            metrics = payload["cold_query"][source]
            for metric in ("spearman", "topk_recall", "ndcg"):
                mean, low, high, count = bootstrap(metrics[metric])
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "queries": count,
                    }
                )
    return rows


def build_causal_tables(studies):
    methods = {
        "sensitivity_atlas_rbf": "Sensitivity RBF",
        "sensitivity_atlas_nearest": "Nearest sample",
        "sensitivity_atlas_scalar": "Scalar sensitivity",
        "activation_atlas_rbf": "Activation atlas RBF",
        "activation_x_gradient_atlas_rbf": "Act×grad atlas RBF",
        "class_source_onehot_reference": "Class one-hot",
        "direct_gradient_oracle": "Direct gradient",
        "target_activation_magnitude": "Target activation",
        "target_activation_x_gradient": "Target act×grad",
        "weight_magnitude": "Weight magnitude",
        "random": "Random",
    }
    effects = []
    paired = []
    for model, path in studies:
        payload = read_json(path)
        for fraction in payload["protocol"]["fractions"]:
            index = causal_index(payload, fraction)
            off_index = causal_index(payload, fraction, require_off_target=True)
            for source, method in methods.items():
                current = [
                    record["degradation"]
                    for (name, _), record in index.items()
                    if name == source
                ]
                if not current:
                    continue
                mean, low, high, count = bootstrap(current)
                selectivity = [
                    record["selectivity"]
                    for (name, _), record in off_index.items()
                    if name == source
                ]
                selectivity_stats = (
                    bootstrap(selectivity) if selectivity else (np.nan,) * 3 + (0,)
                )
                effects.append(
                    {
                        "model": model,
                        "parameter_fraction": fraction,
                        "method": method,
                        "effect_mean": mean,
                        "effect_ci95_low": low,
                        "effect_ci95_high": high,
                        "selectivity_mean": selectivity_stats[0],
                        "selectivity_ci95_low": selectivity_stats[1],
                        "selectivity_ci95_high": selectivity_stats[2],
                        "correct_queries": count,
                        "selectivity_queries": selectivity_stats[3],
                    }
                )
            primary = "sensitivity_atlas_rbf"
            for comparator, comparator_label in methods.items():
                if comparator == primary:
                    continue
                differences = [
                    record["degradation"] - index[(comparator, column)]["degradation"]
                    for (name, column), record in index.items()
                    if name == primary and (comparator, column) in index
                ]
                if differences:
                    mean, low, high, count = bootstrap(differences)
                    paired.append(
                        {
                            "model": model,
                            "parameter_fraction": fraction,
                            "primary": "Sensitivity RBF",
                            "comparator": comparator_label,
                            "paired_difference_mean": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "queries": count,
                        }
                    )
    return effects, paired


def build_functional_table(paths):
    values = {}
    rows = []
    method = "sample_dinov2_rbf_scale_0.1"
    for functional, path in paths.items():
        payload = read_json(path)
        for fraction in payload["protocol"]["fractions"]:
            index = causal_index(payload, fraction)
            current = {
                column: record["degradation"]
                for (name, column), record in index.items()
                if name == method
            }
            values[(functional, fraction)] = current
            mean, low, high, count = bootstrap(list(current.values()))
            rows.append(
                {
                    "ranking_functional": functional,
                    "evaluation_functional": "margin",
                    "parameter_fraction": fraction,
                    "effect_mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "queries": count,
                }
            )
    for fraction in sorted({key[1] for key in values}):
        margin = values[("margin", fraction)]
        for functional in ("class_logit", "loss"):
            current = values[(functional, fraction)]
            columns = sorted(set(margin) & set(current))
            mean, low, high, count = bootstrap(
                [margin[column] - current[column] for column in columns]
            )
            rows.append(
                {
                    "ranking_functional": f"margin_minus_{functional}",
                    "evaluation_functional": "margin",
                    "parameter_fraction": fraction,
                    "effect_mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "queries": count,
                }
            )
    return rows


def build_compression_table(nystrom_path, rff_path, confirmations):
    rows = []
    for family, path in (("Nyström", nystrom_path), ("RFF", rff_path)):
        for row in read_json(path)["rows"]:
            rows.append(
                {
                    "kind": "cold_screen",
                    "model": "ViT-B/16",
                    "family": family,
                    "dimension": row["dimension"],
                    "parameter_fraction": "",
                    "metric": "spearman",
                    "mean": row["mean_spearman"],
                    "ci95_low": "",
                    "ci95_high": "",
                    "queries": 200,
                }
            )
    nystrom = "sample_nystrom_rbf_scale_0.1_landmarks_200"
    for model, path in confirmations:
        payload = read_json(path)
        cold = payload["cold_query"][nystrom]
        mean, low, high, count = bootstrap(cold["spearman"])
        rows.append(
            {
                "kind": "cold_confirmation",
                "model": model,
                "family": "Nyström",
                "dimension": 200,
                "parameter_fraction": "",
                "metric": "spearman",
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "queries": count,
            }
        )
        for fraction in payload["protocol"]["fractions"]:
            index = causal_index(payload, fraction)
            for source, label in (
                (nystrom, "Nyström"),
                ("sample_dinov2_rbf_scale_0.1", "Full kernel"),
                ("scalar_mass", "Scalar"),
            ):
                current = [
                    record["degradation"]
                    for (name, _), record in index.items()
                    if name == source
                ]
                mean, low, high, count = bootstrap(current)
                rows.append(
                    {
                        "kind": "causal_confirmation",
                        "model": model,
                        "family": label,
                        "dimension": 200 if label == "Nyström" else "",
                        "parameter_fraction": fraction,
                        "metric": "margin_degradation",
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "queries": count,
                    }
                )
    return rows


def build_generalization_table(path):
    payload = read_json(path)
    methods = {
        "sample_dinov2_rbf_scale_0.1": "Sensitivity RBF",
        "sample_nearest": "Nearest sample",
        "scalar_mass": "Scalar sensitivity",
        "sample_jl": "JL control",
        "direct_gradient_oracle": "Direct gradient",
        "target_activation_magnitude": "Target activation",
    }
    rows = []
    for source, method in methods.items():
        metrics = payload["cold_query"].get(source)
        if metrics:
            mean, low, high, count = bootstrap(metrics["spearman"])
            rows.append(
                {
                    "scope": "cold_retrieval",
                    "parameter_fraction": "",
                    "method": method,
                    "metric": "spearman",
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "queries": count,
                }
            )
    for fraction in payload["protocol"]["fractions"]:
        index = causal_index(payload, fraction)
        for source, method in methods.items():
            current = [
                record["degradation"]
                for (name, _), record in index.items()
                if name == source
            ]
            if current:
                mean, low, high, count = bootstrap(current)
                rows.append(
                    {
                        "scope": "causal",
                        "parameter_fraction": fraction,
                        "method": method,
                        "metric": "margin_degradation",
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "queries": count,
                    }
                )
    return rows


def plot_cold(rows, output):
    splits = list(dict.fromkeys(row["split"] for row in rows))
    methods = [
        "Sensitivity RBF",
        "Nearest sample",
        "Class one-hot",
        "Scalar sensitivity",
        "JL control",
    ]
    styles = {
        "Sensitivity RBF": (PRIMARY, "o", "-"),
        "Nearest sample": (SECONDARY, "s", "--"),
        "Class one-hot": (GRAY, "^", "-."),
        "Scalar sensitivity": (BLACK, "D", ":"),
        "JL control": (LIGHT, "x", ":"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, metric, ylabel in zip(
        axes,
        ("spearman", "topk_recall"),
        ("Spearman correlation", "Top-5% recall"),
    ):
        for method in methods:
            selected = [
                row
                for row in rows
                if row["method"] == method and row["metric"] == metric
            ]
            by_split = {row["split"]: row for row in selected}
            y = [by_split[split]["mean"] for split in splits]
            low = [by_split[split]["ci95_low"] for split in splits]
            high = [by_split[split]["ci95_high"] for split in splits]
            color, marker, linestyle = styles[method]
            x = np.arange(len(splits))
            ax.errorbar(
                x,
                y,
                yerr=[np.asarray(y) - low, np.asarray(high) - y],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                markersize=4,
                capsize=2,
                label=method,
            )
        ax.set_xticks(np.arange(len(splits)), splits, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    outside_legend(fig, axes, ncol=len(methods))
    save_plot(fig, output)


def plot_causal(rows, output):
    models = ["ViT-B/16", "ViT-L/16", "ConvNeXtV2-T"]
    methods = [
        "Sensitivity RBF",
        "Activation atlas RBF",
        "Act×grad atlas RBF",
        "Class one-hot",
        "Scalar sensitivity",
    ]
    styles = {
        "Sensitivity RBF": (PRIMARY, "o", "-"),
        "Activation atlas RBF": (SECONDARY, "s", "--"),
        "Act×grad atlas RBF": (GRAY, "^", "-."),
        "Class one-hot": (BLACK, "D", ":"),
        "Scalar sensitivity": (LIGHT, "x", ":"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.65), sharey=False)
    for ax, model in zip(axes, models):
        ax.text(0.03, 0.95, model, transform=ax.transAxes, va="top", fontsize=8)
        for method in methods:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model"] == model and row["method"] == method
                ],
                key=lambda row: row["parameter_fraction"],
            )
            x = np.asarray([row["parameter_fraction"] * 100 for row in selected])
            y = np.asarray([row["effect_mean"] for row in selected])
            low = np.asarray([row["effect_ci95_low"] for row in selected])
            high = np.asarray([row["effect_ci95_high"] for row in selected])
            color, marker, linestyle = styles[method]
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                markersize=3,
                label=method,
            )
            ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel("Ablated MLP parameters (%)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel("Target margin degradation")
    outside_legend(fig, axes, ncol=len(methods), fontsize=6.5)
    save_plot(fig, output)


def plot_selectivity(rows, output):
    models = ["ViT-B/16", "ViT-L/16", "ConvNeXtV2-T"]
    methods = [
        "Sensitivity RBF",
        "Activation atlas RBF",
        "Act×grad atlas RBF",
        "Class one-hot",
        "Scalar sensitivity",
    ]
    styles = {
        "Sensitivity RBF": (PRIMARY, "o", "-"),
        "Activation atlas RBF": (SECONDARY, "s", "--"),
        "Act×grad atlas RBF": (GRAY, "^", "-."),
        "Class one-hot": (BLACK, "D", ":"),
        "Scalar sensitivity": (LIGHT, "x", ":"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.65), sharey=False)
    for ax, model in zip(axes, models):
        ax.text(0.03, 0.95, model, transform=ax.transAxes, va="top", fontsize=8)
        for method in methods:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["method"] == method
                    and row["selectivity_queries"]
                ],
                key=lambda row: row["parameter_fraction"],
            )
            x = np.asarray([row["parameter_fraction"] * 100 for row in selected])
            y = np.asarray([row["selectivity_mean"] for row in selected])
            low = np.asarray([row["selectivity_ci95_low"] for row in selected])
            high = np.asarray([row["selectivity_ci95_high"] for row in selected])
            color, marker, linestyle = styles[method]
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                markersize=3,
                label=method,
            )
            ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
        ax.axhline(0, color="#D5D5D5", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Ablated MLP parameters (%)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel("Target-specific margin effect")
    outside_legend(fig, axes, ncol=len(methods), fontsize=6.5)
    save_plot(fig, output)


def plot_functionals(rows, output):
    rows = [row for row in rows if "minus" not in row["ranking_functional"]]
    labels = {"margin": "Margin", "class_logit": "Class logit", "loss": "Loss"}
    colors = {"margin": PRIMARY, "class_logit": SECONDARY, "loss": GRAY}
    fractions = sorted({row["parameter_fraction"] for row in rows})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    width = 0.22
    x = np.arange(len(fractions))
    for offset, functional in zip((-1, 0, 1), labels):
        selected = sorted(
            [row for row in rows if row["ranking_functional"] == functional],
            key=lambda row: row["parameter_fraction"],
        )
        y = np.asarray([row["effect_mean"] for row in selected])
        low = np.asarray([row["ci95_low"] for row in selected])
        high = np.asarray([row["ci95_high"] for row in selected])
        ax.bar(
            x + offset * width,
            y,
            width,
            color=colors[functional],
            label=labels[functional],
        )
        ax.errorbar(
            x + offset * width,
            y,
            yerr=[y - low, high - y],
            fmt="none",
            ecolor=BLACK,
            capsize=2,
            linewidth=0.8,
        )
    ax.set_xticks(x, [f"{fraction * 100:g}%" for fraction in fractions])
    ax.set_xlabel("Ablated MLP parameters")
    ax.set_ylabel("Margin degradation")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    outside_legend(fig, [ax], ncol=len(labels))
    save_plot(fig, output)


def plot_compression(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for family, color, marker in (("Nyström", SECONDARY, "o"), ("RFF", GRAY, "x")):
        selected = sorted(
            [
                row
                for row in rows
                if row["kind"] == "cold_screen" and row["family"] == family
            ],
            key=lambda row: row["dimension"],
        )
        ax.plot(
            [row["dimension"] for row in selected],
            [row["mean"] for row in selected],
            color=color,
            marker=marker,
            label=family,
        )
    ax.axhline(0.90733, color=PRIMARY, linestyle="--", linewidth=1, label="Full kernel")
    ax.axhline(0.82718, color=LIGHT, linestyle=":", linewidth=1, label="Scalar")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Cold Spearman correlation")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)

    ax = axes[1]
    selected = [row for row in rows if row["kind"] == "causal_confirmation"]
    xlabels = ["ViT-B\n0.02%", "ViT-B\n0.1%", "ViT-L\n0.02%", "ViT-L\n0.1%"]
    keys = [
        ("ViT-B/16", 0.0002),
        ("ViT-B/16", 0.001),
        ("ViT-L/16", 0.0002),
        ("ViT-L/16", 0.001),
    ]
    styles = {
        "Full kernel": (PRIMARY, "o"),
        "Nyström": (SECONDARY, "s"),
        "Scalar": (LIGHT, "x"),
    }
    for family, (color, marker) in styles.items():
        current = {
            (row["model"], row["parameter_fraction"]): row
            for row in selected
            if row["family"] == family
        }
        y = np.asarray([current[key]["mean"] for key in keys])
        low = np.asarray([current[key]["ci95_low"] for key in keys])
        high = np.asarray([current[key]["ci95_high"] for key in keys])
        ax.errorbar(
            np.arange(len(keys)),
            y,
            yerr=[y - low, high - y],
            color=color,
            marker=marker,
            linestyle="-",
            capsize=2,
            label=family,
        )
    ax.set_xticks(np.arange(len(keys)), xlabels)
    ax.set_ylabel("Target margin degradation")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    outside_legend(fig, axes, ncol=4)
    save_plot(fig, output)


def plot_heldout(rows, output):
    methods = [
        "Sensitivity RBF",
        "Nearest sample",
        "Scalar sensitivity",
        "Target activation",
        "Direct gradient",
    ]
    styles = {
        "Sensitivity RBF": PRIMARY,
        "Nearest sample": SECONDARY,
        "Scalar sensitivity": LIGHT,
        "Target activation": GRAY,
        "Direct gradient": BLACK,
    }
    causal = [row for row in rows if row["scope"] == "causal"]
    fractions = sorted({row["parameter_fraction"] for row in causal})
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    width = 0.14
    x = np.arange(len(fractions))
    for index, method in enumerate(methods):
        current = sorted(
            [row for row in causal if row["method"] == method],
            key=lambda row: row["parameter_fraction"],
        )
        y = np.asarray([row["mean"] for row in current])
        low = np.asarray([row["ci95_low"] for row in current])
        high = np.asarray([row["ci95_high"] for row in current])
        location = x + (index - 2) * width
        ax.bar(location, y, width, color=styles[method], label=method)
        ax.errorbar(
            location,
            y,
            yerr=[y - low, high - y],
            fmt="none",
            ecolor=BLACK,
            capsize=2,
            linewidth=0.8,
        )
    ax.set_xticks(x, [f"{fraction * 100:g}%" for fraction in fractions])
    ax.set_xlabel("Ablated MLP parameters")
    ax.set_ylabel("Held-out-class margin degradation")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    outside_legend(fig, [ax], ncol=len(methods), fontsize=6.5)
    save_plot(fig, output)


def build_atlas_size_table(path, cost_path=None):
    """Flatten the source-atlas scaling result for plotting and reuse."""
    payload = read_json(path)
    cost = read_json(cost_path) if cost_path is not None else None
    cost_reference_n = cost["protocol"]["source_examples"] if cost else None
    offline = cost["offline"] if cost else None

    def amortization(source_examples):
        if offline is None:
            return {
                "estimated_profile_seconds": "",
                "estimated_single_break_even_queries": "",
                "estimated_batched_break_even_queries": "",
            }
        scale = source_examples / cost_reference_n
        return {
            "estimated_profile_seconds": offline["source_profile_seconds"] * scale,
            "estimated_single_break_even_queries": (
                offline["break_even_queries_from_profile_cost_only"] * scale
            ),
            "estimated_batched_break_even_queries": (
                offline["batched_break_even_queries_from_profile_cost_only"] * scale
            ),
        }

    rows = []
    for source_examples in payload["protocol"]["source_counts"]:
        current = payload["development"][str(source_examples)]
        for method in ("exact_rbf", "prototype_rbf", "scalar_mass"):
            rows.append(
                {
                    "split": "development",
                    "source_examples": source_examples,
                    "kind": "cold_spearman",
                    "comparison": "measured_profile",
                    "method": method,
                    "parameter_fraction": "",
                    "mean": current["cold_query"][method]["mean_spearman"],
                    "ci95_low": "",
                    "ci95_high": "",
                    "queries": current["cold_query"][method]["queries"],
                    **amortization(source_examples),
                }
            )
        for key, comparison in (
            ("paired_causal_vs_scalar", "scalar"),
            ("paired_causal_vs_matched_permutation", "shuffled_pairing"),
        ):
            for method in ("exact_rbf", "prototype_rbf"):
                for fraction, summary in current[key][method].items():
                    rows.append(
                        {
                            "split": "development",
                            "source_examples": source_examples,
                            "kind": "causal_gain",
                            "comparison": comparison,
                            "method": method,
                            "parameter_fraction": float(fraction),
                            "mean": summary["mean"],
                            "ci95_low": summary["lower_95"],
                            "ci95_high": summary["upper_95"],
                            "queries": summary["n"],
                            **amortization(source_examples),
                        }
                    )
    confirmation = payload["confirmation"]
    source_examples = confirmation["source_examples"]
    for method in ("exact_rbf", "prototype_rbf", "scalar_mass"):
        rows.append(
            {
                "split": "confirmation",
                "source_examples": source_examples,
                "kind": "cold_spearman",
                "comparison": "measured_profile",
                "method": method,
                "parameter_fraction": "",
                "mean": confirmation["cold_query"][method]["mean_spearman"],
                "ci95_low": "",
                "ci95_high": "",
                "queries": confirmation["cold_query"][method]["queries"],
                **amortization(source_examples),
            }
        )
    for key, comparison in (
        ("paired_causal_vs_scalar", "scalar"),
        ("paired_causal_vs_matched_permutation", "shuffled_pairing"),
    ):
        for method in ("exact_rbf", "prototype_rbf"):
            for fraction, summary in confirmation[key][method].items():
                rows.append(
                    {
                        "split": "confirmation",
                        "source_examples": source_examples,
                        "kind": "causal_gain",
                        "comparison": comparison,
                        "method": method,
                        "parameter_fraction": float(fraction),
                        "mean": summary["mean"],
                        "ci95_low": summary["lower_95"],
                        "ci95_high": summary["upper_95"],
                        "queries": summary["n"],
                        **amortization(source_examples),
                    }
                )
    return rows, payload["selection"]["source_examples"]


def plot_atlas_size(rows, selected_source_examples, output):
    """Paper-style source-sample scaling figure used by the evidence bundle."""
    development = [row for row in rows if row["split"] == "development"]
    counts = sorted({row["source_examples"] for row in development})
    styles = {
        "exact_rbf": (PRIMARY, "o", "Exact DINO RBF"),
        "prototype_rbf": (SECONDARY, "s", "Prototype map"),
        "scalar_mass": (LIGHT, "x", "Scalar sensitivity"),
    }
    has_cost = any(
        row["estimated_batched_break_even_queries"] != "" for row in development
    )
    panel_count = 4 if has_cost else 3
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(11.0 if has_cost else 8.4, 2.65),
    )
    axes[0].text(0.03, 0.95, "Cold retrieval", transform=axes[0].transAxes, va="top")
    for method in ("exact_rbf", "prototype_rbf", "scalar_mass"):
        selected = sorted(
            [
                row
                for row in development
                if row["kind"] == "cold_spearman" and row["method"] == method
            ],
            key=lambda row: row["source_examples"],
        )
        color, marker, label = styles[method]
        axes[0].plot(
            [row["source_examples"] for row in selected],
            [row["mean"] for row in selected],
            color=color,
            marker=marker,
            linewidth=1.2,
            markersize=3.5,
            label=label,
        )
    axes[0].set_ylabel("Spearman correlation")

    for axis, comparison, panel_label in (
        (axes[1], "scalar", "Causal gain vs scalar"),
        (axes[2], "shuffled_pairing", "Causal gain vs shuffled pairing"),
    ):
        axis.text(0.03, 0.95, panel_label, transform=axis.transAxes, va="top")
        for method in ("exact_rbf", "prototype_rbf"):
            color, marker, label = styles[method]
            fractions = sorted(
                {
                    row["parameter_fraction"]
                    for row in development
                    if row["kind"] == "causal_gain"
                }
            )
            for fraction, linestyle in zip(fractions, ("-", "--")):
                selected = sorted(
                    [
                        row
                        for row in development
                        if row["kind"] == "causal_gain"
                        and row["comparison"] == comparison
                        and row["method"] == method
                        and row["parameter_fraction"] == fraction
                    ],
                    key=lambda row: row["source_examples"],
                )
                x = np.asarray([row["source_examples"] for row in selected])
                y = np.asarray([row["mean"] for row in selected])
                low = np.asarray([row["ci95_low"] for row in selected])
                high = np.asarray([row["ci95_high"] for row in selected])
                axis.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.2,
                    markersize=3,
                    label=f"{label}, {fraction * 100:g}%",
                )
                axis.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
        axis.axhline(0, color="#D5D5D5", linewidth=0.7)
        axis.set_ylabel("Paired margin-degradation gain")
    if has_cost:
        costs = {}
        for row in development:
            costs.setdefault(
                row["source_examples"],
                row["estimated_batched_break_even_queries"],
            )
        axes[3].text(
            0.03,
            0.95,
            "Amortization cost",
            transform=axes[3].transAxes,
            va="top",
        )
        axes[3].plot(
            counts,
            [costs[count] for count in counts],
            color=BLACK,
            marker="D",
            linewidth=1.2,
            markersize=3.5,
        )
        axes[3].set_ylabel("Batched break-even queries")
        axes[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    legend_handles = [
        Line2D([0], [0], color=PRIMARY, marker="o", linewidth=1.2, markersize=3.5),
        Line2D([0], [0], color=SECONDARY, marker="s", linewidth=1.2, markersize=3.5),
        Line2D([0], [0], color=LIGHT, marker="x", linewidth=1.2, markersize=3.5),
        Line2D([0], [0], color=BLACK, linestyle="-", linewidth=1.2),
        Line2D([0], [0], color=BLACK, linestyle="--", linewidth=1.2),
    ]
    fig.legend(
        legend_handles,
        [
            "Exact DINO RBF",
            "Prototype map",
            "Scalar sensitivity",
            "0.02% budget",
            "0.1% budget",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=5,
        frameon=False,
        fontsize=7,
    )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            counts, [f"{count:,}" for count in counts], rotation=35, ha="right"
        )
        axis.set_xlabel("Source images, $N$")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        axis.axvline(
            selected_source_examples,
            color=BLACK,
            linestyle=":",
            linewidth=0.8,
            alpha=0.8,
        )
    axes[-1].text(
        selected_source_examples,
        0.04,
        f"High fidelity $N={selected_source_examples:,}$",
        transform=axes[-1].get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=6.2,
    )
    if has_cost:
        axes[-1].axvline(
            1600,
            color=SECONDARY,
            linestyle="--",
            linewidth=0.9,
        )
        axes[-1].text(
            1600,
            0.04,
            "Cost-balanced $N=1,600$",
            transform=axes[-1].get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=6.2,
            color=SECONDARY,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_plot(fig, output)


def build_language_confirmation_tables(path):
    """Flatten the frozen MMLU functional and semantic-space confirmations."""
    payload = read_json(path)
    functional_rows = []
    for functional, result in payload["functional_confirmation"].items():
        for fraction, budget in result["budgets"].items():
            for method in ("exact_rbf", "prototype"):
                for key, comparator in (
                    ("vs_scalar", "scalar"),
                    ("vs_permutation", "shuffled_pairing"),
                    ("vs_subject_onehot", "subject_onehot"),
                ):
                    summary = budget[method][key]
                    functional_rows.append(
                        {
                            "functional": functional,
                            "method": method,
                            "parameter_fraction": float(fraction),
                            "comparator": comparator,
                            "mean": summary["mean"],
                            "ci95_low": summary["lower_95"],
                            "ci95_high": summary["upper_95"],
                            "queries": summary["n"],
                            "cold_spearman": result["cold_query"][method][
                                "mean_spearman"
                            ],
                        }
                    )
    encoder_rows = []
    for encoder, result in payload["independent_encoder_confirmation"].items():
        for fraction, budget in result["budgets"].items():
            for key, comparator in (
                ("vs_scalar", "scalar"),
                ("vs_permutation", "shuffled_pairing"),
                ("vs_subject_onehot", "subject_onehot"),
            ):
                summary = budget["encoder_rbf"][key]
                encoder_rows.append(
                    {
                        "encoder": encoder,
                        "source_examples": result["source_examples"],
                        "rbf_scale": result["scale"],
                        "parameter_fraction": float(fraction),
                        "comparator": comparator,
                        "mean": summary["mean"],
                        "ci95_low": summary["lower_95"],
                        "ci95_high": summary["upper_95"],
                        "queries": summary["n"],
                        "cold_spearman": result["cold_query"]["encoder_rbf"][
                            "mean_spearman"
                        ],
                    }
                )
    return functional_rows, encoder_rows


def _plot_interval_series(axis, selected, x, offset, color, marker, label):
    means = np.asarray([row["mean"] for row in selected])
    lows = np.asarray([row["ci95_low"] for row in selected])
    highs = np.asarray([row["ci95_high"] for row in selected])
    axis.errorbar(
        x + offset,
        means,
        yerr=[means - lows, highs - means],
        fmt=marker,
        color=color,
        ecolor=color,
        capsize=2,
        markersize=4,
        linewidth=0.9,
        label=label,
    )


def plot_language_functionals(rows, output):
    functionals = ("margin", "answer_loss", "correct_logit")
    labels = ("Margin", "Answer loss", "Correct logit")
    series = (
        ("exact_rbf", "scalar", PRIMARY, "o", "Exact RBF vs scalar"),
        (
            "exact_rbf",
            "shuffled_pairing",
            PRIMARY,
            "x",
            "Exact RBF vs shuffled",
        ),
        ("prototype", "scalar", SECONDARY, "s", "Prototype vs scalar"),
        (
            "prototype",
            "shuffled_pairing",
            SECONDARY,
            "+",
            "Prototype vs shuffled",
        ),
    )
    fractions = sorted({row["parameter_fraction"] for row in rows})
    fig, axes = plt.subplots(1, len(fractions), figsize=(7.2, 2.65), sharey=True)
    x = np.arange(len(functionals))
    offsets = np.linspace(-0.18, 0.18, len(series))
    for axis, fraction in zip(np.asarray(axes).reshape(-1), fractions):
        for offset, (method, comparator, color, marker, label) in zip(offsets, series):
            selected = [
                next(
                    row
                    for row in rows
                    if row["functional"] == functional
                    and row["method"] == method
                    and row["comparator"] == comparator
                    and row["parameter_fraction"] == fraction
                )
                for functional in functionals
            ]
            _plot_interval_series(axis, selected, x, offset, color, marker, label)
        axis.axhline(0, color="#D5D5D5", linewidth=0.7)
        axis.set_xticks(x, labels)
        axis.set_title(f"{fraction * 100:g}% parameter budget", fontsize=8)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel("Paired margin-degradation gain")
    outside_legend(fig, axes, ncol=len(series), fontsize=6.6)
    save_plot(fig, output)


def plot_language_semantic_spaces(rows, output):
    encoders = ("qwen3_embedding_4b_answer", "modernbert_answer")
    labels = ("Qwen3-Embedding-4B", "ModernBERT-base")
    series = (
        ("scalar", PRIMARY, "o", "vs scalar"),
        ("shuffled_pairing", SECONDARY, "s", "vs shuffled pairing"),
        ("subject_onehot", GRAY, "^", "vs subject one-hot"),
    )
    fractions = sorted({row["parameter_fraction"] for row in rows})
    fig, axes = plt.subplots(1, len(fractions), figsize=(7.2, 2.65), sharey=True)
    x = np.arange(len(encoders))
    offsets = np.linspace(-0.13, 0.13, len(series))
    for axis, fraction in zip(np.asarray(axes).reshape(-1), fractions):
        for offset, (comparator, color, marker, label) in zip(offsets, series):
            selected = [
                next(
                    row
                    for row in rows
                    if row["encoder"] == encoder
                    and row["comparator"] == comparator
                    and row["parameter_fraction"] == fraction
                )
                for encoder in encoders
            ]
            _plot_interval_series(axis, selected, x, offset, color, marker, label)
        axis.axhline(0, color="#D5D5D5", linewidth=0.7)
        axis.set_xticks(x, labels)
        axis.set_title(f"{fraction * 100:g}% parameter budget", fontsize=8)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel("Paired margin-degradation gain")
    outside_legend(fig, axes, ncol=len(series), fontsize=7)
    save_plot(fig, output)


def build_language_mechinterp_table(studies):
    """Flatten language attribution-reference and exact-intervention results."""
    rows = []
    for model_label, ablation, path in studies:
        payload = read_json(path)
        for fraction, budget in payload["budgets"].items():
            for key, label in (
                ("activation_taylor_vs_scalar", "activation_attribution"),
                ("direct_parameter_vs_scalar", "direct_parameter"),
                ("activation_x_gradient_vs_scalar", "activation_x_gradient"),
                ("activation_magnitude_vs_scalar", "activation_magnitude"),
            ):
                summary = budget["reference_comparisons"][key]
                rows.append(
                    {
                        "model": model_label,
                        "ablation": ablation,
                        "parameter_fraction": float(fraction),
                        "method": label,
                        "comparator": "scalar",
                        "mean": summary["mean"],
                        "ci95_low": summary["lower_95"],
                        "ci95_high": summary["upper_95"],
                        "queries": summary["n"],
                        "activation_attribution_gap_recovered": "",
                        "direct_parameter_gap_recovered": "",
                    }
                )
            for family, comparisons in budget["atlas_comparisons"].items():
                for key, comparator in (
                    ("vs_scalar", "scalar"),
                    ("vs_shuffled_pairing", "shuffled_pairing"),
                    ("vs_subject_onehot", "subject_onehot"),
                    ("vs_nearest", "nearest"),
                ):
                    summary = comparisons[key]
                    rows.append(
                        {
                            "model": model_label,
                            "ablation": ablation,
                            "parameter_fraction": float(fraction),
                            "method": family,
                            "comparator": comparator,
                            "mean": summary["mean"],
                            "ci95_low": summary["lower_95"],
                            "ci95_high": summary["upper_95"],
                            "queries": summary["n"],
                            "activation_attribution_gap_recovered": comparisons[
                                "activation_taylor_gap_recovered"
                            ],
                            "direct_parameter_gap_recovered": comparisons.get(
                                "direct_parameter_gap_recovered", ""
                            ),
                        }
                    )
    return rows


def build_vision_mechinterp_table(path):
    """Flatten the matched ViT zero-deactivation confirmation."""
    payload = read_json(path)
    rows = []
    method_names = {
        "direct_coordinate_taylor": "activation_attribution",
        "direct_gradient_oracle": "direct_parameter",
        "exact_rbf_scale_0.1": "target_hidden_exact",
        "prototype_800_scale_0.025": "target_hidden_prototype",
    }
    for fraction in payload["protocol"]["fractions"]:
        fraction_key = f"{fraction:g}"
        for source_name, method in method_names.items():
            summary = payload["paired_causal_vs_scalar"][source_name][fraction_key]
            gap = (
                payload.get("gap_recovery", {})
                .get(source_name, {})
                .get(fraction_key, {})
            )
            rows.append(
                {
                    "model": "ViT-B/16",
                    "ablation": "zero",
                    "parameter_fraction": fraction,
                    "method": method,
                    "comparator": "scalar",
                    "mean": summary["mean"],
                    "ci95_low": summary["lower_95"],
                    "ci95_high": summary["upper_95"],
                    "queries": summary["n"],
                    "activation_attribution_gap_recovered": gap.get(
                        "activation_attribution", ""
                    ),
                    "direct_parameter_gap_recovered": gap.get(
                        "direct_parameter_sensitivity", ""
                    ),
                }
            )
        for source_name, method in (
            ("exact_rbf_scale_0.1", "target_hidden_exact"),
            ("prototype_800_scale_0.025", "target_hidden_prototype"),
        ):
            for key, comparator in (
                ("paired_causal_vs_matched_permutation", "shuffled_pairing"),
                ("paired_causal_vs_nearest", "nearest"),
                ("paired_causal_vs_class_onehot", "class_onehot"),
            ):
                summary = payload[key][source_name][fraction_key]
                gap = payload["gap_recovery"][source_name][fraction_key]
                rows.append(
                    {
                        "model": "ViT-B/16",
                        "ablation": "zero",
                        "parameter_fraction": fraction,
                        "method": method,
                        "comparator": comparator,
                        "mean": summary["mean"],
                        "ci95_low": summary["lower_95"],
                        "ci95_high": summary["upper_95"],
                        "queries": summary["n"],
                        "activation_attribution_gap_recovered": gap[
                            "activation_attribution"
                        ],
                        "direct_parameter_gap_recovered": gap[
                            "direct_parameter_sensitivity"
                        ],
                    }
                )
    return rows


def build_parameter_influence_table(path):
    """Flatten the parameter-aligned local-intervention confirmation."""
    payload = read_json(path)
    records = payload["records"]

    def query_means(method, fraction, metric):
        grouped = {}
        for row in records:
            if (
                row["target_correct"]
                and row["method"] == method
                and row["fraction"] == fraction
            ):
                grouped.setdefault(int(row["query_column"]), []).append(row[metric])
        return {key: float(np.mean(values)) for key, values in grouped.items()}

    def paired(method, baseline, fraction, metric="target_squared_susceptibility"):
        left = query_means(method, fraction, metric)
        right = query_means(baseline, fraction, metric)
        return bootstrap([left[key] - right[key] for key in left.keys() & right.keys()])

    rows = []
    methods = {
        "direct_gradient_oracle": "direct_parameter",
        "exact_rbf_scale_0.1": "target_hidden_exact",
        "prototype_800_scale_0.025": "target_hidden_prototype",
    }
    for fraction in payload["protocol"]["fractions"]:
        key = f"{fraction:g}"
        for source, method in methods.items():
            mean, low, high, queries = paired(source, "scalar_mass", fraction)
            rows.append(
                {
                    "model": "ViT-B/16",
                    "assay": "parameter_influence",
                    "parameter_fraction": fraction,
                    "method": method,
                    "comparator": "scalar",
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "queries": queries,
                    "direct_parameter_gap_recovered": (
                        1.0
                        if source == "direct_gradient_oracle"
                        else payload["comparisons"][key][source]["oracle_gap_recovered"]
                    ),
                }
            )
        for source, method in tuple(methods.items())[1:]:
            comparisons = payload["comparisons"][key][source]
            for field, comparator in (
                ("vs_shuffled_pairing", "shuffled_pairing"),
                ("vs_nearest", "nearest"),
                ("vs_class_onehot", "class_onehot"),
                ("off_class_selectivity", "off_class"),
            ):
                summary = comparisons[field]
                rows.append(
                    {
                        "model": "ViT-B/16",
                        "assay": "parameter_influence",
                        "parameter_fraction": fraction,
                        "method": method,
                        "comparator": comparator,
                        "mean": summary["mean"],
                        "ci95_low": summary["lower_95"],
                        "ci95_high": summary["upper_95"],
                        "queries": summary["n"],
                        "direct_parameter_gap_recovered": comparisons[
                            "oracle_gap_recovered"
                        ],
                    }
                )
    return rows


def plot_vision_mechanistic_assays(deactivation_rows, influence_rows, output):
    """Compare the complementary activation- and parameter-space assays."""
    methods = (
        ("direct_parameter", GRAY, "D", "Direct parameter sensitivity"),
        ("target_hidden_exact", PRIMARY, "o", "Exact representation RBF"),
        ("target_hidden_prototype", SECONDARY, "s", "Fixed prototype atlas"),
    )
    panels = (
        (deactivation_rows, "Neuron deactivation"),
        (influence_rows, "Parameter influence"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.7), sharex="col")
    for column, (rows, title) in enumerate(panels):
        fractions = sorted({row["parameter_fraction"] for row in rows})
        x = np.arange(len(fractions))
        for method, color, marker, label in methods:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["method"] == method and row["comparator"] == "scalar"
                ],
                key=lambda row: row["parameter_fraction"],
            )
            _plot_interval_series(axes[0, column], selected, x, 0, color, marker, label)
        for method, color, marker, label in methods[1:]:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["comparator"] == "shuffled_pairing"
                ],
                key=lambda row: row["parameter_fraction"],
            )
            _plot_interval_series(axes[1, column], selected, x, 0, color, marker, label)
        axes[0, column].set_title(title, fontsize=8)
        for row in range(2):
            axes[row, column].axhline(0, color="#D5D5D5", linewidth=0.7)
            axes[row, column].grid(axis="y", color="#E5E5E5", linewidth=0.6)
        axes[1, column].set_xticks(
            x, [f"{fraction * 100:g}%" for fraction in fractions]
        )
        axes[1, column].set_xlabel("MLP parameter budget")
    axes[0, 0].set_ylabel("Causal effect gain\nvs scalar")
    axes[1, 0].set_ylabel("Causal effect gain\nvs shuffled pairing")
    outside_legend(fig, axes, ncol=len(methods), fontsize=6.8)
    save_plot(fig, output)


def build_resolution_table(path):
    """Flatten nested-resolution utility, fidelity, and storage results."""
    payload = read_json(path)
    rows = []
    for resolution in payload["protocol"]["resolutions"]:
        metadata = payload["resolution_metadata"][resolution]
        for assay, oracle in (
            ("neuron_deactivation", "activation_attribution"),
            ("parameter_influence", "direct_parameter"),
        ):
            for method in (oracle, "exact_rbf", "prototype_rbf"):
                values = payload["comparisons"][assay][resolution][method]
                summary = values["vs_scalar"]
                rows.append(
                    {
                        "resolution": resolution,
                        "groups": metadata["groups"],
                        "mean_parameters_per_group": metadata[
                            "mean_parameters_per_group"
                        ],
                        "atlas_bytes_float32": metadata["atlas_bytes_float32"],
                        "assay": assay,
                        "method": method,
                        "metric": "gain_vs_scalar",
                        "mean": summary["mean"],
                        "ci95_low": summary["lower_95"],
                        "ci95_high": summary["upper_95"],
                        "queries": summary["n"],
                        "oracle_gap_recovered": values.get("oracle_gap_recovered", ""),
                    }
                )
        for method in ("exact_rbf", "prototype_rbf", "scalar_mass"):
            values = payload["query_fidelity"][resolution][method]["spearman"]
            mean, low, high, count = bootstrap(values)
            rows.append(
                {
                    "resolution": resolution,
                    "groups": metadata["groups"],
                    "mean_parameters_per_group": metadata["mean_parameters_per_group"],
                    "atlas_bytes_float32": metadata["atlas_bytes_float32"],
                    "assay": "query_fidelity",
                    "method": method,
                    "metric": "spearman",
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "queries": count,
                    "oracle_gap_recovered": "",
                }
            )
    return rows


def plot_resolution_sweep(rows, output):
    resolutions = ("feature", "bundle_4", "bundle_32", "mlp_block")
    labels = ("Feature", "4 features", "32 features", "MLP block")
    x = np.asarray(
        [
            next(
                row["mean_parameters_per_group"]
                for row in rows
                if row["resolution"] == resolution
            )
            for resolution in resolutions
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    panels = (
        (
            "neuron_deactivation",
            (
                ("activation_attribution", GRAY, "D", "Direct reference"),
                ("exact_rbf", PRIMARY, "o", "Exact representation RBF"),
                ("prototype_rbf", SECONDARY, "s", "Fixed prototype atlas"),
            ),
            "Neuron-deactivation gain\nvs scalar",
        ),
        (
            "parameter_influence",
            (
                ("direct_parameter", GRAY, "D", "Direct reference"),
                ("exact_rbf", PRIMARY, "o", "Exact representation RBF"),
                ("prototype_rbf", SECONDARY, "s", "Fixed prototype atlas"),
            ),
            "Parameter-influence gain\nvs scalar",
        ),
        (
            "query_fidelity",
            (
                ("scalar_mass", BLACK, "x", "Scalar sensitivity"),
                ("exact_rbf", PRIMARY, "o", "Exact representation RBF"),
                ("prototype_rbf", SECONDARY, "s", "Fixed prototype atlas"),
            ),
            "Cold-profile Spearman",
        ),
    )
    for axis, (assay, methods, ylabel) in zip(axes, panels):
        for method, color, marker, label in methods:
            selected = [
                next(
                    row
                    for row in rows
                    if row["resolution"] == resolution
                    and row["assay"] == assay
                    and row["method"] == method
                )
                for resolution in resolutions
            ]
            means = np.asarray([row["mean"] for row in selected])
            low = means - np.asarray([row["ci95_low"] for row in selected])
            high = np.asarray([row["ci95_high"] for row in selected]) - means
            axis.errorbar(
                x,
                means,
                yerr=(low, high),
                color=color,
                marker=marker,
                linewidth=1.4,
                markersize=4,
                capsize=2,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_xticks(x, labels, rotation=22, ha="right")
        axis.set_xlabel("Parameter-group resolution")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        if assay != "query_fidelity":
            axis.axhline(0, color="#D5D5D5", linewidth=0.7)
    outside_legend(fig, axes, ncol=4, fontsize=6.8)
    save_plot(fig, output)


def build_cost_accuracy_table(path):
    payload = read_json(path)
    online = payload["online"]
    accuracy = payload["accuracy"]
    rows = [
        {
            "method": "Activation attribution",
            "parameter_budget": "per-query reference",
            "attribution_gap_recovered": 1.0,
            "batch_images_per_second": online["batched_activation_attribution"][
                "mean_images_per_second"
            ],
            "peak_batch_bytes": online["batched_activation_attribution"][
                "peak_allocated_bytes"
            ],
            "persistent_atlas_bytes": 0,
        }
    ]
    feature_online = online["batched_compact_atlas"]["feature"]
    feature_single = online["compact_atlas"]["feature"]
    for fraction, values in accuracy["feature_resolution"].items():
        rows.append(
            {
                "method": "Feature atlas",
                "parameter_budget": f"{float(fraction) * 100:g}%",
                "attribution_gap_recovered": values["prototype"],
                "batch_images_per_second": feature_online["mean_images_per_second"],
                "peak_batch_bytes": feature_online["peak_allocated_bytes"],
                "persistent_atlas_bytes": feature_single["persistent_atlas_bytes"],
            }
        )
    bundle_online = online["batched_compact_atlas"]["bundle_32"]
    bundle_single = online["compact_atlas"]["bundle_32"]
    rows.append(
        {
            "method": "32-feature bundle atlas",
            "parameter_budget": "1/12 MLP parameters",
            "attribution_gap_recovered": accuracy[
                "bundle_32_at_parameter_fraction_1_over_12"
            ]["prototype"],
            "batch_images_per_second": bundle_online["mean_images_per_second"],
            "peak_batch_bytes": bundle_online["peak_allocated_bytes"],
            "persistent_atlas_bytes": bundle_single["persistent_atlas_bytes"],
        }
    )
    return rows


def plot_cost_accuracy(rows, output):
    colors = {
        "Activation attribution": GRAY,
        "Feature atlas": PRIMARY,
        "32-feature bundle atlas": SECONDARY,
    }
    markers = {
        "Activation attribution": "D",
        "Feature atlas": "o",
        "32-feature bundle atlas": "s",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for row in rows:
        label = row["method"]
        axes[0].scatter(
            row["batch_images_per_second"],
            row["attribution_gap_recovered"],
            color=colors[label],
            marker=markers[label],
            label=label,
            zorder=3,
        )
        axes[1].scatter(
            row["peak_batch_bytes"] / 2**30,
            row["attribution_gap_recovered"],
            color=colors[label],
            marker=markers[label],
            label=label,
            zorder=3,
        )
    axes[0].set_xlabel("Batched query throughput (images/s)")
    axes[1].set_xlabel("Peak batched GPU memory (GiB)")
    axes[0].set_ylabel("Activation-attribution gap recovered")
    axes[1].set_ylabel("Activation-attribution gap recovered")
    for axis in axes:
        axis.set_ylim(0.5, 1.04)
        axis.grid(color="#E5E5E5", linewidth=0.6)
    outside_legend(fig, axes, ncol=3, fontsize=6.8)
    save_plot(fig, output)


def build_normalization_table(path):
    payload = read_json(path)
    rows = []
    for fraction in payload["protocol"]["fractions"]:
        current = payload["comparisons"][f"{fraction:g}"]
        for normalization in ("normalized", "raw"):
            for method in ("exact", "prototype"):
                values = current[normalization][method]
                summary = values["vs_scalar"]
                rows.append(
                    {
                        "parameter_fraction": fraction,
                        "normalization": normalization,
                        "method": method,
                        "activation_gap_recovered": values["activation_gap_recovered"],
                        "gain_vs_scalar": summary["mean"],
                        "ci95_low": summary["lower_95"],
                        "ci95_high": summary["upper_95"],
                        "queries": summary["n"],
                    }
                )
    return rows


def plot_normalization(rows, output):
    fig, axis = plt.subplots(figsize=(4.4, 2.8))
    styles = (
        ("normalized", "exact", PRIMARY, "o", "Normalized, exact RBF"),
        ("normalized", "prototype", SECONDARY, "s", "Normalized, prototype"),
        ("raw", "exact", GRAY, "o", "Raw, exact RBF"),
        ("raw", "prototype", LIGHT, "s", "Raw, prototype"),
    )
    fractions = sorted({row["parameter_fraction"] for row in rows})
    x = np.arange(len(fractions))
    for normalization, method, color, marker, label in styles:
        selected = sorted(
            [
                row
                for row in rows
                if row["normalization"] == normalization and row["method"] == method
            ],
            key=lambda row: row["parameter_fraction"],
        )
        axis.plot(
            x,
            [row["activation_gap_recovered"] for row in selected],
            color=color,
            marker=marker,
            linewidth=1.4,
            markersize=4,
            label=label,
        )
    axis.set_xticks(x, [f"{value * 100:g}%" for value in fractions])
    axis.set_xlabel("MLP parameter budget")
    axis.set_ylabel("Activation-attribution gap recovered")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    outside_legend(fig, [axis], ncol=4, fontsize=6.5)
    save_plot(fig, output)


def build_imagenet1k_circuit_table(screen_path, causal_path):
    screen = read_json(screen_path)
    causal = read_json(causal_path)
    methods = {
        "class_onehot": "Categorical class space",
        "class_plus_prototype_1000_scale_0.1_alpha_0.25": (
            "Categorical + semantic space"
        ),
        "combined_class_prototype": "Categorical + semantic space",
        "prototype": "Semantic atlas",
        "nearest": "Nearest example",
        "scalar": "Scalar sensitivity",
        "direct": "Direct per-query gradient",
    }
    rows = []
    for count in screen["protocol"]["group_counts"]:
        key = str(count)
        for source in (
            "class_onehot",
            screen["selected_method"],
            "nearest",
            "scalar",
            "direct",
        ):
            if source not in screen["confirmation"]:
                continue
            values = screen["confirmation"][source][key]["coverage"]
            mean, low, high, queries = bootstrap(values)
            rows.append(
                {
                    "endpoint": "direct_gradient_coverage",
                    "selected_groups": count,
                    "method": methods[source],
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "queries": queries,
                }
            )
        for source, summary in causal["mean_susceptibility"][key].items():
            if source not in methods:
                continue
            rows.append(
                {
                    "endpoint": "exact_parameter_susceptibility",
                    "selected_groups": count,
                    "method": methods[source],
                    "mean": summary["mean"],
                    "ci95_low": summary["lower_95"],
                    "ci95_high": summary["upper_95"],
                    "queries": summary["n"],
                }
            )
        for source in ("combined_class_prototype", "prototype"):
            summary = causal["comparisons"][key][source]["vs_class_onehot"]
            rows.append(
                {
                    "endpoint": "exact_gain_over_categorical_class_space",
                    "selected_groups": count,
                    "method": methods[source],
                    "mean": summary["mean"],
                    "ci95_low": summary["lower_95"],
                    "ci95_high": summary["upper_95"],
                    "queries": summary["n"],
                }
            )
    return rows


def plot_imagenet1k_circuits(rows, output):
    methods = (
        ("Scalar sensitivity", BLACK, "x"),
        ("Nearest example", LIGHT, "^"),
        ("Categorical class space", GRAY, "D"),
        ("Semantic atlas", SECONDARY, "s"),
        ("Categorical + semantic space", PRIMARY, "o"),
        ("Direct per-query gradient", "#C53D43", "P"),
    )
    panels = (
        ("direct_gradient_coverage", "Direct-gradient energy covered"),
        ("exact_parameter_susceptibility", "Exact local parameter susceptibility"),
        (
            "exact_gain_over_categorical_class_space",
            "Exact gain vs categorical space",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 2.9))
    for axis, (endpoint, ylabel) in zip(axes, panels):
        counts = sorted(
            {row["selected_groups"] for row in rows if row["endpoint"] == endpoint}
        )
        x = np.arange(len(counts))
        for method, color, marker in methods:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["endpoint"] == endpoint and row["method"] == method
                ],
                key=lambda row: row["selected_groups"],
            )
            if not selected:
                continue
            means = np.asarray([row["mean"] for row in selected])
            axis.errorbar(
                x,
                means,
                yerr=(
                    means - np.asarray([row["ci95_low"] for row in selected]),
                    np.asarray([row["ci95_high"] for row in selected]) - means,
                ),
                color=color,
                marker=marker,
                linewidth=1.35,
                markersize=4,
                capsize=2,
                label=method,
            )
        axis.set_xticks(x, counts)
        axis.set_xlabel("Selected MLP feature groups")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        if endpoint == "exact_gain_over_categorical_class_space":
            axis.axhline(0, color="#D5D5D5", linewidth=0.7)
    outside_legend(fig, axes, ncol=6, fontsize=5.8)
    save_plot(fig, output)


def build_interpretability_table(path):
    payload = read_json(path)
    metrics = payload["metrics"]
    methods = {
        "prototype": "Semantic atlas",
        "prototype_permuted": "Shuffled semantic pairing",
        "class_onehot": "Categorical class space",
        "nearest": "Nearest example",
        "scalar": "Scalar sensitivity",
    }
    rows = []
    for source, method in methods.items():
        summaries = {
            "direct_gradient_top_group_overlap": metrics["oracle_overlap"][source][
                "summary"
            ],
            "query_class_sensitivity_mass": metrics["class_semantics"][source][
                "query_class_sensitivity_mass"
            ]["summary"],
            "within_class_circuit_jaccard": metrics["circuit_stability"][source][
                "within_class"
            ],
            "between_class_circuit_jaccard": metrics["circuit_stability"][source][
                "between_class"
            ],
        }
        for metric, summary in summaries.items():
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": summary["mean"],
                    "ci95_low": summary["lower_95"],
                    "ci95_high": summary["upper_95"],
                    "queries_or_pairs": summary["n"],
                }
            )
    return rows


def build_backend_tables(screen_path, causal_path):
    screen = read_json(screen_path)
    screen_rows = []
    for name, values in screen["results"].items():
        for metric in ("spearman", "top_37_recall", "coverage_8", "coverage_37"):
            summary = values["metrics"][metric]
            screen_rows.append(
                {
                    "stage": "retained_profile_screen",
                    "method": name,
                    "family": values["family"],
                    "dimension": values["dimension"],
                    "metric": metric,
                    "mean": summary["mean"],
                    "ci95_low": summary["lower_95"],
                    "ci95_high": summary["upper_95"],
                    "queries": summary["n"],
                    "atlas_bytes_float32": values["atlas_bytes_float32"],
                }
            )

    causal = read_json(causal_path)
    labels = {
        "nystrom_400": "Nyström-400",
        "exact_rbf": "Exact RBF",
        "categorical": "Categorical space",
        "nearest": "Nearest example",
        "scalar": "Scalar sensitivity",
        "direct": "Direct per-query gradient",
    }
    causal_rows = []
    for fraction in causal["protocol"]["fractions"]:
        for source, method in labels.items():
            values = [
                row["degradation"]
                for row in causal["records"]
                if row["method"] == source
                and row["fraction"] == fraction
                and row["target_correct"]
            ]
            mean, low, high, queries = bootstrap(values)
            causal_rows.append(
                {
                    "stage": "fresh_causal_confirmation",
                    "method": method,
                    "family": source,
                    "dimension": 400 if source == "nystrom_400" else "",
                    "parameter_fraction": fraction,
                    "metric": "margin_degradation",
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "queries": queries,
                    "atlas_bytes_float32": "",
                }
            )
    return screen_rows, causal_rows


def plot_backend_comparison(screen_rows, causal_rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 2.9))
    backend_styles = (
        ("kmeans_nystrom", PRIMARY, "o", "k-means++ Nyström"),
        ("anchor_nystrom", SECONDARY, "s", "Anchor-Net Nyström"),
        ("tensor_sketch", LIGHT, "^", "TensorSketch"),
    )
    for axis, metric, ylabel in (
        (axes[0], "spearman", "Cold-profile Spearman"),
        (axes[1], "coverage_37", "Direct sensitivity covered (37 groups)"),
    ):
        for family, color, marker, label in backend_styles:
            selected = sorted(
                [
                    row
                    for row in screen_rows
                    if row["family"] == family and row["metric"] == metric
                ],
                key=lambda row: row["dimension"],
            )
            axis.plot(
                [row["dimension"] for row in selected],
                [row["mean"] for row in selected],
                color=color,
                marker=marker,
                linewidth=1.35,
                markersize=4,
                label=label,
            )
        exact = next(
            row
            for row in screen_rows
            if row["method"] == "exact_rbf" and row["metric"] == metric
        )
        linear = next(
            row
            for row in screen_rows
            if row["method"] == "linear_384" and row["metric"] == metric
        )
        axis.axhline(exact["mean"], color="#C53D43", linewidth=1.1, label="Exact RBF")
        axis.axhline(
            linear["mean"], color=BLACK, linewidth=1.0, linestyle=":", label="Linear"
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks([200, 400, 800], ["200", "400", "800"])
        axis.set_xlabel("Feature dimension")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)

    causal_styles = (
        ("Nyström-400", PRIMARY, "o"),
        ("Exact RBF", "#C53D43", "P"),
        ("Categorical space", SECONDARY, "s"),
        ("Nearest example", LIGHT, "^"),
        ("Scalar sensitivity", BLACK, "x"),
        ("Direct per-query gradient", GRAY, "D"),
    )
    fractions = sorted({row["parameter_fraction"] for row in causal_rows})
    x = np.arange(len(fractions))
    for method, color, marker in causal_styles:
        selected = sorted(
            [row for row in causal_rows if row["method"] == method],
            key=lambda row: row["parameter_fraction"],
        )
        means = np.asarray([row["mean"] for row in selected])
        axes[2].errorbar(
            x,
            means,
            yerr=(
                means - np.asarray([row["ci95_low"] for row in selected]),
                np.asarray([row["ci95_high"] for row in selected]) - means,
            ),
            color=color,
            marker=marker,
            linewidth=1.3,
            markersize=4,
            capsize=2,
            label=method,
        )
    axes[2].set_xticks(x, [f"{value * 100:g}%" for value in fractions])
    axes[2].set_xlabel("MLP parameter budget")
    axes[2].set_ylabel("Exact margin degradation")
    axes[2].grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[2].axhline(0, color="#D5D5D5", linewidth=0.7)
    outside_legend(fig, axes, ncol=10, fontsize=4.8)
    save_plot(fig, output)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_timeseries_circuit_table(path):
    """Flatten the frozen cross-dataset forecasting confirmation."""
    payload = read_json(path)
    rows = []
    for dataset, result in payload["datasets"].items():
        for method, value in result["profile_top25_coverage"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "budget": "profile_top25",
                    "selected_groups": 25,
                    "metric": "direct_gradient_top25_coverage",
                    "comparison": method,
                    "mean": value,
                    "ci95_low": "",
                    "ci95_high": "",
                    "queries": 96,
                }
            )
        for fraction, causal in result["causal"].items():
            budget = f"{100 * float(fraction):g}%"
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "selected_groups": causal["selected_groups"],
                    "metric": "oracle_gap_recovered",
                    "comparison": "exact_rbf",
                    "mean": causal["oracle_gap_recovered"],
                    "ci95_low": "",
                    "ci95_high": "",
                    "queries": 128,
                }
            )
            denominator = (
                causal["vs_scalar"]["mean"] - causal["vs_direct_gradient"]["mean"]
            )
            for comparison in ("vs_scalar", "vs_matched_shuffle", "vs_nearest"):
                values = causal[comparison]
                rows.append(
                    {
                        "dataset": dataset,
                        "budget": budget,
                        "selected_groups": causal["selected_groups"],
                        "metric": "paired_exact_gain_fraction_of_direct_scalar_gap",
                        "comparison": comparison,
                        "mean": values["mean"] / denominator,
                        "ci95_low": values["ci_low"] / denominator,
                        "ci95_high": values["ci_high"] / denominator,
                        "queries": values["n"],
                    }
                )
    return rows


def plot_timeseries_circuits(rows, output):
    """Plot cross-dataset profile and exact-circuit evidence."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.05))
    datasets = ("ETTh1", "Weather")
    profile_methods = (
        ("latent_rbf_0.25", "Sensitivity RBF", PRIMARY),
        ("scalar", "Scalar", GRAY),
        ("latent_rbf_0.25_shuffled", "Matched shuffle", LIGHT),
        ("nearest", "Nearest", SECONDARY),
        ("temporal_categorical", "Temporal categorical", "#C9853D"),
    )
    width = 0.15
    x = np.arange(len(datasets))
    for index, (key, label, color) in enumerate(profile_methods):
        values = [
            next(
                row["mean"]
                for row in rows
                if row["dataset"] == dataset
                and row["budget"] == "profile_top25"
                and row["comparison"] == key
            )
            for dataset in datasets
        ]
        axes[0].bar(
            x + (index - 2) * width,
            values,
            width,
            color=color,
            label=label,
        )
    axes[0].set_xticks(x, datasets)
    axes[0].set_ylabel("Top-25 coverage")
    axes[0].set_title("Held-out profiles")

    budget_labels = (("0.02%", "1 group"), ("0.1%", "3 groups"))
    width = 0.34
    for index, dataset in enumerate(datasets):
        values = [
            next(
                row["mean"]
                for row in rows
                if row["dataset"] == dataset
                and row["budget"] == budget
                and row["metric"] == "oracle_gap_recovered"
            )
            for budget, _ in budget_labels
        ]
        axes[1].bar(
            np.arange(2) + (index - 0.5) * width,
            np.asarray(values) * 100,
            width,
            color=(PRIMARY, SECONDARY)[index],
            label=dataset,
        )
    axes[1].set_xticks(np.arange(2), [label for _, label in budget_labels])
    axes[1].set_ylabel("Direct-gradient gap recovered (%)")
    axes[1].set_title("Exact parameter influence")

    comparisons = (
        ("vs_scalar", "vs scalar", GRAY),
        ("vs_matched_shuffle", "vs matched shuffle", LIGHT),
        ("vs_nearest", "vs nearest", "#C9853D"),
    )
    positions = np.arange(4)
    categories = [
        (dataset, budget) for dataset in datasets for budget, _ in budget_labels
    ]
    width = 0.23
    for index, (key, label, color) in enumerate(comparisons):
        selected = [
            next(
                row
                for row in rows
                if row["dataset"] == dataset
                and row["budget"] == budget
                and row["comparison"] == key
                and row["metric"] == "paired_exact_gain_fraction_of_direct_scalar_gap"
            )
            for dataset, budget in categories
        ]
        means = np.asarray([row["mean"] for row in selected]) * 100
        lows = np.asarray([row["ci95_low"] for row in selected]) * 100
        highs = np.asarray([row["ci95_high"] for row in selected]) * 100
        axes[2].bar(
            positions + (index - 1) * width,
            means,
            width,
            color=color,
            label=label,
            yerr=np.stack((means - lows, highs - means)),
            error_kw={"elinewidth": 0.7, "capsize": 1.5},
        )
    axes[2].axhline(0, color=BLACK, linewidth=0.7)
    axes[2].set_xticks(
        positions,
        ("ETT\n1 grp", "ETT\n3 grp", "Weather\n1 grp", "Weather\n3 grp"),
    )
    axes[2].tick_params(axis="x", labelsize=6.5)
    axes[2].set_ylabel("Paired gain / direct-scalar gap (%)")
    axes[2].set_title("Controls under exact perturbations")
    outside_legend(fig, axes, ncol=10, fontsize=4.7)
    save_plot(fig, output)


def build_protein_application_table(feature_path, influence_path):
    """Flatten the two frozen ESM2 causal confirmations."""
    studies = (
        (
            "feature_localization",
            read_json(feature_path),
            "activation_attribution",
            "activation_gap_recovered",
        ),
        (
            "parameter_influence",
            read_json(influence_path),
            "direct_parameter",
            "direct_parameter_gap_recovered",
        ),
    )
    labels = {
        "exact_rbf": "KPSA",
        "matched_shuffle": "Shuffled pairing",
        "nearest": "Nearest example",
        "scalar": "Scalar sensitivity",
        "categorical": "Residue category",
        "direct_parameter": "Direct parameter gradient",
        "activation_attribution": "Activation attribution",
    }
    rows = []
    for application, payload, oracle, gap_key in studies:
        for fraction, methods in payload["summaries"].items():
            selected_groups = next(
                record["selected_groups"]
                for record in payload["records"]
                if f"{record['fraction']:g}" == fraction
            )
            for method, summary in methods.items():
                rows.append(
                    {
                        "application": application,
                        "parameter_fraction": float(fraction),
                        "selected_groups": selected_groups,
                        "method": labels[method],
                        "mean": summary["mean"],
                        "ci95_low": summary["ci95_low"],
                        "ci95_high": summary["ci95_high"],
                        "queries": summary["n"],
                        "gap_recovered": (
                            payload["comparisons"][fraction]["exact_rbf"][gap_key]
                            if method == "exact_rbf"
                            else ""
                        ),
                        "reference": labels[oracle],
                    }
                )
    return rows


def plot_protein_applications(rows, output):
    """Show the same frozen protein atlas under both causal applications."""
    methods = (
        ("KPSA", PRIMARY, "o"),
        ("Nearest example", SECONDARY, "s"),
        ("Residue category", GRAY, "^"),
        ("Scalar sensitivity", BLACK, "x"),
        ("Shuffled pairing", LIGHT, "D"),
    )
    panels = (
        (
            "parameter_influence",
            "Parameter influence",
            "Squared local susceptibility",
            "Direct parameter gradient",
        ),
        (
            "feature_localization",
            "Neuron localization",
            "Absolute masked-margin change",
            "Activation attribution",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.9))
    for axis, (application, title, ylabel, reference) in zip(axes, panels):
        counts = sorted(
            {
                row["selected_groups"]
                for row in rows
                if row["application"] == application
            }
        )
        x = np.arange(len(counts))
        for method, color, marker in methods + ((reference, "#C53D43", "P"),):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["application"] == application and row["method"] == method
                ],
                key=lambda row: row["selected_groups"],
            )
            means = np.asarray([row["mean"] for row in selected])
            axis.errorbar(
                x,
                means,
                yerr=(
                    means - np.asarray([row["ci95_low"] for row in selected]),
                    np.asarray([row["ci95_high"] for row in selected]) - means,
                ),
                color=color,
                marker=marker,
                linewidth=1.35,
                markersize=4,
                capsize=2,
                label=method,
            )
        kpsa = sorted(
            [
                row
                for row in rows
                if row["application"] == application and row["method"] == "KPSA"
            ],
            key=lambda row: row["selected_groups"],
        )
        for location, row in zip(x, kpsa):
            horizontal_offset = 8 if location == 0 else 0
            axis.annotate(
                f"{100 * row['gap_recovered']:.0f}% gap",
                (location, row["mean"]),
                xytext=(horizontal_offset, 8),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=PRIMARY,
            )
        axis.set_xticks(x, counts)
        axis.set_xlabel("Selected ESM2 FF feature groups")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=8)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    outside_legend(fig, axes, ncol=7, fontsize=5.5)
    save_plot(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--strengthening-arc", type=Path)
    args = parser.parse_args()
    arc = args.arc.resolve()
    artifacts = arc / "artifacts"
    vision = artifacts / "vision"
    strengthening_arc = (
        args.strengthening_arc.resolve()
        if args.strengthening_arc
        else arc.parent / "06_strenghening"
    )
    atlas_size_path = (
        strengthening_arc / "artifacts" / "vision" / "vitb16_atlas_size_scaling.json"
    )
    language_final_path = (
        strengthening_arc
        / "artifacts"
        / "language"
        / "mmlu_qwen3_1_7b_final_confirmations.json"
    )
    vision_zero_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_prototype_zero_ablation_confirmation.json"
    )
    parameter_influence_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_parameter_influence_circuits_n6400_confirmation.json"
    )
    resolution_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_nested_parameter_resolution_sweep.json"
    )
    cost_accuracy_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_cost_accuracy_benchmark.json"
    )
    normalization_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_n6400_normalization_ablation.json"
    )
    imagenet1k_screen_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_imagenet1k_combined_kernel_confirmation.json"
    )
    imagenet1k_causal_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_imagenet1k_parameter_influence_space_ablation_confirmation.json"
    )
    interpretability_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_parameter_influence_interpretability_imagenet1000.json"
    )
    backend_screen_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_kernel_backend_comparison_n6400_development.json"
    )
    backend_causal_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_nystrom400_backend_confirmation_n6400.json"
    )
    timeseries_summary_path = (
        strengthening_arc
        / "artifacts"
        / "timeseries"
        / "timeseries_cross_dataset_summary.json"
    )
    protein_feature_path = (
        strengthening_arc
        / "artifacts"
        / "protein"
        / "esm2_t12_feature_localization_causal_n1600_frozen_confirmation.json"
    )
    protein_influence_path = (
        strengthening_arc
        / "artifacts"
        / "protein"
        / "esm2_t12_parameter_influence_causal_n1600_frozen_confirmation.json"
    )
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    (output / "figures").mkdir(parents=True)
    (output / "tables").mkdir()
    (output / "source_results").mkdir()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cold_studies = [
        (
            "ViT-B C1",
            "ViT-B/16",
            vision / "imagenet200_vitb16_sample_dinov2_rbf01_confirmation.json",
        ),
        (
            "ViT-B C2",
            "ViT-B/16",
            vision / "imagenet200_vitb16_sample_dinov2_rbf01_confirmation2.json",
        ),
        (
            "ViT-B C3",
            "ViT-B/16",
            vision / "imagenet200_vitb16_sample_dinov2_rbf01_confirmation3.json",
        ),
        (
            "ViT-B C4",
            "ViT-B/16",
            vision / "imagenet200_vitb16_sample_dinov2_nystrom200_confirmation.json",
        ),
        (
            "ViT-L C1",
            "ViT-L/16",
            vision / "imagenet200_vitl16_sample_dinov2_rbf01_replication.json",
        ),
        (
            "ViT-L C2",
            "ViT-L/16",
            vision / "imagenet200_vitl16_sample_dinov2_nystrom200_confirmation.json",
        ),
        (
            "ConvNeXt",
            "ConvNeXtV2-T",
            vision / "imagenet200_convnextv2tiny_sample_dinov2_rbf01_replication.json",
        ),
    ]
    cold_rows = build_cold_table(cold_studies)
    write_csv(output / "tables" / "cold_retrieval.csv", cold_rows)
    plot_cold(cold_rows, output / "figures" / "cold_retrieval")

    causal_studies = [
        ("ViT-B/16", vision / "imagenet200_vitb16_sample_dinov2_causal_curve.json"),
        ("ViT-L/16", vision / "imagenet200_vitl16_sample_dinov2_causal_curve.json"),
        (
            "ConvNeXtV2-T",
            vision / "imagenet200_convnextv2tiny_sample_dinov2_causal_curve.json",
        ),
    ]
    causal_rows, paired_rows = build_causal_tables(causal_studies)
    write_csv(output / "tables" / "causal_curves.csv", causal_rows)
    write_csv(output / "tables" / "causal_paired_comparisons.csv", paired_rows)
    plot_causal(causal_rows, output / "figures" / "causal_curves")
    plot_selectivity(causal_rows, output / "figures" / "selectivity_curves")

    functional_rows = build_functional_table(
        {
            "margin": vision / "imagenet200_vitb16_sample_dinov2_margin_dev100.json",
            "class_logit": vision
            / "imagenet200_vitb16_sample_dinov2_class_logit_to_margin_dev.json",
            "loss": vision / "imagenet200_vitb16_sample_dinov2_loss_to_margin_dev.json",
        }
    )
    write_csv(output / "tables" / "functional_comparison.csv", functional_rows)
    plot_functionals(functional_rows, output / "figures" / "functional_comparison")

    compression_rows = build_compression_table(
        vision / "imagenet200_vitb16_sample_dinov2_nystrom_screen.json",
        vision / "imagenet200_vitb16_sample_dinov2_rff_screen.json",
        [
            (
                "ViT-B/16",
                vision
                / "imagenet200_vitb16_sample_dinov2_nystrom200_confirmation.json",
            ),
            (
                "ViT-L/16",
                vision
                / "imagenet200_vitl16_sample_dinov2_nystrom200_confirmation.json",
            ),
        ],
    )
    write_csv(output / "tables" / "compression.csv", compression_rows)
    plot_compression(compression_rows, output / "figures" / "compression")

    generalization_rows = build_generalization_table(
        vision / "imagenet200_vitb16_sample_dinov2_heldout_classes.json"
    )
    write_csv(output / "tables" / "heldout_class.csv", generalization_rows)
    plot_heldout(generalization_rows, output / "figures" / "heldout_class")

    if atlas_size_path.exists():
        atlas_size_rows, selected_source_examples = build_atlas_size_table(
            atlas_size_path,
            cost_accuracy_path if cost_accuracy_path.exists() else None,
        )
        write_csv(output / "tables" / "atlas_size_scaling.csv", atlas_size_rows)
        plot_atlas_size(
            atlas_size_rows,
            selected_source_examples,
            output / "figures" / "atlas_size_scaling",
        )

    if vision_zero_path.exists() and parameter_influence_path.exists():
        deactivation_rows = build_vision_mechinterp_table(vision_zero_path)
        influence_rows = build_parameter_influence_table(parameter_influence_path)
        write_csv(
            output / "tables" / "vision_neuron_deactivation.csv",
            deactivation_rows,
        )
        write_csv(
            output / "tables" / "vision_parameter_influence.csv",
            influence_rows,
        )
        plot_vision_mechanistic_assays(
            deactivation_rows,
            influence_rows,
            output / "figures" / "vision_mechanistic_assays",
        )
    if resolution_path.exists():
        resolution_rows = build_resolution_table(resolution_path)
        write_csv(
            output / "tables" / "parameter_resolution.csv",
            resolution_rows,
        )
        plot_resolution_sweep(
            resolution_rows,
            output / "figures" / "parameter_resolution",
        )
    if cost_accuracy_path.exists():
        cost_rows = build_cost_accuracy_table(cost_accuracy_path)
        write_csv(output / "tables" / "vision_cost_accuracy.csv", cost_rows)
        plot_cost_accuracy(cost_rows, output / "figures" / "vision_cost_accuracy")
    if normalization_path.exists():
        normalization_rows = build_normalization_table(normalization_path)
        write_csv(
            output / "tables" / "normalization_ablation.csv",
            normalization_rows,
        )
        plot_normalization(
            normalization_rows,
            output / "figures" / "normalization_ablation",
        )
    if imagenet1k_screen_path.exists() and imagenet1k_causal_path.exists():
        imagenet1k_rows = build_imagenet1k_circuit_table(
            imagenet1k_screen_path,
            imagenet1k_causal_path,
        )
        write_csv(
            output / "tables" / "imagenet1k_parameter_circuits.csv",
            imagenet1k_rows,
        )
        plot_imagenet1k_circuits(
            imagenet1k_rows,
            output / "figures" / "imagenet1k_parameter_circuits",
        )
    if interpretability_path.exists():
        interpretability_rows = build_interpretability_table(interpretability_path)
        write_csv(
            output / "tables" / "imagenet1k_circuit_interpretability.csv",
            interpretability_rows,
        )
    if backend_screen_path.exists() and backend_causal_path.exists():
        backend_rows, backend_causal_rows = build_backend_tables(
            backend_screen_path,
            backend_causal_path,
        )
        write_csv(
            output / "tables" / "kernel_backend_screen.csv",
            backend_rows,
        )
        write_csv(
            output / "tables" / "kernel_backend_causal.csv",
            backend_causal_rows,
        )
        plot_backend_comparison(
            backend_rows,
            backend_causal_rows,
            output / "figures" / "kernel_backend_comparison",
        )
    if timeseries_summary_path.exists():
        timeseries_rows = build_timeseries_circuit_table(timeseries_summary_path)
        write_csv(
            output / "tables" / "timeseries_parameter_circuits.csv",
            timeseries_rows,
        )
        plot_timeseries_circuits(
            timeseries_rows,
            output / "figures" / "timeseries_parameter_circuits",
        )
    if protein_feature_path.exists() and protein_influence_path.exists():
        protein_rows = build_protein_application_table(
            protein_feature_path,
            protein_influence_path,
        )
        write_csv(
            output / "tables" / "protein_causal_applications.csv",
            protein_rows,
        )
        plot_protein_applications(
            protein_rows,
            output / "figures" / "protein_causal_applications",
        )

    strengthening_figures = (
        "parameter_influence_interpretability_imagenet1000",
        "parameter_influence_exemplars_imagenet1000",
    )
    for stem in strengthening_figures:
        for suffix in (".png", ".pdf"):
            source = (
                strengthening_arc
                / "artifacts"
                / "vision"
                / "figures"
                / f"{stem}{suffix}"
            )
            if source.exists():
                shutil.copy2(source, output / "figures" / f"{stem}{suffix}")

    efficiency = read_json(vision / "imagenet200_vitb16_nystrom200_efficiency.json")
    efficiency_rows = []
    for stage, values in efficiency["timing"].items():
        efficiency_rows.append({"kind": "timing", "item": stage, **values})
    for item, value in efficiency["storage"].items():
        efficiency_rows.append(
            {
                "kind": "storage",
                "item": item,
                "mean_ms": value,
                "median_ms": "",
                "q10_ms": "",
                "q90_ms": "",
                "repeats": "",
            }
        )
    write_csv(output / "tables" / "efficiency_storage.csv", efficiency_rows)

    search_rows = [
        {
            "candidate": "Canonical multilingual precision",
            "status": "rejected",
            "gate": "Frozen nonlinear winners failed untouched test",
        },
        {
            "candidate": "ViT class-mean continuous atlas",
            "status": "rejected",
            "gate": "Continuous representation collapsed to scalar",
        },
        {
            "candidate": "ViT sample-level DINO atlas",
            "status": "supported",
            "gate": "Cold, causal, selectivity, controls, and confirmations pass",
        },
        {
            "candidate": "Behavior-specific margin functional",
            "status": "supported",
            "gate": "Beats loss- and class-logit-derived maps",
        },
        {
            "candidate": "Matched activation atlases",
            "status": "supported baseline comparison",
            "gate": (
                "Sensitivity stronger on ViT-B; sparse/large-budget advantages "
                "replicate across models"
            ),
        },
        {
            "candidate": "Held-out ImageNet classes",
            "status": "supporting",
            "gate": "Positive but below 25% oracle-gap promotion threshold",
        },
        {
            "candidate": "WordNet semantic directions",
            "status": "rejected causal",
            "gate": (
                "Retrieval positive; semantic-direction causal interval crosses zero"
            ),
        },
        {
            "candidate": "Qwen multilingual behavior circuit",
            "status": "rejected",
            "gate": "Direct-gradient application premise fails",
        },
        {
            "candidate": "Qwen MMLU sensitivity atlas",
            "status": (
                "supported supplementary" if language_final_path.exists() else "not run"
            ),
            "gate": (
                "Two-model assay clears the direct-parameter gap threshold but is "
                "substantially weaker than vision against activation attribution"
                if language_final_path.exists()
                else "Frozen confirmation artifact unavailable"
            ),
        },
        {
            "candidate": "Input-conditioned parameter influence circuits",
            "status": "supported" if imagenet1k_causal_path.exists() else "not run",
            "gate": (
                "ImageNet-1k combined categorical + semantic space beats each "
                "space ablation, nearest, and scalar at 37 groups under exact "
                "local parameter interventions"
                if imagenet1k_causal_path.exists()
                else "Confirmation artifact unavailable"
            ),
        },
        {
            "candidate": "Time-series parameter influence circuits",
            "status": "supported secondary" if timeseries_summary_path.exists() else "not run",
            "gate": (
                "Frozen Chronos-Bolt RBF circuit beats scalar and matched shuffle "
                "on ETTh1 and Weather and recovers 31--67% of the direct-gradient gap"
                if timeseries_summary_path.exists()
                else "Cross-dataset confirmation artifact unavailable"
            ),
        },
        {
            "candidate": "Protein neuron localization and parameter influence",
            "status": (
                "supported primary"
                if protein_feature_path.exists() and protein_influence_path.exists()
                else "not run"
            ),
            "gate": (
                "One frozen ESM2 atlas passes exact causal gates for both "
                "applications on disjoint confirmations"
                if protein_feature_path.exists() and protein_influence_path.exists()
                else "Frozen confirmation artifacts unavailable"
            ),
        },
        {
            "candidate": "Real-model hierarchy sweep",
            "status": "rejected",
            "gate": "Coarse resolutions nearly scalar",
        },
        {
            "candidate": "ViT-L / ConvNeXt replication",
            "status": "supported",
            "gate": "Frozen sample-level protocol replicates",
        },
        {
            "candidate": "Nyström-200 finite embedding",
            "status": "supported with tradeoff",
            "gate": (
                "Fourfold storage reduction; positive causal confirmation; weaker "
                "than full kernel"
            ),
        },
        {
            "candidate": "Vision source-atlas sample scaling",
            "status": "supported" if atlas_size_path.exists() else "not run",
            "gate": (
                "6,400-source default selected on development and confirmed against "
                "scalar and shuffled pairing"
                if atlas_size_path.exists()
                else "Scaling artifact unavailable"
            ),
        },
        {
            "candidate": "MIB standardized circuit benchmark",
            "status": "not run",
            "gate": (
                "MIB node/edge circuits do not faithfully match parameter-group "
                "intervention coordinates"
            ),
        },
    ]
    write_csv(output / "tables" / "search_outcomes.csv", search_rows)

    source_paths = sorted(
        {path for _, _, path in cold_studies}
        | {path for _, path in causal_studies}
        | {
            vision / "imagenet200_vitb16_sample_dinov2_margin_dev100.json",
            vision / "imagenet200_vitb16_sample_dinov2_class_logit_to_margin_dev.json",
            vision / "imagenet200_vitb16_sample_dinov2_loss_to_margin_dev.json",
            vision / "imagenet200_vitb16_sample_dinov2_heldout_classes.json",
            vision / "imagenet200_vitb16_sample_dinov2_stability_structure.json",
            vision / "imagenet200_vitb16_sample_dinov2_nystrom_screen.json",
            vision / "imagenet200_vitb16_sample_dinov2_rff_screen.json",
            vision / "imagenet200_vitb16_nystrom200_efficiency.json",
            vision / "imagenet200_wordnet_family_queries.json",
            vision / "imagenet200_wordnet_family_causal.json",
        }
    )
    if atlas_size_path.exists():
        source_paths.append(atlas_size_path)
        source_paths.sort()
    if vision_zero_path.exists():
        source_paths.append(vision_zero_path)
    if parameter_influence_path.exists():
        source_paths.append(parameter_influence_path)
    if resolution_path.exists():
        source_paths.append(resolution_path)
    for path in (
        cost_accuracy_path,
        normalization_path,
        imagenet1k_screen_path,
        imagenet1k_causal_path,
        interpretability_path,
        backend_screen_path,
        backend_causal_path,
        timeseries_summary_path,
        protein_feature_path,
        protein_influence_path,
    ):
        if path.exists():
            source_paths.append(path)
    timeseries_artifacts = (
        "patchtst_etth1_profile_premise.json",
        "chronos_etth1_profile_development.json",
        "chronos_weather_profile_transfer.json",
        "chronos_etth1_full_scope_causal_development.json",
        "chronos_etth1_final_decoder_causal_development.json",
        "chronos_etth1_final_decoder_causal_confirmation.json",
        "chronos_weather_final_decoder_causal_confirmation.json",
    )
    for name in timeseries_artifacts:
        path = strengthening_arc / "artifacts" / "timeseries" / name
        if path.exists():
            source_paths.append(path)
    source_paths.sort()
    manifest_sources = []
    for path in source_paths:
        prefix = path.parent.name
        destination = output / "source_results" / f"{prefix}__{path.name}"
        shutil.copy2(path, destination)
        manifest_sources.append(
            {
                "file": str(destination.relative_to(output)),
                "source": (
                    str(path.relative_to(arc))
                    if path.is_relative_to(arc)
                    else str(path.relative_to(arc.parent))
                ),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    shutil.copy2(Path(__file__), output / "generate_bundle.py")
    documentation = (
        strengthening_arc / "artifacts" / "empirical_results_map.md",
        strengthening_arc / "artifacts" / "figure_table_captions.md",
    )
    for path in documentation:
        if path.exists():
            shutil.copy2(path, output / path.name)
    tensor_paths = [
        vision / "imagenet200_vitb16_sample_dinov2_rbf01_confirmation3.pt",
        vision / "imagenet200_vitl16_sample_dinov2_rbf01_replication.pt",
        vision / "imagenet200_convnextv2tiny_sample_dinov2_rbf01_replication.pt",
    ]
    manifest = {
        "contents": (
            "tables, publication-style figures, plotting code, source metric "
            "JSONs, an empirical claim map, and figure/table captions"
        ),
        "primary_claim": (
            "amortized sample-conditioned neuron localization and local "
            "parameter-influence retrieval from a precomputed sensitivity atlas"
        ),
        "figures": sorted(
            str(path.relative_to(output)) for path in (output / "figures").iterdir()
        ),
        "tables": sorted(
            str(path.relative_to(output)) for path in (output / "tables").iterdir()
        ),
        "documentation": sorted(
            path.name for path in documentation if (output / path.name).exists()
        ),
        "source_results": manifest_sources,
        "external_tensor_artifacts": [
            {"path": str(path.relative_to(arc)), "bytes": path.stat().st_size}
            for path in tensor_paths
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output.parent))


if __name__ == "__main__":
    main()
