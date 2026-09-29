"""Package the refined positive-evidence search as tables, plots, and raw metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .paper_figures import (
    application1_gap_series,
    application1_series,
    application2_series,
    plot_application1_efficiency,
    plot_application1_primary,
    plot_application2_primary,
    plot_parameter_resolution_appendix,
)
from .paper_style import (
    CATEGORICAL,
    CONSTANT,
    GRID,
    NEAREST,
    SEMANTIC,
    SHUFFLE,
    apply_paper_style,
    outside_legend,
    save_plot,
)
from .semantic_circuit_evidence_v8 import run_plot as plot_semantic_circuits


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


def build_semantic_circuit_table(vision_path, forecast_path):
    """Flatten the matched cross-modal circuit evidence behind Figure 2."""
    rows = []
    fields = {
        "modality": "",
        "kind": "",
        "budget": "",
        "selection": "",
        "comparison": "",
        "metric": "",
        "similarity_quantile": "",
        "similarity_mean": "",
        "mean": "",
        "ci95_low": "",
        "ci95_high": "",
        "n": "",
    }
    for modality, path in (("vision", vision_path), ("forecasting", forecast_path)):
        payload = read_json(path)
        for budget, methods in payload["causal_comparisons"].items():
            for selection, metrics in methods.items():
                comparison = selection.replace("semantic_", "constant_") + "_matched"
                for metric, summary in metrics.items():
                    rows.append(
                        {
                            **fields,
                            "modality": modality,
                            "kind": "exact_causal_gain",
                            "budget": float(budget),
                            "selection": selection,
                            "comparison": comparison,
                            "metric": metric,
                            "mean": summary["mean"],
                            "ci95_low": summary["lower_95"],
                            "ci95_high": summary["upper_95"],
                            "n": summary["n"],
                        }
                    )
        for budget, methods in payload["structure"]["budgets"].items():
            for selection, bins in methods.items():
                for current in bins:
                    summary = current["jaccard"]
                    rows.append(
                        {
                            **fields,
                            "modality": modality,
                            "kind": "circuit_overlap",
                            "budget": float(budget),
                            "selection": selection,
                            "comparison": "representation_similarity",
                            "metric": "jaccard",
                            "similarity_quantile": "-".join(
                                f"{value:g}" for value in current["similarity_quantile"]
                            ),
                            "similarity_mean": current["similarity_mean"],
                            "mean": summary["mean"],
                            "ci95_low": summary["lower_95"],
                            "ci95_high": summary["upper_95"],
                            "n": summary["n"],
                        }
                    )
        grouped = defaultdict(list)
        for record in payload["selections"]:
            if record["method"] != "semantic_full":
                continue
            if modality == "vision" and not record["target_correct"]:
                continue
            grouped[float(record["budget"])].append(record)
        for budget, records in grouped.items():
            for metric in ("represented_layers", "direct_gradient_energy"):
                mean, low, high, count = bootstrap(
                    [record[metric] for record in records]
                )
                rows.append(
                    {
                        **fields,
                        "modality": modality,
                        "kind": "selection_structure",
                        "budget": budget,
                        "selection": "semantic_full",
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n": count,
                    }
                )
    return rows


def build_budget_retrieval_table(path):
    """Flatten the precision/recall summaries used by the appendix figures."""
    payload = read_json(path)
    fixed_oracle = payload["protocol"].get("fixed_oracle_fraction")
    rows = []
    for method, summaries in payload["summaries"].items():
        display_method = "KPSA-1NN" if method == "Semantic nearest" else method
        for summary in summaries:
            for metric in ("precision", "recall", "f1"):
                values = summary[metric]
                rows.append(
                    {
                        "method": display_method,
                        "oracle_fraction": summary.get("oracle_fraction", fixed_oracle),
                        "predicted_fraction": summary["predicted_fraction"],
                        "retrieval_multiplier": summary.get("retrieval_multiplier", ""),
                        "metric": metric,
                        "mean": values["mean"],
                        "ci95_low": values["ci95_low"],
                        "ci95_high": values["ci95_high"],
                        "queries": values["n"],
                    }
                )
    return rows


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
        "sample_dinov2_rbf_scale_0.1": "Semantic KPSA",
        "sample_nearest": "KPSA-1NN",
        "class_source_onehot_reference": "Categorical KPSA",
        "scalar_mass": "Constant sensitivity",
        "sample_jl": "Random-projection control",
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
        "sensitivity_atlas_rbf": "Semantic KPSA",
        "sensitivity_atlas_nearest": "KPSA-1NN",
        "sensitivity_atlas_scalar": "Constant sensitivity",
        "activation_atlas_rbf": "Activation atlas RBF",
        "activation_x_gradient_atlas_rbf": "Act×grad atlas RBF",
        "class_source_onehot_reference": "Categorical KPSA",
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
                            "primary": "Semantic KPSA",
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
                ("scalar_mass", "Constant"),
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
        "scalar_mass": "Constant sensitivity",
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
        "Semantic KPSA",
        "KPSA-1NN",
        "Categorical KPSA",
        "Constant sensitivity",
        "Random-projection control",
    ]
    styles = {
        "Semantic KPSA": (SEMANTIC, "o", "-"),
        "KPSA-1NN": (NEAREST, "^", ":"),
        "Categorical KPSA": (CATEGORICAL, "s", "--"),
        "Constant sensitivity": (CONSTANT, "x", "-."),
        "Random-projection control": (SHUFFLE, "D", (0, (2, 2))),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8))
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
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    outside_legend(fig, axes, ncol=len(methods))
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
        "scalar": "Constant sensitivity",
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


def build_interpretability_table(path):
    payload = read_json(path)
    metrics = payload["metrics"]
    methods = {
        "prototype": "Semantic atlas",
        "prototype_permuted": "Shuffled semantic pairing",
        "class_onehot": "Categorical class space",
        "nearest": "Nearest example",
        "scalar": "Constant sensitivity",
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
        "scalar": "Constant sensitivity",
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
        "scalar": "Constant sensitivity",
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
    vision_zero_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_prototype_zero_ablation_confirmation.json"
    )
    vision_zero_budget_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_prototype_zero_ablation_confirmation_budgets.json"
    )
    budget_pr_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_app1_budget_precision_recall.json"
    )
    budget_recall_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_app1_budget_recall_multiples.json"
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
    imagenet1k_causal_budget_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_imagenet1k_parameter_influence_space_ablation_confirmation_budgets.json"
    )
    interpretability_legacy_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_parameter_influence_interpretability_imagenet1000.json"
    )
    interpretability_v2_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_parameter_influence_interpretability_imagenet1000_v2.json"
    )
    interpretability_path = (
        interpretability_v2_path
        if interpretability_v2_path.exists()
        else interpretability_legacy_path
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
    etth1_causal_path = (
        strengthening_arc
        / "artifacts"
        / "timeseries"
        / "chronos_etth1_final_decoder_causal_confirmation.json"
    )
    etth1_budget_path = (
        strengthening_arc
        / "artifacts"
        / "timeseries"
        / "chronos_etth1_final_decoder_causal_confirmation_budgets.json"
    )
    weather_causal_path = (
        strengthening_arc
        / "artifacts"
        / "timeseries"
        / "chronos_weather_final_decoder_causal_confirmation.json"
    )
    weather_budget_path = (
        strengthening_arc
        / "artifacts"
        / "timeseries"
        / "chronos_weather_final_decoder_causal_confirmation_budgets.json"
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
    protein_feature_budget_path = (
        strengthening_arc
        / "artifacts"
        / "protein"
        / "esm2_t12_feature_localization_causal_n1600_frozen_confirmation_budgets.json"
    )
    protein_influence_budget_path = (
        strengthening_arc
        / "artifacts"
        / "protein"
        / "esm2_t12_parameter_influence_causal_n1600_frozen_confirmation_budgets.json"
    )
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    (output / "figures").mkdir(parents=True)
    (output / "tables").mkdir()
    (output / "source_results").mkdir()
    (output / "code").mkdir()
    apply_paper_style()

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

    functional_rows = build_functional_table(
        {
            "margin": vision / "imagenet200_vitb16_sample_dinov2_margin_dev100.json",
            "class_logit": vision
            / "imagenet200_vitb16_sample_dinov2_class_logit_to_margin_dev.json",
            "loss": vision / "imagenet200_vitb16_sample_dinov2_loss_to_margin_dev.json",
        }
    )
    write_csv(output / "tables" / "functional_comparison.csv", functional_rows)

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

    generalization_rows = build_generalization_table(
        vision / "imagenet200_vitb16_sample_dinov2_heldout_classes.json"
    )
    write_csv(output / "tables" / "heldout_class.csv", generalization_rows)

    if atlas_size_path.exists():
        atlas_size_rows, _selected_source_examples = build_atlas_size_table(
            atlas_size_path,
            cost_accuracy_path if cost_accuracy_path.exists() else None,
        )
        write_csv(output / "tables" / "atlas_size_scaling.csv", atlas_size_rows)

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
    if resolution_path.exists():
        resolution_rows = build_resolution_table(resolution_path)
        write_csv(
            output / "tables" / "parameter_resolution.csv",
            resolution_rows,
        )
        plot_parameter_resolution_appendix(
            resolution_rows,
            output / "figures" / "paper_parameter_resolution_appendix",
        )
    if cost_accuracy_path.exists():
        cost_rows = build_cost_accuracy_table(cost_accuracy_path)
        write_csv(output / "tables" / "vision_cost_accuracy.csv", cost_rows)
    if budget_pr_path.exists():
        write_csv(
            output / "tables" / "app1_budget_precision_recall.csv",
            build_budget_retrieval_table(budget_pr_path),
        )
    if budget_recall_path.exists():
        write_csv(
            output / "tables" / "app1_budget_recall_multiples.csv",
            build_budget_retrieval_table(budget_recall_path),
        )
    if normalization_path.exists():
        normalization_rows = build_normalization_table(normalization_path)
        write_csv(
            output / "tables" / "normalization_ablation.csv",
            normalization_rows,
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
    if timeseries_summary_path.exists():
        timeseries_rows = build_timeseries_circuit_table(timeseries_summary_path)
        write_csv(
            output / "tables" / "timeseries_parameter_circuits.csv",
            timeseries_rows,
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
    app1_vision_path = (
        vision_zero_budget_path
        if vision_zero_budget_path.exists()
        else vision_zero_path
    )
    app1_protein_path = (
        protein_feature_budget_path
        if protein_feature_budget_path.exists()
        else protein_feature_path
    )
    app1_series = application1_series(app1_vision_path, app1_protein_path)
    app1_gap = application1_gap_series(app1_vision_path, app1_protein_path)
    gap_lookup = {
        (modality, row["budget"], row["method"]): row
        for modality, rows in app1_gap.items()
        for row in rows
    }
    app1_rows = []
    for modality, rows in app1_series.items():
        for row in rows:
            gap = gap_lookup[(modality, row["budget"], row["method"])]
            app1_rows.append(
                {
                    "modality": modality,
                    **row,
                    "gap_recovery_pct": gap["mean"],
                    "gap_ci95_low_pct": gap["ci95_low"],
                    "gap_ci95_high_pct": gap["ci95_high"],
                }
            )
    write_csv(output / "tables" / "paper_application1_primary.csv", app1_rows)
    plot_application1_primary(
        app1_series,
        output / "figures" / "paper_application1_primary",
    )
    plot_application1_efficiency(
        cost_accuracy_path,
        app1_gap["Vision"],
        atlas_size_rows,
        output / "figures" / "paper_application1_efficiency",
    )

    app2_vision_path = (
        imagenet1k_causal_budget_path
        if imagenet1k_causal_budget_path.exists()
        else imagenet1k_causal_path
    )
    app2_etth1_path = (
        etth1_budget_path if etth1_budget_path.exists() else etth1_causal_path
    )
    app2_weather_path = (
        weather_budget_path if weather_budget_path.exists() else weather_causal_path
    )
    app2_protein_path = (
        protein_influence_budget_path
        if protein_influence_budget_path.exists()
        else protein_influence_path
    )
    app2_series = application2_series(
        app2_vision_path,
        app2_etth1_path,
        app2_weather_path,
        app2_protein_path,
    )
    app2_rows = [
        {"modality": modality, "measurement": measurement, **row}
        for modality, measurements in app2_series.items()
        for measurement, rows in measurements.items()
        for row in rows
    ]
    write_csv(output / "tables" / "paper_application2_primary.csv", app2_rows)
    plot_application2_primary(
        app2_series,
        output / "figures" / "paper_application2_primary",
    )

    semantic_circuit_vision_path = (
        strengthening_arc
        / "artifacts"
        / "vision"
        / "vitb16_imagenet1k_semantic_circuit_confirmation.json"
    )
    semantic_circuit_forecast_path = (
        strengthening_arc
        / "artifacts"
        / "timeseries"
        / "chronos_etth1_semantic_circuit_confirmation.json"
    )
    missing_circuit_sources = [
        path
        for path in (semantic_circuit_vision_path, semantic_circuit_forecast_path)
        if not path.exists()
    ]
    if missing_circuit_sources:
        raise FileNotFoundError(
            "missing semantic circuit confirmations: "
            + ", ".join(map(str, missing_circuit_sources))
        )
    plot_semantic_circuits(
        Namespace(
            vision=semantic_circuit_vision_path,
            forecast=semantic_circuit_forecast_path,
            output=output / "figures" / "paper_application2_circuits",
        )
    )
    write_csv(
        output / "tables" / "paper_application2_circuits.csv",
        build_semantic_circuit_table(
            semantic_circuit_vision_path,
            semantic_circuit_forecast_path,
        ),
    )

    strengthening_figures = (
        "parameter_influence_exemplars_imagenet1000",
        "app1_budget_precision_recall",
        "app1_budget_recall_multiples",
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
        budget_pr_path,
        budget_recall_path,
        backend_screen_path,
        backend_causal_path,
        timeseries_summary_path,
        protein_feature_path,
        protein_influence_path,
        app1_vision_path,
        app1_protein_path,
        app2_vision_path,
        app2_etth1_path,
        app2_weather_path,
        app2_protein_path,
        semantic_circuit_vision_path,
        semantic_circuit_forecast_path,
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
    source_paths = sorted(set(source_paths))
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
    code_files = (
        Path(__file__),
        Path(__file__).with_name("paper_figures.py"),
        Path(__file__).with_name("paper_style.py"),
        Path(__file__).with_name("parameter_influence_interpretability_v8.py"),
        Path(__file__).with_name("semantic_circuit_evidence_v8.py"),
        Path(__file__).with_name("vision_budget_precision_recall_v8.py"),
    )
    for path in code_files:
        shutil.copy2(path, output / "code" / path.name)
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
            "sensitivity-circuit retrieval from a precomputed atlas"
        ),
        "figures": sorted(
            str(path.relative_to(output)) for path in (output / "figures").iterdir()
        ),
        "tables": sorted(
            str(path.relative_to(output)) for path in (output / "tables").iterdir()
        ),
        "code": sorted(
            str(path.relative_to(output)) for path in (output / "code").iterdir()
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
