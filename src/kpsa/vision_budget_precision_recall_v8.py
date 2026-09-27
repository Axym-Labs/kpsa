"""Choose KPSA budgets from precision--recall against a fixed oracle set."""

from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .common import save_json, seed_everything
from .paper_style import (
    apply_paper_style,
    finish_axis,
    method_style,
    outside_legend,
    save_plot,
)
from .parameter_groups import ParameterPartition
from .vision_causal_refined import (
    mean_ablation_taylor_scores,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)
from .vision_prototype_strengthening_v8 import construction_indices, make_predictions
from .vision_sample_refined import encode_indices

METHODS = {
    "prototype_800_scale_0.025": "Semantic KPSA",
    "class_onehot": "Categorical KPSA",
    "nearest_encoder": "Semantic nearest",
    "scalar_mass": "Constant sensitivity",
    "direct_coordinate_taylor": "Activation attribution",
}


def _bootstrap(values, seed, draws=5000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n": len(values),
    }


def _weighted_prf(predicted, oracle, sizes):
    intersection = predicted & oracle
    true_mass = float(sizes[intersection].sum())
    predicted_mass = float(sizes[predicted].sum())
    oracle_mass = float(sizes[oracle].sum())
    precision = true_mass / predicted_mass
    recall = true_mass / oracle_mass
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _plot(summaries, output, oracle_fraction, oracle_mode):
    apply_paper_style()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    metrics = (
        (("f1",), ("Oracle top-set overlap",))
        if oracle_mode == "matched"
        else (("precision", "recall", "f1"), ("Precision", "Recall", "F1"))
    )
    fig, axes = plt.subplots(
        1,
        len(metrics[0]),
        figsize=(4.15, 2.75) if oracle_mode == "matched" else (7.15, 2.65),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    for axis, metric, title in zip(
        axes,
        *metrics,
    ):
        for method in METHODS.values():
            if oracle_mode == "fixed" and method == "Activation attribution":
                continue
            rows = summaries.get(method, [])
            if not rows:
                continue
            means = np.asarray([row[metric]["mean"] for row in rows])
            lows = np.asarray([row[metric]["ci95_low"] for row in rows])
            highs = np.asarray([row[metric]["ci95_high"] for row in rows])
            axis.plot(
                [100 * row["predicted_fraction"] for row in rows],
                100 * means,
                label=method,
                **method_style(method),
            )
            axis.fill_between(
                [100 * row["predicted_fraction"] for row in rows],
                100 * lows,
                100 * highs,
                color=method_style(method)["color"],
                alpha=0.10,
                linewidth=0,
            )
        if oracle_mode == "fixed":
            axis.axvline(
                100 * oracle_fraction,
                color="#78B7C5",
                linestyle=":",
                linewidth=0.9,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Predicted MLP parameters")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylim(-2, 42 if oracle_mode == "fixed" else 102)
        if oracle_mode == "fixed":
            axis.text(
                0.97,
                0.04,
                "oracle self-retrieval: 100%",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.9,
                color="#C96868",
            )
        axis.xaxis.set_minor_locator(mticker.NullLocator())
        finish_axis(axis)
    ticks = sorted(
        {100 * row["predicted_fraction"] for rows in summaries.values() for row in rows}
    )
    for axis in axes:
        axis.set_xticks(ticks[::2], [f"{tick:g}%" for tick in ticks[::2]])
    axes[0].set_ylabel("Oracle-set agreement (%)")
    outside_legend(fig, axes, ncol=4 if oracle_mode == "fixed" else 5, y=1.08)
    save_plot(fig, output)


def _plot_recall_multiples(summaries, output, oracle_fractions, multipliers):
    apply_paper_style()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(
        1, len(oracle_fractions), figsize=(7.15, 2.65), sharey=True
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, oracle_fraction in zip(axes, oracle_fractions):
        for method in METHODS.values():
            if method == "Activation attribution":
                continue
            rows = [
                row
                for row in summaries.get(method, [])
                if row["oracle_fraction"] == oracle_fraction
            ]
            if not rows:
                continue
            means = np.asarray([row["recall"]["mean"] for row in rows])
            lows = np.asarray([row["recall"]["ci95_low"] for row in rows])
            highs = np.asarray([row["recall"]["ci95_high"] for row in rows])
            axis.plot(
                [row["retrieval_multiplier"] for row in rows],
                100 * means,
                label=method,
                **method_style(method),
            )
            axis.fill_between(
                [row["retrieval_multiplier"] for row in rows],
                100 * lows,
                100 * highs,
                color=method_style(method)["color"],
                alpha=0.10,
                linewidth=0,
            )
        axis.set_xscale("log", base=10)
        axis.set_xticks(multipliers, [f"{value:g}×" for value in multipliers])
        axis.xaxis.set_minor_locator(mticker.NullLocator())
        axis.set_xlabel("Retrieval/oracle budget")
        axis.set_title(f"Oracle: {100 * oracle_fraction:g}%", loc="left")
        axis.set_ylim(-2, 58)
        axis.text(
            0.97,
            0.04,
            "oracle self-retrieval: 100%",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.9,
            color="#C96868",
        )
        finish_axis(axis)
    axes[0].set_ylabel("Oracle-set recall (%)")
    outside_legend(fig, axes, ncol=4, y=1.08)
    save_plot(fig, output)


def run(args):
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    raw_dataset = ImageFolder(args.data / "validation")
    representation_model = (
        AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True)
        .cuda()
        .eval()
    )
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    )
    construction_features = encode_indices(
        representation_model,
        processor,
        raw_dataset,
        construction_indices(payload["representation_indices"]),
    )
    shifted_target_indices = payload["target_indices"].long() + args.target_shift
    target_features = encode_indices(
        representation_model, processor, raw_dataset, shifted_target_indices
    )
    del representation_model
    gc.collect()
    torch.cuda.empty_cache()

    predictions, _maps, construction_median, source_median = make_predictions(
        payload["source_profiles"],
        payload["source_features"],
        target_features,
        payload["class_source_profiles"],
        construction_features,
        prototype_counts=(800,),
        prototype_scales=(0.025,),
        seed=args.seed,
    )

    model = (
        timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)
        .cuda()
        .eval()
    )
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    zero_reference = {
        item["name"]: torch.zeros(item["count"], device=partition.device)
        for item in layout
    }

    query_columns = (
        torch.linspace(0, len(shifted_target_indices) - 1, args.queries)
        .round()
        .long()
        .unique()
    )
    valid_columns = None
    if args.causal_result:
        import json

        causal = json.loads(args.causal_result.read_text())
        valid_columns = {
            int(row["query_column"])
            for row in causal["records"]
            if row["target_correct"]
        }

    records = []
    actual_oracle_fractions = []
    if args.oracle_mode == "multiple":
        budget_settings = [
            (oracle_fraction, multiplier, oracle_fraction * multiplier)
            for oracle_fraction in args.oracle_fractions
            for multiplier in args.retrieval_multipliers
        ]
    else:
        budget_settings = [
            (
                fraction if args.oracle_mode == "matched" else args.oracle_fraction,
                (
                    1.0
                    if args.oracle_mode == "matched"
                    else fraction / args.oracle_fraction
                ),
                fraction,
            )
            for fraction in args.predicted_fractions
        ]
    for progress, column in enumerate(query_columns.tolist(), start=1):
        if valid_columns is not None and column not in valid_columns:
            continue
        image, label = dataset[int(shifted_target_indices[column])]
        oracle_scores = mean_ablation_taylor_scores(
            model,
            image,
            int(label),
            "margin",
            partition,
            layout,
            zero_reference,
        )
        scores = {
            name: predictions[name][:, column]
            for name in METHODS
            if name != "direct_coordinate_taylor"
        }
        scores["direct_coordinate_taylor"] = oracle_scores
        for name, method_scores in scores.items():
            for oracle_fraction, multiplier, fraction in budget_settings:
                oracle, actual_oracle = select_parameter_budget(
                    oracle_scores.clamp_min(0), sizes, scope, oracle_fraction
                )
                actual_oracle_fractions.append(actual_oracle)
                selected, actual = select_parameter_budget(
                    method_scores.clamp_min(0), sizes, scope, fraction
                )
                precision, recall, f1 = _weighted_prf(selected, oracle, sizes)
                records.append(
                    {
                        "query_column": column,
                        "method": METHODS[name],
                        "oracle_fraction": oracle_fraction,
                        "retrieval_multiplier": multiplier,
                        "predicted_fraction": fraction,
                        "actual_predicted_fraction": actual,
                        "actual_oracle_fraction": actual_oracle,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )
        if progress % 10 == 0:
            print(
                f"budget precision-recall {progress}/{len(query_columns)}", flush=True
            )

    summaries = defaultdict(list)
    for method in METHODS.values():
        for budget_index, (oracle_fraction, multiplier, fraction) in enumerate(
            budget_settings
        ):
            selected = [
                row
                for row in records
                if row["method"] == method
                and row["oracle_fraction"] == oracle_fraction
                and row["retrieval_multiplier"] == multiplier
            ]
            if not selected:
                continue
            summaries[method].append(
                {
                    "oracle_fraction": oracle_fraction,
                    "retrieval_multiplier": multiplier,
                    "predicted_fraction": fraction,
                    **{
                        metric: _bootstrap(
                            [row[metric] for row in selected],
                            args.seed + 100 * budget_index + metric_index,
                        )
                        for metric_index, metric in enumerate(
                            ("precision", "recall", "f1")
                        )
                    },
                }
            )
    semantic = summaries["Semantic KPSA"]
    best_semantic = max(semantic, key=lambda row: row["f1"]["mean"])
    result = {
        "setting": "vitb16_fixed_oracle_budget_precision_recall",
        "protocol": {
            "oracle_budget_mode": args.oracle_mode,
            "fixed_oracle_fraction": (
                args.oracle_fraction if args.oracle_mode == "fixed" else None
            ),
            "predicted_fractions": list(args.predicted_fractions),
            "oracle_fractions": list(args.oracle_fractions),
            "retrieval_multipliers": list(args.retrieval_multipliers),
            "selection": "parameter-mass budget within coupled MLP features",
            "reference": "positive gradient-times-activation scores for zero ablation",
            "queries": len({row["query_column"] for row in records}),
            "construction_examples": int(payload["source_profiles"].shape[1]),
            "construction_median_squared_distance": construction_median,
            "source_median_squared_distance": source_median,
            "mean_actual_oracle_fraction": float(np.mean(actual_oracle_fractions)),
        },
        "best_semantic_f1": best_semantic,
        "summaries": dict(summaries),
        "records": records,
    }
    save_json(args.output, result)
    if args.oracle_mode == "multiple":
        _plot_recall_multiples(
            summaries,
            args.figure,
            args.oracle_fractions,
            args.retrieval_multipliers,
        )
    else:
        _plot(summaries, args.figure, args.oracle_fraction, args.oracle_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--causal-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--target-shift", type=int, default=1)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--oracle-fraction", type=float, default=0.001)
    parser.add_argument(
        "--oracle-mode", choices=("matched", "fixed", "multiple"), default="matched"
    )
    parser.add_argument(
        "--oracle-fractions",
        type=float,
        nargs="+",
        default=(0.0002, 0.0005, 0.001),
    )
    parser.add_argument(
        "--retrieval-multipliers", type=float, nargs="+", default=(1, 2, 5, 10)
    )
    parser.add_argument(
        "--predicted-fractions",
        type=float,
        nargs="+",
        default=(
            0.00005,
            0.0001,
            0.0002,
            0.00035,
            0.0005,
            0.00075,
            0.001,
            0.0015,
            0.002,
            0.003,
            0.005,
        ),
    )
    parser.add_argument("--seed", type=int, default=91_027)
    args = parser.parse_args()
    args.predicted_fractions = tuple(args.predicted_fractions)
    args.oracle_fractions = tuple(args.oracle_fractions)
    args.retrieval_multipliers = tuple(args.retrieval_multipliers)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
