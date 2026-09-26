"""Scale the vision sensitivity atlas and choose a source-sample default."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .kernel_sensitivity import (
    fit_prototype_response_map,
    median_squared_distance,
)
from .refined_analysis import paired_t_summary
from .representation_sensitivity import profile_ranking_metrics
from .vision_causal_refined import (
    calibrate_activation_means,
    functional_value,
    mean_ablate_mlp_activations,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)
from .vision_sample_refined import encode_indices, profile_functional_images


def offset_major_indices(
    representation_indices: torch.Tensor,
    offsets: range | tuple[int, ...] | list[int],
) -> torch.Tensor:
    """Return class-balanced prefixes by placing offsets before classes."""
    starts = torch.unique(representation_indices.long() // 50 * 50, sorted=True)
    offsets = torch.as_tensor(list(offsets), dtype=torch.long)
    if not len(offsets) or int(offsets.min()) < 0 or int(offsets.max()) >= 50:
        raise ValueError("ImageNet validation offsets must lie in [0, 49]")
    return (offsets[:, None] + starts[None, :]).reshape(-1)


def balanced_pairing_permutation(
    samples: int,
    classes: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Shuffle class pairings separately in every balanced source block."""
    if samples % classes:
        raise ValueError("samples must contain complete class-balanced blocks")
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for block in range(samples // classes):
        blocks.append(block * classes + torch.randperm(classes, generator=generator))
    return torch.cat(blocks)


def _rbf_similarity(left, right, bandwidth_squared):
    cosine = F.normalize(left.float(), dim=1) @ F.normalize(right.float(), dim=1).T
    squared_distance = (2 - 2 * cosine).clamp_min(0)
    return torch.exp(-squared_distance / (2 * bandwidth_squared))


def _summary_metrics(predicted, measured):
    values = profile_ranking_metrics(predicted, measured)
    return {
        key: values[key]
        for key in (
            "groups",
            "queries",
            "top_count",
            "mean_spearman",
            "mean_topk_recall",
            "mean_ndcg",
            "mean_cosine",
        )
    }


def make_predictions(
    source_profiles,
    source_features,
    target_features,
    source_responses,
    target_responses,
    *,
    exact_scale,
    classes,
    seed,
):
    """Query exact and compact kernels plus matched shuffled-pair controls."""
    samples = source_profiles.shape[1]
    permutation = balanced_pairing_permutation(
        samples, classes, seed=seed + samples
    )
    source_median = median_squared_distance(source_features)
    exact = _rbf_similarity(
        source_features,
        target_features,
        source_median * exact_scale**2,
    )
    exact_permuted = _rbf_similarity(
        source_features[permutation],
        target_features,
        source_median * exact_scale**2,
    )
    prototype = source_responses @ target_responses.T
    prototype_permuted = source_responses[permutation] @ target_responses.T
    profiles = source_profiles.cuda()

    def aggregate(similarity, source_matrix=profiles):
        return (source_matrix @ similarity.cuda() / samples).cpu()

    predictions = {
        "exact_rbf": aggregate(exact),
        "exact_rbf_permuted": aggregate(exact_permuted),
        "prototype_rbf": aggregate(prototype),
        "prototype_rbf_permuted": aggregate(prototype_permuted),
    }
    predictions["scalar_mass"] = source_profiles.mean(1, keepdim=True).expand(
        -1, len(target_features)
    )
    del profiles
    return predictions, source_median


def paired(records, method, baseline, fraction):
    left = {
        int(row["query_column"]): float(row["degradation"])
        for row in records
        if row["target_correct"]
        and row["method"] == method
        and row["fraction"] == fraction
    }
    right = {
        int(row["query_column"]): float(row["degradation"])
        for row in records
        if row["target_correct"]
        and row["method"] == baseline
        and row["fraction"] == fraction
    }
    return paired_t_summary([left[key] - right[key] for key in left.keys() & right])


def prepare_queries(model, dataset, target_indices, partition, queries):
    columns = torch.linspace(0, len(target_indices) - 1, queries).round().long().unique()
    prepared = []
    for column in columns.tolist():
        image, label = dataset[int(target_indices[column])]
        label = int(label)
        baseline = functional_value(model, image, label, "margin")
        predicted = int(model(image[None].to(partition.device)).argmax(1))
        prepared.append((column, image, label, baseline, predicted == label))
    return prepared


def evaluate_causal(
    model,
    prepared_queries,
    predictions,
    partition,
    layout,
    scope,
    sizes,
    means,
    fractions,
    *,
    comparison_pairs=(
        ("exact_rbf", "exact_rbf_permuted"),
        ("prototype_rbf", "prototype_rbf_permuted"),
    ),
    scalar_method="scalar_mass",
):
    records = []
    for column, image, label, baseline, correct in prepared_queries:
        for method, scores in predictions.items():
            for fraction in fractions:
                selected, actual = select_parameter_budget(
                    scores[:, column].clamp_min(0), sizes, scope, fraction
                )
                with mean_ablate_mlp_activations(layout, selected, means):
                    intervened = functional_value(model, image, label, "margin")
                records.append(
                    {
                        "query_column": column,
                        "target_correct": correct,
                        "method": method,
                        "fraction": fraction,
                        "actual_scoped_parameter_fraction": actual,
                        "selected_groups": int(selected.sum()),
                        "degradation": baseline - intervened,
                    }
                )
    comparisons = {}
    matched = {}
    for method, permuted in comparison_pairs:
        comparisons[method] = {
            f"{fraction:g}": paired(
                records, method, scalar_method, fraction
            )
            for fraction in fractions
        }
        matched[method] = {
            f"{fraction:g}": paired(
                records, method, permuted, fraction
            )
            for fraction in fractions
        }
    return records, comparisons, matched


def select_default(development, fractions):
    """Choose the smallest gated N within 95% of the best prototype gain."""
    eligible = []
    for samples, result in development.items():
        scalar = result["paired_causal_vs_scalar"]["prototype_rbf"]
        matched = result["paired_causal_vs_matched_permutation"]["prototype_rbf"]
        gated = all(
            scalar[f"{fraction:g}"]["lower_95"] > 0
            and matched[f"{fraction:g}"]["lower_95"] > 0
            for fraction in fractions
        )
        score = sum(scalar[f"{fraction:g}"]["mean"] for fraction in fractions) / len(
            fractions
        )
        if gated:
            eligible.append((int(samples), score))
    used_gate = bool(eligible)
    if not eligible:
        eligible = [
            (
                int(samples),
                sum(
                    result["paired_causal_vs_scalar"]["prototype_rbf"][
                        f"{fraction:g}"
                    ]["mean"]
                    for fraction in fractions
                )
                / len(fractions),
            )
            for samples, result in development.items()
        ]
    best = max(score for _, score in eligible)
    threshold = 0.95 * best if best >= 0 else best / 0.95
    selected = min(samples for samples, score in eligible if score >= threshold)
    return {
        "source_examples": selected,
        "best_mean_gain": best,
        "threshold": threshold,
        "required_positive_95_ci_vs_scalar_and_permutation": True,
        "gate_satisfied_by_at_least_one_candidate": used_gate,
        "rule": (
            "smallest source count with both prototype causal gains having positive "
            "95% CIs versus scalar and pairing permutation, and mean gain across "
            "budgets at least 95% of the best gated candidate"
        ),
    }


def plot_curve(result, path):
    import matplotlib.pyplot as plt

    counts = [int(value) for value in result["protocol"]["source_counts"]]
    development = result["development"]
    fractions = result["protocol"]["fractions"]
    colors = {"exact_rbf": "#16697A", "prototype_rbf": "#DB6400"}
    labels = {"exact_rbf": "Exact DINO RBF", "prototype_rbf": "Prototype map"}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.75), constrained_layout=True)
    for method in ("exact_rbf", "prototype_rbf", "scalar_mass"):
        axes[0].plot(
            counts,
            [development[str(n)]["cold_query"][method]["mean_spearman"] for n in counts],
            marker="o",
            linewidth=2,
            color=colors.get(method, "#777777"),
            label=labels.get(method, "Scalar"),
        )
    axes[0].set_ylabel("Mean Spearman")
    axes[0].set_title("Cold profile prediction")
    for axis, comparison, title in (
        (axes[1], "paired_causal_vs_scalar", "Causal gain over scalar"),
        (
            axes[2],
            "paired_causal_vs_matched_permutation",
            "Causal gain over shuffled pairing",
        ),
    ):
        for method in ("exact_rbf", "prototype_rbf"):
            for fraction, linestyle in zip(fractions, ("-", "--")):
                summaries = [
                    development[str(n)][comparison][method][f"{fraction:g}"]
                    for n in counts
                ]
                means = [value["mean"] for value in summaries]
                low = [value["mean"] - value["lower_95"] for value in summaries]
                high = [value["upper_95"] - value["mean"] for value in summaries]
                axis.errorbar(
                    counts,
                    means,
                    yerr=[low, high],
                    marker="o",
                    linewidth=1.8,
                    capsize=2,
                    linestyle=linestyle,
                    color=colors[method],
                    label=f"{labels[method]}, {100*fraction:.2g}%",
                )
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_ylabel("Paired margin-degradation gain")
        axis.set_title(title)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(counts, [f"{n:,}" for n in counts], rotation=30)
        axis.set_xlabel("Source images N")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=7)
    selected = result["selection"]["source_examples"]
    for axis in axes:
        axis.axvline(selected, color="#8B1E3F", linewidth=1, alpha=0.75)
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument(
        "--source-counts",
        type=int,
        nargs="+",
        default=[200, 400, 800, 1600, 3200, 6400, 8600],
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--exact-scale", type=float, default=0.1)
    parser.add_argument("--prototype-scale", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=91_027)
    args = parser.parse_args()
    seed_everything(args.seed)
    if sorted(args.source_counts) != args.source_counts:
        parser.error("source counts must be increasing")

    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    class_starts = torch.unique(
        payload["representation_indices"].long() // 50 * 50, sorted=True
    )
    classes = len(class_starts)
    if any(count % classes for count in args.source_counts):
        parser.error("every source count must be divisible by selected classes")
    max_per_class = max(args.source_counts) // classes
    if max_per_class > 43:
        parser.error("source roles are reserved for at most offsets 0--42")

    raw_dataset = ImageFolder(args.data / "validation")
    source_indices = offset_major_indices(
        payload["representation_indices"], range(max_per_class)
    )
    construction_indices = offset_major_indices(
        payload["representation_indices"], range(43, 47)
    )
    development_indices = offset_major_indices(
        payload["representation_indices"], [48]
    )
    confirmation_indices = offset_major_indices(
        payload["representation_indices"], [49]
    )
    calibration_indices = offset_major_indices(
        payload["representation_indices"], [47]
    )
    role_sets = [
        set(value.tolist())
        for value in (
            source_indices,
            construction_indices,
            development_indices,
            confirmation_indices,
            calibration_indices,
        )
    ]
    if any(role_sets[i] & role_sets[j] for i in range(5) for j in range(i)):
        raise RuntimeError("source, construction, target, and calibration roles overlap")

    encoder = AutoModel.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    ).cuda().eval()
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    )
    encoded = encode_indices(
        encoder,
        processor,
        raw_dataset,
        torch.cat(
            [
                source_indices,
                construction_indices,
                development_indices,
                confirmation_indices,
            ]
        ),
    )
    source_end = len(source_indices)
    construction_end = source_end + len(construction_indices)
    development_end = construction_end + len(development_indices)
    source_features = encoded[:source_end]
    construction_features = encoded[source_end:construction_end]
    development_features = encoded[construction_end:development_end]
    confirmation_features = encoded[development_end:]
    print(f"encoded {len(encoded):,} DINOv2 representations", flush=True)
    del encoder, encoded
    gc.collect()
    torch.cuda.empty_cache()

    construction_median = median_squared_distance(construction_features)
    feature_map = fit_prototype_response_map(
        construction_features,
        len(construction_features),
        bandwidth_squared=construction_median * args.prototype_scale**2,
        seed=args.seed + 800,
    )
    source_responses = feature_map.transform(source_features).float()
    development_responses = feature_map.transform(development_features).float()
    confirmation_responses = feature_map.transform(confirmation_features).float()

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    ).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    fractions = (0.0002, 0.001)

    started = time.perf_counter()
    source_profiles, source_error, source_seconds = profile_functional_images(
        model,
        dataset,
        source_indices,
        partition,
        functional="margin",
        weight_mode="normalized",
    )
    print(
        f"profiled {len(source_indices):,} source images in {source_seconds:.1f}s",
        flush=True,
    )
    development_profiles, development_error, development_profile_seconds = (
        profile_functional_images(
            model,
            dataset,
            development_indices,
            partition,
            functional="margin",
            weight_mode="normalized",
        )
    )
    confirmation_profiles, confirmation_error, confirmation_profile_seconds = (
        profile_functional_images(
            model,
            dataset,
            confirmation_indices,
            partition,
            functional="margin",
            weight_mode="normalized",
        )
    )
    print("profiled development and confirmation targets", flush=True)
    means = calibrate_activation_means(model, dataset, calibration_indices, layout)
    development_queries = prepare_queries(
        model, dataset, development_indices, partition, args.queries
    )
    confirmation_queries = prepare_queries(
        model, dataset, confirmation_indices, partition, args.queries
    )

    development = {}
    for samples in args.source_counts:
        predictions, source_median = make_predictions(
            source_profiles[:, :samples],
            source_features[:samples],
            development_features,
            source_responses[:samples],
            development_responses,
            exact_scale=args.exact_scale,
            classes=classes,
            seed=args.seed,
        )
        cold = {
            method: _summary_metrics(predictions[method], development_profiles)
            for method in ("exact_rbf", "prototype_rbf", "scalar_mass")
        }
        records, comparisons, matched = evaluate_causal(
            model,
            development_queries,
            predictions,
            partition,
            layout,
            scope,
            sizes,
            means,
            fractions,
        )
        development[str(samples)] = {
            "source_median_squared_distance": source_median,
            "cold_query": cold,
            "paired_causal_vs_scalar": comparisons,
            "paired_causal_vs_matched_permutation": matched,
            "correct_causal_queries": sum(row[4] for row in development_queries),
            "records": records,
        }
        print(f"completed development N={samples:,}", flush=True)
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

    selection = select_default(development, fractions)
    selected = selection["source_examples"]
    print(f"selected N={selected:,}; running confirmation", flush=True)
    confirmation_predictions, confirmation_median = make_predictions(
        source_profiles[:, :selected],
        source_features[:selected],
        confirmation_features,
        source_responses[:selected],
        confirmation_responses,
        exact_scale=args.exact_scale,
        classes=classes,
        seed=args.seed,
    )
    confirmation_cold = {
        method: _summary_metrics(confirmation_predictions[method], confirmation_profiles)
        for method in ("exact_rbf", "prototype_rbf", "scalar_mass")
    }
    confirmation_records, confirmation_comparisons, confirmation_matched = evaluate_causal(
        model,
        confirmation_queries,
        confirmation_predictions,
        partition,
        layout,
        scope,
        sizes,
        means,
        fractions,
    )
    result = {
        "setting": "vitb16_dinov2_atlas_source_size_scaling",
        "model": {
            "name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "groups": partition.n_groups,
        },
        "protocol": {
            "source_counts": args.source_counts,
            "source_offsets": [0, max_per_class - 1],
            "construction_offsets": [43, 46],
            "development_offset": 48,
            "confirmation_offset": 49,
            "calibration_offset": 47,
            "selected_classes": classes,
            "causal_queries_per_split": args.queries,
            "fractions": list(fractions),
            "sensitivity_weight": "normalized",
            "exact_scale": args.exact_scale,
            "prototype_count": len(construction_features),
            "prototype_scale": args.prototype_scale,
            "construction_median_squared_distance": construction_median,
            "pairing_control": (
                "feature/profile pairings shuffled across classes separately within "
                "each balanced per-offset source block"
            ),
        },
        "fidelity": {
            "source_max_partition_relative_error": source_error,
            "development_max_partition_relative_error": development_error,
            "confirmation_max_partition_relative_error": confirmation_error,
        },
        "timing": {
            "source_profile_seconds": source_seconds,
            "development_profile_seconds": development_profile_seconds,
            "confirmation_profile_seconds": confirmation_profile_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "development": development,
        "selection": selection,
        "confirmation": {
            "source_examples": selected,
            "source_median_squared_distance": confirmation_median,
            "cold_query": confirmation_cold,
            "paired_causal_vs_scalar": confirmation_comparisons,
            "paired_causal_vs_matched_permutation": confirmation_matched,
            "correct_causal_queries": sum(row[4] for row in confirmation_queries),
            "records": confirmation_records,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plot_curve(result, args.figure)
    print(f"wrote {args.output} and {args.figure}", flush=True)


if __name__ == "__main__":
    main()
