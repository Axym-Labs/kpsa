"""Nested parameter-resolution sweep for the two selected vision assays."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .refined_analysis import paired_t_summary
from .representation_sensitivity import profile_ranking_metrics
from .vision_atlas_size_scaling_v8 import (
    make_predictions as make_scaled_predictions,
)
from .vision_atlas_size_scaling_v8 import (
    offset_major_indices,
)
from .vision_causal_refined import (
    functional_value,
    mean_ablation_taylor_scores,
    mlp_feature_layout,
    select_parameter_budget,
    zero_ablate_mlp_activations,
)
from .vision_sample_refined import encode_indices, profile_functional_images


def nested_feature_assignment(partition, layout, bundle_size):
    """Map scoped MLP features into within-layer contiguous bundles."""
    if bundle_size < 1:
        raise ValueError("bundle size must be positive")
    assignment = torch.full((partition.n_groups,), -1, dtype=torch.long)
    labels = []
    next_group = 0
    for item in layout:
        count = int(item["count"])
        width = min(bundle_size, count)
        local_groups = (count + width - 1) // width
        local = torch.arange(count) // width + next_group
        start = int(item["offset"])
        assignment[start : start + count] = local
        labels.extend(
            f"{item['name']}[{begin}:{min(begin + width, count)}]"
            for begin in range(0, count, width)
        )
        next_group += local_groups
    if not bool((assignment >= 0).any()):
        raise ValueError("layout contains no parameter groups")
    return assignment, labels


def aggregate_groups(values, assignment, groups):
    """Add fine-group values into a nested coarse partition."""
    if values.shape[0] != len(assignment):
        raise ValueError("values and assignment must share their group axis")
    scoped = assignment >= 0
    shape = (groups, *values.shape[1:])
    output = torch.zeros(shape, dtype=values.dtype, device=values.device)
    output.index_add_(0, assignment[scoped].to(values.device), values[scoped])
    return output


def expand_selection(selected, assignment):
    """Expand selected coarse groups back to the global fine partition."""
    output = torch.zeros(len(assignment), dtype=torch.bool)
    scoped = assignment >= 0
    output[scoped] = selected[assignment[scoped]]
    return output


def _paired(records, assay, resolution, method, baseline, metric):
    left = {
        int(row["query_column"]): float(row[metric])
        for row in records
        if row["assay"] == assay
        and row["resolution"] == resolution
        and row["method"] == method
        and row["target_correct"]
    }
    right = {
        int(row["query_column"]): float(row[metric])
        for row in records
        if row["assay"] == assay
        and row["resolution"] == resolution
        and row["method"] == baseline
        and row["target_correct"]
    }
    return paired_t_summary([left[key] - right[key] for key in left.keys() & right])


def _gap_recovery(method, scalar, oracle):
    denominator = oracle["mean"]
    return method["mean"] / denominator if denominator > 0 else None


def run_resolution_study(
    model,
    dataset,
    target_indices,
    target_profiles,
    predictions,
    *,
    queries=100,
    parameter_fraction=1 / 12,
    representation_dimension=800,
):
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    sizes = partition.sizes.detach().double().cpu()
    query_columns = torch.linspace(
        0, len(target_indices) - 1, min(queries, len(target_indices))
    ).round().long().unique()
    score_grid = {
        "exact_rbf": predictions["exact_rbf"],
        "prototype_rbf": predictions["prototype_rbf"],
        "scalar_mass": predictions["scalar_mass"],
        "direct_parameter": target_profiles,
    }
    resolution_specs = (
        ("feature", 1),
        ("bundle_4", 4),
        ("bundle_32", 32),
        ("mlp_block", max(int(item["count"]) for item in layout)),
    )
    prepared = []
    taylor_profiles = torch.zeros_like(target_profiles)
    zero_reference = {
        item["name"]: torch.zeros(item["count"], device=partition.device)
        for item in layout
    }
    for column in query_columns.tolist():
        image, label = dataset[int(target_indices[column])]
        label = int(label)
        baseline = functional_value(model, image, label, "margin")
        correct = bool(baseline > 0)
        taylor_profiles[:, column] = mean_ablation_taylor_scores(
            model,
            image,
            label,
            "margin",
            partition,
            layout,
            zero_reference,
        ).float()
        prepared.append((column, image, label, baseline, correct))
    score_grid["activation_attribution"] = taylor_profiles

    records = []
    fidelity = {}
    metadata = {}
    for resolution, bundle_size in resolution_specs:
        assignment, labels = nested_feature_assignment(
            partition, layout, bundle_size
        )
        groups = len(labels)
        coarse_sizes = aggregate_groups(sizes, assignment, groups)
        coarse_truth = aggregate_groups(target_profiles, assignment, groups)
        coarse_scores = {
            method: aggregate_groups(scores, assignment, groups)
            for method, scores in score_grid.items()
        }
        metadata[resolution] = {
            "groups": groups,
            "mean_parameters_per_group": float(coarse_sizes.mean()),
            "atlas_bytes_float32": groups * representation_dimension * 4,
        }
        fidelity[resolution] = {}
        for method in ("exact_rbf", "prototype_rbf", "scalar_mass"):
            metrics = profile_ranking_metrics(
                coarse_scores[method], coarse_truth
            )
            fidelity[resolution][method] = {
                key: metrics[key]
                for key in (
                    "mean_spearman",
                    "mean_topk_recall",
                    "mean_ndcg",
                    "mean_cosine",
                    "spearman",
                )
            }

        for column, image, label, baseline, correct in prepared:
            for method, scores in coarse_scores.items():
                selected_coarse, actual = select_parameter_budget(
                    scores[:, column].clamp_min(0),
                    coarse_sizes,
                    torch.ones(groups, dtype=torch.bool),
                    parameter_fraction,
                )
                selected_fine = expand_selection(selected_coarse, assignment)
                with zero_ablate_mlp_activations(layout, selected_fine):
                    intervened = functional_value(model, image, label, "margin")
                coverage = float(target_profiles[:, column][selected_fine].sum())
                records.extend(
                    (
                        {
                            "assay": "neuron_deactivation",
                            "resolution": resolution,
                            "method": method,
                            "query_column": column,
                            "target_correct": correct,
                            "requested_parameter_fraction": parameter_fraction,
                            "actual_parameter_fraction": actual,
                            "selected_coarse_groups": int(selected_coarse.sum()),
                            "selected_fine_groups": int(selected_fine.sum()),
                            "metric": baseline - intervened,
                        },
                        {
                            "assay": "parameter_influence",
                            "resolution": resolution,
                            "method": method,
                            "query_column": column,
                            "target_correct": correct,
                            "requested_parameter_fraction": parameter_fraction,
                            "actual_parameter_fraction": actual,
                            "selected_coarse_groups": int(selected_coarse.sum()),
                            "selected_fine_groups": int(selected_fine.sum()),
                            "metric": coverage,
                        },
                    )
                )
        print(f"completed resolution {resolution}", flush=True)

    comparisons = {}
    for assay in ("neuron_deactivation", "parameter_influence"):
        comparisons[assay] = {}
        oracle = (
            "activation_attribution"
            if assay == "neuron_deactivation"
            else "direct_parameter"
        )
        for resolution, _ in resolution_specs:
            current = {}
            oracle_gap = _paired(
                records, assay, resolution, oracle, "scalar_mass", "metric"
            )
            for method in ("exact_rbf", "prototype_rbf"):
                summary = _paired(
                    records,
                    assay,
                    resolution,
                    method,
                    "scalar_mass",
                    "metric",
                )
                current[method] = {
                    "vs_scalar": summary,
                    "oracle_gap_recovered": _gap_recovery(
                        summary, None, oracle_gap
                    ),
                }
            current[oracle] = {"vs_scalar": oracle_gap}
            comparisons[assay][resolution] = current
    return {
        "protocol": {
            "resolutions": [name for name, _ in resolution_specs],
            "parameter_fraction": parameter_fraction,
            "causal_queries": len(query_columns),
            "sensitivity_weight": "normalized",
            "neuron_intervention": "zero coupled MLP feature activations",
            "parameter_influence_metric": (
                "direct-gradient energy captured; exact local parameter-noise "
                "endpoint validated separately at fine resolution"
            ),
            "representation_dimension": representation_dimension,
        },
        "resolution_metadata": metadata,
        "query_fidelity": fidelity,
        "comparisons": comparisons,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=6400)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    seed_everything(271_828)

    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    retained = torch.load(args.input, map_location="cpu", weights_only=True)
    classes = len(
        torch.unique(retained["representation_indices"].long() // 50 * 50)
    )
    if args.source_count % classes:
        parser.error("source count must contain complete class-balanced blocks")
    per_class = args.source_count // classes
    if not 1 <= per_class <= 43:
        parser.error("source count must use between 1 and 43 images/class")
    source_indices = offset_major_indices(
        retained["representation_indices"], range(per_class)
    )
    construction_indices = offset_major_indices(
        retained["representation_indices"], range(43, 47)
    )
    target_indices = offset_major_indices(retained["representation_indices"], [49])

    raw_dataset = ImageFolder(args.data / "validation")
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
        torch.cat((source_indices, construction_indices, target_indices)),
    )
    source_end = len(source_indices)
    construction_end = source_end + len(construction_indices)
    source_features = encoded[:source_end]
    construction_features = encoded[source_end:construction_end]
    target_features = encoded[construction_end:]
    del encoder, encoded
    gc.collect()
    torch.cuda.empty_cache()

    construction_median = median_squared_distance(construction_features)
    feature_map = fit_prototype_response_map(
        construction_features,
        800,
        bandwidth_squared=construction_median * 0.025**2,
        seed=91_027 + 800,
    )
    source_responses = feature_map.transform(source_features).float()
    target_responses = feature_map.transform(target_features).float()

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    ).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    started = time.perf_counter()
    source_profiles, source_error, source_seconds = profile_functional_images(
        model,
        dataset,
        source_indices,
        partition,
        functional="margin",
        weight_mode="normalized",
    )
    target_profiles, target_error, target_seconds = profile_functional_images(
        model,
        dataset,
        target_indices,
        partition,
        functional="margin",
        weight_mode="normalized",
    )
    predictions, source_median = make_scaled_predictions(
        source_profiles,
        source_features,
        target_features,
        source_responses,
        target_responses,
        exact_scale=0.1,
        classes=classes,
        seed=91_027,
    )
    predictions = {
        "exact_rbf": predictions["exact_rbf"],
        "prototype_rbf": predictions["prototype_rbf"],
        "scalar_mass": predictions["scalar_mass"],
    }
    del source_profiles, source_features, source_responses, target_responses
    gc.collect()
    torch.cuda.empty_cache()

    result = run_resolution_study(
        model,
        dataset,
        target_indices,
        target_profiles,
        predictions,
        queries=args.queries,
    )
    result.update(
        {
            "setting": "vitb16_nested_parameter_resolution_sweep",
            "model": {
                "name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "parameters": sum(p.numel() for p in model.parameters()),
                "groups": partition.n_groups,
            },
            "atlas": {
                "source_examples": args.source_count,
                "source_median_squared_distance": source_median,
                "construction_median_squared_distance": construction_median,
            },
            "fidelity": {
                "source_max_partition_relative_error": source_error,
                "target_max_partition_relative_error": target_error,
            },
            "timing": {
                "source_profile_seconds": source_seconds,
                "target_profile_seconds": target_seconds,
                "total_seconds": time.perf_counter() - started,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
