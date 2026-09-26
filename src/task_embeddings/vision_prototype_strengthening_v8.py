"""Fixed-prototype and matched-locality controls for the vision atlas."""

from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .kernel_sensitivity import (
    PrototypeResponseMap,
    fit_empirical_rbf_index,
    fit_prototype_response_map,
    fit_weighted_kernel_mean_index,
    median_squared_distance,
)
from .refined_analysis import paired_t_summary
from .representation_sensitivity import profile_ranking_metrics
from .vision_causal_refined import (
    calibrate_activation_means,
    functional_value,
    mean_ablate_mlp_activations,
    mean_ablation_taylor_scores,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
    zero_ablate_mlp_activations,
)
from .vision_sample_refined import encode_indices, profile_functional_images


def construction_indices(representation_indices: torch.Tensor) -> torch.Tensor:
    """Select four disjoint ImageNet validation images per represented class."""
    starts = torch.unique(representation_indices.long() // 50 * 50, sorted=True)
    return torch.cat([start + torch.arange(20, 24) for start in starts])


def source_rebuild_indices(
    representation_indices: torch.Tensor,
    start_offset: int,
) -> torch.Tensor:
    """Choose four replacement source images per represented ImageNet class."""
    if not 0 <= start_offset <= 46:
        raise ValueError("four-image source block must fit within each class")
    starts = torch.unique(representation_indices.long() // 50 * 50, sorted=True)
    return torch.cat(
        [start + torch.arange(start_offset, start_offset + 4) for start in starts]
    )


def make_predictions(
    source_profiles,
    source_features,
    target_features,
    class_profiles,
    construction_features,
    *,
    prototype_counts=(200, 400, 800),
    prototype_scales=(0.025, 0.05, 0.1, 0.25),
    exact_scale=0.1,
    seed=91_027,
):
    source = source_profiles.detach().double().cpu()
    source_features = F.normalize(source_features.detach().double().cpu(), dim=1)
    target_features = F.normalize(target_features.detach().double().cpu(), dim=1)
    construction = F.normalize(construction_features.detach().double().cpu(), dim=1)
    construction_median = median_squared_distance(construction)
    source_median = median_squared_distance(source_features)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(source_features), generator=generator)
    cosine = source_features @ target_features.T
    permuted_cosine = source_features[permutation] @ target_features.T
    exact = fit_empirical_rbf_index(
        source,
        source_features,
        bandwidth_squared=source_median * exact_scale**2,
    )
    exact_permuted = fit_empirical_rbf_index(
        source,
        source_features[permutation],
        bandwidth_squared=source_median * exact_scale**2,
    )
    predictions = {
        f"exact_rbf_scale_{exact_scale:g}": exact.query(
            target_features, mass_weighted=True
        ).float(),
        f"exact_rbf_scale_{exact_scale:g}_permuted": exact_permuted.query(
            target_features, mass_weighted=True
        ).float(),
        "linear_encoder": (source @ cosine / len(source_features)).float(),
        "nearest_encoder": source[:, cosine.argmax(0)].float(),
        "nearest_encoder_permuted": source[:, permuted_cosine.argmax(0)].float(),
        "class_onehot": class_profiles.detach().float().cpu(),
        "scalar_mass": source.mean(1, keepdim=True)
        .expand(-1, len(target_features))
        .float(),
    }
    maps = {}
    for count in prototype_counts:
        base_map = fit_prototype_response_map(
            construction,
            count,
            bandwidth_squared=construction_median * prototype_scales[0] ** 2,
            seed=seed + count,
        )
        for scale in prototype_scales:
            name = f"prototype_{count}_scale_{scale:g}"
            feature_map = PrototypeResponseMap(
                base_map.prototypes,
                construction_median * scale**2,
            )
            maps[name] = feature_map
            index = fit_weighted_kernel_mean_index(
                source.T,
                source_features,
                feature_map,
                weight_mode="normalized",
            )
            permuted = fit_weighted_kernel_mean_index(
                source.T,
                source_features[permutation],
                feature_map,
                weight_mode="normalized",
            )
            predictions[name] = index.query(
                target_features, mass_weighted=True
            ).float()
            predictions[name + "_permuted"] = permuted.query(
                target_features, mass_weighted=True
            ).float()
            del index, permuted
    return predictions, maps, construction_median, source_median


def paired(records, method, baseline, fraction, metric="degradation"):
    left = {
        int(row["query_column"]): float(row[metric])
        for row in records
        if row["target_correct"]
        and row["method"] == method
        and row["fraction"] == fraction
    }
    right = {
        int(row["query_column"]): float(row[metric])
        for row in records
        if row["target_correct"]
        and row["method"] == baseline
        and row["fraction"] == fraction
    }
    return paired_t_summary([left[key] - right[key] for key in left.keys() & right.keys()])


def run_study(
    model,
    dataset,
    payload,
    construction_features,
    target_features,
    *,
    target_shift=0,
    queries=24,
    fractions=(0.0002, 0.001),
    prototype_counts=(200, 400, 800),
    prototype_scales=(0.025, 0.05, 0.1, 0.25),
    weight_mode="normalized",
    ablation="mean",
    seed=91_027,
):
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    source_profiles = payload["source_profiles"]
    source_features = payload["source_features"]
    class_profiles = payload["class_source_profiles"]
    source_indices = payload["source_indices"].long()
    target_indices = payload["target_indices"].long() + target_shift
    if target_shift or weight_mode == "raw":
        target_profiles, max_error, target_seconds = profile_functional_images(
            model,
            dataset,
            target_indices,
            partition,
            functional="margin",
            weight_mode=weight_mode,
        )
    else:
        target_profiles = payload["target_profiles"]
        max_error = 0.0
        target_seconds = 0.0
    predictions, _maps, construction_median, source_median = make_predictions(
        source_profiles,
        source_features,
        target_features,
        class_profiles,
        construction_features,
        prototype_counts=prototype_counts,
        prototype_scales=prototype_scales,
        seed=seed,
    )
    predictions["direct_gradient_oracle"] = target_profiles.detach().float().cpu()
    cold = {
        method: profile_ranking_metrics(scores.double(), target_profiles.double())
        for method, scores in predictions.items()
    }

    calibration_columns = torch.linspace(0, len(source_indices) - 1, 64).round().long()
    means = calibrate_activation_means(
        model, dataset, source_indices[calibration_columns], layout
    )
    attribution_reference = (
        means
        if ablation == "mean"
        else {name: torch.zeros_like(value) for name, value in means.items()}
    )
    query_columns = torch.linspace(0, len(target_indices) - 1, queries).round().long().unique()
    records = []
    started = time.perf_counter()
    for column in query_columns.tolist():
        image, label = dataset[int(target_indices[column])]
        label = int(label)
        baseline = functional_value(model, image, label, "margin")
        prediction = int(model(image[None].to(partition.device)).argmax(1))
        off_column = (column + len(target_indices) // 2) % len(target_indices)
        off_image, off_label = dataset[int(target_indices[off_column])]
        off_label = int(off_label)
        off_baseline = functional_value(model, off_image, off_label, "margin")
        same_index = int(target_indices[column]) + 1
        same_image, same_label = dataset[same_index]
        same_label = int(same_label)
        if same_label != label:
            raise RuntimeError("same-class companion crossed an ImageNet class boundary")
        same_baseline = functional_value(model, same_image, same_label, "margin")
        same_prediction = int(model(same_image[None].to(partition.device)).argmax(1))
        taylor = mean_ablation_taylor_scores(
            model,
            image,
            label,
            "margin",
            partition,
            layout,
            attribution_reference,
        )
        score_grid = {**predictions, "direct_coordinate_taylor": taylor[:, None]}
        for method, scores in score_grid.items():
            for fraction in fractions:
                method_scores = (
                    scores[:, 0] if method == "direct_coordinate_taylor" else scores[:, column]
                )
                selected, actual = select_parameter_budget(
                    method_scores.clamp_min(0), sizes, scope, fraction
                )
                if ablation == "mean":
                    intervention = mean_ablate_mlp_activations(
                        layout, selected, means
                    )
                elif ablation == "zero":
                    intervention = zero_ablate_mlp_activations(layout, selected)
                else:
                    raise ValueError("ablation must be mean or zero")
                with intervention:
                    intervened = functional_value(model, image, label, "margin")
                    off_intervened = functional_value(
                        model, off_image, off_label, "margin"
                    )
                    same_intervened = functional_value(
                        model, same_image, same_label, "margin"
                    )
                degradation = baseline - intervened
                off_degradation = off_baseline - off_intervened
                same_degradation = same_baseline - same_intervened
                records.append(
                    {
                        "query_column": column,
                        "class_index": label,
                        "target_correct": prediction == label,
                        "method": method,
                        "fraction": fraction,
                        "actual_scoped_parameter_fraction": actual,
                        "selected_groups": int(selected.sum()),
                        "degradation": degradation,
                        "off_target_degradation": off_degradation,
                        "selectivity": degradation - off_degradation,
                        "same_class_index": same_index,
                        "same_class_correct": same_prediction == same_label,
                        "same_class_degradation": same_degradation,
                        "within_class_selectivity": degradation - same_degradation,
                    }
                )
    prototype_methods = [
        name
        for name in predictions
        if name.startswith("prototype_") and not name.endswith("_permuted")
    ]
    comparisons = defaultdict(dict)
    evaluated_methods = sorted({row["method"] for row in records})
    for method in evaluated_methods:
        if method == "scalar_mass":
            continue
        for fraction in fractions:
            comparisons[method][f"{fraction:g}"] = paired(
                records, method, "scalar_mass", fraction
            )
    matched = {
        method: {
            f"{fraction:g}": paired(
                records, method, method + "_permuted", fraction
            )
            for fraction in fractions
        }
        for method in prototype_methods
    }
    exact_name = "exact_rbf_scale_0.1"
    matched[exact_name] = {
        f"{fraction:g}": paired(
            records, exact_name, exact_name + "_permuted", fraction
        )
        for fraction in fractions
    }
    best_prototype = max(
        prototype_methods,
        key=lambda method: sum(
            comparisons[method][f"{fraction:g}"]["mean"] for fraction in fractions
        ),
    )
    gates = {
        method: all(
            comparisons[method][f"{fraction:g}"]["lower_95"] > 0
            and matched[method][f"{fraction:g}"]["lower_95"] > 0
            for fraction in fractions
        )
        for method in (exact_name, best_prototype)
    }
    specificity = defaultdict(dict)
    for method in evaluated_methods:
        for fraction in fractions:
            values = [
                row["within_class_selectivity"]
                for row in records
                if row["method"] == method
                and row["fraction"] == fraction
                and row["target_correct"]
                and row["same_class_correct"]
            ]
            if len(values) >= 2:
                specificity[method][f"{fraction:g}"] = paired_t_summary(values)
    return {
        "setting": "vitb16_fixed_prototype_representation_strengthening",
        "model": {
            "name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "groups": partition.n_groups,
        },
        "protocol": {
            "phase": "development" if not target_shift else "confirmation",
            "source_examples": source_profiles.shape[1],
            "source_rebuild_offset": payload.get("source_rebuild_offset"),
            "construction_examples": len(construction_features),
            "construction_indices": "offsets 20--23 within each selected ImageNet class",
            "target_shift": target_shift,
            "target_examples": len(target_indices),
            "causal_queries": len(query_columns),
            "fractions": list(fractions),
            "sensitivity_weight": weight_mode,
            "ablation": ablation,
            "activation_attribution": (
                "gradient * activation"
                if ablation == "zero"
                else "gradient * (activation - calibration_mean)"
            ),
            "prototype_counts": list(prototype_counts),
            "prototype_scales": list(prototype_scales),
            "construction_median_squared_distance": construction_median,
            "source_median_squared_distance": source_median,
            "best_prototype": best_prototype,
            "selection_rule": (
                "largest mean paired gain over scalar across both budgets; "
                "must also beat matched local permutation at both budgets"
            ),
        },
        "fidelity": {"max_partition_relative_error": max_error},
        "timing": {
            "source_profile_seconds": payload.get("source_profile_seconds", 0.0),
            "target_profile_seconds": target_seconds,
            "causal_seconds": time.perf_counter() - started,
        },
        "cold_query": cold,
        "paired_causal_vs_scalar": dict(comparisons),
        "paired_causal_vs_matched_permutation": matched,
        "within_class_selectivity": dict(specificity),
        "development_gates": gates,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-shift", type=int, default=0)
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--rebuild-source-offset", type=int)
    parser.add_argument(
        "--weight-mode", choices=("raw", "normalized"), default="normalized"
    )
    parser.add_argument("--ablation", choices=("mean", "zero"), default="mean")
    parser.add_argument("--prototype-counts", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument(
        "--prototype-scales", type=float, nargs="+", default=[0.025, 0.05, 0.1, 0.25]
    )
    args = parser.parse_args()
    if args.weight_mode == "raw" and args.rebuild_source_offset is None:
        parser.error("raw sensitivity requires --rebuild-source-offset")
    seed_everything(91_027)
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    raw_dataset = ImageFolder(args.data / "validation")
    representation_model = AutoModel.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    ).cuda().eval()
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    )
    construction = construction_indices(payload["representation_indices"])
    construction_features = encode_indices(
        representation_model, processor, raw_dataset, construction
    )
    rebuilt_indices = None
    rebuilt_features = None
    if args.rebuild_source_offset is not None:
        rebuilt_indices = source_rebuild_indices(
            payload["representation_indices"], args.rebuild_source_offset
        )
        rebuilt_features = encode_indices(
            representation_model, processor, raw_dataset, rebuilt_indices
        )
    target_indices = payload["target_indices"].long() + args.target_shift
    target_features = (
        payload["target_features"]
        if not args.target_shift
        else encode_indices(
            representation_model, processor, raw_dataset, target_indices
        )
    )
    del representation_model
    gc.collect()
    torch.cuda.empty_cache()

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    ).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    if rebuilt_indices is not None:
        rebuilt_profiles, source_error, source_seconds = profile_functional_images(
            model,
            dataset,
            rebuilt_indices,
            ParameterPartition(model, "swiglu"),
            functional="margin",
            weight_mode=args.weight_mode,
        )
        selected_classes = len(rebuilt_indices) // 4
        payload = dict(payload)
        payload["source_profiles"] = rebuilt_profiles
        payload["source_features"] = rebuilt_features
        payload["class_source_profiles"] = rebuilt_profiles.reshape(
            rebuilt_profiles.shape[0], selected_classes, 4
        ).mean(2)
        payload["source_rebuild_offset"] = args.rebuild_source_offset
        payload["source_profile_seconds"] = source_seconds
        payload["source_rebuild_partition_error"] = source_error
    result = run_study(
        model,
        dataset,
        payload,
        construction_features,
        target_features,
        target_shift=args.target_shift,
        queries=args.queries,
        prototype_counts=tuple(args.prototype_counts),
        prototype_scales=tuple(args.prototype_scales),
        weight_mode=args.weight_mode,
        ablation=args.ablation,
    )
    if rebuilt_indices is not None:
        result["fidelity"]["source_rebuild_partition_error"] = payload[
            "source_rebuild_partition_error"
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
