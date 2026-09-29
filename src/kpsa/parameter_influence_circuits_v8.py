"""Sensitivity-circuit experiments for the vision atlas.

The sensitivity atlas stores squared parameter-gradient energy.  Its natural
causal estimand is therefore susceptibility to an isotropic local parameter
intervention, rather than the effect of one privileged ablation direction.
This module evaluates that estimand with antithetic Rademacher perturbations
and exact forward passes.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .parameter_groups import ParameterPartition
from .refined_analysis import paired_t_summary
from .vision_atlas_size_scaling_v8 import (
    make_predictions as make_scaled_predictions,
)
from .vision_atlas_size_scaling_v8 import (
    offset_major_indices,
)
from .vision_causal_refined import (
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)
from .vision_prototype_strengthening_v8 import (
    construction_indices,
    make_predictions,
)
from .vision_sample_refined import encode_indices, profile_functional_images


def _group_seed(seed: int, spec_index: int, group_index: int) -> int:
    """Mix intervention, tensor, and group identifiers into a stable RNG seed."""
    modulus = 2**63 - 25
    return int(
        (
            seed * 6_364_136_223_846_793_005
            + (spec_index + 1) * 1_442_695_040_888_963_407
            + (group_index + 1) * 2_862_933_555_777_941_757
        )
        % modulus
    )


def _axis_slice(tensor: torch.Tensor, axis: int, index: int) -> tuple:
    key = [slice(None)] * tensor.ndim
    key[axis] = index
    return tuple(key)


@contextmanager
def additive_group_noise(
    partition: ParameterPartition,
    selected: torch.Tensor,
    standard_deviation: float,
    *,
    seed: int,
    sign: int = 1,
):
    """Add deterministic iid Rademacher noise to selected parameter groups.

    Every parameter coordinate receives ``+/- standard_deviation``.  Reusing
    the seed with the opposite sign gives an antithetic intervention.  The
    original parameters are restored exactly on exit.
    """
    if selected.shape != (partition.n_groups,) or selected.dtype != torch.bool:
        raise ValueError("selection must be one Boolean flag per parameter group")
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("standard deviation must be finite and positive")
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or 1")

    changes: list[tuple[torch.nn.Parameter, tuple | None, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for spec_index, spec in enumerate(partition.slices):
                local = selected[spec.offset : spec.offset + spec.count]
                indices = local.nonzero().flatten().tolist()
                for local_index in indices:
                    group_index = spec.offset + local_index
                    key = (
                        None
                        if spec.axis is None
                        else _axis_slice(spec.parameter, spec.axis, local_index)
                    )
                    view = spec.parameter if key is None else spec.parameter[key]
                    original = view.detach().clone()
                    generator = torch.Generator(device=view.device).manual_seed(
                        _group_seed(seed, spec_index, group_index)
                    )
                    noise = torch.empty_like(view)
                    noise.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
                    noise.mul_(standard_deviation * sign)
                    view.add_(noise)
                    changes.append((spec.parameter, key, original))
        yield
    finally:
        with torch.no_grad():
            for parameter, key, original in reversed(changes):
                view = parameter if key is None else parameter[key]
                view.copy_(original)


@torch.no_grad()
def scoped_parameter_rms(partition: ParameterPartition, scope: torch.Tensor) -> float:
    """RMS of all parameter coordinates owned by groups in ``scope``."""
    if scope.shape != (partition.n_groups,) or scope.dtype != torch.bool:
        raise ValueError("scope must be one Boolean flag per parameter group")
    squared = torch.zeros((), dtype=torch.float64, device=partition.device)
    count = 0
    for spec in partition.slices:
        local = scope[spec.offset : spec.offset + spec.count]
        indices = local.nonzero().flatten().to(spec.parameter.device)
        if not len(indices):
            continue
        if spec.axis is None:
            values = spec.parameter
        else:
            values = spec.parameter.index_select(spec.axis, indices)
        squared += values.double().square().sum()
        count += values.numel()
    if count == 0:
        raise ValueError("scope contains no parameter coordinates")
    return float((squared / count).sqrt())


@torch.no_grad()
def margin_values(model, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Correct-class logit margins for a batch of images."""
    logits = model(images.to(next(model.parameters()).device)).float()
    labels = labels.to(logits.device)
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    alternatives = logits.clone()
    alternatives.scatter_(1, labels[:, None], -torch.inf)
    return correct - alternatives.max(1).values


def _mean_by_query(records, method, fraction, metric):
    values = defaultdict(list)
    for row in records:
        if (
            row["target_correct"]
            and row["method"] == method
            and row["fraction"] == fraction
        ):
            values[int(row["query_column"])].append(float(row[metric]))
    return {query: sum(current) / len(current) for query, current in values.items()}


def paired_metric(records, method, baseline, fraction, metric):
    """Paired summary after averaging perturbation directions per query."""
    left = _mean_by_query(records, method, fraction, metric)
    right = _mean_by_query(records, baseline, fraction, metric)
    keys = left.keys() & right.keys()
    return paired_t_summary([left[key] - right[key] for key in keys])


def _gap_recovery(method, scalar, oracle):
    denominator = oracle["mean"] - scalar["mean"]
    return (method["mean"] - scalar["mean"]) / denominator if denominator > 0 else None


def run_study(
    model,
    dataset,
    payload,
    construction_features,
    *,
    queries=24,
    fractions=(0.0002, 0.001),
    directions=4,
    noise_scale=0.03,
    prototype_count=800,
    prototype_scale=0.025,
    target_shift=0,
    weight_mode="normalized",
    precomputed_predictions=None,
    prediction_metadata=None,
    seed=314_159,
):
    """Evaluate sparse retrieved parameter subnetworks under local noise."""
    if directions < 1:
        raise ValueError("directions must be positive")
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout).cpu()
    sizes = partition.sizes.detach().double().cpu()
    parameter_rms = scoped_parameter_rms(partition, scope)
    noise_std = parameter_rms * noise_scale

    if precomputed_predictions is None:
        predictions, _maps, construction_median, source_median = make_predictions(
            payload["source_profiles"],
            payload["source_features"],
            payload["target_features"],
            payload["class_source_profiles"],
            construction_features,
            prototype_counts=(prototype_count,),
            prototype_scales=(prototype_scale,),
            exact_scale=0.1,
            seed=91_027,
        )
        source_examples = payload["source_profiles"].shape[1]
    else:
        predictions = dict(precomputed_predictions)
        prediction_metadata = prediction_metadata or {}
        construction_median = prediction_metadata[
            "construction_median_squared_distance"
        ]
        source_median = prediction_metadata["source_median_squared_distance"]
        source_examples = int(prediction_metadata["source_examples"])
    predictions["direct_gradient_oracle"] = payload["target_profiles"].float()
    methods = {
        name: predictions[name]
        for name in (
            "exact_rbf_scale_0.1",
            "exact_rbf_scale_0.1_permuted",
            f"prototype_{prototype_count}_scale_{prototype_scale:g}",
            f"prototype_{prototype_count}_scale_{prototype_scale:g}_permuted",
            "nearest_encoder",
            "class_onehot",
            "scalar_mass",
            "direct_gradient_oracle",
        )
    }

    target_indices = payload["target_indices"].long()
    query_columns = (
        torch.linspace(0, len(target_indices) - 1, min(queries, len(target_indices)))
        .round()
        .long()
        .unique()
    )
    records = []
    started = time.perf_counter()
    for position, column in enumerate(query_columns.tolist()):
        target_index = int(target_indices[column])
        target_image, target_label = dataset[target_index]
        target_label = int(target_label)
        same_index = target_index + (1 if target_index % 50 < 49 else -1)
        same_image, same_label = dataset[same_index]
        same_label = int(same_label)
        if same_label != target_label:
            raise RuntimeError("same-class companion crossed a class boundary")
        off_column = (column + len(target_indices) // 2) % len(target_indices)
        off_image, off_label = dataset[int(target_indices[off_column])]
        off_label = int(off_label)
        images = torch.stack((target_image, same_image, off_image))
        labels = torch.tensor((target_label, same_label, off_label), dtype=torch.long)
        intact = margin_values(model, images[:1], labels[:1])
        target_correct = bool(intact[0] > 0)

        for method, scores in methods.items():
            for fraction in fractions:
                selected, actual = select_parameter_budget(
                    scores[:, column].clamp_min(0), sizes, scope, fraction
                )
                energy_coverage = float(
                    payload["target_profiles"][:, column][selected].sum()
                )
                for direction in range(directions):
                    intervention_seed = seed + position * 100_003 + direction * 1_009
                    with additive_group_noise(
                        partition,
                        selected,
                        noise_std,
                        seed=intervention_seed,
                        sign=1,
                    ):
                        plus = margin_values(model, images, labels).cpu()
                    with additive_group_noise(
                        partition,
                        selected,
                        noise_std,
                        seed=intervention_seed,
                        sign=-1,
                    ):
                        minus = margin_values(model, images, labels).cpu()
                    response = (plus - minus) / (2 * noise_std)
                    squared = response.square()
                    records.append(
                        {
                            "query_column": column,
                            "target_correct": target_correct,
                            "class_index": target_label,
                            "method": method,
                            "fraction": fraction,
                            "direction": direction,
                            "actual_scoped_parameter_fraction": actual,
                            "selected_groups": int(selected.sum()),
                            "gradient_energy_coverage": energy_coverage,
                            "target_squared_susceptibility": float(squared[0]),
                            "same_class_squared_susceptibility": float(squared[1]),
                            "off_class_squared_susceptibility": float(squared[2]),
                            "off_class_selectivity": float(squared[0] - squared[2]),
                            "within_class_selectivity": float(squared[0] - squared[1]),
                        }
                    )
        print(
            f"parameter influence {position + 1}/{len(query_columns)}",
            flush=True,
        )

    evaluated = tuple(methods)
    comparisons = {}
    summaries = {}
    for fraction in fractions:
        key = f"{fraction:g}"
        current = {}
        for method in evaluated:
            values = _mean_by_query(
                records,
                method,
                fraction,
                "target_squared_susceptibility",
            )
            current[method] = paired_t_summary(list(values.values()))
        scalar = current["scalar_mass"]
        oracle = current["direct_gradient_oracle"]
        summaries[key] = current
        comparisons[key] = {
            method: {
                "vs_scalar": paired_metric(
                    records,
                    method,
                    "scalar_mass",
                    fraction,
                    "target_squared_susceptibility",
                ),
                "vs_nearest": paired_metric(
                    records,
                    method,
                    "nearest_encoder",
                    fraction,
                    "target_squared_susceptibility",
                ),
                "vs_class_onehot": paired_metric(
                    records,
                    method,
                    "class_onehot",
                    fraction,
                    "target_squared_susceptibility",
                ),
                "oracle_gap_recovered": _gap_recovery(current[method], scalar, oracle),
                "off_class_selectivity": paired_t_summary(
                    list(
                        _mean_by_query(
                            records,
                            method,
                            fraction,
                            "off_class_selectivity",
                        ).values()
                    )
                ),
                "within_class_selectivity": paired_t_summary(
                    list(
                        _mean_by_query(
                            records,
                            method,
                            fraction,
                            "within_class_selectivity",
                        ).values()
                    )
                ),
            }
            for method in (
                "exact_rbf_scale_0.1",
                f"prototype_{prototype_count}_scale_{prototype_scale:g}",
            )
        }
        comparisons[key]["exact_rbf_scale_0.1"]["vs_shuffled_pairing"] = paired_metric(
            records,
            "exact_rbf_scale_0.1",
            "exact_rbf_scale_0.1_permuted",
            fraction,
            "target_squared_susceptibility",
        )
        prototype = f"prototype_{prototype_count}_scale_{prototype_scale:g}"
        comparisons[key][prototype]["vs_shuffled_pairing"] = paired_metric(
            records,
            prototype,
            prototype + "_permuted",
            fraction,
            "target_squared_susceptibility",
        )

    return {
        "setting": "vitb16_input_conditioned_parameter_influence_circuits",
        "model": {
            "name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
        },
        "protocol": {
            "functional": "correct-class logit margin",
            "intervention": (
                "antithetic iid Rademacher additive noise on retrieved parameter "
                "coordinates; exact forward central difference"
            ),
            "intervention_estimand": (
                "expected squared local margin response per unit coordinate-noise "
                "variance"
            ),
            "source_examples": source_examples,
            "sensitivity_weight": weight_mode,
            "target_examples": len(target_indices),
            "target_shift": target_shift,
            "causal_queries": len(query_columns),
            "correct_query_filter_applied_only_in_summary": True,
            "fractions": list(fractions),
            "directions_per_query_method_budget": directions,
            "noise_scale_relative_to_scoped_parameter_rms": noise_scale,
            "scoped_parameter_rms": parameter_rms,
            "coordinate_noise_standard_deviation": noise_std,
            "prototype_count": prototype_count,
            "prototype_scale": prototype_scale,
            "construction_median_squared_distance": construction_median,
            "source_median_squared_distance": source_median,
            "paired_noise": (
                "same group-coordinate directions for all retrieval methods"
            ),
        },
        "timing": {"causal_seconds": time.perf_counter() - started},
        "mean_susceptibility": summaries,
        "comparisons": comparisons,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0002, 0.001])
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--target-shift", type=int, default=0)
    parser.add_argument("--source-count", type=int, default=800)
    parser.add_argument(
        "--weight-mode", choices=("raw", "normalized"), default="normalized"
    )
    args = parser.parse_args()
    if args.weight_mode == "raw" and args.source_count == 800:
        parser.error("raw mode requires rebuilding the source atlas")

    seed_everything(314_159)
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
    if args.source_count == 800:
        construction = construction_indices(payload["representation_indices"])
        construction_features = encode_indices(
            representation_model, processor, raw_dataset, construction
        )
        shifted_target_indices = payload["target_indices"].long() + args.target_shift
        shifted_target_features = (
            payload["target_features"]
            if args.target_shift == 0
            else encode_indices(
                representation_model,
                processor,
                raw_dataset,
                shifted_target_indices,
            )
        )
        source_features = None
        feature_map = None
    else:
        class_starts = torch.unique(
            payload["representation_indices"].long() // 50 * 50, sorted=True
        )
        classes = len(class_starts)
        if args.source_count % classes:
            parser.error("source count must contain complete class-balanced blocks")
        per_class = args.source_count // classes
        if not 1 <= per_class <= 43:
            parser.error("scaled source count must use between 1 and 43 images/class")
        source_indices = offset_major_indices(
            payload["representation_indices"], range(per_class)
        )
        construction = offset_major_indices(
            payload["representation_indices"], range(43, 47)
        )
        target_offset = 49 + args.target_shift
        if not 0 <= target_offset < 50:
            parser.error("scaled target offset must remain within its ImageNet class")
        shifted_target_indices = offset_major_indices(
            payload["representation_indices"], [target_offset]
        )
        encoded = encode_indices(
            representation_model,
            processor,
            raw_dataset,
            torch.cat((source_indices, construction, shifted_target_indices)),
        )
        source_end = len(source_indices)
        construction_end = source_end + len(construction)
        source_features = encoded[:source_end]
        construction_features = encoded[source_end:construction_end]
        shifted_target_features = encoded[construction_end:]
        construction_median = median_squared_distance(construction_features)
        feature_map = fit_prototype_response_map(
            construction_features,
            800,
            bandwidth_squared=construction_median * 0.025**2,
            seed=91_027 + 800,
        )
    del representation_model
    gc.collect()
    torch.cuda.empty_cache()

    model = (
        timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)
        .cuda()
        .eval()
    )
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    precomputed_predictions = None
    prediction_metadata = None
    if args.source_count != 800:
        partition = ParameterPartition(model, "swiglu")
        source_profiles, source_error, source_profile_seconds = (
            profile_functional_images(
                model,
                dataset,
                source_indices,
                partition,
                functional="margin",
                weight_mode=args.weight_mode,
            )
        )
        shifted_target_profiles, partition_error, profile_seconds = (
            profile_functional_images(
                model,
                dataset,
                shifted_target_indices,
                partition,
                functional="margin",
                weight_mode=args.weight_mode,
            )
        )
        source_responses = feature_map.transform(source_features).float()
        target_responses = feature_map.transform(shifted_target_features).float()
        scaled, source_median = make_scaled_predictions(
            source_profiles,
            source_features,
            shifted_target_features,
            source_responses,
            target_responses,
            exact_scale=0.1,
            classes=classes,
            seed=91_027,
        )
        cosine = source_features.float() @ shifted_target_features.float().T
        class_profiles = source_profiles.reshape(
            source_profiles.shape[0], per_class, classes
        ).mean(1)
        precomputed_predictions = {
            "exact_rbf_scale_0.1": scaled["exact_rbf"],
            "exact_rbf_scale_0.1_permuted": scaled["exact_rbf_permuted"],
            "prototype_800_scale_0.025": scaled["prototype_rbf"],
            "prototype_800_scale_0.025_permuted": scaled["prototype_rbf_permuted"],
            "nearest_encoder": source_profiles[:, cosine.argmax(0)],
            "class_onehot": class_profiles,
            "scalar_mass": scaled["scalar_mass"],
        }
        prediction_metadata = {
            "source_examples": args.source_count,
            "source_median_squared_distance": source_median,
            "construction_median_squared_distance": construction_median,
        }
        if args.weight_mode == "raw":
            totals = source_profiles.double().sum(0)
            group_sums = source_profiles.double().sum(1)
            group_square_sums = source_profiles.double().square().sum(1)
            group_ess = group_sums.square() / group_square_sums.clamp_min(1e-300)
            prediction_metadata["raw_total_energy"] = {
                "min": float(totals.min()),
                "q25": float(totals.quantile(0.25)),
                "median": float(totals.median()),
                "q75": float(totals.quantile(0.75)),
                "q95": float(totals.quantile(0.95)),
                "q99": float(totals.quantile(0.99)),
                "max": float(totals.max()),
                "iqr_over_median": float(
                    (totals.quantile(0.75) - totals.quantile(0.25))
                    / totals.median().clamp_min(1e-300)
                ),
                "mean_group_effective_sample_size": float(group_ess.mean()),
                "median_group_effective_sample_size": float(group_ess.median()),
            }
        payload = dict(payload)
        payload["target_indices"] = shifted_target_indices
        payload["target_features"] = shifted_target_features
        payload["target_profiles"] = shifted_target_profiles
        del source_profiles, source_responses, target_responses, scaled, cosine
        gc.collect()
        torch.cuda.empty_cache()
    elif args.target_shift:
        shifted_target_profiles, partition_error, profile_seconds = (
            profile_functional_images(
                model,
                dataset,
                shifted_target_indices,
                ParameterPartition(model, "swiglu"),
                functional="margin",
                weight_mode=args.weight_mode,
            )
        )
        payload = dict(payload)
        payload["target_indices"] = shifted_target_indices
        payload["target_features"] = shifted_target_features
        payload["target_profiles"] = shifted_target_profiles
    result = run_study(
        model,
        dataset,
        payload,
        construction_features,
        queries=args.queries,
        fractions=tuple(args.fractions),
        directions=args.directions,
        noise_scale=args.noise_scale,
        target_shift=args.target_shift,
        weight_mode=args.weight_mode,
        precomputed_predictions=precomputed_predictions,
        prediction_metadata=prediction_metadata,
    )
    if args.target_shift or args.source_count != 800:
        result["fidelity"] = {"max_partition_relative_error": partition_error}
        result["timing"]["target_profile_seconds"] = profile_seconds
    if args.source_count != 800:
        result["fidelity"]["source_max_partition_relative_error"] = source_error
        result["timing"]["source_profile_seconds"] = source_profile_seconds
        if "raw_total_energy" in prediction_metadata:
            result["raw_total_energy"] = prediction_metadata["raw_total_energy"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
