"""Sample-level DINOv2 atlas and activation-aligned causal evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .kernel_sensitivity import (
    fit_empirical_rbf_index,
    fit_nystrom_rbf,
    fit_weighted_kernel_mean_index,
    median_squared_distance,
)
from .parameter_groups import ParameterPartition
from .representation_sensitivity import (
    partition_gradient_energy,
    profile_ranking_metrics,
    sensitivity_weights,
)
from .vision_causal_refined import (
    activation_feature_scores,
    calibrate_activation_means,
    functional_value,
    mean_ablate_mlp_activations,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)


def profile_functional_images(
    model,
    dataset,
    indices,
    partition,
    functional="margin",
    weight_mode="normalized",
):
    profiles = torch.empty(partition.n_groups, len(indices), dtype=torch.float32)
    max_error = 0.0
    device = next(model.parameters()).device
    started = time.perf_counter()
    for column, index in enumerate(indices.tolist()):
        image, label = dataset[index]
        model.zero_grad(set_to_none=True)
        logits = model(image[None].to(device)).float()
        if functional == "class_logit":
            objective = logits[0, int(label)]
        elif functional == "margin":
            alternatives = logits[0].clone()
            alternatives[int(label)] = -torch.inf
            objective = logits[0, int(label)] - alternatives.max()
        elif functional == "loss":
            objective = F.cross_entropy(
                logits, torch.tensor([int(label)], device=logits.device)
            )
        else:
            raise ValueError(functional)
        objective.backward()
        energy = partition_gradient_energy(partition)
        direct = sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        max_error = max(
            max_error,
            float((energy.sum() - direct).abs() / direct.clamp_min(1e-30)),
        )
        profiles[:, column] = sensitivity_weights(
            energy.detach().double().cpu()[None], weight_mode
        )[0].float()
    model.zero_grad(set_to_none=True)
    return profiles, max_error, time.perf_counter() - started


@torch.no_grad()
def encode_indices(representation_model, processor, raw_dataset, indices, batch_size=32):
    device = next(representation_model.parameters()).device
    values = []
    for local in indices.split(batch_size):
        images = [raw_dataset[int(index)][0] for index in local]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = representation_model(pixel_values=pixels)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        values.append(F.normalize(pooled.float(), dim=1).cpu())
    return torch.cat(values)


def sample_predictions(
    source_profiles,
    source_features,
    target_features,
    *,
    rbf_scales=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    nystrom_landmarks=(),
    nystrom_landmark_method="kmeans++",
    seed=23_041,
):
    source = source_profiles.double()
    source_features = F.normalize(source_features.double(), dim=1)
    target_features = F.normalize(target_features.double(), dim=1)
    cosine = source_features @ target_features.T
    bandwidth = median_squared_distance(source_features)
    generator = torch.Generator().manual_seed(seed)
    random_source = F.normalize(
        torch.randn(source_features.shape, generator=generator, dtype=torch.float64), dim=1
    )
    random_target = F.normalize(
        torch.randn(target_features.shape, generator=generator, dtype=torch.float64), dim=1
    )
    permutation = torch.randperm(len(source_features), generator=generator)
    kernels = {
        "sample_dinov2_shifted_cosine": ((cosine + 1) / 2).clamp(0, 1),
        "sample_permuted": ((source_features[permutation] @ target_features.T + 1) / 2).clamp(0, 1),
        "sample_jl": ((random_source @ random_target.T + 1) / 2).clamp(0, 1),
    }
    predictions = {
        name: source @ kernel / len(source_features) for name, kernel in kernels.items()
    }
    for scale in rbf_scales:
        exact = fit_empirical_rbf_index(
            source,
            source_features,
            bandwidth_squared=bandwidth * scale**2,
        )
        predictions[f"sample_dinov2_rbf_scale_{scale:g}"] = exact.query(
            target_features, mass_weighted=True
        )
    for landmarks in nystrom_landmarks:
        if not 1 <= landmarks <= len(source_features):
            raise ValueError("Nyström landmark count must fit the source atlas")
        feature_map = fit_nystrom_rbf(
            source_features,
            landmarks,
            bandwidth_squared=bandwidth * 0.1**2,
            landmark_method=nystrom_landmark_method,
            seed=seed,
            eigenvalue_floor=1e-6,
        )
        index = fit_weighted_kernel_mean_index(
            source.T,
            source_features,
            feature_map,
            weight_mode="precomputed",
        )
        predictions[f"sample_nystrom_rbf_scale_0.1_landmarks_{landmarks}"] = (
            index.query(target_features, mass_weighted=True)
        )
    predictions["sample_nearest"] = source[:, cosine.argmax(0)]
    predictions["scalar_mass"] = source.mean(1, keepdim=True).expand(-1, len(target_features))
    return predictions


def run_study(
    model,
    representation_model,
    processor,
    dataset,
    raw_dataset,
    retained_payload,
    *,
    queries=24,
    fractions=(0.0002, 0.001),
    calibration_images=64,
    target_shift=0,
    rbf_scales=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    nystrom_landmarks=(),
    recompute_class_references=False,
    functional="margin",
    evaluation_functional=None,
    evaluate_off_target=False,
    weight_mode="normalized",
    seed=23_041,
):
    evaluation_functional = evaluation_functional or functional
    partition = ParameterPartition(model, "swiglu")
    source_indices = retained_payload["source_indices"].long()
    target_indices = retained_payload["target_indices"].long() + target_shift
    representation_indices = retained_payload.get("representation_indices")
    if representation_indices is None:
        representation_indices = torch.cat(
            [torch.arange(int(index) - 4, int(index)) for index in source_indices]
        )
    source_profiles, max_error, profile_seconds = profile_functional_images(
        model,
        dataset,
        representation_indices,
        partition,
        functional,
        weight_mode,
    )
    source_features = encode_indices(
        representation_model, processor, raw_dataset, representation_indices
    )
    target_features = encode_indices(
        representation_model, processor, raw_dataset, target_indices
    )
    retained_groups = retained_payload[functional]["target_profiles"].shape[0]
    if (
        target_shift
        or retained_groups != partition.n_groups
        or recompute_class_references
        or weight_mode == "raw"
    ):
        target_profiles, target_error, target_seconds = profile_functional_images(
            model, dataset, target_indices, partition, functional, weight_mode
        )
        target_profiles = target_profiles.double()
        max_error = max(max_error, target_error)
    else:
        target_profiles = retained_payload[functional]["target_profiles"].double()
        target_seconds = 0.0
    if (
        retained_groups != partition.n_groups
        or recompute_class_references
        or weight_mode == "raw"
    ):
        class_source_profiles, source_error, class_source_seconds = profile_functional_images(
            model, dataset, source_indices, partition, functional, weight_mode
        )
        max_error = max(max_error, source_error)
        class_source_profiles = class_source_profiles.double()
    else:
        class_source_profiles = retained_payload[functional]["source_profiles"].double()
        class_source_seconds = 0.0
    predictions = sample_predictions(
        source_profiles,
        source_features,
        target_features,
        rbf_scales=rbf_scales,
        nystrom_landmarks=nystrom_landmarks,
        seed=seed,
    )
    predictions["class_source_onehot_reference"] = class_source_profiles
    predictions["direct_gradient_oracle"] = target_profiles
    cold_query = {
        method: profile_ranking_metrics(values, target_profiles)
        for method, values in predictions.items()
    }

    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    positions = torch.linspace(
        0, len(source_indices) - 1, min(calibration_images, len(source_indices))
    ).round().long().unique()
    means = calibrate_activation_means(model, dataset, source_indices[positions], layout)
    weight = partition.weight_energy().detach().double().cpu() * sizes
    predictions["weight_magnitude"] = weight[:, None].expand(-1, len(target_indices))
    predictions["random"] = torch.rand(
        target_profiles.shape,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    query_columns = torch.linspace(
        0, len(target_indices) - 1, min(queries, len(target_indices))
    ).round().long().unique()
    records = []
    for column in query_columns.tolist():
        image, label = dataset[int(target_indices[column])]
        label = int(label)
        baseline = functional_value(model, image, label, evaluation_functional)
        prediction = int(model(image[None].to(partition.device)).argmax(1))
        source_image, source_label = dataset[int(source_indices[column])]
        if int(source_label) != label:
            raise RuntimeError("source and target class orders differ")
        source_magnitude, source_act_grad = activation_feature_scores(
            model, source_image, label, evaluation_functional, partition, layout
        )
        target_magnitude, target_act_grad = activation_feature_scores(
            model, image, label, evaluation_functional, partition, layout
        )
        off_target = None
        if evaluate_off_target:
            off_column = (column + len(target_indices) // 2) % len(target_indices)
            off_image, off_label = dataset[int(target_indices[off_column])]
            off_label = int(off_label)
            off_baseline = functional_value(
                model, off_image, off_label, evaluation_functional
            )
            off_prediction = int(model(off_image[None].to(partition.device)).argmax(1))
            off_target = (off_column, off_image, off_label, off_baseline, off_prediction)
        score_grid = {
            **{method: values[:, column] for method, values in predictions.items()},
            "source_activation_magnitude": source_magnitude,
            "source_activation_x_gradient": source_act_grad,
            "target_activation_magnitude": target_magnitude,
            "target_activation_x_gradient": target_act_grad,
        }
        for method, scores in score_grid.items():
            for fraction in fractions:
                selected, actual = select_parameter_budget(
                    scores.clamp_min(0), sizes, scope, fraction
                )
                with mean_ablate_mlp_activations(layout, selected, means):
                    intervened = functional_value(
                        model, image, label, evaluation_functional
                    )
                    off_intervened = (
                        functional_value(
                            model, off_target[1], off_target[2], evaluation_functional
                        )
                        if off_target is not None
                        else None
                    )
                direction = -1 if evaluation_functional == "loss" else 1
                degradation = direction * (baseline - intervened)
                record = {
                    "method": method,
                    "query_column": column,
                    "class_index": label,
                    "target_correct": prediction == label,
                    "requested_parameter_fraction": fraction,
                    "actual_scoped_parameter_fraction": actual,
                    "selected_groups": int(selected.sum()),
                    "baseline": baseline,
                    "intervened": intervened,
                    "degradation": degradation,
                }
                if off_target is not None:
                    off_degradation = direction * (off_target[3] - off_intervened)
                    record.update(
                        {
                            "off_target_query_column": off_target[0],
                            "off_target_class_index": off_target[2],
                            "off_target_correct": off_target[4] == off_target[2],
                            "off_target_baseline": off_target[3],
                            "off_target_intervened": off_intervened,
                            "off_target_degradation": off_degradation,
                            "selectivity": degradation - off_degradation,
                        }
                    )
                records.append(record)
    return (
        {
            "setting": "imagenet_sample_level_dinov2",
            "protocol": {
                "functional": functional,
                "evaluation_functional": evaluation_functional,
                "atlas_samples": len(representation_indices),
                "atlas_samples_per_class": len(representation_indices) // len(target_indices),
                "target_queries": len(target_indices),
                "target_shift_from_development_image": target_shift,
                "causal_queries": len(query_columns),
                "intervention": "source-calibration mean ablation at coupled MLP fc2 inputs",
                "fractions": list(fractions),
                "representation": "full normalized frozen DINOv2-Small per-image CLS",
                "rbf_scales": list(rbf_scales),
                "nystrom_landmarks": list(nystrom_landmarks),
                "off_target_evaluation": evaluate_off_target,
                "sensitivity_weight": weight_mode,
            },
            "fidelity": {"max_partition_relative_error": max_error},
            "timing": {
                "source_profile_seconds": profile_seconds,
                "target_profile_seconds": target_seconds,
                "class_source_profile_seconds": class_source_seconds,
            },
            "cold_query": cold_query,
            "records": records,
        },
        {
            "source_profiles": source_profiles,
            "target_profiles": target_profiles.float(),
            "class_source_profiles": class_source_profiles.float(),
            "source_features": source_features,
            "target_features": target_features,
            "representation_indices": representation_indices,
            "source_indices": source_indices,
            "target_indices": target_indices,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--target-shift", type=int, default=0)
    parser.add_argument("--rbf-scales", type=float, nargs="+", default=[0.1])
    parser.add_argument("--nystrom-landmarks", type=int, nargs="*", default=[])
    parser.add_argument("--representation-model", default="facebook/dinov2-small")
    parser.add_argument(
        "--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k"
    )
    parser.add_argument("--recompute-class-references", action="store_true")
    parser.add_argument(
        "--functional", choices=("margin", "class_logit", "loss"), default="margin"
    )
    parser.add_argument(
        "--evaluation-functional",
        choices=("margin", "class_logit", "loss"),
        default=None,
    )
    parser.add_argument("--evaluate-off-target", action="store_true")
    parser.add_argument(
        "--weight-mode", choices=("raw", "normalized"), default="normalized"
    )
    parser.add_argument("--no-tensors", action="store_true")
    args = parser.parse_args()
    seed_everything(23_041)
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    model_name = args.model
    model = timm.create_model(model_name, pretrained=True).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    raw_dataset = ImageFolder(args.data / "validation")
    representation_model = AutoModel.from_pretrained(
        args.representation_model, local_files_only=True
    ).cuda().eval()
    processor = AutoImageProcessor.from_pretrained(
        args.representation_model, local_files_only=True
    )
    retained = torch.load(args.profiles, map_location="cpu", weights_only=True)
    result, tensors = run_study(
        model,
        representation_model,
        processor,
        dataset,
        raw_dataset,
        retained,
        queries=args.queries,
        target_shift=args.target_shift,
        rbf_scales=tuple(args.rbf_scales),
        nystrom_landmarks=tuple(args.nystrom_landmarks),
        recompute_class_references=args.recompute_class_references,
        functional=args.functional,
        evaluation_functional=args.evaluation_functional,
        evaluate_off_target=args.evaluate_off_target,
        weight_mode=args.weight_mode,
    )
    result["model"] = {
        "name": model_name,
        "architecture": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.with_suffix(".json"), result)
    if not args.no_tensors:
        torch.save(tensors, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
