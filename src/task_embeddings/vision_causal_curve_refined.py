"""Frozen causal curves with parameter-sensitivity and activation-atlas baselines."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .vision_causal_refined import (
    activation_feature_scores,
    calibrate_activation_means,
    functional_value,
    mean_ablate_mlp_activations,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)
from .vision_sample_refined import sample_predictions


def rbf_kernel(source_features, target_features, scale):
    source_features = F.normalize(source_features.float(), dim=1)
    target_features = F.normalize(target_features.float(), dim=1)
    source_distance = torch.cdist(source_features, source_features).square()
    positive = source_distance[source_distance > 0]
    bandwidth = positive.median() if len(positive) else torch.tensor(1.0)
    target_distance = torch.cdist(source_features, target_features).square()
    return torch.exp(-target_distance / (2 * bandwidth.clamp_min(1e-12) * scale**2))


def activation_atlas_predictions(
    model,
    dataset,
    source_indices,
    source_features,
    target_features,
    partition,
    layout,
    scope,
    *,
    rbf_scale,
):
    local_indices = scope.nonzero().flatten()
    magnitude = torch.empty(len(local_indices), len(source_indices), dtype=torch.float32)
    act_grad = torch.empty_like(magnitude)
    started = time.perf_counter()
    for column, index in enumerate(source_indices.tolist()):
        image, label = dataset[index]
        current_magnitude, current_act_grad = activation_feature_scores(
            model, image, int(label), "margin", partition, layout
        )
        magnitude[:, column] = current_magnitude[local_indices].float()
        act_grad[:, column] = current_act_grad[local_indices].float()
    magnitude /= magnitude.sum(0, keepdim=True).clamp_min(1e-30)
    act_grad /= act_grad.sum(0, keepdim=True).clamp_min(1e-30)
    kernel = rbf_kernel(source_features, target_features, rbf_scale)
    nearest = F.normalize(source_features.float(), dim=1) @ F.normalize(
        target_features.float(), dim=1
    ).T
    nearest = nearest.argmax(0)
    local_predictions = {
        "activation_atlas_rbf": magnitude @ kernel / len(source_indices),
        "activation_atlas_scalar": magnitude.mean(1, keepdim=True).expand(
            -1, len(target_features)
        ),
        "activation_atlas_nearest": magnitude[:, nearest],
        "activation_x_gradient_atlas_rbf": act_grad @ kernel / len(source_indices),
        "activation_x_gradient_atlas_scalar": act_grad.mean(1, keepdim=True).expand(
            -1, len(target_features)
        ),
        "activation_x_gradient_atlas_nearest": act_grad[:, nearest],
    }
    predictions = {}
    for method, values in local_predictions.items():
        predictions[method] = torch.zeros(
            partition.n_groups, len(target_features), dtype=torch.float32
        )
        predictions[method][local_indices] = values
    return predictions, time.perf_counter() - started


def run_study(
    model,
    dataset,
    payload,
    *,
    rbf_scale=0.1,
    queries=100,
    fractions=(0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01),
    seed=23_041,
):
    partition = ParameterPartition(model, "swiglu")
    if partition.n_groups != len(payload["source_profiles"]):
        raise ValueError("atlas and model partition have different group counts")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    sensitivity = sample_predictions(
        payload["source_profiles"],
        payload["source_features"],
        payload["target_features"],
        rbf_scales=(rbf_scale,),
        seed=seed,
    )
    predictions = {
        "sensitivity_atlas_rbf": sensitivity[
            f"sample_dinov2_rbf_scale_{rbf_scale:g}"
        ],
        "sensitivity_atlas_nearest": sensitivity["sample_nearest"],
        "sensitivity_atlas_scalar": sensitivity["scalar_mass"],
        "class_source_onehot_reference": payload["class_source_profiles"].double(),
        "direct_gradient_oracle": payload["target_profiles"].double(),
    }
    activation_predictions, activation_seconds = activation_atlas_predictions(
        model,
        dataset,
        payload["representation_indices"].long(),
        payload["source_features"],
        payload["target_features"],
        partition,
        layout,
        scope,
        rbf_scale=rbf_scale,
    )
    predictions.update(activation_predictions)
    representation_indices = payload["representation_indices"].long()
    positions = torch.linspace(0, len(representation_indices) - 1, 64).round().long()
    means = calibrate_activation_means(
        model, dataset, representation_indices[positions], layout
    )
    predictions["weight_magnitude"] = (
        partition.weight_energy().detach().double().cpu() * sizes
    )[:, None].expand(-1, len(payload["target_indices"]))
    predictions["random"] = torch.rand(
        payload["target_profiles"].shape,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    target_indices = payload["target_indices"].long()
    query_columns = torch.linspace(
        0, len(target_indices) - 1, min(queries, len(target_indices))
    ).round().long().unique()
    records = []
    for column in query_columns.tolist():
        image, label = dataset[int(target_indices[column])]
        label = int(label)
        baseline = functional_value(model, image, label, "margin")
        prediction = int(model(image[None].to(partition.device)).argmax(1))
        target_magnitude, target_act_grad = activation_feature_scores(
            model, image, label, "margin", partition, layout
        )
        off_column = (column + len(target_indices) // 2) % len(target_indices)
        off_image, off_label = dataset[int(target_indices[off_column])]
        off_label = int(off_label)
        off_baseline = functional_value(model, off_image, off_label, "margin")
        off_prediction = int(model(off_image[None].to(partition.device)).argmax(1))
        score_grid = {
            **{method: values[:, column] for method, values in predictions.items()},
            "target_activation_magnitude": target_magnitude,
            "target_activation_x_gradient": target_act_grad,
        }
        for method, scores in score_grid.items():
            for fraction in fractions:
                selected, actual = select_parameter_budget(
                    scores.clamp_min(0), sizes, scope, fraction
                )
                with mean_ablate_mlp_activations(layout, selected, means):
                    intervened = functional_value(model, image, label, "margin")
                    off_intervened = functional_value(
                        model, off_image, off_label, "margin"
                    )
                degradation = baseline - intervened
                off_degradation = off_baseline - off_intervened
                records.append(
                    {
                        "method": method,
                        "query_column": column,
                        "class_index": label,
                        "target_correct": prediction == label,
                        "off_target_correct": off_prediction == off_label,
                        "requested_parameter_fraction": fraction,
                        "actual_scoped_parameter_fraction": actual,
                        "selected_groups": int(selected.sum()),
                        "degradation": degradation,
                        "off_target_degradation": off_degradation,
                        "selectivity": degradation - off_degradation,
                    }
                )
    return {
        "setting": "imagenet_sample_level_causal_curve",
        "model": {
            "name": getattr(model, "pretrained_cfg", {}).get("hf_hub_id", type(model).__name__),
            "architecture": type(model).__name__,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "protocol": {
            "functional": "correct-class logit margin",
            "representation": "full normalized frozen DINOv2-Small per-image CLS",
            "rbf_scale": rbf_scale,
            "target_queries": len(target_indices),
            "causal_queries": len(query_columns),
            "fractions": list(fractions),
            "intervention": "calibration-mean ablation at coupled MLP fc2 inputs",
            "activation_atlas_normalization": "per-source-example group-relative share within the MLP-feature scope",
        },
        "timing": {"activation_atlas_profile_seconds": activation_seconds},
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k"
    )
    parser.add_argument("--rbf-scale", type=float, default=0.1)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01],
    )
    args = parser.parse_args()
    seed_everything(23_041)
    import timm
    from torchvision.datasets import ImageFolder

    model = timm.create_model(args.model, pretrained=True).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    payload = torch.load(args.atlas, map_location="cpu", weights_only=True)
    result = run_study(
        model,
        dataset,
        payload,
        rbf_scale=args.rbf_scale,
        queries=args.queries,
        fractions=tuple(args.fractions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
