"""Held-out-class cold retrieval and causal evaluation for a vision atlas."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .parameter_groups import ParameterPartition
from .representation_sensitivity import profile_ranking_metrics
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


def heldout_predictions(payload, *, folds=5, rbf_scale=0.1, seed=23_041):
    source_profiles = payload["source_profiles"].double()
    source_features = payload["source_features"]
    target_features = payload["target_features"]
    classes = len(target_features)
    samples_per_class = len(source_features) // classes
    output = {}
    fold_ids = torch.arange(classes) % folds
    for fold in range(folds):
        target_columns = fold_ids.eq(fold).nonzero().flatten()
        source_classes = fold_ids.ne(fold)
        source_columns = (
            source_classes[:, None].expand(-1, samples_per_class).flatten()
        )
        source_columns = source_columns.nonzero().flatten()
        current = sample_predictions(
            source_profiles[:, source_columns],
            source_features[source_columns],
            target_features[target_columns],
            rbf_scales=(rbf_scale,),
            seed=seed + fold,
        )
        for method, values in current.items():
            if method not in output:
                output[method] = torch.empty(
                    len(source_profiles), classes, dtype=torch.float64
                )
            output[method][:, target_columns] = values
    return output, fold_ids


def run_study(
    model,
    dataset,
    payload,
    *,
    folds=5,
    rbf_scale=0.1,
    queries=100,
    fractions=(0.0002, 0.001),
    seed=23_041,
):
    partition = ParameterPartition(model, "swiglu")
    if partition.n_groups != len(payload["source_profiles"]):
        raise ValueError("atlas and model partition have different group counts")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    predictions, fold_ids = heldout_predictions(
        payload, folds=folds, rbf_scale=rbf_scale, seed=seed
    )
    target_profiles = payload["target_profiles"].double()
    predictions["direct_gradient_oracle"] = target_profiles
    cold_query = {
        method: profile_ranking_metrics(values, target_profiles)
        for method, values in predictions.items()
    }

    representation_indices = payload["representation_indices"].long()
    positions = torch.linspace(0, len(representation_indices) - 1, 64).round().long()
    means = calibrate_activation_means(
        model, dataset, representation_indices[positions], layout
    )
    predictions["weight_magnitude"] = (
        partition.weight_energy().detach().double().cpu() * sizes
    )[:, None].expand(-1, len(payload["target_indices"]))
    predictions["random"] = torch.rand(
        target_profiles.shape,
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
                        "heldout_fold": int(fold_ids[column]),
                        "target_correct": prediction == label,
                        "off_target_correct": off_prediction == off_label,
                        "requested_parameter_fraction": fraction,
                        "actual_scoped_parameter_fraction": actual,
                        "selected_groups": int(selected.sum()),
                        "baseline": baseline,
                        "intervened": intervened,
                        "degradation": degradation,
                        "off_target_degradation": off_degradation,
                        "selectivity": degradation - off_degradation,
                    }
                )
    return {
        "setting": "imagenet_heldout_class_sample_level_dinov2",
        "model": {
            "name": getattr(model, "pretrained_cfg", {}).get("hf_hub_id", type(model).__name__),
            "architecture": type(model).__name__,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "protocol": {
            "heldout_class_folds": folds,
            "rule": "class index modulo fold count; all source images from the query fold excluded",
            "source_classes_per_query": len(target_indices) * (folds - 1) // folds,
            "source_samples_per_class": len(representation_indices) // len(target_indices),
            "target_queries": len(target_indices),
            "causal_queries": len(query_columns),
            "rbf_scale": rbf_scale,
            "functional": "correct-class logit margin",
            "intervention": "calibration-mean ablation at coupled MLP fc2 inputs",
            "fractions": list(fractions),
        },
        "fidelity": {
            "source_mass_error": float(
                (payload["source_profiles"].sum(0).double() - 1).abs().max()
            ),
            "target_mass_error": float(
                (payload["target_profiles"].sum(0).double() - 1).abs().max()
            ),
        },
        "cold_query": cold_query,
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rbf-scale", type=float, default=0.1)
    parser.add_argument("--queries", type=int, default=100)
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
        folds=args.folds,
        rbf_scale=args.rbf_scale,
        queries=args.queries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
