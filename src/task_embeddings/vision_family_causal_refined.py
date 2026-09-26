"""Causal validation for cold WordNet-family sensitivity directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .vision_causal_refined import (
    calibrate_activation_means,
    functional_value,
    mean_ablate_mlp_activations,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)
from .vision_family_refined import pair_holdout_direction_predictions


def aggregate_families(values: torch.Tensor, members: list[list[int]]) -> torch.Tensor:
    return torch.stack(
        [values[:, torch.tensor(local)].mean(1) for local in members], dim=1
    )


def aggregate_family_features(values: torch.Tensor, members: list[list[int]]):
    return torch.stack(
        [torch.nn.functional.normalize(values[torch.tensor(local)].mean(0), dim=0) for local in members]
    )


def choose_correct_columns(model, dataset, target_indices, members, count=2):
    chosen = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for local in members:
            correct = []
            for column in local:
                image, label = dataset[int(target_indices[column])]
                prediction = int(model(image[None].to(device)).argmax(1))
                if prediction == int(label):
                    correct.append(column)
                if len(correct) == count:
                    break
            chosen.append(correct or local[:1])
    return chosen


def run_family_causal(
    model,
    dataset,
    payload,
    family_payload,
    *,
    queries=24,
    examples_per_family=2,
    fractions=(0.0002, 0.001),
    calibration_images=64,
    seed=23_041,
):
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    members = family_payload["families"]["member_columns"]
    pairs = torch.tensor(
        family_payload["families"]["direction_pairs"][:queries], dtype=torch.long
    )
    source = aggregate_families(payload["margin"]["source_profiles"].double(), members)
    target = aggregate_families(payload["margin"]["target_profiles"].double(), members)
    features = aggregate_family_features(
        payload["feature_maps"]["frozen_input_encoder"].double(), members
    )
    predictions = pair_holdout_direction_predictions(source, features, pairs, seed=seed)
    predictions["direct_gradient_oracle"] = torch.stack(
        [target[:, int(left)] - target[:, int(right)] for left, right in pairs], 1
    )
    scalar = payload["margin"]["source_profiles"].double().mean(1)
    predictions["scalar_mass"] = scalar[:, None].expand(-1, len(pairs))
    weight = partition.weight_energy().detach().double().cpu() * sizes
    predictions["weight_magnitude"] = weight[:, None].expand(-1, len(pairs))
    predictions["random"] = torch.rand(
        target.shape[0], len(pairs), generator=torch.Generator().manual_seed(seed), dtype=torch.float64
    )

    source_indices = payload["source_indices"].long()
    target_indices = payload["target_indices"].long()
    reference_indices = payload.get("representation_indices")
    if reference_indices is None:
        reference_indices = torch.cat(
            [torch.arange(int(index) - 4, int(index)) for index in source_indices]
        )
    positions = torch.linspace(
        0, len(reference_indices) - 1, min(calibration_images, len(reference_indices))
    ).round().long().unique()
    means = calibrate_activation_means(
        model, dataset, reference_indices[positions], layout
    )
    selected_columns = choose_correct_columns(
        model, dataset, target_indices, members, count=examples_per_family
    )
    records = []
    for pair_index, (left, right) in enumerate(pairs.tolist()):
        for orientation, (target_family, off_family, sign) in enumerate(
            ((left, right, 1.0), (right, left, -1.0))
        ):
            target_examples = selected_columns[target_family]
            off_examples = selected_columns[off_family]
            target_baselines = [
                functional_value(
                    model,
                    dataset[int(target_indices[column])][0],
                    int(dataset[int(target_indices[column])][1]),
                    "margin",
                )
                for column in target_examples
            ]
            off_baselines = [
                functional_value(
                    model,
                    dataset[int(target_indices[column])][0],
                    int(dataset[int(target_indices[column])][1]),
                    "margin",
                )
                for column in off_examples
            ]
            for method, values in predictions.items():
                scores = values[:, pair_index] * sign
                if method in {"scalar_mass", "weight_magnitude", "random"}:
                    scores = values[:, pair_index]
                for fraction in fractions:
                    selected, actual = select_parameter_budget(
                        scores.clamp_min(0), sizes, scope, fraction
                    )
                    with mean_ablate_mlp_activations(layout, selected, means):
                        target_values = [
                            functional_value(
                                model,
                                dataset[int(target_indices[column])][0],
                                int(dataset[int(target_indices[column])][1]),
                                "margin",
                            )
                            for column in target_examples
                        ]
                        off_values = [
                            functional_value(
                                model,
                                dataset[int(target_indices[column])][0],
                                int(dataset[int(target_indices[column])][1]),
                                "margin",
                            )
                            for column in off_examples
                        ]
                    target_effect = sum(
                        baseline - value
                        for baseline, value in zip(target_baselines, target_values)
                    ) / len(target_values)
                    off_effect = sum(
                        baseline - value
                        for baseline, value in zip(off_baselines, off_values)
                    ) / len(off_values)
                    records.append(
                        {
                            "pair_index": pair_index,
                            "orientation": orientation,
                            "target_family": target_family,
                            "off_family": off_family,
                            "method": method,
                            "requested_parameter_fraction": fraction,
                            "actual_scoped_parameter_fraction": actual,
                            "selected_groups": int(selected.sum()),
                            "target_examples": target_examples,
                            "off_examples": off_examples,
                            "target_degradation": target_effect,
                            "off_target_degradation": off_effect,
                            "selectivity": target_effect - off_effect,
                        }
                    )
    return {
        "protocol": {
            "functional": "correct-class logit margin",
            "query": "pair-holdout WordNet-family direction in full DINOv2-Small space",
            "intervention": "disjoint-calibration mean ablation at coupled MLP fc2 inputs",
            "pairs": len(pairs),
            "orientations_per_pair": 2,
            "examples_per_family_maximum": examples_per_family,
            "fractions": list(fractions),
            "calibration_images": len(positions),
        },
        "resources": {
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
            "parameters": partition.n_parameters,
            "scoped_parameters": int(sizes[scope].sum()),
        },
        "family_names": family_payload["families"]["family_names"],
        "pairs": pairs.tolist(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--family-queries", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=24)
    args = parser.parse_args()
    seed_everything(23_041)
    import timm
    from torchvision.datasets import ImageFolder

    model_name = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    model = timm.create_model(model_name, pretrained=True).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    payload = torch.load(args.profiles, map_location="cpu", weights_only=True)
    family_payload = json.loads(args.family_queries.read_text())
    result = run_family_causal(
        model, dataset, payload, family_payload, queries=args.queries
    )
    result["model"] = model_name
    result["profiles"] = str(args.profiles)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
