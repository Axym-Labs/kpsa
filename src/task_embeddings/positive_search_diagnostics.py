"""Fast exact-resolution diagnostics for refined sensitivity artifacts."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from .common import save_json
from .domain_optimizer import ParameterPartition, partition_hierarchy
from .qwen_sensitivity import cold_fold_predictions
from .representation_sensitivity import coarsen_profiles, profile_ranking_metrics


def analyze_resolutions(
    partition: ParameterPartition,
    source: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor,
    *,
    folds: int,
    seed: int,
) -> dict[str, object]:
    if source.shape != target.shape or source.shape[0] != partition.n_groups:
        raise ValueError("profiles must match the reconstructed fine partition")
    output: dict[str, object] = {
        "fine": {
            "groups": partition.n_groups,
            "source_onehot_reference": profile_ranking_metrics(source, target),
            "direct_gradient_oracle": profile_ranking_metrics(target, target),
        }
    }
    for resolution in ("bundle", "sublayer", "layer"):
        assignment, labels = partition_hierarchy(partition, resolution)
        coarse_source = coarsen_profiles(source, assignment, groups=len(labels))
        coarse_target = coarsen_profiles(target, assignment, groups=len(labels))
        predictions = cold_fold_predictions(
            coarse_source, features, folds=folds, seed=seed
        )
        metrics = {
            method: profile_ranking_metrics(predicted, coarse_target)
            for method, predicted in predictions.items()
        }
        metrics["source_onehot_reference"] = profile_ranking_metrics(
            coarse_source, coarse_target
        )
        metrics["direct_gradient_oracle"] = profile_ranking_metrics(
            coarse_target, coarse_target
        )
        output[resolution] = {
            "groups": len(labels),
            "labels": labels,
            "metrics": metrics,
            "source_mass_error": float((coarse_source.sum(0) - 1).abs().max()),
            "target_mass_error": float((coarse_target.sum(0) - 1).abs().max()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("vision", "qwen3"), required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    payload = torch.load(args.profiles, map_location="cpu", weights_only=False, mmap=True)
    if args.kind == "vision":
        import timm

        model = timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=False
        )
        partition = ParameterPartition(model, "swiglu", cache_sizes=False)
        studies = {}
        for functional in ("class_logit", "margin", "loss"):
            studies[functional] = {
                name: analyze_resolutions(
                    partition,
                    payload[functional]["source_profiles"].double(),
                    payload[functional]["target_profiles"].double(),
                    values.double(),
                    folds=args.folds,
                    seed=23_041,
                )
                for name, values in payload["feature_maps"].items()
            }
        result = {"kind": args.kind, "studies": studies}
    else:
        from transformers import AutoConfig, AutoModelForCausalLM

        configuration = AutoConfig.from_pretrained(
            "Qwen/Qwen3-1.7B", local_files_only=True
        )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(configuration)
        partition = ParameterPartition(model, "swiglu", cache_sizes=False)
        features = payload["features"]
        if args.features is not None:
            features = torch.load(
                args.features, map_location="cpu", weights_only=False, mmap=True
            )["features"]
        result = {
            "kind": args.kind,
            "studies": analyze_resolutions(
                partition,
                payload["source_profiles"].double(),
                payload["target_profiles"].double(),
                features.double(),
                folds=args.folds,
                seed=17_071,
            ),
        }
    del model, partition
    gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
