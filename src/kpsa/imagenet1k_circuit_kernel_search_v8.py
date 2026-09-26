"""Frozen development/confirmation search for ImageNet-1k circuit retrieval."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .kernel_sensitivity import (
    PrototypeResponseMap,
    fit_prototype_response_map,
    median_squared_distance,
)
from .parameter_groups import ParameterPartition
from .parameter_influence_interpretability_v8 import (
    _indices_from_starts,
    _matrix_product,
    _profile_scoped_normalized,
)
from .refined_analysis import paired_t_summary
from .vision_atlas_size_scaling_v8 import balanced_pairing_permutation
from .vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from .vision_sample_refined import encode_indices


def _rbf(left: torch.Tensor, right: torch.Tensor, bandwidth_squared: float):
    cosine = F.normalize(left.float(), dim=1) @ F.normalize(right.float(), dim=1).T
    distance = (2 - 2 * cosine).clamp_min(0)
    return torch.exp(-distance / (2 * bandwidth_squared))


def _evaluate(scores, targets, correct, counts=(8, 37)):
    values = {}
    for count in counts:
        selected = torch.topk(scores, count, dim=0).indices.T
        oracle = torch.topk(targets, count, dim=0).indices.T
        overlap = []
        coverage = []
        for query in correct.nonzero().flatten().tolist():
            left = set(selected[query].tolist())
            right = set(oracle[query].tolist())
            overlap.append(len(left & right) / count)
            coverage.append(float(targets[selected[query], query].sum()))
        values[str(count)] = {
            "overlap": overlap,
            "coverage": coverage,
            "overlap_summary": paired_t_summary(overlap),
            "coverage_summary": paired_t_summary(coverage),
        }
    return values


def _selection_score(evaluation):
    return sum(
        evaluation[str(count)]["coverage_summary"]["mean"]
        for count in (8, 37)
    )


def _paired(left, right):
    return paired_t_summary([float(a) - float(b) for a, b in zip(left, right)])


def run_search(
    *,
    data_path: Path,
    output: Path,
    source_count: int,
    development_offset: int,
    confirmation_offset: int,
):
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    raw_dataset = ImageFolder(data_path / "validation")
    if len(raw_dataset) != 50_000:
        raise ValueError("expected the 50,000-image ImageNet validation set")
    class_starts = torch.arange(0, len(raw_dataset), 50)
    classes = len(class_starts)
    if source_count % classes:
        raise ValueError("source count must contain complete class blocks")
    per_class = source_count // classes
    source_indices = _indices_from_starts(class_starts, range(per_class))
    construction_indices = _indices_from_starts(class_starts, [43])
    development_indices = _indices_from_starts(
        class_starts, [development_offset]
    )
    confirmation_indices = _indices_from_starts(
        class_starts, [confirmation_offset]
    )

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
            (
                source_indices,
                construction_indices,
                development_indices,
                confirmation_indices,
            )
        ),
    )
    source_end = len(source_indices)
    construction_end = source_end + len(construction_indices)
    development_end = construction_end + len(development_indices)
    source_features = encoded[:source_end]
    construction_features = encoded[source_end:construction_end]
    development_features = encoded[construction_end:development_end]
    confirmation_features = encoded[development_end:]
    query_features = torch.cat((development_features, confirmation_features))
    construction_median = median_squared_distance(construction_features)
    source_median = median_squared_distance(source_features)
    del encoder, encoded
    gc.collect()
    torch.cuda.empty_cache()

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    ).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(data_path / "validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    source_weights, source_error, source_seconds, _ = _profile_scoped_normalized(
        model, dataset, source_indices, partition, scope
    )
    development_weights, development_error, development_seconds, development_correct = (
        _profile_scoped_normalized(
            model, dataset, development_indices, partition, scope
        )
    )
    confirmation_weights, confirmation_error, confirmation_seconds, confirmation_correct = (
        _profile_scoped_normalized(
            model, dataset, confirmation_indices, partition, scope
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    targets = torch.cat((development_weights, confirmation_weights), dim=1)
    permutation = balanced_pairing_permutation(source_count, classes, seed=91_027)
    candidates = {}

    for scale in (0.025, 0.05, 0.1, 0.25):
        kernel = _rbf(
            source_features,
            query_features,
            source_median * scale**2,
        )
        candidates[f"exact_scale_{scale:g}"] = _matrix_product(
            source_weights, kernel
        ) / source_count
        candidates[f"exact_scale_{scale:g}_permuted"] = _matrix_product(
            source_weights, kernel[permutation]
        ) / source_count
        del kernel

    for count in (800, 1000):
        base = fit_prototype_response_map(
            construction_features,
            count,
            bandwidth_squared=construction_median * 0.025**2,
            seed=91_027 + count,
        )
        for scale in (0.025, 0.05, 0.1):
            feature_map = PrototypeResponseMap(
                base.prototypes, construction_median * scale**2
            )
            source_responses = feature_map.transform(source_features).float()
            query_responses = feature_map.transform(query_features).float()
            atlas = _matrix_product(source_weights, source_responses) / source_count
            shuffled = (
                _matrix_product(source_weights, source_responses[permutation])
                / source_count
            )
            candidates[f"prototype_{count}_scale_{scale:g}"] = _matrix_product(
                atlas, query_responses.T
            )
            candidates[
                f"prototype_{count}_scale_{scale:g}_permuted"
            ] = _matrix_product(shuffled, query_responses.T)
            del source_responses, query_responses, atlas, shuffled

    cosine = source_features @ query_features.T
    candidates["nearest"] = source_weights[:, cosine.argmax(0)]
    class_scores = source_weights.reshape(
        len(source_weights), per_class, classes
    ).mean(1).repeat(1, 2)
    candidates["class_onehot"] = class_scores
    candidates["scalar"] = source_weights.mean(1, keepdim=True).expand(
        -1, 2 * classes
    )
    candidates["direct"] = targets

    # This is an additive positive-definite kernel on the direct-sum space:
    # alpha * k_DINO + (1-alpha) * k_class.  The 1/classes factor puts the
    # class conditional mean on the same empirical-kernel expectation scale.
    class_kernel_scores = class_scores / classes
    for semantic in ("exact_scale_0.1", "prototype_1000_scale_0.1"):
        for alpha in (0.1, 0.25, 0.5, 0.75, 0.9):
            name = f"class_plus_{semantic}_alpha_{alpha:g}"
            candidates[name] = (
                (1 - alpha) * class_kernel_scores
                + alpha * candidates[semantic]
            )
            candidates[name + "_permuted"] = (
                (1 - alpha) * class_kernel_scores
                + alpha * candidates[semantic + "_permuted"]
            )

    development = {}
    confirmation = {}
    for method, scores in candidates.items():
        development[method] = _evaluate(
            scores[:, :classes],
            development_weights,
            development_correct,
        )
        confirmation[method] = _evaluate(
            scores[:, classes:],
            confirmation_weights,
            confirmation_correct,
        )

    eligible = [
        method
        for method in candidates
        if method.startswith(("prototype_", "exact_", "class_plus_"))
        and not method.endswith("_permuted")
    ]
    best = max(eligible, key=lambda method: _selection_score(development[method]))
    comparisons = {}
    for count in (8, 37):
        key = str(count)
        comparisons[key] = {}
        for baseline in (
            best + "_permuted",
            "class_onehot",
            "nearest",
            "scalar",
        ):
            comparisons[key][f"{best}_vs_{baseline}"] = {
                metric: _paired(
                    confirmation[best][key][metric],
                    confirmation[baseline][key][metric],
                )
                for metric in ("overlap", "coverage")
            }

    result = {
        "setting": "vitb16_imagenet1k_circuit_kernel_search",
        "protocol": {
            "source_examples": source_count,
            "source_examples_per_class": per_class,
            "classes": classes,
            "construction_examples": len(construction_indices),
            "development_offset": development_offset,
            "confirmation_offset": confirmation_offset,
            "group_counts": [8, 37],
            "candidate_selection": (
                "largest summed direct-gradient coverage on the development offset"
            ),
        },
        "selected_method": best,
        "development": development,
        "confirmation": confirmation,
        "comparisons": comparisons,
        "fidelity": {
            "source_max_partition_relative_error": source_error,
            "development_max_partition_relative_error": development_error,
            "confirmation_max_partition_relative_error": confirmation_error,
        },
        "timing": {
            "source_profile_seconds": source_seconds,
            "development_profile_seconds": development_seconds,
            "confirmation_profile_seconds": confirmation_seconds,
        },
        "geometry": {
            "source_median_squared_distance": source_median,
            "construction_median_squared_distance": construction_median,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=8000)
    parser.add_argument("--development-offset", type=int, default=48)
    parser.add_argument("--confirmation-offset", type=int, default=49)
    args = parser.parse_args()
    seed_everything(577_215)
    run_search(
        data_path=args.data,
        output=args.output,
        source_count=args.source_count,
        development_offset=args.development_offset,
        confirmation_offset=args.confirmation_offset,
    )


if __name__ == "__main__":
    main()
