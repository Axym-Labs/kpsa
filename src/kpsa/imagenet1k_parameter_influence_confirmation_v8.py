"""Exact local-parameter confirmation for the selected ImageNet-1k circuit atlas."""

from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .parameter_groups import ParameterPartition
from .parameter_influence_circuits_v8 import (
    additive_group_noise,
    margin_values,
    scoped_parameter_rms,
)
from .parameter_influence_interpretability_v8 import (
    _indices_from_starts,
    _matrix_product,
    _profile_scoped_normalized,
)
from .refined_analysis import paired_t_summary
from .vision_atlas_size_scaling_v8 import balanced_pairing_permutation
from .vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from .vision_sample_refined import encode_indices


def _mean_by_query(records, method, group_count, metric):
    values = defaultdict(list)
    for row in records:
        if (
            row["target_correct"]
            and row["method"] == method
            and row["selected_groups"] == group_count
        ):
            values[int(row["query_column"])].append(float(row[metric]))
    return {query: sum(current) / len(current) for query, current in values.items()}


def _paired(records, method, baseline, group_count, metric):
    left = _mean_by_query(records, method, group_count, metric)
    right = _mean_by_query(records, baseline, group_count, metric)
    return paired_t_summary(
        [left[query] - right[query] for query in left.keys() & right.keys()]
    )


def _gap_recovery(method, scalar, oracle):
    denominator = oracle["mean"] - scalar["mean"]
    return (
        (method["mean"] - scalar["mean"]) / denominator
        if denominator > 0
        else None
    )


def run_confirmation(
    *,
    data_path: Path,
    output: Path,
    source_count: int,
    queries: int,
    directions: int,
    noise_scale: float,
    target_offset: int,
):
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    raw_dataset = ImageFolder(data_path / "validation")
    class_starts = torch.arange(0, len(raw_dataset), 50)
    classes = len(class_starts)
    if classes != 1000 or source_count % classes:
        raise ValueError("expected balanced ImageNet-1k sources")
    per_class = source_count // classes
    source_indices = _indices_from_starts(class_starts, range(per_class))
    construction_indices = _indices_from_starts(class_starts, [43])
    target_indices = _indices_from_starts(class_starts, [target_offset])

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
    construction_median = median_squared_distance(construction_features)
    feature_map = fit_prototype_response_map(
        construction_features,
        1000,
        bandwidth_squared=construction_median * 0.1**2,
        seed=91_027 + 1000,
    )
    source_responses = feature_map.transform(source_features).float()
    target_responses = feature_map.transform(target_features).float()
    del encoder, encoded, construction_features
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
    scope_indices = scope.nonzero().flatten()
    source_weights, source_error, source_seconds, _ = _profile_scoped_normalized(
        model, dataset, source_indices, partition, scope
    )
    target_weights, target_error, target_seconds, _ = _profile_scoped_normalized(
        model, dataset, target_indices, partition, scope
    )

    permutation = balanced_pairing_permutation(source_count, classes, seed=91_027)
    atlas = _matrix_product(source_weights, source_responses) / source_count
    shuffled_atlas = (
        _matrix_product(source_weights, source_responses[permutation]) / source_count
    )
    prototype = _matrix_product(atlas, target_responses.T)
    prototype_permuted = _matrix_product(shuffled_atlas, target_responses.T)
    class_mean = source_weights.reshape(
        len(source_weights), per_class, classes
    ).mean(1)
    class_kernel = class_mean / classes
    alpha = 0.25
    combined = (1 - alpha) * class_kernel + alpha * prototype
    combined_permuted = (
        (1 - alpha) * class_kernel + alpha * prototype_permuted
    )
    cosine = source_features @ target_features.T
    methods = {
        "combined_class_prototype": combined,
        "combined_permuted": combined_permuted,
        "prototype": prototype,
        "prototype_permuted": prototype_permuted,
        "class_onehot": class_mean,
        "nearest": source_weights[:, cosine.argmax(0)],
        "scalar": source_weights.mean(1, keepdim=True).expand(-1, classes),
        "direct": target_weights,
    }
    parameter_rms = scoped_parameter_rms(partition, scope)
    noise_std = parameter_rms * noise_scale
    query_columns = torch.linspace(0, classes - 1, queries).round().long().unique()
    records = []
    started = time.perf_counter()
    for position, column in enumerate(query_columns.tolist()):
        target_index = int(target_indices[column])
        target_image, target_label = dataset[target_index]
        same_image, same_label = dataset[target_index + 1]
        off_column = (column + classes // 2) % classes
        off_image, off_label = dataset[int(target_indices[off_column])]
        target_label, same_label, off_label = map(
            int, (target_label, same_label, off_label)
        )
        images = torch.stack((target_image, same_image, off_image))
        labels = torch.tensor((target_label, same_label, off_label))
        intact = margin_values(model, images[:1], labels[:1]).cpu()
        target_correct = bool(intact[0] > 0)
        for method, scores in methods.items():
            for group_count in (8, 37):
                selected_local = torch.topk(
                    scores[:, column], group_count
                ).indices
                selected = torch.zeros(partition.n_groups, dtype=torch.bool)
                selected[scope_indices[selected_local]] = True
                coverage = float(target_weights[selected_local, column].sum())
                for direction in range(directions):
                    seed = 314_159 + position * 100_003 + direction * 1_009
                    with additive_group_noise(
                        partition, selected, noise_std, seed=seed, sign=1
                    ):
                        plus = margin_values(model, images, labels).cpu()
                    with additive_group_noise(
                        partition, selected, noise_std, seed=seed, sign=-1
                    ):
                        minus = margin_values(model, images, labels).cpu()
                    squared = ((plus - minus) / (2 * noise_std)).square()
                    records.append(
                        {
                            "query_column": column,
                            "target_correct": target_correct,
                            "method": method,
                            "selected_groups": group_count,
                            "direction": direction,
                            "gradient_energy_coverage": coverage,
                            "target_squared_susceptibility": float(squared[0]),
                            "same_class_squared_susceptibility": float(squared[1]),
                            "off_class_squared_susceptibility": float(squared[2]),
                            "off_class_selectivity": float(squared[0] - squared[2]),
                            "within_class_selectivity": float(
                                squared[0] - squared[1]
                            ),
                        }
                    )
        print(
            f"ImageNet-1k parameter influence {position + 1}/{len(query_columns)}",
            flush=True,
        )

    summaries = {}
    comparisons = {}
    for group_count in (8, 37):
        key = str(group_count)
        summaries[key] = {}
        for method in methods:
            values = _mean_by_query(
                records,
                method,
                group_count,
                "target_squared_susceptibility",
            )
            summaries[key][method] = paired_t_summary(list(values.values()))
        comparisons[key] = {}
        for method in ("combined_class_prototype", "prototype"):
            pairing_control = (
                "combined_permuted"
                if method == "combined_class_prototype"
                else "prototype_permuted"
            )
            comparisons[key][method] = {
                f"vs_{baseline}": _paired(
                    records,
                    method,
                    baseline,
                    group_count,
                    "target_squared_susceptibility",
                )
                for baseline in (
                    pairing_control,
                    "class_onehot",
                    "nearest",
                    "scalar",
                )
            }
            comparisons[key][method]["off_class_selectivity"] = paired_t_summary(
                list(
                    _mean_by_query(
                        records,
                        method,
                        group_count,
                        "off_class_selectivity",
                    ).values()
                )
            )
            comparisons[key][method]["within_class_selectivity"] = paired_t_summary(
                list(
                    _mean_by_query(
                        records,
                        method,
                        group_count,
                        "within_class_selectivity",
                    ).values()
                )
            )
            comparisons[key][method]["direct_gap_recovered"] = _gap_recovery(
                summaries[key][method],
                summaries[key]["scalar"],
                summaries[key]["direct"],
            )

    result = {
        "setting": "vitb16_imagenet1k_parameter_influence_confirmation",
        "protocol": {
            "classes": classes,
            "source_examples": source_count,
            "source_examples_per_class": per_class,
            "construction_examples": len(construction_indices),
            "target_offset": target_offset,
            "causal_queries": len(query_columns),
            "directions_per_query_method_budget": directions,
            "selected_groups": [8, 37],
            "combined_kernel": (
                "0.75 * class-delta kernel + 0.25 * DINO prototype kernel"
            ),
            "prototype_count": 1000,
            "prototype_scale": 0.1,
            "noise_scale_relative_to_scoped_parameter_rms": noise_scale,
            "coordinate_noise_standard_deviation": noise_std,
            "intervention": (
                "antithetic iid Rademacher additive noise; exact forward "
                "central difference"
            ),
        },
        "fidelity": {
            "source_max_partition_relative_error": source_error,
            "target_max_partition_relative_error": target_error,
        },
        "timing": {
            "source_profile_seconds": source_seconds,
            "target_profile_seconds": target_seconds,
            "causal_seconds": time.perf_counter() - started,
        },
        "mean_susceptibility": summaries,
        "comparisons": comparisons,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=8000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--target-offset", type=int, default=46)
    args = parser.parse_args()
    seed_everything(314_159)
    run_confirmation(
        data_path=args.data,
        output=args.output,
        source_count=args.source_count,
        queries=args.queries,
        directions=args.directions,
        noise_scale=args.noise_scale,
        target_offset=args.target_offset,
    )


if __name__ == "__main__":
    main()
