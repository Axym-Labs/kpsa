"""Cross-modal evidence for semantic KPSA sensitivity circuits.

The study holds the KPSA query space fixed and asks whether retrieved groups
form causally useful, input-specific circuits beyond a final-layer/readout
effect.  Vision and forecasting use the same 0.1% and 0.5% group budgets,
full/non-late/late scopes, layer-matched constant controls, and exact
antithetic parameter perturbations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from .common import save_json, seed_everything
from .kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .paper_style import (
    NEAREST,
    SEMANTIC,
    SEMANTIC_LIGHT,
    SHUFFLE,
    finish_axis,
    outside_legend,
    panel_title,
    save_plot,
)
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
from .timeseries_parameter_influence_v8 import (
    DEFAULT_MODEL,
    ChronosSensitivity,
    _collect,
    _load_values,
)
from .vision_atlas_size_scaling_v8 import balanced_pairing_permutation
from .vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from .vision_sample_refined import encode_indices

DEFAULT_BUDGETS = (0.001, 0.005)


def _restricted_topk(scores: torch.Tensor, eligible: torch.Tensor, count: int):
    """Return global indices for the largest eligible scores."""
    eligible = eligible.long().cpu()
    if count > len(eligible):
        raise ValueError("selection budget exceeds eligible groups")
    local = torch.topk(scores[eligible], count).indices
    return eligible[local].sort().values


def _layer_matched_topk(
    scores: torch.Tensor,
    reference: torch.Tensor,
    *,
    units_per_layer: int,
) -> torch.Tensor:
    """Select by ``scores`` while exactly matching reference layer counts."""
    selected = []
    layers = torch.div(reference, units_per_layer, rounding_mode="floor")
    for layer in torch.unique(layers, sorted=True).tolist():
        count = int((layers == layer).sum())
        start = layer * units_per_layer
        eligible = torch.arange(start, start + units_per_layer)
        selected.append(_restricted_topk(scores, eligible, count))
    return torch.cat(selected).sort().values


def _selection_grid(
    semantic: torch.Tensor,
    constant: torch.Tensor,
    direct: torch.Tensor,
    *,
    count: int,
    units_per_layer: int,
    late_layers: int,
) -> dict[str, torch.Tensor]:
    groups = len(semantic)
    layer_count = groups // units_per_layer
    late_start = (layer_count - late_layers) * units_per_layer
    full = _restricted_topk(semantic, torch.arange(groups), count)
    nonlate = _restricted_topk(semantic, torch.arange(late_start), count)
    late = _restricted_topk(semantic, torch.arange(late_start, groups), count)
    return {
        "semantic_full": full,
        "semantic_nonlate": nonlate,
        "semantic_late": late,
        "constant_full_matched": _layer_matched_topk(
            constant, full, units_per_layer=units_per_layer
        ),
        "constant_nonlate_matched": _layer_matched_topk(
            constant, nonlate, units_per_layer=units_per_layer
        ),
        "constant_late_matched": _layer_matched_topk(
            constant, late, units_per_layer=units_per_layer
        ),
        "direct_full": _restricted_topk(direct, torch.arange(groups), count),
    }


def _selection_record(
    selected: torch.Tensor,
    truth: torch.Tensor,
    *,
    units_per_layer: int,
) -> dict:
    layers = torch.div(selected, units_per_layer, rounding_mode="floor")
    layer_count = len(truth) // units_per_layer
    counts = torch.bincount(layers, minlength=layer_count)
    return {
        "selected_groups": len(selected),
        "represented_layers": int((counts > 0).sum()),
        "layer_counts": counts.tolist(),
        "direct_gradient_energy": float(truth[selected].sum()),
    }


def _mean_by_query(records, method, budget, metric, *, valid=None):
    grouped = defaultdict(list)
    for row in records:
        if row["method"] != method or row["budget"] != budget:
            continue
        if valid is not None and not row.get(valid, False):
            continue
        grouped[int(row["query"])].append(float(row[metric]))
    return {query: float(np.mean(values)) for query, values in grouped.items()}


def _paired_records(records, left, right, budget, metric, *, valid=None):
    left_values = _mean_by_query(records, left, budget, metric, valid=valid)
    right_values = _mean_by_query(records, right, budget, metric, valid=valid)
    keys = sorted(left_values.keys() & right_values.keys())
    return paired_t_summary([left_values[key] - right_values[key] for key in keys])


def _summarize_causal(records, budgets, *, valid=None):
    methods = sorted({row["method"] for row in records})
    metrics = sorted(
        key
        for key in records[0]
        if key.endswith("susceptibility") or key == "input_specificity"
    )
    summaries, comparisons = {}, {}
    for budget in budgets:
        key = f"{budget:g}"
        summaries[key] = {}
        for method in methods:
            summaries[key][method] = {
                metric: paired_t_summary(
                    list(
                        _mean_by_query(
                            records, method, budget, metric, valid=valid
                        ).values()
                    )
                )
                for metric in metrics
            }
        comparisons[key] = {}
        for scope in ("full", "nonlate", "late"):
            semantic = f"semantic_{scope}"
            constant = f"constant_{scope}_matched"
            comparisons[key][semantic] = {
                metric: _paired_records(
                    records,
                    semantic,
                    constant,
                    budget,
                    metric,
                    valid=valid,
                )
                for metric in metrics
            }
    return summaries, comparisons


def _jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    left_set, right_set = set(left.tolist()), set(right.tolist())
    return len(left_set & right_set) / len(left_set | right_set)


def _overlap_by_similarity(
    features: torch.Tensor,
    semantic_scores: torch.Tensor,
    shuffled_scores: torch.Tensor,
    *,
    counts: dict[float, int],
    seed: int,
    max_pairs: int = 20_000,
) -> dict:
    """Bin pairwise circuit Jaccard by representation cosine similarity."""
    n = len(features)
    all_pairs = n * (n - 1) // 2
    if all_pairs <= max_pairs:
        first, second = torch.triu_indices(n, n, offset=1)
    else:
        generator = torch.Generator().manual_seed(seed)
        first = torch.randint(n, (max_pairs * 2,), generator=generator)
        second = torch.randint(n, (max_pairs * 2,), generator=generator)
        keep = first != second
        first, second = first[keep][:max_pairs], second[keep][:max_pairs]
    similarities = (features[first] * features[second]).sum(1).numpy()
    order = np.argsort(similarities)
    if len(order) >= 500:
        quantile_edges = (0.0, 0.5, 0.8, 0.95, 0.99, 1.0)
        bins = [
            order[round(low * len(order)) : round(high * len(order))]
            for low, high in pairwise(quantile_edges)
        ]
    else:
        bins = np.array_split(order, 5)
        quantile_edges = tuple(np.linspace(0, 1, 6))
    result = {
        "pairs": len(first),
        "similarity_quantile_edges": list(quantile_edges),
        "budgets": {},
    }
    for budget, count in counts.items():
        selections = {
            "semantic": torch.topk(semantic_scores, count, dim=0).indices.T,
            "shuffled_pairing": torch.topk(shuffled_scores, count, dim=0).indices.T,
        }
        budget_result = {}
        for method, selected in selections.items():
            jaccard = np.asarray(
                [
                    _jaccard(selected[int(a)], selected[int(b)])
                    for a, b in zip(first, second)
                ]
            )
            rows = []
            for bin_index, indices in enumerate(bins):
                values = jaccard[indices]
                summary = paired_t_summary(values.tolist())
                rows.append(
                    {
                        "similarity_quantile": [
                            float(quantile_edges[bin_index]),
                            float(quantile_edges[bin_index + 1]),
                        ],
                        "similarity_mean": float(similarities[indices].mean()),
                        "similarity_low": float(similarities[indices].min()),
                        "similarity_high": float(similarities[indices].max()),
                        "jaccard": summary,
                    }
                )
            budget_result[method] = rows
        result["budgets"][f"{budget:g}"] = budget_result
    return result


def run_forecast(args) -> None:
    _frame, values = _load_values(args)
    experiment = ChronosSensitivity(
        args.model,
        values,
        device=args.device,
        context_length=args.context_length,
        horizon=args.horizon,
    )
    atlas = torch.load(args.atlas_cache, map_location="cpu", weights_only=True)
    source_features = atlas["source_features"].double()
    source_profiles = atlas["source_profiles"].double()
    median_distance = float(atlas["median_squared_distance"])
    permutation = atlas["permutation"]
    query_ends = (
        np.linspace(args.query_start, args.query_stop, args.queries).round().astype(int)
    )
    query_features, query_profiles = _collect(
        experiment, query_ends, label="forecast circuit queries"
    )
    denominator = 2 * median_distance * args.rbf_scale**2
    kernel = torch.exp(
        -torch.cdist(source_features, query_features).square() / denominator
    )
    shuffled_kernel = torch.exp(
        -torch.cdist(source_features[permutation], query_features).square()
        / denominator
    )
    semantic = source_profiles @ kernel / len(source_features)
    shuffled = source_profiles @ shuffled_kernel / len(source_features)
    constant = source_profiles.mean(1)
    counts = {
        budget: max(1, math.ceil(budget * experiment.groups)) for budget in args.budgets
    }
    similarity = query_features @ query_features.T
    near_similarity = similarity.clone()
    near_similarity.fill_diagonal_(-torch.inf)
    far_similarity = similarity.clone()
    far_similarity.fill_diagonal_(torch.inf)
    near = near_similarity.argmax(1)
    far = far_similarity.argmin(1)
    structural = _overlap_by_similarity(
        query_features,
        semantic,
        shuffled,
        counts=counts,
        seed=args.seed,
    )
    if args.structure_only:
        save_json(
            args.output,
            {
                "setting": "chronos_semantic_circuit_structure",
                "modality": "forecasting",
                "protocol": {
                    "source_examples": len(source_features),
                    "queries": len(query_ends),
                    "budgets": list(args.budgets),
                    "selected_groups": {f"{k:g}": v for k, v in counts.items()},
                },
                "structure": structural,
            },
        )
        return
    parameter_rms = experiment.parameter_rms()
    noise_std = parameter_rms * args.noise_scale
    selections, records = [], []
    started = time.time()
    late_layers = max(1, len(experiment.ff_pairs) // 4)
    for query, end in enumerate(query_ends):
        contexts = torch.stack(
            (
                experiment.window(int(end)),
                experiment.window(int(query_ends[int(near[query])])),
                experiment.window(int(query_ends[int(far[query])])),
            )
        )
        for budget, count in counts.items():
            grid = _selection_grid(
                semantic[:, query],
                constant,
                query_profiles[:, query].double(),
                count=count,
                units_per_layer=experiment.units_per_layer,
                late_layers=late_layers,
            )
            for method, selected in grid.items():
                selections.append(
                    {
                        "query": query,
                        "end": int(end),
                        "budget": budget,
                        "method": method,
                        **_selection_record(
                            selected,
                            query_profiles[:, query],
                            units_per_layer=experiment.units_per_layer,
                        ),
                    }
                )
                for direction in range(args.directions):
                    intervention_seed = args.seed + query * 100_003 + direction * 1_009
                    with experiment.additive_noise(
                        selected, noise_std, seed=intervention_seed, sign=1
                    ):
                        plus = experiment.functionals(contexts)
                    with experiment.additive_noise(
                        selected, noise_std, seed=intervention_seed, sign=-1
                    ):
                        minus = experiment.functionals(contexts)
                    squared = ((plus - minus) / (2 * noise_std)).square()
                    records.append(
                        {
                            "query": query,
                            "end": int(end),
                            "budget": budget,
                            "method": method,
                            "direction": direction,
                            "target_susceptibility": float(squared[0]),
                            "near_susceptibility": float(squared[1]),
                            "far_susceptibility": float(squared[2]),
                            "input_specificity": float(squared[0] - squared[2]),
                        }
                    )
        print(
            f"forecast semantic circuits {query + 1}/{len(query_ends)} "
            f"elapsed {time.time() - started:.1f}s",
            flush=True,
        )
    summaries, comparisons = _summarize_causal(records, args.budgets)
    save_json(
        args.output,
        {
            "setting": "chronos_semantic_parameter_influence_circuits",
            "modality": "forecasting",
            "model": args.model,
            "dataset": args.dataset_name,
            "protocol": {
                "source_examples": len(source_features),
                "queries": len(query_ends),
                "query_end_range": [args.query_start, args.query_stop],
                "groups": experiment.groups,
                "layers": len(experiment.ff_pairs),
                "units_per_layer": experiment.units_per_layer,
                "budgets": list(args.budgets),
                "selected_groups": {f"{k:g}": v for k, v in counts.items()},
                "nonlate_layers": list(range(len(experiment.ff_pairs) - late_layers)),
                "late_layers": list(
                    range(
                        len(experiment.ff_pairs) - late_layers, len(experiment.ff_pairs)
                    )
                ),
                "representation": "final Chronos encoder regression-token state",
                "rbf_scale": args.rbf_scale,
                "directions": args.directions,
                "noise_scale_relative_to_ff_parameter_rms": args.noise_scale,
                "coordinate_noise_standard_deviation": noise_std,
                "near_context": "highest cosine among held-out query representations",
                "far_context": "lowest cosine among held-out query representations",
            },
            "structure": structural,
            "selections": selections,
            "causal_summaries": summaries,
            "causal_comparisons": comparisons,
            "records": records,
            "elapsed_seconds": time.time() - started,
        },
    )


def run_vision(args) -> None:
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    raw_dataset = ImageFolder(args.data / "validation")
    if len(raw_dataset) != 50_000:
        raise ValueError("expected ImageNet-1k validation with 50,000 images")
    class_starts = torch.arange(0, len(raw_dataset), 50)
    classes = len(class_starts)
    if args.source_count % classes:
        raise ValueError("source count must be divisible by 1,000 classes")
    per_class = args.source_count // classes
    source_indices = _indices_from_starts(class_starts, range(per_class))
    construction_indices = _indices_from_starts(class_starts, [43])
    query_offsets = (46, 47, 48)
    query_indices = _indices_from_starts(class_starts, query_offsets)
    target_indices = query_indices[:classes]
    query_columns = torch.linspace(0, classes - 1, args.queries).round().long().unique()

    encoder = (
        AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True)
        .cuda()
        .eval()
    )
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    )
    encoded = encode_indices(
        encoder,
        processor,
        raw_dataset,
        torch.cat((source_indices, construction_indices, query_indices)),
    )
    source_end = len(source_indices)
    construction_end = source_end + len(construction_indices)
    source_features = encoded[:source_end]
    construction_features = encoded[source_end:construction_end]
    query_features = encoded[construction_end:]
    construction_median = median_squared_distance(construction_features)
    feature_map = fit_prototype_response_map(
        construction_features,
        1000,
        bandwidth_squared=construction_median * 0.1**2,
        seed=91_027 + 1000,
    )
    source_responses = feature_map.transform(source_features).float()
    query_responses = feature_map.transform(query_features).float()
    del encoder, encoded, construction_features
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
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    scope_indices = scope.nonzero().flatten()
    units_per_layer = int(layout[0]["count"])
    groups = len(scope_indices)
    source_weights, source_error, source_seconds, _ = _profile_scoped_normalized(
        model, dataset, source_indices, partition, scope
    )
    target_weights, target_error, target_seconds, correct = _profile_scoped_normalized(
        model, dataset, target_indices[query_columns], partition, scope
    )
    permutation = balanced_pairing_permutation(args.source_count, classes, seed=91_027)
    atlas = _matrix_product(source_weights, source_responses) / args.source_count
    shuffled_atlas = (
        _matrix_product(source_weights, source_responses[permutation])
        / args.source_count
    )
    semantic = _matrix_product(atlas, query_responses.T)
    shuffled = _matrix_product(shuffled_atlas, query_responses.T)
    constant = source_weights.mean(1)
    counts = {budget: max(1, math.ceil(budget * groups)) for budget in args.budgets}
    structural = _overlap_by_similarity(
        query_features,
        semantic,
        shuffled,
        counts=counts,
        seed=args.seed,
    )
    if args.structure_only:
        save_json(
            args.output,
            {
                "setting": "vitb16_semantic_circuit_structure",
                "modality": "vision",
                "protocol": {
                    "source_examples": args.source_count,
                    "queries": len(query_features),
                    "budgets": list(args.budgets),
                    "selected_groups": {f"{k:g}": v for k, v in counts.items()},
                },
                "structure": structural,
            },
        )
        return
    parameter_rms = scoped_parameter_rms(partition, scope)
    noise_std = parameter_rms * args.noise_scale
    late_layers = max(1, len(layout) // 4)
    selections, records = [], []
    started = time.time()
    for position, column in enumerate(query_columns.tolist()):
        target_index = int(target_indices[column])
        same_index = target_index + 1
        off_column = (column + classes // 2) % classes
        off_index = int(target_indices[off_column])
        samples = (dataset[target_index], dataset[same_index], dataset[off_index])
        images = torch.stack(tuple(sample[0] for sample in samples))
        labels = torch.tensor(tuple(int(sample[1]) for sample in samples))
        for budget, count in counts.items():
            grid = _selection_grid(
                semantic[:, column],
                constant,
                target_weights[:, position].double(),
                count=count,
                units_per_layer=units_per_layer,
                late_layers=late_layers,
            )
            for method, selected_local in grid.items():
                selections.append(
                    {
                        "query": position,
                        "class": column,
                        "budget": budget,
                        "method": method,
                        "target_correct": bool(correct[position]),
                        **_selection_record(
                            selected_local,
                            target_weights[:, position],
                            units_per_layer=units_per_layer,
                        ),
                    }
                )
                selected = torch.zeros(partition.n_groups, dtype=torch.bool)
                selected[scope_indices[selected_local]] = True
                for direction in range(args.directions):
                    intervention_seed = (
                        args.seed + position * 100_003 + direction * 1_009
                    )
                    with additive_group_noise(
                        partition, selected, noise_std, seed=intervention_seed, sign=1
                    ):
                        plus = margin_values(model, images, labels).cpu()
                    with additive_group_noise(
                        partition, selected, noise_std, seed=intervention_seed, sign=-1
                    ):
                        minus = margin_values(model, images, labels).cpu()
                    squared = ((plus - minus) / (2 * noise_std)).square()
                    records.append(
                        {
                            "query": position,
                            "class": column,
                            "budget": budget,
                            "method": method,
                            "direction": direction,
                            "target_correct": bool(correct[position]),
                            "target_susceptibility": float(squared[0]),
                            "same_class_susceptibility": float(squared[1]),
                            "off_class_susceptibility": float(squared[2]),
                            "input_specificity": float(squared[0] - squared[2]),
                        }
                    )
        print(
            f"vision semantic circuits {position + 1}/{len(query_columns)} "
            f"elapsed {time.time() - started:.1f}s",
            flush=True,
        )
    summaries, comparisons = _summarize_causal(
        records, args.budgets, valid="target_correct"
    )
    save_json(
        args.output,
        {
            "setting": "vitb16_semantic_parameter_influence_circuits",
            "modality": "vision",
            "model": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "dataset": "ImageNet-1k validation",
            "protocol": {
                "source_examples": args.source_count,
                "queries": len(query_columns),
                "correct_queries": int(correct.sum()),
                "groups": groups,
                "layers": len(layout),
                "units_per_layer": units_per_layer,
                "budgets": list(args.budgets),
                "selected_groups": {f"{k:g}": v for k, v in counts.items()},
                "query_offsets": list(query_offsets),
                "nonlate_layers": list(range(len(layout) - late_layers)),
                "late_layers": list(range(len(layout) - late_layers, len(layout))),
                "representation": "pretrained DINOv2-small CLS state",
                "kernel": "1,000-prototype RBF response map at scale 0.1",
                "directions": args.directions,
                "noise_scale_relative_to_ff_parameter_rms": args.noise_scale,
                "coordinate_noise_standard_deviation": noise_std,
            },
            "fidelity": {
                "source_max_partition_relative_error": source_error,
                "target_max_partition_relative_error": target_error,
            },
            "timing": {
                "source_profile_seconds": source_seconds,
                "target_profile_seconds": target_seconds,
                "causal_seconds": time.time() - started,
            },
            "structure": structural,
            "selections": selections,
            "causal_summaries": summaries,
            "causal_comparisons": comparisons,
            "records": records,
        },
    )


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _selection_heatmap(payload: dict, budget: float, rows: int = 12):
    selected = [
        row
        for row in payload["selections"]
        if row["method"] == "semantic_full"
        and math.isclose(float(row["budget"]), budget)
    ]
    selected.sort(key=lambda row: int(row["query"]))
    positions = np.linspace(0, len(selected) - 1, rows).round().astype(int)
    chosen = [selected[position] for position in positions]
    matrix = np.asarray(
        [
            np.asarray(row["layer_counts"], dtype=float) / row["selected_groups"]
            for row in chosen
        ]
    )
    identifiers = [int(row.get("class", row.get("end"))) for row in chosen]
    return matrix, identifiers


def _query_metric(payload, method, budget, metric):
    valid = payload["modality"] != "vision"
    grouped = defaultdict(list)
    for row in payload["records"]:
        if row["method"] != method or not math.isclose(row["budget"], budget):
            continue
        if not valid and not row["target_correct"]:
            continue
        grouped[int(row["query"])].append(float(row[metric]))
    return {query: float(np.mean(values)) for query, values in grouped.items()}


def _normalized_gain(payload, semantic, constant, budget, metric, *, seed):
    left = _query_metric(payload, semantic, budget, metric)
    low = _query_metric(payload, constant, budget, metric)
    high = _query_metric(payload, "direct_full", budget, "target_susceptibility")
    queries = sorted(left.keys() & low.keys() & high.keys())
    left_values = np.asarray([left[query] for query in queries])
    low_values = np.asarray([low[query] for query in queries])
    high_values = np.asarray([high[query] for query in queries])
    point = 100 * (left_values.mean() - low_values.mean()) / high_values.mean()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(queries), size=(5000, len(queries)))
    sampled = (
        100
        * (left_values[draws].mean(1) - low_values[draws].mean(1))
        / high_values[draws].mean(1)
    )
    low_ci, high_ci = np.quantile(sampled, (0.025, 0.975))
    return float(point), float(low_ci), float(high_ci), len(queries)


def _plot_structure_panel(axis, payload, letter, title):
    budgets = payload["protocol"]["budgets"]
    styles = {
        (budgets[0], "semantic"): (SEMANTIC_LIGHT, "o", "-", "0.1% KPSA"),
        (budgets[1], "semantic"): (SEMANTIC, "s", "-", "0.5% KPSA"),
        (budgets[0], "shuffled_pairing"): (
            SHUFFLE,
            "o",
            ":",
            "0.1% shuffled",
        ),
        (budgets[1], "shuffled_pairing"): (
            NEAREST,
            "s",
            ":",
            "0.5% shuffled",
        ),
    }
    for budget in budgets:
        budget_rows = payload["structure"]["budgets"][f"{budget:g}"]
        for method in ("semantic", "shuffled_pairing"):
            rows = budget_rows[method]
            means = np.asarray([row["jaccard"]["mean"] for row in rows])
            # Absolute overlap can be high for a collapsed control.  The
            # circuit claim concerns how much overlap changes with semantic
            # proximity, so anchor each curve at its least-similar quintile.
            anchor = means[0]
            means = means - anchor
            x = np.asarray([row["similarity_mean"] for row in rows])
            color, marker, linestyle, label = styles[(budget, method)]
            axis.plot(
                x,
                means,
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
    panel_title(axis, letter, title)
    axis.set_xlabel("Representation cosine similarity")
    axis.set_ylabel("Jaccard gain from\nleast-similar half")
    finish_axis(axis)


def _plot_gain_panel(axis, payload, letter, title):
    budgets = payload["protocol"]["budgets"]
    pairs = (
        ("semantic_full", "constant_full_matched", "Full stack"),
        ("semantic_nonlate", "constant_nonlate_matched", "Non-late"),
        ("semantic_late", "constant_late_matched", "Final quarter"),
    )
    x = np.arange(len(pairs))
    for budget, color, marker, offset, label in (
        (budgets[0], SEMANTIC_LIGHT, "o", -0.09, "0.1% KPSA"),
        (budgets[1], SEMANTIC, "s", 0.09, "0.5% KPSA"),
    ):
        values = [
            _normalized_gain(
                payload,
                semantic,
                constant,
                budget,
                "target_susceptibility",
                seed=91_027 + position + round(1e6 * budget),
            )
            for position, (semantic, constant, _name) in enumerate(pairs)
        ]
        means = np.asarray([row[0] for row in values])
        low = np.asarray([row[1] for row in values])
        high = np.asarray([row[2] for row in values])
        axis.errorbar(
            x + offset,
            means,
            yerr=[means - low, high - means],
            color=color,
            marker=marker,
            linestyle="none",
            label=label,
        )
    panel_title(axis, letter, title)
    axis.set_xticks(x, [item[2] for item in pairs])
    axis.set_ylabel("Gain over matched constant\n(% of direct gradient)")
    finish_axis(axis, zero_line=True)


def run_plot(args) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    vision = _read_json(args.vision)
    forecast = _read_json(args.forecast)
    budget = float(vision["protocol"]["budgets"][-1])
    cmap = LinearSegmentedColormap.from_list("kpsa_circuit", ["#FFFFFF", SEMANTIC])
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.75))
    display_records = {}
    for row, (payload, label) in enumerate(
        ((vision, "Vision"), (forecast, "Forecasting"))
    ):
        matrix, identifiers = _selection_heatmap(payload, budget)
        axis = axes[row, 0]
        axis.imshow(matrix, cmap=cmap, vmin=0, vmax=max(0.15, matrix.max()))
        panel_title(axis, "AD"[row], f"{label}: circuit across depth")
        axis.set_xlabel("Transformer block")
        axis.set_ylabel("Held-out input")
        axis.set_xticks(range(matrix.shape[1]), range(1, matrix.shape[1] + 1))
        axis.set_yticks(range(len(matrix)), range(1, len(matrix) + 1))
        axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
        display_records[label.lower()] = {
            "row_identifiers": identifiers,
            "cell_value": "share of the 0.5% semantic KPSA selection in block",
        }
    _plot_structure_panel(axes[0, 1], vision, "B", "Vision: geometry predicts overlap")
    _plot_gain_panel(axes[0, 2], vision, "C", "Vision: exact causal utility")
    _plot_structure_panel(
        axes[1, 1], forecast, "E", "Forecasting: geometry predicts overlap"
    )
    _plot_gain_panel(axes[1, 2], forecast, "F", "Forecasting: exact causal utility")
    outside_legend(fig, axes, ncol=4, y=1.02)
    save_plot(fig, args.output)
    save_json(
        args.output.with_name(args.output.name + "_display.json"),
        {
            "figure": args.output.name,
            "vision_source": str(args.vision),
            "forecast_source": str(args.forecast),
            "display_rows": display_records,
            "causal_y_axis": (
                "semantic minus layer-count-matched constant mean exact squared "
                "susceptibility, normalized by direct-gradient mean susceptibility"
            ),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--output", type=Path, required=True)
    shared.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    shared.add_argument("--queries", type=int, default=64)
    shared.add_argument("--directions", type=int, default=8)
    shared.add_argument("--noise-scale", type=float, default=0.03)
    shared.add_argument("--seed", type=int, default=260927)
    shared.add_argument("--structure-only", action="store_true")

    vision = subparsers.add_parser("vision", parents=[shared])
    vision.add_argument("--data", type=Path, required=True)
    vision.add_argument("--source-count", type=int, default=8000)

    forecast = subparsers.add_parser("forecast", parents=[shared])
    forecast.add_argument("--data", type=Path, required=True)
    forecast.add_argument("--dataset-name", default="ETTh1")
    forecast.add_argument("--target-column", default="OT")
    forecast.add_argument("--normalization-end", type=int, default=8640)
    forecast.add_argument("--query-start", type=int, default=12300)
    forecast.add_argument("--query-stop", type=int, default=16800)
    forecast.add_argument("--context-length", type=int, default=512)
    forecast.add_argument("--horizon", type=int, default=24)
    forecast.add_argument("--model", default=DEFAULT_MODEL)
    forecast.add_argument("--device", default="cuda")
    forecast.add_argument("--atlas-cache", type=Path, required=True)
    forecast.add_argument("--rbf-scale", type=float, default=0.25)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--vision", type=Path, required=True)
    plot.add_argument("--forecast", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.stage == "plot":
        run_plot(args)
        return
    args.budgets = tuple(args.budgets)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.stage == "vision":
        run_vision(args)
    else:
        run_forecast(args)


if __name__ == "__main__":
    main()
