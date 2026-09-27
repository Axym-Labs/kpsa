"""Interpretability assays for input-conditioned parameter-influence circuits."""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from .common import save_json, seed_everything
from .kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .parameter_groups import ParameterPartition
from .positive_evidence_bundle import (
    BLACK,
    GRAY,
    LIGHT,
    PRIMARY,
    SECONDARY,
    outside_legend,
    save_plot,
)
from .refined_analysis import paired_t_summary
from .representation_sensitivity import (
    partition_gradient_energy,
    sensitivity_weights,
)
from .vision_atlas_size_scaling_v8 import balanced_pairing_permutation
from .vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from .vision_sample_refined import encode_indices

METHOD_LABELS = {
    "prototype": "Sensitivity atlas",
    "prototype_permuted": "Shuffled pairing",
    "class_onehot": "Categorical class space",
    "nearest": "Nearest example",
    "scalar": "Scalar sensitivity",
}


def _topk_sets(scores: torch.Tensor, count: int) -> torch.Tensor:
    return torch.topk(scores, count, dim=0).indices.T.contiguous()


def _overlap(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    values = []
    for first, second in zip(left.tolist(), right.tolist()):
        values.append(len(set(first) & set(second)) / len(first))
    return values


def _jaccard(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    values = []
    for first, second in zip(left.tolist(), right.tolist()):
        first_set, second_set = set(first), set(second)
        values.append(len(first_set & second_set) / len(first_set | second_set))
    return values


def _summary(values) -> dict[str, float]:
    return paired_t_summary([float(value) for value in values])


def _paired(values, reference) -> dict[str, float]:
    return _summary([float(a) - float(b) for a, b in zip(values, reference)])


def _matrix_product(left: torch.Tensor, right: torch.Tensor, chunk=4096):
    output = torch.empty(left.shape[0], right.shape[1], dtype=torch.float32)
    right_gpu = right.float().cuda()
    for start in range(0, len(left), chunk):
        stop = min(start + chunk, len(left))
        output[start:stop] = (left[start:stop].float().cuda() @ right_gpu).cpu()
    return output


def _indices_from_starts(class_starts: torch.Tensor, offsets) -> torch.Tensor:
    offsets = torch.as_tensor(list(offsets), dtype=torch.long)
    return (offsets[:, None] + class_starts[None, :]).reshape(-1)


def _profile_scoped_normalized(model, dataset, indices, partition, scope):
    """Profile normalized sensitivity while retaining only scoped MLP groups."""
    profiles = torch.empty(int(scope.sum()), len(indices), dtype=torch.float32)
    correct = torch.empty(len(indices), dtype=torch.bool)
    max_error = 0.0
    device = next(model.parameters()).device
    started = time.perf_counter()
    for column, index in enumerate(indices.tolist()):
        image, label = dataset[index]
        label = int(label)
        model.zero_grad(set_to_none=True)
        logits = model(image[None].to(device)).float()
        correct[column] = int(logits.argmax(1)) == label
        alternatives = logits[0].clone()
        alternatives[label] = -torch.inf
        objective = logits[0, label] - alternatives.max()
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
        normalized = sensitivity_weights(
            energy.detach().double().cpu()[None], "normalized"
        )[0].float()
        profiles[:, column] = normalized[scope]
    model.zero_grad(set_to_none=True)
    return profiles, max_error, time.perf_counter() - started, correct


def _group_labels(layout):
    labels = []
    for layer, item in enumerate(layout):
        labels.extend(
            {
                "layer": layer,
                "feature": feature,
                "name": f"L{layer} MLP feature {feature}",
            }
            for feature in range(int(item["count"]))
        )
    return labels


def _selected_class_metrics(
    selections: torch.Tensor,
    class_distributions: torch.Tensor,
    correct: torch.Tensor,
):
    class_count = class_distributions.shape[1]
    entropy_denominator = math.log(class_count)
    query_mass = []
    dominant_match = []
    purity = []
    for query, groups in enumerate(selections):
        if not bool(correct[query]):
            continue
        distributions = class_distributions[groups]
        query_mass.append(float(distributions[:, query].mean()))
        dominant_match.append(float((distributions.argmax(1) == query).float().mean()))
        entropy = -(
            distributions
            * distributions.clamp_min(torch.finfo(distributions.dtype).tiny).log()
        ).sum(1)
        purity.append(float((1 - entropy / entropy_denominator).mean()))
    return {
        "query_class_sensitivity_mass": query_mass,
        "dominant_class_match": dominant_match,
        "class_purity": purity,
    }


def _save_quantitative_figure(metrics, output):
    methods = list(METHOD_LABELS)
    colors = {
        "prototype": PRIMARY,
        "prototype_permuted": LIGHT,
        "class_onehot": SECONDARY,
        "nearest": GRAY,
        "scalar": BLACK,
    }
    markers = {
        "prototype": "o",
        "prototype_permuted": "x",
        "class_onehot": "s",
        "nearest": "^",
        "scalar": "D",
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.7))
    for method in methods:
        label = METHOD_LABELS[method]
        oracle = metrics["oracle_overlap"][method]["summary"]
        class_mass = metrics["class_semantics"][method]["query_class_sensitivity_mass"][
            "summary"
        ]
        x = methods.index(method)
        for axis, summary in zip(axes[:2], (oracle, class_mass)):
            axis.errorbar(
                x,
                summary["mean"],
                yerr=[
                    [summary["mean"] - summary["lower_95"]],
                    [summary["upper_95"] - summary["mean"]],
                ],
                color=colors[method],
                marker=markers[method],
                capsize=2,
                linestyle="none",
                label=label,
            )
        stability = metrics["circuit_stability"][method]
        axes[2].plot(
            [0, 1],
            [stability["within_class"]["mean"], stability["between_class"]["mean"]],
            color=colors[method],
            marker=markers[method],
            linewidth=1.1,
            markersize=4,
            label=label,
        )
    short = ["Semantic", "Shuffle", "Categorical", "Nearest", "Scalar"]
    axes[0].set_xticks(range(len(methods)), short, rotation=30, ha="right")
    axes[1].set_xticks(range(len(methods)), short, rotation=30, ha="right")
    axes[0].set_ylabel("Direct-gradient top-8 overlap")
    axes[1].set_ylabel("Query-class sensitivity mass")
    axes[1].axhline(1 / 200, color="#D5D5D5", linewidth=0.8, linestyle=":")
    axes[2].set_xticks([0, 1], ["Same class", "Different class"])
    axes[2].set_ylabel("Circuit Jaccard similarity")
    for axis in axes:
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    outside_legend(fig, axes, ncol=len(methods), fontsize=6.5)
    save_plot(fig, output)


def _save_exemplar_figure(
    *,
    output: Path,
    raw_dataset,
    target_indices,
    source_indices,
    selected_queries,
    selected_groups,
    source_weights,
    source_responses,
    query_responses,
    group_labels,
    categories,
):
    columns = 9
    fig, axes = plt.subplots(
        len(selected_queries), columns, figsize=(11.4, 1.55 * len(selected_queries))
    )
    axes = np.asarray(axes).reshape(len(selected_queries), columns)
    for row, query in enumerate(selected_queries):
        group = int(selected_groups[query, 0])
        weights = source_weights[group]
        unconditional = torch.topk(weights, 4).indices
        similarity = source_responses @ query_responses[query]
        conditional = torch.topk(weights * similarity, 4).indices
        images = [raw_dataset[int(target_indices[query])][0]]
        images.extend(
            raw_dataset[int(source_indices[index])][0] for index in unconditional
        )
        images.extend(
            raw_dataset[int(source_indices[index])][0] for index in conditional
        )
        query_label = int(target_indices[query] // 50)
        row_label = f"{categories[query_label]}\n{group_labels[group]['name']}"
        for column, (axis, image) in enumerate(zip(axes[row], images)):
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(
                    row_label, fontsize=6, rotation=0, ha="right", va="center"
                )
            if row == 0:
                titles = (
                    ["Query"]
                    + ["Sensitive exemplar"] * 4
                    + ["Query-weighted exemplar"] * 4
                )
                axis.set_title(titles[column], fontsize=6)
    fig.tight_layout(w_pad=0.15, h_pad=0.35)
    save_plot(fig, output)


def _display_classes(correct: torch.Tensor, count: int) -> torch.Tensor:
    """Choose a deterministic, circuit-score-independent spread of correct classes."""
    candidates = correct.nonzero().flatten()
    if not len(candidates):
        raise ValueError("the circuit display requires at least one correct query")
    count = min(count, len(candidates))
    positions = torch.linspace(0, len(candidates) - 1, count).round().long()
    return candidates[positions]


def _jaccard_matrix(selections: torch.Tensor) -> np.ndarray:
    sets = [set(row) for row in selections.tolist()]
    matrix = np.empty((len(sets), len(sets)), dtype=np.float32)
    for row, left in enumerate(sets):
        for column, right in enumerate(sets):
            matrix[row, column] = len(left & right) / len(left | right)
    return matrix


def _save_circuit_structure_figure(
    *,
    output: Path,
    selections: torch.Tensor,
    class_distributions: torch.Tensor,
    correct: torch.Tensor,
    categories,
    group_labels,
    classes: int,
    query_offsets,
    full_stability,
    display_class_count: int = 16,
):
    """Show how class sensitivity composes into stable per-image circuits."""
    display_classes = _display_classes(correct, display_class_count)
    blocks = [
        selections[offset * classes : (offset + 1) * classes]
        for offset in range(len(query_offsets))
    ]
    sample_selections = torch.stack(
        [block[display_classes] for block in blocks], dim=1
    ).reshape(-1, selections.shape[1])

    displayed_groups = torch.unique(sample_selections.flatten(), sorted=True)
    displayed_mass = class_distributions[displayed_groups][:, display_classes]
    dominant_display_class = displayed_mass.argmax(1)
    group_order = sorted(
        range(len(displayed_groups)),
        key=lambda index: (
            int(dominant_display_class[index]),
            int(group_labels[int(displayed_groups[index])]["layer"]),
            int(group_labels[int(displayed_groups[index])]["feature"]),
        ),
    )
    displayed_groups = displayed_groups[group_order]
    displayed_mass = displayed_mass[group_order]
    dominant_display_class = dominant_display_class[group_order]
    group_to_column = {
        int(group): column for column, group in enumerate(displayed_groups)
    }

    uniform_mass = 1 / classes
    enrichment = torch.log2(
        displayed_mass.T.clamp_min(torch.finfo(displayed_mass.dtype).tiny)
        / uniform_mass
    ).clamp(-3, 3)
    circuit_map = np.full(
        (len(sample_selections), len(displayed_groups)), np.nan, dtype=np.float32
    )
    for row, groups in enumerate(sample_selections.tolist()):
        for rank, group in enumerate(groups):
            circuit_map[row, group_to_column[group]] = 1 - 0.65 * rank / max(
                1, len(groups) - 1
            )
    similarities = _jaccard_matrix(sample_selections)

    fig = plt.figure(figsize=(12.0, 5.6))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.35, 1.0), wspace=0.32)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    class_names = [categories[int(index)] for index in display_classes]
    class_labels = [
        f"{position + 1}. {name}" for position, name in enumerate(class_names)
    ]

    image = axes[0].imshow(
        enrichment.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )
    axes[0].set_title("A  Class sensitivity atlas", loc="left", fontsize=9)
    axes[0].set_ylabel("ImageNet class")
    axes[0].set_yticks(range(len(class_labels)), class_labels, fontsize=6.5)
    axes[0].set_xticks([])
    axes[0].set_xlabel("Retrieved groups (ordered by dominant displayed class)")
    colorbar = fig.colorbar(image, ax=axes[0], orientation="horizontal", pad=0.08)
    colorbar.set_label("Sensitivity enrichment (log$_2$ vs. uniform)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    circuit_cmap = LinearSegmentedColormap.from_list(
        "circuit_rank", ["#B8D7EA", PRIMARY]
    )
    circuit_cmap.set_bad("white")
    axes[1].imshow(
        circuit_map,
        aspect="auto",
        interpolation="nearest",
        cmap=circuit_cmap,
        vmin=0.35,
        vmax=1,
    )
    axes[1].set_title("B  Retrieved circuit by image", loc="left", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_xlabel("Same parameter groups and order as A")
    sample_centers = (
        np.arange(len(display_classes)) * len(query_offsets)
        + (len(query_offsets) - 1) / 2
    )
    axes[1].set_yticks(sample_centers, class_labels, fontsize=6.5)
    axes[1].set_ylabel("Three held-out images per class")

    similarity_image = axes[2].imshow(
        similarities,
        aspect="equal",
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("C  Circuit overlap by image", loc="left", fontsize=9)
    axes[2].set_xticks(sample_centers, range(1, len(display_classes) + 1), fontsize=6)
    axes[2].set_yticks(sample_centers, range(1, len(display_classes) + 1), fontsize=6)
    axes[2].set_xlabel("Class index")
    axes[2].set_ylabel("Class index")
    axes[2].text(
        0.98,
        0.98,
        (
            f"All {classes:,} classes\n"
            f"same: {full_stability['within_class']['mean']:.3f}\n"
            f"different: {full_stability['between_class']['mean']:.3f}"
        ),
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    colorbar = fig.colorbar(
        similarity_image, ax=axes[2], orientation="horizontal", pad=0.08
    )
    colorbar.set_label("Top-group Jaccard", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    for axis in axes:
        for position in range(1, len(display_classes)):
            if axis is axes[0]:
                axis.axhline(position - 0.5, color="white", linewidth=0.45)
            elif axis is axes[1]:
                axis.axhline(
                    position * len(query_offsets) - 0.5,
                    color="#D9D9D9",
                    linewidth=0.45,
                )
            else:
                boundary = position * len(query_offsets) - 0.5
                axis.axhline(boundary, color="white", linewidth=0.4)
                axis.axvline(boundary, color="white", linewidth=0.4)

    changes = (
        (dominant_display_class[1:] != dominant_display_class[:-1]).nonzero().flatten()
    )
    for boundary in (changes + 1).tolist():
        axes[0].axvline(boundary - 0.5, color="white", linewidth=0.5)
        axes[1].axvline(boundary - 0.5, color="#D9D9D9", linewidth=0.5)
    axes[1].text(
        0.5,
        -0.145,
        "Darker cells are higher-ranked members of each top-8 circuit",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.19)
    save_plot(fig, output)
    return {
        "class_selection_rule": (
            "evenly spaced quantiles of correctly classified offset-49 queries"
        ),
        "display_class_indices": display_classes.tolist(),
        "display_class_names": class_names,
        "display_group_count": len(displayed_groups),
        "display_group_order": (
            "dominant sensitivity mass among displayed classes, then layer and feature"
        ),
        "display_groups": [group_labels[int(group)] for group in displayed_groups],
        "sample_offsets": list(query_offsets),
        "sample_top_groups": sample_selections.tolist(),
        "full_class_stability": full_stability,
    }


def run_study(
    *,
    retained_path: Path,
    data_path: Path,
    output: Path,
    figure_dir: Path,
    source_count: int,
    top_groups: int,
    all_classes: bool,
):
    import timm
    from torchvision.datasets import ImageFolder
    from torchvision.models import ViT_B_16_Weights
    from transformers import AutoImageProcessor, AutoModel

    retained = torch.load(retained_path, map_location="cpu", weights_only=True)
    raw_dataset = ImageFolder(data_path / "validation")
    if all_classes:
        if len(raw_dataset) != 50_000:
            raise ValueError("all-class ImageNet validation requires 50,000 images")
        class_starts = torch.arange(0, len(raw_dataset), 50)
    else:
        class_starts = torch.unique(
            retained["representation_indices"].long() // 50 * 50, sorted=True
        )
    classes = len(class_starts)
    if source_count % classes:
        raise ValueError("source count must contain balanced class blocks")
    per_class = source_count // classes
    source_indices = _indices_from_starts(class_starts, range(per_class))
    construction_offsets = [43] if all_classes else range(43, 47)
    construction_indices = _indices_from_starts(class_starts, construction_offsets)
    query_offsets = (47, 48, 49)
    query_indices = _indices_from_starts(class_starts, query_offsets)
    target_indices = query_indices[-classes:]

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
        800,
        bandwidth_squared=construction_median * 0.025**2,
        seed=91_027 + 800,
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
    dataset = ImageFolder(data_path / "validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    source_weights, source_error, source_seconds, _ = _profile_scoped_normalized(
        model,
        dataset,
        source_indices,
        partition,
        scope,
    )
    target_weights, target_error, target_seconds, correct = _profile_scoped_normalized(
        model,
        dataset,
        target_indices,
        partition,
        scope,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    permutation = balanced_pairing_permutation(source_count, classes, seed=91_027)
    atlas = _matrix_product(source_weights, source_responses) / source_count
    shuffled_atlas = (
        _matrix_product(source_weights, source_responses[permutation]) / source_count
    )
    prototype_scores = _matrix_product(atlas, query_responses.T)
    shuffled_scores = _matrix_product(shuffled_atlas, query_responses.T)
    cosine = source_features @ query_features.T
    nearest_scores = source_weights[:, cosine.argmax(0)]
    class_scores = source_weights.reshape(len(source_weights), per_class, classes).mean(
        1
    )
    scalar_scores = source_weights.mean(1, keepdim=True).expand(-1, len(query_indices))
    target_block = slice(2 * classes, 3 * classes)
    score_grid = {
        "prototype": prototype_scores,
        "prototype_permuted": shuffled_scores,
        "class_onehot": class_scores.repeat(1, len(query_offsets)),
        "nearest": nearest_scores,
        "scalar": scalar_scores,
    }
    selections = {
        method: _topk_sets(scores, top_groups) for method, scores in score_grid.items()
    }
    oracle_selection = _topk_sets(target_weights, top_groups)

    oracle_values = {}
    class_values = {}
    class_mass = source_weights.reshape(len(source_weights), per_class, classes).sum(1)
    class_distributions = class_mass / class_mass.sum(1, keepdim=True).clamp_min(
        torch.finfo(class_mass.dtype).tiny
    )
    for method in METHOD_LABELS:
        selected = selections[method][target_block]
        overlaps = _overlap(selected[correct], oracle_selection[correct])
        oracle_values[method] = {"values": overlaps, "summary": _summary(overlaps)}
        raw_metrics = _selected_class_metrics(selected, class_distributions, correct)
        class_values[method] = {
            name: {"values": values, "summary": _summary(values)}
            for name, values in raw_metrics.items()
        }

    comparisons = {"oracle_overlap": {}, "query_class_sensitivity_mass": {}}
    for baseline in METHOD_LABELS:
        if baseline == "prototype":
            continue
        comparisons["oracle_overlap"][f"prototype_vs_{baseline}"] = _paired(
            oracle_values["prototype"]["values"],
            oracle_values[baseline]["values"],
        )
        comparisons["query_class_sensitivity_mass"][f"prototype_vs_{baseline}"] = (
            _paired(
                class_values["prototype"]["query_class_sensitivity_mass"]["values"],
                class_values[baseline]["query_class_sensitivity_mass"]["values"],
            )
        )

    stability = {}
    for method, selected in selections.items():
        blocks = [
            selected[offset * classes : (offset + 1) * classes]
            for offset in range(len(query_offsets))
        ]
        within = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            within.extend(_jaccard(blocks[first], blocks[second]))
        between = []
        for block in blocks:
            between.extend(_jaccard(block, block.roll(73, dims=0)))
        stability[method] = {
            "within_class": _summary(within),
            "between_class": _summary(between),
            "within_minus_between": _paired(within, between),
        }

    group_labels = _group_labels(layout)
    correct_indices = correct.nonzero().flatten()
    exemplar_positions = torch.linspace(0, len(correct_indices) - 1, 4).round().long()
    selected_queries = correct_indices[exemplar_positions].tolist()
    categories = ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_suffix = "imagenet1000" if all_classes else "imagenet200"
    quantitative_stem = f"parameter_influence_interpretability_{figure_suffix}"
    exemplar_stem = f"parameter_influence_exemplars_{figure_suffix}"
    structure_stem = f"parameter_influence_structure_{figure_suffix}"
    metrics = {
        "oracle_overlap": oracle_values,
        "class_semantics": class_values,
        "comparisons": comparisons,
        "circuit_stability": stability,
    }
    _save_quantitative_figure(metrics, figure_dir / quantitative_stem)
    _save_exemplar_figure(
        output=figure_dir / exemplar_stem,
        raw_dataset=raw_dataset,
        target_indices=target_indices,
        source_indices=source_indices,
        selected_queries=selected_queries,
        selected_groups=selections["prototype"][target_block],
        source_weights=source_weights,
        source_responses=source_responses,
        query_responses=query_responses[target_block],
        group_labels=group_labels,
        categories=categories,
    )
    structure_record = _save_circuit_structure_figure(
        output=figure_dir / structure_stem,
        selections=selections["prototype"],
        class_distributions=class_distributions,
        correct=correct,
        categories=categories,
        group_labels=group_labels,
        classes=classes,
        query_offsets=query_offsets,
        full_stability=stability["prototype"],
    )

    exemplar_records = []
    for query in selected_queries:
        group = int(selections["prototype"][target_block][query, 0])
        weights = source_weights[group]
        similarity = source_responses @ query_responses[target_block][query]
        unconditional = torch.topk(weights, 4).indices.tolist()
        conditional = torch.topk(weights * similarity, 4).indices.tolist()
        exemplar_records.append(
            {
                "query_column": query,
                "query_dataset_index": int(target_indices[query]),
                "query_class": categories[int(target_indices[query] // 50)],
                "group": group_labels[group],
                "unconditional_source_indices": [
                    int(source_indices[index]) for index in unconditional
                ],
                "query_weighted_source_indices": [
                    int(source_indices[index]) for index in conditional
                ],
            }
        )
    result = {
        "setting": "vitb16_parameter_influence_circuit_interpretability",
        "protocol": {
            "source_examples": source_count,
            "classes": classes,
            "construction_examples": len(construction_indices),
            "query_offsets": list(query_offsets),
            "correct_confirmation_queries": int(correct.sum()),
            "selected_groups_per_query": top_groups,
            "selection_budget": "0.02% scoped MLP parameter budget (8 coupled features)",
            "semantic_statistic": (
                f"sensitivity mass across the {classes} represented ImageNet classes"
            ),
            "circuit_stability": (
                "top-group Jaccard across held-out same-class images versus a "
                "fixed class derangement"
            ),
        },
        "fidelity": {
            "source_max_partition_relative_error": source_error,
            "target_max_partition_relative_error": target_error,
        },
        "timing": {
            "source_profile_seconds": source_seconds,
            "target_profile_seconds": target_seconds,
        },
        "metrics": metrics,
        "structure_visualization": structure_record,
        "exemplars": exemplar_records,
        "figures": [
            f"{quantitative_stem}.png",
            f"{quantitative_stem}.pdf",
            f"{exemplar_stem}.png",
            f"{exemplar_stem}.pdf",
            f"{structure_stem}.png",
            f"{structure_stem}.pdf",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=6400)
    parser.add_argument("--top-groups", type=int, default=8)
    parser.add_argument("--all-classes", action="store_true")
    args = parser.parse_args()
    seed_everything(161_803)
    run_study(
        retained_path=args.input,
        data_path=args.data,
        output=args.output,
        figure_dir=args.figure_dir,
        source_count=args.source_count,
        top_groups=args.top_groups,
        all_classes=args.all_classes,
    )


if __name__ == "__main__":
    main()
