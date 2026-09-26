"""Matched raw-versus-normalized vision atlas ablation at the selected scale."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .common import save_json, seed_everything
from .kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .parameter_groups import ParameterPartition
from .parameter_influence_circuits_v8 import margin_values
from .refined_analysis import paired_t_summary
from .representation_sensitivity import profile_ranking_metrics
from .vision_atlas_size_scaling_v8 import (
    make_predictions as make_scaled_predictions,
)
from .vision_atlas_size_scaling_v8 import (
    offset_major_indices,
)
from .vision_causal_refined import (
    functional_value,
    mean_ablation_taylor_scores,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
    zero_ablate_mlp_activations,
)
from .vision_sample_refined import encode_indices, profile_functional_images


@torch.no_grad()
def dataset_margins(model, dataset, indices, batch_size=32):
    """Compute correct-class margins for a fixed dataset index list."""
    values = []
    for local in indices.split(batch_size):
        samples = [dataset[int(index)] for index in local]
        images = torch.stack([sample[0] for sample in samples])
        labels = torch.tensor([int(sample[1]) for sample in samples])
        values.append(margin_values(model, images, labels).cpu())
    return torch.cat(values)


def _paired(records, method, baseline, fraction):
    left = {
        int(row["query_column"]): float(row["degradation"])
        for row in records
        if row["target_correct"]
        and row["method"] == method
        and row["fraction"] == fraction
    }
    right = {
        int(row["query_column"]): float(row["degradation"])
        for row in records
        if row["target_correct"]
        and row["method"] == baseline
        and row["fraction"] == fraction
    }
    return paired_t_summary([left[key] - right[key] for key in left.keys() & right])


def _raw_diagnostics(source_raw, source_margins):
    totals = source_raw.double().sum(0)
    group_sums = source_raw.double().sum(1)
    group_square_sums = source_raw.double().square().sum(1)
    group_ess = group_sums.square() / group_square_sums.clamp_min(1e-300)
    log_totals = totals.log().numpy()
    margins = source_margins.double().numpy()
    return {
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
        "log_total_vs_margin_pearson": float(np.corrcoef(log_totals, margins)[0, 1]),
        "total_vs_margin_spearman": float(spearmanr(totals.numpy(), margins).statistic),
        "correct_fraction": float((source_margins > 0).float().mean()),
    }


def run_causal(
    model,
    dataset,
    target_indices,
    predictions,
    *,
    queries=100,
    fractions=(0.0002, 0.001),
):
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout).cpu()
    sizes = partition.sizes.detach().double().cpu()
    zero_reference = {
        item["name"]: torch.zeros(item["count"], device=partition.device)
        for item in layout
    }
    columns = torch.linspace(
        0, len(target_indices) - 1, min(queries, len(target_indices))
    ).round().long().unique()
    records = []
    for position, column in enumerate(columns.tolist()):
        image, label = dataset[int(target_indices[column])]
        label = int(label)
        baseline = functional_value(model, image, label, "margin")
        correct = baseline > 0
        taylor = mean_ablation_taylor_scores(
            model,
            image,
            label,
            "margin",
            partition,
            layout,
            zero_reference,
        )
        local_predictions = {**predictions, "activation_attribution": taylor[:, None]}
        for method, scores in local_predictions.items():
            for fraction in fractions:
                values = scores[:, 0] if method == "activation_attribution" else scores[:, column]
                selected, actual = select_parameter_budget(
                    values.clamp_min(0), sizes, scope, fraction
                )
                with zero_ablate_mlp_activations(layout, selected):
                    intervened = functional_value(model, image, label, "margin")
                records.append(
                    {
                        "query_column": column,
                        "target_correct": correct,
                        "method": method,
                        "fraction": fraction,
                        "actual_parameter_fraction": actual,
                        "degradation": baseline - intervened,
                    }
                )
        print(f"normalization ablation {position + 1}/{len(columns)}", flush=True)

    comparisons = {}
    for fraction in fractions:
        key = f"{fraction:g}"
        comparisons[key] = {}
        for mode in ("normalized", "raw"):
            scalar = f"{mode}_scalar"
            oracle = _paired(records, "activation_attribution", scalar, fraction)
            current = {"activation_attribution_vs_scalar": oracle}
            for family in ("exact", "prototype"):
                method = f"{mode}_{family}"
                summary = _paired(records, method, scalar, fraction)
                other = "raw" if mode == "normalized" else "normalized"
                current[family] = {
                    "vs_scalar": summary,
                    "vs_shuffled_pairing": _paired(
                        records, method, f"{mode}_{family}_permuted", fraction
                    ),
                    f"vs_{other}": _paired(
                        records, method, f"{other}_{family}", fraction
                    ),
                    "activation_gap_recovered": (
                        summary["mean"] / oracle["mean"]
                        if oracle["mean"] > 0
                        else None
                    ),
                }
            comparisons[key][mode] = current
    return records, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=6400)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    seed_everything(161_803)

    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    retained = torch.load(args.input, map_location="cpu", weights_only=True)
    classes = len(
        torch.unique(retained["representation_indices"].long() // 50 * 50)
    )
    if args.source_count % classes:
        parser.error("source count must contain complete class-balanced blocks")
    per_class = args.source_count // classes
    source_indices = offset_major_indices(
        retained["representation_indices"], range(per_class)
    )
    construction_indices = offset_major_indices(
        retained["representation_indices"], range(43, 47)
    )
    target_indices = offset_major_indices(retained["representation_indices"], [49])

    raw_dataset = ImageFolder(args.data / "validation")
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
    del encoder, encoded
    gc.collect()
    torch.cuda.empty_cache()

    construction_median = median_squared_distance(construction_features)
    feature_map = fit_prototype_response_map(
        construction_features,
        800,
        bandwidth_squared=construction_median * 0.025**2,
        seed=91_027 + 800,
    )
    source_responses = feature_map.transform(source_features).float()
    target_responses = feature_map.transform(target_features).float()

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    ).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    started = time.perf_counter()
    source_raw, source_error, source_seconds = profile_functional_images(
        model,
        dataset,
        source_indices,
        partition,
        functional="margin",
        weight_mode="raw",
    )
    target_raw, target_error, target_seconds = profile_functional_images(
        model,
        dataset,
        target_indices,
        partition,
        functional="margin",
        weight_mode="raw",
    )
    source_margins = dataset_margins(model, dataset, source_indices)
    diagnostics = _raw_diagnostics(source_raw, source_margins)

    raw_predictions, source_median = make_scaled_predictions(
        source_raw,
        source_features,
        target_features,
        source_responses,
        target_responses,
        exact_scale=0.1,
        classes=classes,
        seed=91_027,
    )
    source_raw.div_(source_raw.sum(0, keepdim=True).clamp_min(1e-30))
    target_normalized = target_raw / target_raw.sum(0, keepdim=True).clamp_min(1e-30)
    normalized_predictions, _ = make_scaled_predictions(
        source_raw,
        source_features,
        target_features,
        source_responses,
        target_responses,
        exact_scale=0.1,
        classes=classes,
        seed=91_027,
    )
    predictions = {}
    for mode, current in (
        ("raw", raw_predictions),
        ("normalized", normalized_predictions),
    ):
        predictions.update(
            {
                f"{mode}_exact": current["exact_rbf"],
                f"{mode}_exact_permuted": current["exact_rbf_permuted"],
                f"{mode}_prototype": current["prototype_rbf"],
                f"{mode}_prototype_permuted": current["prototype_rbf_permuted"],
                f"{mode}_scalar": current["scalar_mass"],
            }
        )
    cold = {
        "raw": {
            method: profile_ranking_metrics(raw_predictions[method], target_raw)
            for method in ("exact_rbf", "prototype_rbf", "scalar_mass")
        },
        "normalized": {
            method: profile_ranking_metrics(
                normalized_predictions[method], target_normalized
            )
            for method in ("exact_rbf", "prototype_rbf", "scalar_mass")
        },
    }
    del source_raw, source_features, source_responses, target_responses
    del raw_predictions, normalized_predictions
    gc.collect()
    torch.cuda.empty_cache()

    records, comparisons = run_causal(
        model,
        dataset,
        target_indices,
        predictions,
        queries=args.queries,
    )
    result = {
        "setting": "vitb16_raw_vs_normalized_sensitivity",
        "model": {
            "name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "parameters": sum(p.numel() for p in model.parameters()),
            "groups": partition.n_groups,
        },
        "protocol": {
            "source_examples": args.source_count,
            "target_offset": 49,
            "construction_offsets": [43, 46],
            "queries": args.queries,
            "fractions": [0.0002, 0.001],
            "intervention": "zero coupled MLP feature activations",
            "exact_scale": 0.1,
            "prototype_count": 800,
            "prototype_scale": 0.025,
            "source_median_squared_distance": source_median,
            "construction_median_squared_distance": construction_median,
        },
        "raw_total_energy": diagnostics,
        "fidelity": {
            "source_max_partition_relative_error": source_error,
            "target_max_partition_relative_error": target_error,
        },
        "timing": {
            "source_profile_seconds": source_seconds,
            "target_profile_seconds": target_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "cold_query": cold,
        "comparisons": comparisons,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
