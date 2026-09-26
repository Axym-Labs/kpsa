"""Analyze broader RBF kernels from a retained development run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import save_json
from .kernel_reproduction_v8 import summarize_causal_records
from .kernel_sensitivity import median_squared_distance, rbf_kernel


def _distribution(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().double().cpu().numpy()
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.1)),
        "q90": float(np.quantile(array, 0.9)),
    }


def analyze(
    tensors: dict[str, torch.Tensor],
    result: dict,
    *,
    scales: tuple[float, ...],
    narrow_scale: float,
) -> dict[str, object]:
    source = tensors["source_features"].double()
    target = tensors["target_features"].double()
    base_bandwidth = median_squared_distance(source)
    causal = summarize_causal_records(result["records"])
    scalar = result["cold_query"]["scalar_mass"]
    samples_per_class = int(result["protocol"]["atlas_samples_per_class"])
    source_class = torch.arange(len(source)) // samples_per_class
    target_class = torch.arange(len(target))
    per_scale = {}
    for scale in scales:
        method = f"sample_dinov2_rbf_scale_{scale:g}"
        if method not in result["cold_query"]:
            raise ValueError(f"retained result does not contain {method}")
        kernel = rbf_kernel(
            source,
            target,
            bandwidth_squared=base_bandwidth * scale**2,
        )
        probabilities = kernel / kernel.sum(dim=0, keepdim=True).clamp_min(1e-300)
        effective_neighbors = probabilities.square().sum(dim=0).reciprocal()
        maximum_weight = probabilities.max(dim=0).values
        same_class = source_class[:, None] == target_class[None, :]
        same_class_mass = (probabilities * same_class).sum(dim=0)
        metrics = result["cold_query"][method]
        paired = causal["paired_vs_scalar"][method]
        causal_positive = all(
            summary["lower_95"] > 0 for summary in paired.values()
        )
        retrieval_positive = (
            metrics["mean_spearman"] - scalar["mean_spearman"] >= 0.02
            or metrics["mean_topk_recall"] - scalar["mean_topk_recall"] >= 0.03
        )
        per_scale[f"{scale:g}"] = {
            "bandwidth_squared": base_bandwidth * scale**2,
            "variance_relative_to_narrow": (scale / narrow_scale) ** 2,
            "cold_query": {
                key: metrics[key]
                for key in (
                    "mean_spearman",
                    "mean_topk_recall",
                    "mean_ndcg",
                    "mean_cosine",
                )
            },
            "cold_delta_vs_scalar": {
                "mean_spearman": metrics["mean_spearman"]
                - scalar["mean_spearman"],
                "mean_topk_recall": metrics["mean_topk_recall"]
                - scalar["mean_topk_recall"],
                "mean_ndcg": metrics["mean_ndcg"] - scalar["mean_ndcg"],
            },
            "kernel_neighborhood": {
                "effective_neighbors": _distribution(effective_neighbors),
                "maximum_normalized_weight": _distribution(maximum_weight),
                "same_class_kernel_mass": _distribution(same_class_mass),
            },
            "paired_causal_vs_scalar": paired,
            "passes_small_screen": bool(retrieval_positive and causal_positive),
            "screen_components": {
                "retrieval_positive": bool(retrieval_positive),
                "causal_positive_at_both_budgets": bool(causal_positive),
            },
        }
    broader_passes = [
        scale
        for scale in scales
        if scale > narrow_scale and per_scale[f"{scale:g}"]["passes_small_screen"]
    ]
    return {
        "setting": "broader_rbf_development_screen",
        "data_role": "development; no confirmation selection",
        "base_median_squared_distance": base_bandwidth,
        "narrow_reference_scale": narrow_scale,
        "scales": list(scales),
        "screen_gate": (
            "at least +.02 Spearman or +.03 top-5% recall over scalar, and "
            "paired causal 95% intervals above zero at both budgets"
        ),
        "broader_scales_passing": broader_passes,
        "per_scale": per_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--narrow-scale", type=float, default=0.1)
    args = parser.parse_args()
    tensors = torch.load(args.tensors, map_location="cpu", weights_only=True)
    result = json.loads(args.result.read_text())
    analysis = analyze(
        tensors,
        result,
        scales=tuple(args.scales),
        narrow_scale=args.narrow_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, analysis)


if __name__ == "__main__":
    main()
