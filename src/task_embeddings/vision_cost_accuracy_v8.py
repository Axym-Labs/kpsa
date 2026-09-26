"""Amortized cost benchmark for the vision sensitivity atlas.

The benchmark compares an optimized, action-matched activation attribution
query with the complete compact-atlas query path.  The latter includes image
preprocessing, a DINOv2-S forward pass, prototype responses, the stored-atlas
matrix product, and selection.  The atlas construction cost is reported
separately so that the break-even query count is explicit.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .vision_atlas_size_scaling_v8 import offset_major_indices
from .vision_causal_refined import (
    mean_ablation_taylor,
    mean_ablation_taylor_scores,
    mlp_feature_layout,
)
from .vision_sample_refined import encode_indices


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "standard_error": float(array.std(ddof=1) / math.sqrt(len(array))),
        "n": len(array),
    }


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure(callable_, repeats: int) -> tuple[dict[str, float], int]:
    times = []
    peaks = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _synchronize()
        started = time.perf_counter()
        callable_()
        _synchronize()
        times.append((time.perf_counter() - started) * 1000)
        peaks.append(torch.cuda.max_memory_allocated())
    return _percentiles(times), int(max(peaks))


def _atlas_query(
    raw_image,
    atlas,
    encoder,
    processor,
    prototypes,
    bandwidth_squared,
):
    pixels = processor(images=[raw_image], return_tensors="pt")[
        "pixel_values"
    ].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = encoder(pixel_values=pixels)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
    value = F.normalize(pooled.float(), dim=1)
    distance = torch.cdist(value, prototypes).square()
    log_response = -distance / (2 * bandwidth_squared)
    response = F.normalize(
        torch.exp(log_response - log_response.max(dim=1, keepdim=True).values),
        dim=1,
    )
    scores = atlas @ response.T
    torch.topk(scores[:, 0], min(96, len(scores)))


def _encoder_batch(images, encoder, processor):
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        encoder(pixel_values=pixels)


def _atlas_query_batch(
    raw_images,
    atlas,
    encoder,
    processor,
    prototypes,
    bandwidth_squared,
):
    pixels = processor(images=raw_images, return_tensors="pt")[
        "pixel_values"
    ].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = encoder(pixel_values=pixels)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
    values = F.normalize(pooled.float(), dim=1)
    distance = torch.cdist(values, prototypes).square()
    log_response = -distance / (2 * bandwidth_squared)
    responses = F.normalize(
        torch.exp(log_response - log_response.max(dim=1, keepdim=True).values),
        dim=1,
    )
    scores = atlas @ responses.T
    torch.topk(scores, min(96, len(scores)), dim=0)


def optimized_zero_activation_attribution(
    model,
    image: torch.Tensor,
    label: int,
    partition: ParameterPartition,
    layout,
) -> torch.Tensor:
    """Compute gradient-times-activation without parameter-gradient storage."""
    captured = {}
    handles = []
    for item in layout:
        name = item["name"]

        def capture(_module, inputs, name=name):
            captured[name] = inputs[0]

        handles.append(item["module"].register_forward_pre_hook(capture))
    try:
        logits = model(image[None].to(partition.device)).float()
        alternatives = logits[0].clone()
        alternatives[label] = -torch.inf
        objective = logits[0, label] - alternatives.max()
        activations = tuple(captured[item["name"]] for item in layout)
        gradients = torch.autograd.grad(objective, activations)
        scores = torch.zeros(partition.n_groups, dtype=torch.float64)
        zero = None
        for item, activation, gradient in zip(layout, activations, gradients):
            if zero is None or zero.shape[-1] != activation.shape[-1]:
                zero = torch.zeros(
                    activation.shape[-1],
                    dtype=activation.dtype,
                    device=activation.device,
                )
            local = mean_ablation_taylor(
                activation.detach().float(), gradient.detach().float(), zero
            )
            start = int(item["offset"])
            scores[start : start + int(item["count"])] = local.double().cpu()
        return scores
    finally:
        for handle in handles:
            handle.remove()


def optimized_zero_activation_attribution_batch(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    partition: ParameterPartition,
    layout,
) -> torch.Tensor:
    """Batched action-matched attributions for independent examples."""
    captured = {}
    handles = []
    for item in layout:
        name = item["name"]

        def capture(_module, inputs, name=name):
            captured[name] = inputs[0]

        handles.append(item["module"].register_forward_pre_hook(capture))
    try:
        logits = model(images.to(partition.device)).float()
        labels = labels.to(logits.device)
        rows = torch.arange(len(labels), device=logits.device)
        alternatives = logits.clone()
        alternatives[rows, labels] = -torch.inf
        objective = (logits[rows, labels] - alternatives.max(1).values).sum()
        activations = tuple(captured[item["name"]] for item in layout)
        gradients = torch.autograd.grad(objective, activations)
        scores = torch.zeros(
            len(images), partition.n_groups, dtype=torch.float32, device="cpu"
        )
        for item, activation, gradient in zip(layout, activations, gradients):
            local = gradient.detach().float() * activation.detach().float()
            dims = tuple(range(1, local.ndim - 1))
            if dims:
                local = local.sum(dim=dims)
            start = int(item["offset"])
            scores[:, start : start + int(item["count"])] = local.cpu()
        torch.topk(scores, min(96, scores.shape[1]), dim=1)
        return scores
    finally:
        for handle in handles:
            handle.remove()


def _load_accuracy(resolution_path: Path, normalization_path: Path) -> dict:
    resolution = json.loads(resolution_path.read_text())
    normalization = json.loads(normalization_path.read_text())
    bundle = resolution["comparisons"]["neuron_deactivation"]["bundle_32"]
    feature = {}
    for fraction, block in normalization["comparisons"].items():
        feature[fraction] = {
            family: block["normalized"][family]["activation_gap_recovered"]
            for family in ("exact", "prototype")
        }
    return {
        "metric": "fraction of activation-attribution gain over scalar recovered",
        "bundle_32_at_parameter_fraction_1_over_12": {
            "exact": bundle["exact_rbf"]["oracle_gap_recovered"],
            "prototype": bundle["prototype_rbf"]["oracle_gap_recovered"],
        },
        "feature_resolution": feature,
        "activation_attribution": 1.0,
        "scalar": 0.0,
    }


def run_benchmark(
    *,
    retained_path: Path,
    data_path: Path,
    resolution_path: Path,
    normalization_path: Path,
    queries: int,
    warmup: int,
) -> dict:
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    retained = torch.load(retained_path, map_location="cpu", weights_only=True)
    starts = torch.unique(retained["representation_indices"].long() // 50 * 50)
    target_indices = offset_major_indices(retained["representation_indices"], [49])
    positions = torch.linspace(0, len(target_indices) - 1, queries).round().long().unique()
    target_indices = target_indices[positions]

    raw_dataset = ImageFolder(data_path / "validation")
    encoder = AutoModel.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    ).cuda().eval()
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    )
    construction_indices = offset_major_indices(
        retained["representation_indices"], range(43, 47)
    )
    construction_features = encode_indices(
        encoder, processor, raw_dataset, construction_indices
    )
    construction_median = median_squared_distance(construction_features)
    feature_map = fit_prototype_response_map(
        construction_features,
        800,
        bandwidth_squared=construction_median * 0.025**2,
        seed=91_027 + 800,
    )
    prototypes = F.normalize(feature_map.prototypes.float().cuda(), dim=1)
    bandwidth_squared = float(feature_map.bandwidth_squared)

    # Actual values do not affect matrix-product timing.  Both stored tensors
    # have exactly the selected float32 atlas shapes.
    generator = torch.Generator(device="cuda").manual_seed(27_1828)
    atlases = {
        "bundle_32": torch.randn(1152, 800, generator=generator, device="cuda"),
        "feature": torch.randn(36864, 800, generator=generator, device="cuda"),
    }
    raw_images = [raw_dataset[int(index)][0] for index in target_indices]

    for _ in range(warmup):
        _atlas_query(
            raw_images[0],
            atlases["feature"],
            encoder,
            processor,
            prototypes,
            bandwidth_squared,
        )
    atlas_timings = {}
    for name, atlas in atlases.items():
        cursor = 0

        def call(atlas=atlas, encoder=encoder, prototypes=prototypes):
            nonlocal cursor
            _atlas_query(
                raw_images[cursor % len(raw_images)],
                atlas,
                encoder,
                processor,
                prototypes,
                bandwidth_squared,
            )
            cursor += 1

        timing, peak = _measure(call, len(raw_images))
        atlas_timings[name] = {
            "milliseconds_per_query": timing,
            "peak_allocated_bytes": peak,
            "persistent_atlas_bytes": int(atlas.numel() * atlas.element_size()),
        }

    # Batched encoder throughput is useful when an analysis contains many images.
    batch_size = min(32, len(raw_images))
    batch = raw_images[:batch_size]

    def batch_encoder(encoder=encoder):
        _encoder_batch(batch, encoder, processor)

    for _ in range(warmup):
        batch_encoder()
    batch_timing, batch_peak = _measure(
        batch_encoder, max(5, len(raw_images) // batch_size)
    )
    atlas_batch = {}
    for name, atlas in atlases.items():

        def batch_atlas(
            atlas=atlas,
            encoder=encoder,
            prototypes=prototypes,
        ):
            _atlas_query_batch(
                batch,
                atlas,
                encoder,
                processor,
                prototypes,
                bandwidth_squared,
            )

        for _ in range(warmup):
            batch_atlas()
        timing, peak = _measure(batch_atlas, 10)
        atlas_batch[name] = {
            "batch_size": batch_size,
            "milliseconds_per_batch": timing,
            "mean_images_per_second": float(
                batch_size / (timing["mean"] / 1000)
            ),
            "peak_allocated_bytes": peak,
        }
    del construction_features, prototypes, atlases, encoder
    torch.cuda.empty_cache()

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    ).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(data_path / "validation", transform=transform)
    samples = [dataset[int(index)] for index in target_indices]
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)

    # Confirm that the optimized baseline matches the implementation used in
    # the causal study before benchmarking it.
    image0, label0 = samples[0]
    optimized = optimized_zero_activation_attribution(
        model, image0, int(label0), partition, layout
    )
    zero_reference = {
        item["name"]: torch.zeros(item["count"], device=partition.device)
        for item in layout
    }
    reference = mean_ablation_taylor_scores(
        model,
        image0,
        int(label0),
        "margin",
        partition,
        layout,
        zero_reference,
    )
    denominator = reference.abs().max().clamp_min(1e-12)
    relative_error = float((optimized - reference).abs().max() / denominator)
    if relative_error > 2e-5:
        raise RuntimeError(f"optimized attribution mismatch: {relative_error:.3e}")

    for _ in range(warmup):
        optimized_zero_activation_attribution(
            model, image0, int(label0), partition, layout
        )
    cursor = 0

    def attribution_query():
        nonlocal cursor
        image, label = samples[cursor % len(samples)]
        optimized_zero_activation_attribution(
            model, image, int(label), partition, layout
        )
        cursor += 1

    attribution_timing, attribution_peak = _measure(
        attribution_query, len(samples)
    )

    batch_images = torch.stack([image for image, _ in samples[:batch_size]])
    batch_labels = torch.tensor([int(label) for _, label in samples[:batch_size]])
    batch_scores = optimized_zero_activation_attribution_batch(
        model, batch_images, batch_labels, partition, layout
    )
    batch_relative_error = float(
        (batch_scores[0].double() - optimized).abs().max() / denominator
    )
    if batch_relative_error > 2e-5:
        raise RuntimeError(
            f"batched attribution mismatch: {batch_relative_error:.3e}"
        )

    def attribution_batch():
        optimized_zero_activation_attribution_batch(
            model, batch_images, batch_labels, partition, layout
        )

    for _ in range(warmup):
        attribution_batch()
    attribution_batch_timing, attribution_batch_peak = _measure(
        attribution_batch, 10
    )

    resolution = json.loads(resolution_path.read_text())
    source_profile_seconds = float(resolution["timing"]["source_profile_seconds"])
    atlas_ms = atlas_timings["bundle_32"]["milliseconds_per_query"]["mean"]
    attribution_ms = attribution_timing["mean"]
    savings = attribution_ms - atlas_ms
    break_even = (
        source_profile_seconds * 1000 / savings if savings > 0 else None
    )
    atlas_batch_ms_per_image = (
        atlas_batch["bundle_32"]["milliseconds_per_batch"]["mean"] / batch_size
    )
    attribution_batch_ms_per_image = (
        attribution_batch_timing["mean"] / batch_size
    )
    batch_savings = attribution_batch_ms_per_image - atlas_batch_ms_per_image
    batch_break_even = (
        source_profile_seconds * 1000 / batch_savings
        if batch_savings > 0
        else None
    )
    return {
        "setting": "vitb16_amortized_atlas_cost_accuracy",
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
        "protocol": {
            "queries": len(samples),
            "warmup_repeats": warmup,
            "atlas_query_includes": [
                "CPU image preprocessing",
                "DINOv2-S forward",
                "800 fixed-prototype responses",
                "stored-atlas matrix product",
                "top-k selection",
            ],
            "activation_attribution": (
                "optimized gradient-times-activation; autograd requested only "
                "intermediate activation gradients"
            ),
            "source_classes": len(starts),
            "source_examples": 6400,
        },
        "correctness": {
            "max_relative_error_vs_causal_study_attribution": relative_error,
            "batch_vs_single_max_relative_error": batch_relative_error,
        },
        "online": {
            "compact_atlas": atlas_timings,
            "activation_attribution": {
                "milliseconds_per_query": attribution_timing,
                "peak_allocated_bytes": attribution_peak,
            },
            "batched_compact_atlas": atlas_batch,
            "batched_activation_attribution": {
                "batch_size": batch_size,
                "milliseconds_per_batch": attribution_batch_timing,
                "mean_images_per_second": float(
                    batch_size / (attribution_batch_timing["mean"] / 1000)
                ),
                "peak_allocated_bytes": attribution_batch_peak,
            },
            "dinov2_batch": {
                "batch_size": batch_size,
                "milliseconds_per_batch": batch_timing,
                "mean_images_per_second": float(
                    batch_size / (batch_timing["mean"] / 1000)
                ),
                "peak_allocated_bytes": batch_peak,
            },
        },
        "offline": {
            "source_profile_seconds": source_profile_seconds,
            "source_profile_seconds_per_example": source_profile_seconds / 6400,
            "break_even_queries_from_profile_cost_only": break_even,
            "batched_break_even_queries_from_profile_cost_only": batch_break_even,
            "note": (
                "The break-even excludes source DINO encoding and one-time "
                "prototype fitting, so it is a lower bound on full construction."
            ),
        },
        "accuracy": _load_accuracy(resolution_path, normalization_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    seed_everything(27_1828)
    result = run_benchmark(
        retained_path=args.input,
        data_path=args.data,
        resolution_path=args.resolution,
        normalization_path=args.normalization,
        queries=args.queries,
        warmup=args.warmup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
