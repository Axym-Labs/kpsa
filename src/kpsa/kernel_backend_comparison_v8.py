"""Matched-dimension kernel backend comparison from retained ViT profiles."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .kernel_sensitivity import (
    LinearKernelMap,
    fit_anchor_nystrom_rbf,
    fit_nystrom_rbf,
    fit_tensor_sketch_polynomial,
    median_squared_distance,
    rbf_kernel,
)
from .refined_analysis import paired_t_summary


def prepare_retained_profiles(
    *,
    data_path: Path,
    template_path: Path,
    output: Path,
    source_count: int,
    target_offsets: tuple[int, ...],
):
    """Profile the current balanced source default once for backend studies."""
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    from .parameter_groups import ParameterPartition
    from .vision_sample_refined import encode_indices, profile_functional_images

    template = torch.load(template_path, map_location="cpu", weights_only=True)
    starts = torch.unique(
        template["representation_indices"].long() // 50 * 50,
        sorted=True,
    )
    classes = len(starts)
    if source_count % classes:
        raise ValueError("source count must contain balanced class rounds")
    offsets = torch.arange(source_count // classes)
    source_indices = (offsets[:, None] + starts[None, :]).reshape(-1)
    target_indices = {
        offset: starts + offset for offset in target_offsets
    }

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    ).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(data_path / "validation", transform=transform)
    raw_dataset = ImageFolder(data_path / "validation")
    partition = ParameterPartition(model, "swiglu")
    source_profiles, source_error, source_seconds = profile_functional_images(
        model,
        dataset,
        source_indices,
        partition,
        "margin",
        "normalized",
    )
    target_payloads = {}
    max_error = source_error
    target_seconds = 0.0
    for offset, indices in target_indices.items():
        profiles, error, seconds = profile_functional_images(
            model,
            dataset,
            indices,
            partition,
            "margin",
            "normalized",
        )
        target_payloads[str(offset)] = {
            "target_profiles": profiles,
            "target_indices": indices,
        }
        max_error = max(max_error, error)
        target_seconds += seconds

    encoder = AutoModel.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    ).cuda().eval()
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small", local_files_only=True
    )
    all_indices = torch.cat(
        [source_indices, *(target_indices[offset] for offset in target_offsets)]
    )
    encoded = encode_indices(encoder, processor, raw_dataset, all_indices)
    source_features = encoded[:source_count]
    cursor = source_count
    for offset in target_offsets:
        target_payloads[str(offset)]["target_features"] = encoded[
            cursor : cursor + classes
        ]
        cursor += classes
    class_profiles = source_profiles.reshape(
        len(source_profiles), source_count // classes, classes
    ).mean(1)
    payload = {
        "source_profiles": source_profiles,
        "source_features": source_features,
        "class_source_profiles": class_profiles,
        "source_indices": source_indices,
        "target_splits": target_payloads,
        "protocol": {
            "classes": classes,
            "source_examples": source_count,
            "source_offsets": offsets.tolist(),
            "target_offsets": list(target_offsets),
        },
        "fidelity": {"max_partition_relative_error": max_error},
        "timing": {
            "source_profile_seconds": source_seconds,
            "target_profile_seconds": target_seconds,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    del model, encoder, encoded, source_profiles
    gc.collect()
    torch.cuda.empty_cache()


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=0)
    ranks = torch.empty_like(values)
    sequence = torch.arange(
        len(values), device=values.device, dtype=values.dtype
    )[:, None]
    ranks.scatter_(0, order, sequence.expand_as(order))
    return ranks


def _spearman(predicted: torch.Tensor, target_ranks: torch.Tensor) -> list[float]:
    predicted_ranks = _ranks(predicted)
    predicted_ranks -= predicted_ranks.mean(0, keepdim=True)
    target = target_ranks - target_ranks.mean(0, keepdim=True)
    numerator = (predicted_ranks * target).sum(0)
    denominator = predicted_ranks.square().sum(0).sqrt() * target.square().sum(
        0
    ).sqrt()
    return (numerator / denominator.clamp_min(1e-30)).cpu().tolist()


def _top_recall(
    predicted: torch.Tensor,
    target_top: torch.Tensor,
    count: int,
) -> list[float]:
    selected = predicted.topk(count, dim=0).indices.T
    target = target_top[:count].T
    values = []
    for left, right in zip(selected.tolist(), target.tolist()):
        values.append(len(set(left) & set(right)) / count)
    return values


def _coverage(
    predicted: torch.Tensor,
    target: torch.Tensor,
    count: int,
) -> list[float]:
    selected = predicted.topk(count, dim=0).indices
    return target.gather(0, selected).sum(0).cpu().tolist()


def _summary(values):
    return paired_t_summary([float(value) for value in values])


def _evaluate(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_ranks: torch.Tensor,
    target_top: torch.Tensor,
):
    return {
        "spearman": _summary(_spearman(predicted, target_ranks)),
        "top_37_recall": _summary(_top_recall(predicted, target_top, 37)),
        "coverage_8": _summary(_coverage(predicted, target, 8)),
        "coverage_37": _summary(_coverage(predicted, target, 37)),
    }


def _query_from_features(source_profiles, source_features, target_features):
    atlas = source_profiles @ source_features / source_profiles.shape[1]
    return atlas @ target_features.T


def run_comparison(
    *,
    retained_path: Path,
    output: Path,
    ranks: tuple[int, ...],
    target_offset: int | None,
):
    retained = torch.load(retained_path, map_location="cpu", weights_only=True)
    if "target_splits" in retained:
        if target_offset is None:
            raise ValueError("target offset is required for a multi-split cache")
        target = retained["target_splits"][str(target_offset)]
        target_profiles_cpu = target["target_profiles"]
        target_representations = target["target_features"].double()
    else:
        target_profiles_cpu = retained["target_profiles"]
        target_representations = retained["target_features"].double()
    source_profiles = retained["source_profiles"].float().cuda()
    target_profiles = target_profiles_cpu.float().cuda()
    source_representations = retained["source_features"].double()
    target_ranks = _ranks(target_profiles)
    target_top = target_profiles.topk(max(37, max(ranks)), dim=0).indices

    median = median_squared_distance(source_representations)
    bandwidth = median * 0.1**2
    exact_kernel = rbf_kernel(
        source_representations,
        target_representations,
        bandwidth_squared=bandwidth,
    ).float()
    results = {}
    timings = {}

    def record(name, predicted, *, dimension, kernel_error, reference_kernel):
        results[name] = {
            "family": name.rsplit("_", 1)[0],
            "dimension": dimension,
            "atlas_bytes_float32": int(len(source_profiles) * dimension * 4),
            "kernel_relative_frobenius_error": kernel_error,
            "kernel_error_reference": reference_kernel,
            "metrics": _evaluate(
                predicted,
                target_profiles,
                target_ranks,
                target_top,
            ),
        }

    started = time.perf_counter()
    exact_scores = source_profiles @ exact_kernel.cuda() / source_profiles.shape[1]
    timings["exact_rbf_query_seconds"] = time.perf_counter() - started
    record(
        "exact_rbf",
        exact_scores,
        dimension=len(source_representations),
        kernel_error=0.0,
        reference_kernel="RBF scale 0.1",
    )
    record(
        "scalar",
        source_profiles.mean(1, keepdim=True).expand_as(target_profiles),
        dimension=1,
        kernel_error=None,
        reference_kernel="constant",
    )
    cosine = source_representations @ target_representations.T
    record(
        "nearest",
        source_profiles[:, cosine.argmax(0).cuda()],
        dimension=len(source_representations),
        kernel_error=None,
        reference_kernel="nearest sample",
    )
    record(
        "categorical",
        retained["class_source_profiles"].float().cuda(),
        dimension=retained["class_source_profiles"].shape[1],
        kernel_error=None,
        reference_kernel="class delta",
    )

    linear = LinearKernelMap(
        input_dimension=source_representations.shape[1], normalize_inputs=True
    )
    linear_source = linear.transform(source_representations).float()
    linear_target = linear.transform(target_representations).float()
    record(
        "linear_384",
        _query_from_features(
            source_profiles, linear_source.cuda(), linear_target.cuda()
        ),
        dimension=linear.dimension,
        kernel_error=0.0,
        reference_kernel="linear cosine",
    )

    for rank in ranks:
        for family, fitter in (
            ("kmeans_nystrom", fit_nystrom_rbf),
            ("anchor_nystrom", fit_anchor_nystrom_rbf),
        ):
            started = time.perf_counter()
            feature_map = fitter(
                source_representations,
                rank,
                bandwidth_squared=bandwidth,
                seed=91_027 + rank,
            )
            source_features = feature_map.transform(source_representations).float()
            target_features = feature_map.transform(target_representations).float()
            approximate_kernel = source_features @ target_features.T
            kernel_error = float(
                (approximate_kernel - exact_kernel).norm()
                / exact_kernel.norm().clamp_min(1e-30)
            )
            predicted = _query_from_features(
                source_profiles,
                source_features.cuda(),
                target_features.cuda(),
            )
            timings[f"{family}_{rank}_seconds"] = time.perf_counter() - started
            record(
                f"{family}_{rank}",
                predicted,
                dimension=rank,
                kernel_error=kernel_error,
                reference_kernel="RBF scale 0.1",
            )

        started = time.perf_counter()
        tensor = fit_tensor_sketch_polynomial(
            source_representations.shape[1],
            rank,
            offset=1.0,
            seed=91_027 + rank,
        )
        source_features = tensor.transform(source_representations).float()
        target_features = tensor.transform(target_representations).float()
        normalized_source = torch.nn.functional.normalize(
            source_representations, dim=1
        )
        normalized_target = torch.nn.functional.normalize(
            target_representations, dim=1
        )
        polynomial_kernel = (1 + normalized_source @ normalized_target.T).square()
        approximate_kernel = source_features @ target_features.T
        kernel_error = float(
            (approximate_kernel - polynomial_kernel).norm()
            / polynomial_kernel.norm().clamp_min(1e-30)
        )
        predicted = _query_from_features(
            source_profiles,
            source_features.cuda(),
            target_features.cuda(),
        )
        timings[f"tensor_sketch_{rank}_seconds"] = time.perf_counter() - started
        record(
            f"tensor_sketch_{rank}",
            predicted,
            dimension=rank,
            kernel_error=kernel_error,
            reference_kernel="degree-2 polynomial",
        )

    payload = {
        "setting": "vitb16_matched_dimension_kernel_backend_comparison",
        "protocol": {
            "source_examples": source_profiles.shape[1],
            "queries": target_profiles.shape[1],
            "parameter_groups": source_profiles.shape[0],
            "ranks": list(ranks),
            "rbf_scale": 0.1,
            "rbf_bandwidth_squared": bandwidth,
            "tensor_sketch_kernel": "(1 + cosine_similarity)^2",
            "anchor_net": (
                "two-level scrambled Halton anchors in the full normalized "
                "representation box, followed by nearest-data landmarks"
            ),
            "selection_metric": (
                "held-out direct-sensitivity coverage; exact intervention "
                "confirmation required only if a new backend wins"
            ),
            "target_offset": target_offset,
        },
        "results": results,
        "timing": timings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, payload)


def run_causal_confirmation(
    *,
    retained_path: Path,
    data_path: Path,
    output: Path,
    target_offset: int,
    rank: int,
    queries: int,
):
    """Confirm the selected compact backend under the exact neuron action."""
    import timm
    from torchvision.datasets import ImageFolder

    from .parameter_groups import ParameterPartition
    from .vision_atlas_size_scaling_v8 import evaluate_causal, prepare_queries
    from .vision_causal_refined import (
        calibrate_activation_means,
        mlp_feature_layout,
        mlp_feature_scope,
    )

    retained = torch.load(retained_path, map_location="cpu", weights_only=True)
    target = retained["target_splits"][str(target_offset)]
    source_profiles = retained["source_profiles"].double()
    source_features = retained["source_features"].double()
    target_profiles = target["target_profiles"].double()
    target_features = target["target_features"].double()
    source_count = source_profiles.shape[1]
    bandwidth = median_squared_distance(source_features) * 0.1**2
    feature_map = fit_nystrom_rbf(
        source_features,
        rank,
        bandwidth_squared=bandwidth,
        seed=91_027 + rank,
    )
    nystrom_source = feature_map.transform(source_features).float().cuda()
    nystrom_target = feature_map.transform(target_features).float().cuda()
    profiles_gpu = source_profiles.float().cuda()
    nystrom = _query_from_features(
        profiles_gpu,
        nystrom_source,
        nystrom_target,
    ).double().cpu()
    exact_kernel = rbf_kernel(
        source_features,
        target_features,
        bandwidth_squared=bandwidth,
    ).float().cuda()
    exact = (profiles_gpu @ exact_kernel / source_count).double().cpu()
    cosine = source_features @ target_features.T
    predictions = {
        f"nystrom_{rank}": nystrom,
        "exact_rbf": exact,
        "categorical": retained["class_source_profiles"].double(),
        "nearest": source_profiles[:, cosine.argmax(0)],
        "scalar": source_profiles.mean(1, keepdim=True).expand_as(target_profiles),
        "direct": target_profiles,
    }
    del profiles_gpu, exact_kernel, nystrom_source, nystrom_target
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
    sizes = partition.sizes.detach().double().cpu()
    starts = torch.unique(
        retained["source_indices"].long() // 50 * 50,
        sorted=True,
    )
    means = calibrate_activation_means(model, dataset, starts + 47, layout)
    prepared = prepare_queries(
        model,
        dataset,
        target["target_indices"],
        partition,
        queries,
    )
    records, _, _ = evaluate_causal(
        model,
        prepared,
        predictions,
        partition,
        layout,
        scope,
        sizes,
        means,
        (0.0002, 0.001),
        comparison_pairs=(),
        scalar_method="scalar",
    )
    retained_methods = {
        f"nystrom_{rank}",
        "exact_rbf",
        "categorical",
        "nearest",
        "scalar",
        "direct",
    }
    records = [row for row in records if row["method"] in retained_methods]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        output,
        {
            "setting": "vitb16_n6400_kernel_backend_causal_confirmation",
            "protocol": {
                "source_examples": source_count,
                "target_offset": target_offset,
                "causal_queries": len(prepared),
                "selected_backend": f"kmeans++ Nystrom rank {rank}",
                "fractions": [0.0002, 0.001],
                "intervention": (
                    "source-calibration mean ablation at coupled MLP fc2 inputs"
                ),
            },
            "records": records,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--target-offset", type=int)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--source-count", type=int, default=6400)
    parser.add_argument("--target-offsets", type=int, nargs="+", default=[48, 49])
    parser.add_argument("--causal-output", type=Path)
    parser.add_argument("--causal-data", type=Path)
    parser.add_argument("--causal-target-offset", type=int, default=49)
    parser.add_argument("--causal-rank", type=int, default=400)
    parser.add_argument("--causal-queries", type=int, default=100)
    args = parser.parse_args()
    seed_everything(91_027)
    if args.data:
        if args.template is None:
            raise ValueError("--template is required with --data")
        prepare_retained_profiles(
            data_path=args.data,
            template_path=args.template,
            output=args.retained,
            source_count=args.source_count,
            target_offsets=tuple(args.target_offsets),
        )
    run_comparison(
        retained_path=args.retained,
        output=args.output,
        ranks=tuple(args.ranks),
        target_offset=args.target_offset,
    )
    if args.causal_output:
        if args.causal_data is None:
            raise ValueError("--causal-data is required with --causal-output")
        run_causal_confirmation(
            retained_path=args.retained,
            data_path=args.causal_data,
            output=args.causal_output,
            target_offset=args.causal_target_offset,
            rank=args.causal_rank,
            queries=args.causal_queries,
        )


if __name__ == "__main__":
    main()
