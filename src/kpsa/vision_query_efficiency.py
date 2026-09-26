"""Measure online query cost and storage for the compressed vision atlas."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .parameter_groups import ParameterPartition
from .representation_sensitivity import (
    partition_gradient_energy,
    profile_ranking_metrics,
)


def nystrom_state(payload, *, landmarks=200, scale=0.1):
    source_features = F.normalize(payload["source_features"].float(), dim=1)
    target_features = F.normalize(payload["target_features"].float(), dim=1)
    source_distance = torch.cdist(source_features, source_features).square()
    bandwidth = source_distance[source_distance > 0].median()
    indices = torch.linspace(0, len(source_features) - 1, landmarks).round().long().unique()
    landmark_kernel = torch.exp(
        -source_distance[indices][:, indices] / (2 * bandwidth * scale**2)
    ).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(landmark_kernel)
    inverse_root = (
        eigenvectors * eigenvalues.clamp_min(1e-6).rsqrt()
    ) @ eigenvectors.T
    source_map = torch.exp(
        -source_distance[:, indices] / (2 * bandwidth * scale**2)
    ).double() @ inverse_root
    group_embedding = (
        payload["source_profiles"].double() @ source_map / len(source_features)
    )
    target_map = torch.exp(
        -torch.cdist(target_features, source_features[indices]).square()
        / (2 * bandwidth * scale**2)
    ).double() @ inverse_root
    return {
        "landmark_features": source_features[indices],
        "bandwidth": bandwidth,
        "inverse_root": inverse_root,
        "group_embedding": group_embedding,
        "target_map": target_map,
    }


def timed_cuda(function, *, warmup=5, repeats=25):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    values = torch.tensor(samples)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(values.median()),
        "q10_ms": float(values.quantile(0.1)),
        "q90_ms": float(values.quantile(0.9)),
        "repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmarks", type=int, default=200)
    parser.add_argument("--rbf-scale", type=float, default=0.1)
    parser.add_argument(
        "--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k"
    )
    parser.add_argument("--representation-model", default="facebook/dinov2-small")
    args = parser.parse_args()
    seed_everything(23_041)
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    payload = torch.load(args.atlas, map_location="cpu", weights_only=True)
    state = nystrom_state(
        payload, landmarks=args.landmarks, scale=args.rbf_scale
    )
    float_prediction = state["group_embedding"] @ state["target_map"].T
    bf16_prediction = (
        state["group_embedding"].cuda().bfloat16()
        @ state["target_map"].cuda().bfloat16().T
    ).float().cpu()
    target_profiles = payload["target_profiles"].double()
    cold = {
        "float32": profile_ranking_metrics(float_prediction, target_profiles),
        "bfloat16": profile_ranking_metrics(bf16_prediction.double(), target_profiles),
    }

    target_model = timm.create_model(args.model, pretrained=True).cuda().eval()
    partition = ParameterPartition(target_model, "swiglu")
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(target_model), is_training=False
    )
    target_dataset = ImageFolder(args.data / "validation", transform=transform)
    raw_dataset = ImageFolder(args.data / "validation")
    representation_model = AutoModel.from_pretrained(
        args.representation_model, local_files_only=True
    ).cuda().eval()
    processor = AutoImageProcessor.from_pretrained(
        args.representation_model, local_files_only=True
    )
    target_index = int(payload["target_indices"][0])
    target_image, target_label = target_dataset[target_index]
    raw_image = raw_dataset[target_index][0]
    pixels = processor(images=[raw_image], return_tensors="pt")["pixel_values"].cuda()
    group_embedding = state["group_embedding"].cuda().bfloat16()
    landmarks = state["landmark_features"].cuda()
    inverse_root = state["inverse_root"].cuda().float()
    bandwidth = state["bandwidth"].cuda()
    pooled_holder = {}

    def encode():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = representation_model(pixel_values=pixels)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        pooled_holder["value"] = F.normalize(pooled.float(), dim=1)

    def atlas_query():
        pooled = pooled_holder["value"]
        kernel = torch.exp(
            -torch.cdist(pooled, landmarks).square()
            / (2 * bandwidth * args.rbf_scale**2)
        )
        query_map = kernel @ inverse_root
        return group_embedding @ query_map.bfloat16().T

    def encode_and_query():
        encode()
        atlas_query()

    label = int(target_label)

    def direct_gradient():
        target_model.zero_grad(set_to_none=True)
        logits = target_model(target_image[None].cuda()).float()
        alternatives = logits[0].clone()
        alternatives[label] = -torch.inf
        margin = logits[0, label] - alternatives.max()
        margin.backward()
        partition_gradient_energy(partition)

    encode()
    timing = {
        "frozen_encoder_only": timed_cuda(encode),
        "atlas_query_after_encoding": timed_cuda(atlas_query),
        "encoder_plus_atlas_query": timed_cuda(encode_and_query),
        "direct_target_gradient_profile": timed_cuda(
            direct_gradient, warmup=2, repeats=20
        ),
    }
    dimensions = state["group_embedding"].shape
    storage = {
        "groups": dimensions[0],
        "embedding_dimension": dimensions[1],
        "float32_megabytes": state["group_embedding"].numel() * 4 / 1e6,
        "bfloat16_megabytes": state["group_embedding"].numel() * 2 / 1e6,
        "full_source_profile_float32_megabytes": payload["source_profiles"].numel()
        * 4
        / 1e6,
        "target_model_float32_megabytes": sum(
            parameter.numel() for parameter in target_model.parameters()
        )
        * 4
        / 1e6,
    }
    result = {
        "protocol": {
            "device": torch.cuda.get_device_name(),
            "model": args.model,
            "representation_model": args.representation_model,
            "landmarks": dimensions[1],
            "rbf_scale": args.rbf_scale,
            "timing_scope": "preprocessed single-image GPU execution with synchronization",
        },
        "storage": storage,
        "cold_query_precision_check": {
            precision: {
                key: value
                for key, value in metrics.items()
                if key.startswith("mean_")
            }
            for precision, metrics in cold.items()
        },
        "timing": timing,
    }
    save_json(args.output, result)


if __name__ == "__main__":
    main()
