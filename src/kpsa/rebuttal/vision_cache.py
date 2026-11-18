"""Cache everything the vision retrieval-baseline comparison needs.

Reproduces the Figure 5 / 6,400-source neuron-localization setting
(vision_normalization_ablation_v8): 200 evenly spaced ImageNet classes,
sources at within-class offsets 0..31 (offset-major, so every prefix of
length 200*k is class balanced), prototype-construction images at offsets
43..46, and queries at offsets 47, 48 and 49 (49 is the paper's
confirmation offset). For every image we store the DINOv2-S query
representation; for sources and queries the exact normalized sensitivity
profile restricted to the 36,864 coupled MLP features (normalized over the
complete partition, as in the paper) and its raw total energy; for queries
additionally the zero-deactivation gradient-times-activation scores and the
intact margin.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..parameter_groups import ParameterPartition
from ..representation_sensitivity import partition_gradient_energy
from ..vision_causal_refined import (
    mean_ablation_taylor_scores,
    mlp_feature_layout,
    mlp_feature_scope,
)
from ..vision_sample_refined import encode_indices

MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"


def class_starts(dataset_targets, classes=200, total_classes=1000):
    selected = torch.linspace(0, total_classes - 1, classes).round().long().unique()
    targets = torch.as_tensor(dataset_targets)
    starts = []
    for label in selected.tolist():
        where = (targets == label).nonzero().flatten()
        if len(where) != 50 or int(where[-1] - where[0]) != 49:
            raise RuntimeError("validation split must hold 50 contiguous images per class")
        starts.append(int(where[0]))
    return selected, torch.tensor(starts)


def offset_major(starts, offsets):
    offsets = torch.as_tensor(list(offsets), dtype=torch.long)
    return (offsets[:, None] + starts[None, :]).reshape(-1)


def profile_scope(model, dataset, indices, partition, scope):
    """Exact per-image profiles (paper's loop), keeping scope rows + totals."""
    device = next(model.parameters()).device
    scope_index = scope.nonzero().flatten()
    profiles = torch.empty(len(indices), len(scope_index), dtype=torch.float32)
    totals = torch.empty(len(indices), dtype=torch.float64)
    scope_mass = torch.empty(len(indices), dtype=torch.float64)
    margins = torch.empty(len(indices), dtype=torch.float32)
    correct = torch.empty(len(indices), dtype=torch.bool)
    started = time.perf_counter()
    for row, index in enumerate(indices.tolist()):
        image, label = dataset[index]
        model.zero_grad(set_to_none=True)
        logits = model(image[None].to(device)).float()
        alternatives = logits[0].clone()
        alternatives[int(label)] = -torch.inf
        objective = logits[0, int(label)] - alternatives.max()
        objective.backward()
        energy = partition_gradient_energy(partition).detach().double().cpu()
        total = energy.sum()
        share = energy / total
        profiles[row] = share[scope_index].float()
        totals[row] = total
        scope_mass[row] = share[scope_index].sum()
        margins[row] = float(objective.detach())
        correct[row] = bool(int(logits.argmax(1)) == int(label))
        if row % 500 == 0:
            print(f"  profiled {row}/{len(indices)}", flush=True)
    model.zero_grad(set_to_none=True)
    return profiles, totals, scope_mass, margins, correct, time.perf_counter() - started


def attribution_scope(model, dataset, indices, partition, layout, scope):
    zero = {
        item["name"]: torch.zeros(item["count"], device=partition.device)
        for item in layout
    }
    scope_index = scope.nonzero().flatten()
    out = torch.empty(len(indices), len(scope_index), dtype=torch.float32)
    started = time.perf_counter()
    for row, index in enumerate(indices.tolist()):
        image, label = dataset[index]
        scores = mean_ablation_taylor_scores(
            model, image, int(label), "margin", partition, layout, zero
        )
        out[row] = scores[scope_index].float()
    return out, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path.home() / "main/data/imagenet-1k-256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-offsets", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(0)

    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    raw = ImageFolder(args.data / "validation")
    selected, starts = class_starts(raw.targets)
    roles = {
        "source": offset_major(starts, range(args.source_offsets)),
        "prototype": offset_major(starts, range(43, 47)),
        "query47": offset_major(starts, [47]),
        "query48": offset_major(starts, [48]),
        "query49": offset_major(starts, [49]),
    }
    flat = torch.cat(list(roles.values()))
    if len(set(flat.tolist())) != len(flat):
        raise RuntimeError("roles overlap")

    encoder = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).cuda().eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
    t0 = time.perf_counter()
    features = {name: encode_indices(encoder, processor, raw, idx) for name, idx in roles.items()}
    encode_seconds = time.perf_counter() - t0
    print(f"encoded in {encode_seconds:.1f}s", flush=True)
    del encoder
    torch.cuda.empty_cache()

    model = timm.create_model(MODEL, pretrained=True).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()

    payload = {
        "classes": selected,
        "class_starts": starts,
        "roles": roles,
        "features": features,
        "labels": {name: torch.tensor([raw.targets[int(i)] for i in idx]) for name, idx in roles.items()},
        "scope_sizes": sizes[scope].float(),
        "timing": {"dinov2_encode_seconds_all_roles": encode_seconds,
                   "images_encoded": len(flat)},
    }
    for name in ("source", "query47", "query48", "query49"):
        prof, tot, mass, marg, corr, secs = profile_scope(model, dataset, roles[name], partition, scope)
        payload[f"{name}_profiles"] = prof
        payload[f"{name}_totals"] = tot
        payload[f"{name}_scope_mass"] = mass
        payload[f"{name}_margins"] = marg
        payload[f"{name}_correct"] = corr
        payload["timing"][f"{name}_profile_seconds"] = secs
        print(f"{name}: {len(prof)} profiles in {secs:.1f}s", flush=True)
        if name.startswith("query"):
            attr, asecs = attribution_scope(model, dataset, roles[name], partition, layout, scope)
            payload[f"{name}_attribution"] = attr
            payload["timing"][f"{name}_attribution_seconds"] = asecs
    payload["torch"] = torch.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
