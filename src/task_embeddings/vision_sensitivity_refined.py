"""Modern ViT experiments for representation-conditioned parameter sensitivity."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .qwen_sensitivity import cold_fold_predictions
from .representation_sensitivity import (
    group_relative_shares,
    partition_gradient_energy,
    profile_ranking_metrics,
)


@dataclass(frozen=True)
class VisionSensitivityProfiles:
    task_profiles: torch.Tensor
    wall_seconds: float
    max_partition_relative_error: float


def profile_class_sensitivity(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    partition,
    *,
    functional: str,
) -> VisionSensitivityProfiles:
    """Measure one complete group-relative profile per labeled image."""
    if functional not in {"class_logit", "margin", "loss"}:
        raise ValueError("unsupported scalar functional")
    if len(images) != len(labels) or not len(images):
        raise ValueError("images and labels must be nonempty and aligned")
    profiles = torch.zeros(partition.n_groups, len(images), dtype=torch.float64)
    device = next(model.parameters()).device
    max_relative_error = 0.0
    started = time.perf_counter()
    model.eval()
    for column, (image, label) in enumerate(zip(images, labels)):
        model.zero_grad(set_to_none=True)
        logits = model(image[None].to(device)).float()
        target = int(label)
        if functional == "class_logit":
            objective = logits[0, target]
        elif functional == "margin":
            alternatives = logits[0].clone()
            alternatives[target] = -torch.inf
            objective = logits[0, target] - alternatives.max()
        else:
            objective = F.cross_entropy(
                logits, torch.tensor([target], device=device)
            )
        objective.backward()
        energy = partition_gradient_energy(partition)
        direct = sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        relative_error = float((energy.sum() - direct).abs() / direct.clamp_min(1e-30))
        max_relative_error = max(max_relative_error, relative_error)
        profiles[:, column] = group_relative_shares(
            energy.detach().double().cpu()[None]
        )[0]
    model.zero_grad(set_to_none=True)
    return VisionSensitivityProfiles(
        task_profiles=profiles,
        wall_seconds=time.perf_counter() - started,
        max_partition_relative_error=max_relative_error,
    )


def run_vision_profiles(
    model,
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    folds: int = 4,
    seed: int = 23_041,
) -> tuple[dict, dict[str, object]]:
    """Compare functionals and representations on disjoint class examples."""
    feature_maps = (
        {"frozen_input_encoder": features}
        if isinstance(features, torch.Tensor)
        else dict(features)
    )
    if not feature_maps:
        raise ValueError("at least one representation is required")
    if len(source_images) != len(target_images) or any(
        len(labels) != len(values) for values in feature_maps.values()
    ):
        raise ValueError("source, target, labels, and features must align by class")
    partition = ParameterPartition(model, "swiglu")
    output = {}
    primary_name = "frozen_input_encoder"
    if primary_name not in feature_maps:
        primary_name = next(iter(feature_maps))
    tensors: dict[str, object] = {
        "features": feature_maps[primary_name].detach().float().cpu(),
        "feature_maps": {
            name: values.detach().float().cpu() for name, values in feature_maps.items()
        },
    }
    errors = []
    for functional in ("class_logit", "margin", "loss"):
        source = profile_class_sensitivity(
            model, source_images, labels, partition, functional=functional
        )
        target = profile_class_sensitivity(
            model, target_images, labels, partition, functional=functional
        )
        representation_metrics = {}
        for name, values in feature_maps.items():
            predictions = cold_fold_predictions(
                source.task_profiles, values, seed=seed, folds=folds
            )
            metrics = {
                method: profile_ranking_metrics(
                    predicted, target.task_profiles, top_fraction=0.05
                )
                for method, predicted in predictions.items()
            }
            metrics["direct_gradient_oracle"] = profile_ranking_metrics(
                target.task_profiles, target.task_profiles, top_fraction=0.05
            )
            representation_metrics[name] = metrics
        metrics = representation_metrics[primary_name]
        output[functional] = {
            "cold_query": metrics,
            "representations": representation_metrics,
            "primary_representation": primary_name,
            "replicate": profile_ranking_metrics(
                source.task_profiles, target.task_profiles, top_fraction=0.05
            ),
            "source_seconds": source.wall_seconds,
            "target_seconds": target.wall_seconds,
        }
        tensors[functional] = {
            "source_profiles": source.task_profiles.float(),
            "target_profiles": target.task_profiles.float(),
        }
        errors.extend(
            [source.max_partition_relative_error, target.max_partition_relative_error]
        )
    return (
        {
            "functionals": output,
            "fidelity": {"max_partition_relative_error": max(errors)},
            "groups": partition.n_groups,
        },
        tensors,
    )


def run_imagenet_study(
    model,
    data_root: Path,
    *,
    representation_model=None,
    representation_processor=None,
    representation_source: str | None = None,
    classes: int = 200,
    representation_images: int = 4,
    folds: int = 4,
) -> tuple[dict, dict[str, object]]:
    """Run disjoint representation/source/target profiling on ImageNet classes."""
    import timm
    from sklearn.decomposition import PCA
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    image_directory = data_root / "validation"
    dataset = ImageFolder(image_directory, transform=transform)
    raw_dataset = ImageFolder(image_directory)
    if not 2 <= classes <= len(dataset.classes):
        raise ValueError("class count must fit the ImageNet catalogue")
    selected_classes = (
        torch.linspace(0, len(dataset.classes) - 1, classes).round().long().unique()
    )
    if len(selected_classes) != classes:
        raise RuntimeError("class selection did not produce the requested count")
    by_class = {int(label): [] for label in selected_classes}
    for index, label in enumerate(dataset.targets):
        if label in by_class:
            by_class[label].append(index)
    required = representation_images + 2
    if any(len(indices) < required for indices in by_class.values()):
        raise ValueError("selected classes lack enough disjoint images")
    representation_indices = [
        index
        for label in selected_classes.tolist()
        for index in by_class[label][:representation_images]
    ]
    source_indices = [
        by_class[label][representation_images] for label in selected_classes.tolist()
    ]
    target_indices = [
        by_class[label][representation_images + 1]
        for label in selected_classes.tolist()
    ]

    device = next(model.parameters()).device
    target_cls_features = []
    target_logit_features = []
    model.eval()
    with torch.no_grad():
        for images, _ in DataLoader(
            Subset(dataset, representation_indices), batch_size=32, num_workers=4
        ):
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                feature_output = model.forward_features(images.to(device))
                if isinstance(feature_output, dict):
                    value = feature_output.get(
                        "x_norm_clstoken", next(iter(feature_output.values()))
                    )
                    logits = model(images.to(device))
                else:
                    value = feature_output
                    logits = model.forward_head(feature_output)
            if value.ndim == 3:
                value = value[:, 0]
            target_cls_features.append(value.detach().float().cpu())
            target_logit_features.append(logits.detach().float().cpu())
    target_cls_features = torch.cat(target_cls_features)
    target_logit_features = torch.cat(target_logit_features)

    if representation_model is not None:
        if representation_processor is None or representation_source is None:
            raise ValueError("the frozen representation model needs a processor and source")
        representation_model.eval()
        encoder_features = []
        with torch.no_grad():
            for start in range(0, len(representation_indices), 32):
                batch_indices = representation_indices[start : start + 32]
                raw_images = [raw_dataset[index][0] for index in batch_indices]
                processed = representation_processor(
                    images=raw_images, return_tensors="pt"
                )
                pixels = processed["pixel_values"].to(device)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    encoded = representation_model(pixel_values=pixels)
                value = getattr(encoded, "pooler_output", None)
                if value is None:
                    value = encoded.last_hidden_state[:, 0]
                encoder_features.append(value.detach().float().cpu())
        encoder_features = torch.cat(encoder_features)
    else:
        # Useful for unit-scale/local callers; paper runs provide an independent
        # encoder trained with a representation objective.
        encoder_features = target_cls_features
        representation_source = "target_model_penultimate_cls"

    encoder_full = F.normalize(encoder_features, dim=1).reshape(
        classes, representation_images, encoder_features.shape[1]
    )
    encoder_class_features = F.normalize(encoder_full.mean(1), dim=1)
    target_cls = F.normalize(target_cls_features, dim=1).reshape(
        classes, representation_images, target_cls_features.shape[1]
    )
    target_logit = F.normalize(target_logit_features, dim=1).reshape(
        classes, representation_images, target_logit_features.shape[1]
    )
    pca_dimension = min(32, classes - 1, encoder_features.shape[1])
    projection = PCA(n_components=pca_dimension, random_state=23_041).fit_transform(
        encoder_features.numpy()
    )
    projected = torch.from_numpy(projection).reshape(
        classes, representation_images, pca_dimension
    )
    pca_class_features = F.normalize(projected.mean(1).float(), dim=1)
    class_features = {
        "frozen_input_encoder": encoder_class_features,
        "target_vit_penultimate_cls": F.normalize(target_cls.mean(1), dim=1),
        "target_vit_class_logits": F.normalize(target_logit.mean(1), dim=1),
        f"frozen_input_encoder_pca_{pca_dimension}": pca_class_features,
    }

    def stack(indices):
        examples = [dataset[index] for index in indices]
        return torch.stack([example[0] for example in examples]), torch.tensor(
            [example[1] for example in examples]
        )

    source_images, source_labels = stack(source_indices)
    target_images, target_labels = stack(target_indices)
    if not torch.equal(source_labels, target_labels):
        raise RuntimeError("source and target class orders differ")
    result, tensors = run_vision_profiles(
        model,
        source_images,
        target_images,
        source_labels,
        class_features,
        folds=folds,
    )
    with torch.no_grad():
        source_accuracy = []
        target_accuracy = []
        for images, labels, output in (
            (source_images, source_labels, source_accuracy),
            (target_images, target_labels, target_accuracy),
        ):
            for batch, local_labels in zip(images.split(32), labels.split(32)):
                logits = model(batch.to(device)).float().cpu()
                output.extend((logits.argmax(1) == local_labels).tolist())
    result.update(
        {
            "setting": "imagenet_pretrained_vit",
            "model": {
                "architecture": type(model).__name__,
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            "protocol": {
                "classes": classes,
                "selected_class_indices": selected_classes.tolist(),
                "representation_images_per_class": representation_images,
                "source_gradient_images_per_class": 1,
                "target_gradient_images_per_class": 1,
                "folds": folds,
                "primary_representation": (
                    f"{representation_source} ({encoder_features.shape[1]}D), "
                    "per-image and class-mean L2 normalized"
                ),
                "representation_ablations": [
                    f"target classifier penultimate CLS ({target_cls_features.shape[1]}D)",
                    f"target classifier class-logit vector ({target_logit_features.shape[1]}D)",
                    f"frozen input encoder PCA-{pca_dimension}",
                ],
                "onehot_cold_query": "unavailable by construction",
            },
            "accuracy": {
                "source_top1": sum(source_accuracy) / len(source_accuracy),
                "target_top1": sum(target_accuracy) / len(target_accuracy),
            },
        }
    )
    tensors.update(
        {
            "selected_class_indices": selected_classes,
            "representation_indices": torch.tensor(representation_indices),
            "source_indices": torch.tensor(source_indices),
            "target_indices": torch.tensor(target_indices),
        }
    )
    return result, tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k"
    )
    parser.add_argument("--representation-model", default="facebook/dinov2-small")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=200)
    parser.add_argument("--representation-images", type=int, default=4)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    seed_everything(23_041)
    import timm
    from transformers import AutoImageProcessor, AutoModel

    model = timm.create_model(args.model, pretrained=True).cuda()
    representation_processor = AutoImageProcessor.from_pretrained(
        args.representation_model, local_files_only=True
    )
    representation_model = AutoModel.from_pretrained(
        args.representation_model, local_files_only=True
    ).cuda()
    result, tensors = run_imagenet_study(
        model,
        args.data,
        representation_model=representation_model,
        representation_processor=representation_processor,
        representation_source=args.representation_model,
        classes=args.classes,
        representation_images=args.representation_images,
        folds=args.folds,
    )
    result["model"]["source"] = args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.with_suffix(".json"), result)
    torch.save(tensors, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
