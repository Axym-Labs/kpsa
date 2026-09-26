from __future__ import annotations

import argparse
import copy
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets as tv_datasets
from torchvision import transforms

from .applications import continual_learning_metrics
from .common import (
    EPS,
    OnlineAccumulator,
    accelerator_peak_memory,
    arc_artifact_dir,
    build_task_atlas,
    jl_task_representation,
    layer_balanced_indices,
    normalized_rows,
    random_layer_balanced_scores,
    reset_accelerator_peak_memory,
    save_json,
    seed_everything,
    source_fidelity,
    task_aligned_importance,
)
from .vision import (
    ClassHomogeneousBatchSampler,
    GradientMeanBuffer,
    cross_entropy_residual_norm_sq,
    fine_to_coarse_mapping,
)


def continual_class_groups(fine_to_coarse: Sequence[int]) -> dict[str, list[list[int]]]:
    """Partition 100 fine classes into related and superclass-mixed experiences."""
    buckets: dict[int, list[int]] = {}
    for fine, coarse in enumerate(fine_to_coarse):
        buckets.setdefault(int(coarse), []).append(fine)
    coarse_ids = sorted(buckets)
    related = [sorted(buckets[coarse]) for coarse in coarse_ids]
    mixed_buckets = {coarse: sorted(values) for coarse, values in buckets.items()}
    dissimilar: list[list[int]] = []
    for group_index in range(len(coarse_ids)):
        chosen_coarse = [
            coarse_ids[(group_index * 5 + offset) % len(coarse_ids)]
            for offset in range(5)
        ]
        dissimilar.append([mixed_buckets[coarse].pop(0) for coarse in chosen_coarse])
    if any(mixed_buckets[coarse] for coarse in coarse_ids):
        raise ValueError("expected five fine classes per CIFAR-100 superclass")
    return {"related": related, "dissimilar": dissimilar}


class Dinov2FeatureProbe(nn.Module):
    """DINOv2 classifier with interventions on disjoint MLP features."""

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self._selected: list[torch.Tensor] | None = None
        self._mode: Literal["zero", "mean", "retain"] = "zero"
        self._means: list[torch.Tensor] | None = None
        self._capture = False
        self._captured: list[torch.Tensor | None] = [None] * len(self.mlps)
        self._hook_handles = [
            mlp.activation.register_forward_hook(self._make_feature_hook(layer))
            for layer, mlp in enumerate(self.mlps)
        ]

    @property
    def mlps(self) -> list[nn.Module]:
        return [layer.mlp for layer in self.backbone.encoder.layer]

    @property
    def layer_sizes(self) -> list[int]:
        return [mlp.fc1.out_features for mlp in self.mlps]

    @property
    def n_modules(self) -> int:
        return sum(self.layer_sizes)

    def _make_feature_hook(self, layer: int):
        def hook(
            _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ):
            features = output
            if self._selected is not None:
                selected = self._selected[layer].to(features.device)
                mask = torch.zeros(
                    self.layer_sizes[layer],
                    device=features.device,
                    dtype=features.dtype,
                )
                mask[selected] = 1.0
                if self._mode == "retain":
                    features = features * mask
                else:
                    keep = 1.0 - mask
                    if self._mode == "zero":
                        features = features * keep
                    else:
                        if self._means is None:
                            raise ValueError("mean intervention requires feature means")
                        means = self._means[layer].to(
                            device=features.device, dtype=features.dtype
                        )
                        features = features * keep + means.view(1, 1, -1) * mask
            if self._capture:
                if features.requires_grad:
                    features.retain_grad()
                self._captured[layer] = features
            return features

        return hook

    def set_intervention(
        self,
        selected: Sequence[torch.Tensor] | None,
        *,
        mode: Literal["zero", "mean", "retain"] = "zero",
        means: Sequence[torch.Tensor] | None = None,
    ) -> None:
        self._selected = (
            None if selected is None else [value.detach().cpu() for value in selected]
        )
        self._mode = mode
        self._means = (
            None if means is None else [value.detach().cpu() for value in means]
        )

    def forward(
        self, pixel_values: torch.Tensor, *, capture: bool = False
    ) -> torch.Tensor:
        self._capture = capture
        if capture:
            self._captured = [None] * len(self.mlps)
        outputs = self.backbone(pixel_values, interpolate_pos_encoding=True)
        return self.classifier(outputs.last_hidden_state[:, 0])

    def captured_activations(self) -> list[torch.Tensor]:
        if any(value is None for value in self._captured):
            raise RuntimeError("forward(capture=True) is required")
        return [value for value in self._captured if value is not None]

    def clear_capture(self) -> None:
        self._capture = False
        self._captured = [None] * len(self.mlps)

    def module_owned_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for mlp in self.mlps:
            parameters.append(mlp.fc1.weight)
            if mlp.fc1.bias is not None:
                parameters.append(mlp.fc1.bias)
            parameters.append(mlp.fc2.weight)
        return parameters

    def block_grad_norms(self) -> torch.Tensor:
        scores = []
        for mlp in self.mlps:
            score = mlp.fc1.weight.grad.float().square().sum(dim=1)
            if mlp.fc1.bias is not None:
                score = score + mlp.fc1.bias.grad.float().square()
            score = score + mlp.fc2.weight.grad.float().square().sum(dim=0)
            scores.append(score)
        return torch.cat(scores)

    @torch.no_grad()
    def block_weight_norms(self) -> torch.Tensor:
        scores = []
        for mlp in self.mlps:
            score = mlp.fc1.weight.float().square().sum(dim=1)
            if mlp.fc1.bias is not None:
                score = score + mlp.fc1.bias.float().square()
            score = score + mlp.fc2.weight.float().square().sum(dim=0)
            scores.append(score.sqrt())
        return torch.cat(scores).cpu()

    @torch.no_grad()
    def apply_feature_gradient_scale(
        self, importance: torch.Tensor, strength: float
    ) -> None:
        offset = 0
        for size, mlp in zip(self.layer_sizes, self.mlps):
            local = importance[offset : offset + size].float()
            scale = (1.0 / (1.0 + strength * local)).to(
                device=mlp.fc1.weight.device, dtype=mlp.fc1.weight.grad.dtype
            )
            mlp.fc1.weight.grad.mul_(scale[:, None])
            if mlp.fc1.bias is not None:
                mlp.fc1.bias.grad.mul_(scale)
            mlp.fc2.weight.grad.mul_(scale[None, :])
            offset += size


@dataclass(frozen=True)
class VisionConfig:
    seed: int = 1
    model_name: str = "facebook/dinov2-small"
    data_root: str = "/home/davwis/main/data/cifar-100"
    image_size: int = 224
    train_per_class: int = 100
    epochs: int = 3
    train_steps: int | None = 10000
    batch_size: int = 16
    tasks_per_update: int = 10
    reference_per_class: int = 2
    evaluation_per_class: int = 5
    workers: int = 8
    intervention_fractions: tuple[float, ...] = (0.01, 0.05, 0.10)
    pruning_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    application_tasks: int = 20
    recovery_steps: int = 5
    cl_steps_per_experience: int = 3
    cl_strength: float = 8.0


def _transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, evaluation_transform


def _balanced_indices(labels: Sequence[int], per_class: int, seed: int) -> list[int]:
    labels_array = np.asarray(labels)
    generator = np.random.default_rng(seed)
    output: list[int] = []
    for class_id in sorted(np.unique(labels_array)):
        candidates = np.flatnonzero(labels_array == class_id)
        generator.shuffle(candidates)
        output.extend(candidates[:per_class].tolist())
    return output


def cache_subset(dataset, indices: Sequence[int]) -> TensorDataset:
    images, targets = [], []
    for index in indices:
        image, target = dataset[index]
        images.append(image)
        targets.append(target)
    return TensorDataset(torch.stack(images), torch.tensor(targets, dtype=torch.long))


def semantic_class_representation(
    class_names: Sequence[str], dimension: int = 32
) -> torch.Tensor:
    """Frozen MiniLM class-description embeddings, reduced without sensitivities."""
    from transformers import AutoModel, AutoTokenizer

    model_path = snapshot_download(
        "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    descriptions = [f"an image of a {name.replace('_', ' ')}" for name in class_names]
    batches = []
    with torch.no_grad():
        for start in range(0, len(descriptions), 32):
            encoded = tokenizer(
                descriptions[start : start + 32],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            batches.append(pooled)
    full = torch.cat(batches).float().numpy()
    reduced = PCA(n_components=dimension, random_state=0).fit_transform(full)
    return normalized_rows(torch.from_numpy(reduced).float())


def freeze_interpolated_position_embeddings(backbone: nn.Module) -> None:
    """Remove DINOv2's nondeterministic CUDA bicubic-backward path."""
    backbone.embeddings.position_embeddings.requires_grad_(False)


def load_pretrained_probe(config: VisionConfig) -> Dinov2FeatureProbe:
    from transformers import Dinov2Model

    snapshot = snapshot_download(config.model_name, local_files_only=True)
    backbone = Dinov2Model.from_pretrained(snapshot, local_files_only=True)
    freeze_interpolated_position_embeddings(backbone)
    return Dinov2FeatureProbe(backbone, num_classes=100)


def clone_probe(model: Dinov2FeatureProbe, device: torch.device) -> Dinov2FeatureProbe:
    from transformers import Dinov2Model

    clone = Dinov2FeatureProbe(Dinov2Model(copy.deepcopy(model.backbone.config)), 100)
    clone.load_state_dict(model.state_dict())
    freeze_interpolated_position_embeddings(clone.backbone)
    return clone.to(device)


def train_model(
    model: Dinov2FeatureProbe,
    loader: DataLoader,
    representations: dict[str, torch.Tensor],
    device: torch.device,
    steps: int,
    tasks_per_update: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict]:
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 2e-5},
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.05,
    )
    buffer = GradientMeanBuffer(model.parameters())
    accumulator = OnlineAccumulator(
        model.n_modules,
        representations,
        device,
        beta=0.995,
        late_fraction=0.7,
        total_steps=steps,
    )
    started = time.perf_counter()
    accumulator_seconds = 0.0
    trace = []
    model.train()
    for step, (images, targets) in enumerate(loader):
        if step >= steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        task = int(targets[0])
        if not bool((targets == task).all()):
            raise RuntimeError(
                "online vision accumulation requires homogeneous class batches"
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        scoring_started = time.perf_counter()
        accumulator.update(
            model.block_grad_norms(),
            task,
            cross_entropy_residual_norm_sq(logits.detach(), targets),
            step,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        accumulator_seconds += time.perf_counter() - scoring_started
        buffer.add()
        if (step + 1) % tasks_per_update == 0 or step == steps - 1:
            buffer.apply_mean()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "batch_accuracy": float(
                        (logits.argmax(dim=1) == targets).float().mean()
                    ),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (
        accumulator.finalize(),
        accumulator.amplitudes(),
        {
            "wall_seconds": elapsed,
            "steps": steps,
            "optimizer_updates": math.ceil(steps / tasks_per_update),
            "examples_per_second": steps
            * loader.batch_sampler.batch_size
            / max(elapsed, EPS),
            "online_accumulator_seconds_including_sync": accumulator_seconds,
            "online_accumulator_fraction_upper_bound": accumulator_seconds
            / max(elapsed, EPS),
            "trace": trace,
        },
    )


def posthoc_profiles(
    model: Dinov2FeatureProbe,
    reference: TensorDataset,
    reference_per_class: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[torch.Tensor], dict]:
    rows = {name: [] for name in ("ief", "raw", "activation", "actgrad")}
    task_ids = []
    mean_sums = [torch.zeros(size, device=device) for size in model.layer_sizes]
    mean_counts = [0 for _ in model.layer_sizes]
    started = time.perf_counter()
    model.eval()
    model.set_intervention(None)
    seen = [0] * 100
    for image, target_tensor in reference:
        task = int(target_tensor)
        if seen[task] >= reference_per_class:
            continue
        seen[task] += 1
        model.zero_grad(set_to_none=True)
        image = image[None].to(device)
        target = target_tensor[None].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(image, capture=True)
            loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        residual_norm_sq = cross_entropy_residual_norm_sq(logits.detach(), target)
        raw = model.block_grad_norms().detach().float().cpu()
        activations = model.captured_activations()
        activation = torch.cat(
            [value.detach().float().abs().mean(dim=(0, 1)) for value in activations]
        ).cpu()
        actgrad = torch.cat(
            [
                (value.detach().float() * value.grad.detach().float())
                .abs()
                .mean(dim=(0, 1))
                for value in activations
            ]
        ).cpu()
        for layer, value in enumerate(activations):
            mean_sums[layer].add_(value.detach().float().sum(dim=(0, 1)))
            mean_counts[layer] += value.shape[0] * value.shape[1]
        rows["raw"].append(raw)
        rows["ief"].append(raw / residual_norm_sq.cpu().clamp_min(EPS))
        rows["activation"].append(activation)
        rows["actgrad"].append(actgrad)
        task_ids.append(task)
    tasks = torch.tensor(task_ids)
    atlases, amplitudes = {}, {}
    for name, values in rows.items():
        atlases[name], amplitudes[name] = build_task_atlas(
            torch.stack(values), tasks, n_tasks=100
        )
    means = [
        (total / max(1, count)).detach().cpu()
        for total, count in zip(mean_sums, mean_counts)
    ]
    model.clear_capture()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (
        atlases,
        amplitudes,
        means,
        {
            "wall_seconds": elapsed,
            "samples": len(task_ids),
            "samples_per_second": len(task_ids) / max(elapsed, EPS),
        },
    )


@torch.no_grad()
def evaluate(
    model: Dinov2FeatureProbe,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    loss_sum = torch.zeros(100, device=device)
    correct = torch.zeros(100, device=device)
    counts = torch.zeros(100, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            losses = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
        loss_sum.scatter_add_(0, targets, losses.float())
        correct.scatter_add_(0, targets, (logits.argmax(dim=1) == targets).float())
        counts.scatter_add_(0, targets, torch.ones_like(losses, dtype=torch.float32))
    return (loss_sum / counts.clamp_min(1)).cpu(), (correct / counts.clamp_min(1)).cpu()


def _score_methods(
    model: Dinov2FeatureProbe,
    atlases: dict[str, torch.Tensor],
    amplitudes: dict[str, torch.Tensor],
    online: dict[str, dict[str, torch.Tensor]],
    online_amplitudes: dict[str, torch.Tensor],
    representations: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, torch.Tensor]:
    methods = {}
    for name in ("onehot", "semantic32", "jl32"):
        representation = representations[name]
        methods[f"post_ief_{name}"] = task_aligned_importance(
            atlases["ief"] @ normalized_rows(representation),
            representation,
            amplitudes["ief"],
        )
    for statistic in ("raw", "activation", "actgrad"):
        representation = representations["semantic32"]
        methods[f"post_{statistic}_semantic32"] = task_aligned_importance(
            atlases[statistic] @ representation, representation, amplitudes[statistic]
        )
    methods["online_late_raw_semantic32"] = task_aligned_importance(
        online["late_raw"]["semantic32"],
        representations["semantic32"],
        online_amplitudes["late_raw"],
    )
    methods["weight_magnitude"] = model.block_weight_norms()[:, None].repeat(1, 100)
    methods["random"] = torch.stack(
        [
            random_layer_balanced_scores(model.layer_sizes, seed + task)
            for task in range(100)
        ],
        dim=1,
    )
    return methods


def query_tasks_by_superclass(fine_to_coarse: Sequence[int], limit: int) -> list[int]:
    tasks = []
    for coarse in sorted(set(fine_to_coarse)):
        tasks.append(
            next(fine for fine, value in enumerate(fine_to_coarse) if value == coarse)
        )
    return tasks[:limit]


def interpretability_study(
    model: Dinov2FeatureProbe,
    evaluation: TensorDataset,
    methods: dict[str, torch.Tensor],
    means: list[torch.Tensor],
    fine_to_coarse: Sequence[int],
    config: VisionConfig,
    device: torch.device,
) -> dict:
    baseline_loss, baseline_accuracy = evaluate(model, evaluation, device)
    query_tasks = query_tasks_by_superclass(fine_to_coarse, config.application_tasks)
    records = []
    coarse_tensor = torch.tensor(fine_to_coarse)
    for method_name, scores in methods.items():
        for fraction in config.intervention_fractions:
            for task in query_tasks:
                selected = layer_balanced_indices(
                    scores[:, task], model.layer_sizes, fraction
                )
                for mode in ("zero", "mean"):
                    model.set_intervention(
                        selected, mode=mode, means=means if mode == "mean" else None
                    )
                    _loss, accuracy = evaluate(model, evaluation, device)
                    drop = baseline_accuracy - accuracy
                    same_family = (coarse_tensor == fine_to_coarse[task]) & (
                        torch.arange(100) != task
                    )
                    other = coarse_tensor != fine_to_coarse[task]
                    records.append(
                        {
                            "method": method_name,
                            "fraction": fraction,
                            "task": task,
                            "intervention": mode,
                            "target_accuracy_drop": float(drop[task]),
                            "same_superclass_drop": float(drop[same_family].mean()),
                            "other_class_drop": float(drop[other].mean()),
                            "target_selectivity": float(
                                drop[task] - drop[other].mean()
                            ),
                            "family_selectivity": float(
                                drop[same_family].mean() - drop[other].mean()
                            ),
                        }
                    )
    model.set_intervention(None)
    return {
        "baseline_loss_mean": float(baseline_loss.mean()),
        "baseline_accuracy_mean": float(baseline_accuracy.mean()),
        "query_tasks": query_tasks,
        "records": records,
    }


def _recovery_accuracy(
    base_model: Dinov2FeatureProbe,
    retained: Sequence[torch.Tensor],
    task: int,
    train_dataset,
    train_indices_by_class: dict[int, list[int]],
    evaluation: TensorDataset,
    config: VisionConfig,
    device: torch.device,
) -> float:
    model = clone_probe(base_model, device)
    model.set_intervention(retained, mode="retain")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    subset = Subset(train_dataset, train_indices_by_class[task])
    loader = DataLoader(
        subset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    iterator = iter(loader)
    model.train()
    for _ in range(config.recovery_steps):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss = torch.nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        optimizer.step()
    return float(evaluate(model, evaluation, device)[1][task])


def pruning_study(
    model: Dinov2FeatureProbe,
    evaluation: TensorDataset,
    methods: dict[str, torch.Tensor],
    fine_to_coarse: Sequence[int],
    train_dataset,
    train_indices_by_class: dict[int, list[int]],
    config: VisionConfig,
    device: torch.device,
) -> dict:
    _, baseline_accuracy = evaluate(model, evaluation, device)
    query_tasks = query_tasks_by_superclass(fine_to_coarse, config.application_tasks)
    records = []
    recovery_methods = {"post_ief_semantic32", "post_activation_semantic32", "random"}
    for method_name, scores in methods.items():
        for fraction in config.pruning_fractions:
            for task in query_tasks:
                retained = layer_balanced_indices(
                    scores[:, task], model.layer_sizes, fraction
                )
                model.set_intervention(retained, mode="retain")
                _, accuracy = evaluate(model, evaluation, device)
                record = {
                    "method": method_name,
                    "retained_fraction": fraction,
                    "task": task,
                    "baseline_accuracy": float(baseline_accuracy[task]),
                    "zero_shot_accuracy": float(accuracy[task]),
                    "zero_shot_overall_accuracy": float(accuracy.mean()),
                    "recovered_accuracy": float("nan"),
                }
                if (
                    config.recovery_steps > 0
                    and method_name in recovery_methods
                    and abs(fraction - 0.25) < 1e-8
                    and task in query_tasks[:2]
                ):
                    model.set_intervention(None)
                    record["recovered_accuracy"] = _recovery_accuracy(
                        model,
                        retained,
                        task,
                        train_dataset,
                        train_indices_by_class,
                        evaluation,
                        config,
                        device,
                    )
                records.append(record)
    model.set_intervention(None)
    return {"query_tasks": query_tasks, "records": records}


def continual_learning_study(
    base_model: Dinov2FeatureProbe,
    evaluation: TensorDataset,
    methods: dict[str, torch.Tensor],
    fine_to_coarse: Sequence[int],
    train_dataset,
    train_indices_by_class: dict[int, list[int]],
    config: VisionConfig,
    device: torch.device,
) -> dict:
    groups = continual_class_groups(fine_to_coarse)
    selected_methods = {
        "post_ief": methods["post_ief_semantic32"],
        "online_raw_ef": methods["online_late_raw_semantic32"],
        "raw_ef": methods["post_raw_semantic32"],
        "activation": methods["post_activation_semantic32"],
        "random": methods["random"],
        "none": torch.zeros_like(methods["random"]),
    }
    output = {}
    for order_name, experiences in groups.items():
        output[order_name] = {}
        for method_name, task_scores in selected_methods.items():
            model = clone_probe(base_model, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            protection = torch.zeros(model.n_modules)
            history = torch.full((len(experiences), len(experiences)), float("nan"))
            for stage, classes in enumerate(experiences):
                indices = [
                    index for task in classes for index in train_indices_by_class[task]
                ]
                loader = DataLoader(
                    Subset(train_dataset, indices),
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=0,
                )
                iterator = iter(loader)
                model.train()
                for _ in range(config.cl_steps_per_experience):
                    try:
                        images, targets = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        images, targets = next(iterator)
                    images, targets = images.to(device), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        loss = torch.nn.functional.cross_entropy(model(images), targets)
                    loss.backward()
                    if stage > 0 and method_name != "none":
                        model.apply_feature_gradient_scale(
                            protection, config.cl_strength
                        )
                    optimizer.step()
                _, class_accuracy = evaluate(model, evaluation, device)
                for seen_stage, seen_classes in enumerate(experiences[: stage + 1]):
                    history[stage, seen_stage] = class_accuracy[seen_classes].mean()
                importance = task_scores[:, classes].mean(dim=1)
                importance = importance / importance.mean().clamp_min(EPS)
                protection = torch.maximum(protection, importance)
            output[order_name][method_name] = {
                "experiences": experiences,
                "history": history,
                "summary": continual_learning_metrics(history, higher_is_better=True),
            }
    return output


def run_experiment(config: VisionConfig, *, device: torch.device | None = None) -> dict:
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_started = time.perf_counter()
    reset_accelerator_peak_memory(device)
    train_transform, evaluation_transform = _transforms(config.image_size)
    train_dataset = tv_datasets.CIFAR100(
        config.data_root, train=True, download=False, transform=train_transform
    )
    evaluation_raw = tv_datasets.CIFAR100(
        config.data_root, train=False, download=False, transform=evaluation_transform
    )
    train_indices = _balanced_indices(
        train_dataset.targets, config.train_per_class, config.seed
    )
    train_subset = Subset(train_dataset, train_indices)
    train_labels = [train_dataset.targets[index] for index in train_indices]
    steps = config.train_steps
    if steps is None:
        steps = config.epochs * math.ceil(len(train_indices) / config.batch_size)
    sampler = ClassHomogeneousBatchSampler(
        train_labels, config.batch_size, steps, config.seed
    )
    train_loader = DataLoader(
        train_subset,
        batch_sampler=sampler,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )
    evaluation_indices = _balanced_indices(
        evaluation_raw.targets, config.evaluation_per_class, config.seed + 1
    )
    evaluation = cache_subset(evaluation_raw, evaluation_indices)
    fine_to_coarse = fine_to_coarse_mapping(Path(config.data_root))
    semantic = semantic_class_representation(train_dataset.classes, dimension=32)
    representations = {
        "onehot": torch.eye(100),
        "semantic32": semantic,
        "jl32": jl_task_representation(100, 32, config.seed + 32),
    }
    model = load_pretrained_probe(config).to(device)
    online, online_amplitudes, train_metrics = train_model(
        model, train_loader, representations, device, steps, config.tasks_per_update
    )
    atlases, amplitudes, means, posthoc_metrics = posthoc_profiles(
        model, evaluation, config.reference_per_class, device
    )
    fidelity_representations = {
        "onehot": torch.eye(100),
        "semantic32": semantic,
        **{
            f"jl{dimension}": jl_task_representation(
                100, dimension, config.seed + dimension
            )
            for dimension in (4, 8, 16, 32, 64)
        },
    }
    source_metrics = {
        name: source_fidelity(atlases["ief"], representation)
        for name, representation in fidelity_representations.items()
    }
    methods = _score_methods(
        model,
        atlases,
        amplitudes,
        online,
        online_amplitudes,
        representations,
        config.seed,
    )
    by_class = {
        class_id: [
            index
            for index, target in enumerate(train_dataset.targets)
            if target == class_id
        ]
        for class_id in range(100)
    }
    interpretability = interpretability_study(
        model, evaluation, methods, means, fine_to_coarse, config, device
    )
    pruning = pruning_study(
        model,
        evaluation,
        methods,
        fine_to_coarse,
        train_dataset,
        by_class,
        config,
        device,
    )
    continual = continual_learning_study(
        model,
        evaluation,
        methods,
        fine_to_coarse,
        train_dataset,
        by_class,
        config,
        device,
    )
    _, baseline_accuracy = evaluate(model, evaluation, device)
    return {
        "setting": "vision_dinov2_cifar100",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "model": {
            "requested": "DINOv3 ViT-S",
            "used": config.model_name,
            "fallback_reason": "official DINOv3 repository requires access approval",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "n_modules": model.n_modules,
            "module_definition": "MLP fc1 row plus fc2 column",
            "frozen_parameter": "backbone.embeddings.position_embeddings",
            "deterministic_cuda": True,
        },
        "train": train_metrics,
        "posthoc": posthoc_metrics,
        "efficiency": {
            "total_wall_seconds": time.perf_counter() - run_started,
            **accelerator_peak_memory(device),
        },
        "baseline_accuracy_mean": float(baseline_accuracy.mean()),
        "sanity": {
            "chance_accuracy": 0.01,
            "passes_20pct_accuracy_gate": bool(float(baseline_accuracy.mean()) >= 0.20),
        },
        "source_fidelity": source_metrics,
        "interpretability": interpretability,
        "pruning": pruning,
        "continual_learning": continual,
        "_arrays": {
            **{f"atlas_{name}": value for name, value in atlases.items()},
            **{f"amplitude_{name}": value for name, value in amplitudes.items()},
            "semantic_representation": semantic,
            **{
                f"online_{variant}_{name}": value
                for variant, reps in online.items()
                for name, value in reps.items()
            },
        },
        "_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }


def smoke_config(seed: int) -> VisionConfig:
    return VisionConfig(
        seed=seed,
        train_per_class=2,
        train_steps=10,
        batch_size=2,
        tasks_per_update=5,
        reference_per_class=1,
        evaluation_per_class=1,
        workers=2,
        intervention_fractions=(0.05,),
        pruning_fractions=(0.25,),
        application_tasks=1,
        recovery_steps=0,
        cl_steps_per_experience=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=arc_artifact_dir("02_exploratory", "vision"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()
    config = smoke_config(args.seed) if args.smoke else VisionConfig(seed=args.seed)
    if args.train_steps is not None:
        config = VisionConfig(**{**asdict(config), "train_steps": args.train_steps})
    result = run_experiment(config)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    arrays = result.pop("_arrays")
    state_dict = result.pop("_state_dict")
    metrics_path = args.artifact_dir / f"vision_v3_metrics_seed{args.seed}.json"
    save_json(metrics_path, result)
    np.savez_compressed(
        args.artifact_dir / f"vision_v3_embeddings_seed{args.seed}.npz",
        **{name: value.numpy() for name, value in arrays.items()},
    )
    torch.save(
        {"model": state_dict, "configuration": asdict(config)},
        args.artifact_dir / f"vision_v3_seed{args.seed}.pt",
    )
    print(f"VISION_V3_RESULT={metrics_path}")


if __name__ == "__main__":
    main()
