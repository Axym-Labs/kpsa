from __future__ import annotations

import argparse
import math
import pickle
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from .common import (
    EPS,
    OnlineAccumulator,
    PosthocAccumulator,
    TangentModule,
    WallTimer,
    arc_artifact_dir,
    embedding_fidelity,
    layer_balanced_indices,
    normalized_rows,
    query_scores,
    random_layer_balanced_scores,
    save_json,
    seed_everything,
    select_evenly,
    tangent_kernel_metrics,
)


def cross_entropy_residual_norm_sq(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Squared norm of d(mean cross entropy)/d(logits)."""
    probabilities = logits.float().softmax(dim=-1)
    residual = probabilities
    residual[torch.arange(len(targets), device=logits.device), targets] -= 1.0
    residual = residual / len(targets)
    return residual.square().sum()


class ClassHomogeneousBatchSampler:
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        steps: int,
        seed: int,
    ) -> None:
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed
        labels_array = np.asarray(labels)
        self.classes = sorted(np.unique(labels_array).tolist())
        self.indices = {
            class_id: np.flatnonzero(labels_array == class_id)
            for class_id in self.classes
        }

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed)
        schedule = []
        while len(schedule) < self.steps:
            schedule.extend(generator.permutation(self.classes).tolist())
        for class_id in schedule[: self.steps]:
            yield generator.choice(
                self.indices[class_id], size=self.batch_size, replace=True
            ).tolist()


class GradientMeanBuffer:
    """Average independent homogeneous-task gradients before an optimizer step."""

    def __init__(self, parameters) -> None:
        self.parameters = [
            parameter for parameter in parameters if parameter.requires_grad
        ]
        self.buffers = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.count = 0

    @torch.no_grad()
    def add(self) -> None:
        for parameter, buffer in zip(self.parameters, self.buffers):
            if parameter.grad is not None:
                buffer.add_(parameter.grad)
        self.count += 1

    @torch.no_grad()
    def apply_mean(self) -> None:
        if self.count == 0:
            raise RuntimeError("cannot apply an empty gradient buffer")
        for parameter, buffer in zip(self.parameters, self.buffers):
            parameter.grad = buffer / self.count
            buffer.zero_()
        self.count = 0


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class CifarResNet18(nn.Module):
    """CIFAR-adapted ResNet-18 with probes on layer3/layer4 conv channels."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        backbone = resnet18(
            weights=None, num_classes=num_classes, norm_layer=_group_norm
        )
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        self.backbone = backbone
        self.probes: list[tuple[nn.Conv2d, nn.GroupNorm, str]] = []
        for stage_name in ("layer3", "layer4"):
            stage = getattr(backbone, stage_name)
            for block_id, block in enumerate(stage):
                self.probes.extend(
                    [
                        (block.conv1, block.bn1, f"{stage_name}.{block_id}.conv1"),
                        (block.conv2, block.bn2, f"{stage_name}.{block_id}.conv2"),
                    ]
                )
        self._masks: list[torch.Tensor | None] = [None] * len(self.probes)
        self._capture = False
        self.activations: list[torch.Tensor | None] = [None] * len(self.probes)
        self._hook_handles = []
        for index, (_, norm, _) in enumerate(self.probes):
            self._hook_handles.append(
                norm.register_forward_hook(self._make_hook(index))
            )

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            value = output
            mask = self._masks[index]
            if mask is not None:
                value = value * mask[None, :, None, None]
            if self._capture:
                value.retain_grad()
                self.activations[index] = value
            return value

        return hook

    @property
    def layer_sizes(self) -> list[int]:
        return [conv.out_channels for conv, _, _ in self.probes]

    @property
    def n_modules(self) -> int:
        return sum(self.layer_sizes)

    @property
    def module_names(self) -> list[str]:
        return [
            f"{name}.channel_{channel}"
            for _conv, norm, name in self.probes
            for channel in range(norm.num_channels)
        ]

    def set_ablation(self, selected: Sequence[torch.Tensor] | None) -> None:
        if selected is None:
            self._masks = [None] * len(self.probes)
            return
        if len(selected) != len(self.probes):
            raise ValueError(
                "one selected-index tensor is required per probed convolution"
            )
        device = next(self.parameters()).device
        masks = []
        for size, indices in zip(self.layer_sizes, selected):
            mask = torch.ones(size, device=device)
            mask[indices.to(device)] = 0
            masks.append(mask)
        self._masks = masks

    def forward(self, x: torch.Tensor, *, capture: bool = False) -> torch.Tensor:
        self._capture = capture
        self.activations = [None] * len(self.probes)
        output = self.backbone(x)
        self._capture = False
        return output

    def block_grad_norms(self) -> torch.Tensor:
        pieces = []
        for conv, norm, _ in self.probes:
            score = conv.weight.grad.float().square().flatten(1).sum(dim=1)
            if norm.weight is not None:
                score = score + norm.weight.grad.float().square()
            if norm.bias is not None:
                score = score + norm.bias.grad.float().square()
            pieces.append(score)
        return torch.cat(pieces)

    def block_weight_norms(self) -> torch.Tensor:
        pieces = []
        for conv, norm, _ in self.probes:
            score = conv.weight.detach().float().square().flatten(1).sum(dim=1)
            if norm.weight is not None:
                score = score + norm.weight.detach().float().square()
            if norm.bias is not None:
                score = score + norm.bias.detach().float().square()
            pieces.append(score.sqrt())
        return torch.cat(pieces).cpu()

    def block_grad_rows(self, global_indices: Sequence[int]) -> list[torch.Tensor]:
        offsets = np.cumsum([0] + self.layer_sizes)
        result = []
        for index in global_indices:
            probe_id = int(np.searchsorted(offsets[1:], index, side="right"))
            local = index - int(offsets[probe_id])
            conv, norm, _ = self.probes[probe_id]
            parts = [conv.weight.grad[local].detach().float().flatten()]
            if norm.weight is not None:
                parts.append(norm.weight.grad[local : local + 1].detach().float())
            if norm.bias is not None:
                parts.append(norm.bias.grad[local : local + 1].detach().float())
            result.append(torch.cat(parts))
        return result


def cifar_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


def fine_to_coarse_mapping(dataset_root: Path) -> list[int]:
    path = dataset_root / "cifar-100-python" / "train"
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    mapping: dict[int, int] = {}
    for fine, coarse in zip(payload["fine_labels"], payload["coarse_labels"]):
        if fine in mapping and mapping[fine] != coarse:
            raise ValueError("CIFAR-100 fine class maps to multiple coarse classes")
        mapping[fine] = coarse
    return [mapping[i] for i in range(100)]


def cache_dataset(dataset: datasets.CIFAR100) -> TensorDataset:
    images, labels = [], []
    for image, label in dataset:
        images.append(image)
        labels.append(label)
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


def balanced_reference_indices(labels: Sequence[int], per_class: int) -> list[int]:
    labels_array = np.asarray(labels)
    indices = []
    for class_id in sorted(np.unique(labels_array)):
        class_indices = np.flatnonzero(labels_array == class_id)
        indices.extend(class_indices[:per_class].tolist())
    return indices


def train_model(
    model: CifarResNet18,
    loader: DataLoader,
    representations: dict[str, torch.Tensor],
    device: torch.device,
    steps: int,
    tasks_per_update: int,
) -> tuple[dict, dict]:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    optimizer_updates = math.ceil(steps / tasks_per_update)
    warmup_updates = max(1, math.ceil(0.05 * optimizer_updates))

    def lr_multiplier(update: int) -> float:
        if update < warmup_updates:
            return (update + 1) / warmup_updates
        progress = (update - warmup_updates) / max(
            1, optimizer_updates - warmup_updates
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    gradient_buffer = GradientMeanBuffer(model.parameters())
    accumulator = OnlineAccumulator(
        model.n_modules,
        representations,
        device,
        beta=0.995,
        late_fraction=0.7,
        total_steps=steps,
    )
    timer = WallTimer()
    accumulation_seconds = 0.0
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
                "online task accumulator requires class-homogeneous batches"
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        t0 = time.perf_counter()
        accumulator.update(
            model.block_grad_norms(),
            task,
            cross_entropy_residual_norm_sq(logits.detach(), targets),
            step,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        accumulation_seconds += time.perf_counter() - t0
        gradient_buffer.add()
        if (step + 1) % tasks_per_update == 0 or step == steps - 1:
            gradient_buffer.apply_mean()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
        if step % 250 == 0 or step == steps - 1:
            accuracy = (logits.argmax(dim=1) == targets).float().mean()
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "batch_accuracy": float(accuracy),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
    elapsed = timer.elapsed()
    return accumulator.finalize(), {
        "wall_seconds": elapsed,
        "steps": steps,
        "optimizer_updates": optimizer_updates,
        "tasks_per_update": tasks_per_update,
        "examples_per_second": steps * loader.batch_sampler.batch_size / elapsed,
        "accumulator_seconds_including_sync": accumulation_seconds,
        "accumulator_fraction_upper_bound": accumulation_seconds / elapsed,
        "trace": trace,
    }


def posthoc_profiles(
    model: CifarResNet18,
    dataset: TensorDataset,
    reference_indices: Sequence[int],
    representations: dict[str, torch.Tensor],
    device: torch.device,
    *,
    tangent_per_class: int = 1,
) -> tuple[dict, dict, dict, list[TangentModule], torch.Tensor]:
    accumulator = PosthocAccumulator(model.n_modules, representations, device)
    selected_modules = select_evenly(model.layer_sizes, per_layer=2)
    tangent_rows: dict[int, list[torch.Tensor]] = {idx: [] for idx in selected_modules}
    tangent_tasks: list[int] = []
    tangent_seen = [0] * 100
    timer = WallTimer()
    model.eval()
    for index in reference_indices:
        image, target_tensor = dataset[index]
        target = int(target_tensor)
        images = image[None].to(device)
        targets = target_tensor[None].to(device)
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images, capture=True)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        residual_norm_sq = cross_entropy_residual_norm_sq(logits.detach(), targets)
        raw = model.block_grad_norms()
        ief = raw / residual_norm_sq.clamp_min(EPS)
        activations = [a for a in model.activations if a is not None]
        activation = torch.cat(
            [a.detach().float().abs().mean(dim=(0, 2, 3)) for a in activations]
        )
        actgrad = torch.cat(
            [
                (a.detach().float() * a.grad.detach().float()).abs().mean(dim=(0, 2, 3))
                for a in activations
            ]
        )
        accumulator.update(
            target, ief=ief, raw=raw, activation=activation, actgrad=actgrad
        )
        if tangent_seen[target] < tangent_per_class:
            scale = residual_norm_sq.sqrt().clamp_min(EPS)
            for module_index, row in zip(
                selected_modules, model.block_grad_rows(selected_modules)
            ):
                tangent_rows[module_index].append((row / scale).cpu())
            tangent_tasks.append(target)
            tangent_seen[target] += 1
    elapsed = timer.elapsed()
    embeddings, amplitudes = accumulator.finalize()
    modules = [
        TangentModule(name=model.module_names[index], gradients=torch.stack(rows))
        for index, rows in tangent_rows.items()
    ]
    return (
        embeddings,
        amplitudes,
        {
            "wall_seconds": elapsed,
            "samples": len(reference_indices),
            "samples_per_second": len(reference_indices) / elapsed,
            "tangent_modules": len(modules),
            "tangent_samples": len(tangent_tasks),
        },
        modules,
        torch.tensor(tangent_tasks),
    )


@torch.no_grad()
def evaluate_classification(
    model: CifarResNet18,
    loader: DataLoader,
    device: torch.device,
    selected: Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.set_ablation(selected)
    loss_sum = torch.zeros(100, device=device)
    correct = torch.zeros(100, device=device)
    count = torch.zeros(100, device=device)
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
        count.scatter_add_(0, targets, torch.ones_like(losses, dtype=torch.float32))
    model.set_ablation(None)
    return (loss_sum / count.clamp_min(1)).cpu(), (correct / count.clamp_min(1)).cpu()


def causal_study(
    model: CifarResNet18,
    loader: DataLoader,
    methods: dict[str, torch.Tensor],
    query_tasks: Sequence[int],
    device: torch.device,
    fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict:
    baseline_loss, baseline_accuracy = evaluate_classification(model, loader, device)
    records = []
    for method_name, score_matrix in methods.items():
        for fraction in fractions:
            for task in query_tasks:
                selected = layer_balanced_indices(
                    score_matrix[:, task], model.layer_sizes, fraction
                )
                loss, accuracy = evaluate_classification(
                    model, loader, device, selected
                )
                delta_accuracy = baseline_accuracy - accuracy
                delta_loss = loss - baseline_loss
                non_target_mask = torch.ones(100, dtype=torch.bool)
                non_target_mask[task] = False
                records.append(
                    {
                        "method": method_name,
                        "fraction": fraction,
                        "task": task,
                        "target_accuracy_drop": float(delta_accuracy[task]),
                        "nontarget_accuracy_drop": float(
                            delta_accuracy[non_target_mask].mean()
                        ),
                        "selective_accuracy_drop": float(
                            delta_accuracy[task]
                            - delta_accuracy[non_target_mask].mean()
                        ),
                        "target_loss_increase": float(delta_loss[task]),
                        "selective_loss_increase": float(
                            delta_loss[task] - delta_loss[non_target_mask].mean()
                        ),
                    }
                )
    summary = {}
    for method in methods:
        summary[method] = {}
        for fraction in fractions:
            rows = [
                r
                for r in records
                if r["method"] == method and r["fraction"] == fraction
            ]
            summary[method][str(fraction)] = {
                "selective_accuracy_drop_mean": float(
                    np.mean([r["selective_accuracy_drop"] for r in rows])
                ),
                "selective_accuracy_drop_std": float(
                    np.std([r["selective_accuracy_drop"] for r in rows])
                ),
                "target_accuracy_drop_mean": float(
                    np.mean([r["target_accuracy_drop"] for r in rows])
                ),
                "selective_loss_increase_mean": float(
                    np.mean([r["selective_loss_increase"] for r in rows])
                ),
            }
    return {
        "baseline_accuracy_mean": float(baseline_accuracy.mean()),
        "baseline_loss_mean": float(baseline_loss.mean()),
        "query_tasks": list(query_tasks),
        "records": records,
        "summary": summary,
    }


def task_structure_metrics(
    embedding: torch.Tensor,
    representation: torch.Tensor,
    fine_to_coarse: Sequence[int],
) -> dict[str, float]:
    clusters = KMeans(n_clusters=20, n_init=10, random_state=0).fit_predict(
        embedding.numpy()
    )
    preferred_fine = query_scores(embedding, representation).argmax(dim=1).numpy()
    preferred_coarse = np.asarray(fine_to_coarse)[preferred_fine]
    return {
        "cluster_vs_preferred_coarse_nmi": float(
            normalized_mutual_info_score(preferred_coarse, clusters)
        )
    }


def compression_overlap(
    full_scores: torch.Tensor,
    compressed_scores: torch.Tensor,
    layer_sizes: Sequence[int],
) -> dict[str, float]:
    output = {}
    for fraction in (0.01, 0.05, 0.10):
        values = []
        for task in range(100):
            full = layer_balanced_indices(full_scores[:, task], layer_sizes, fraction)
            compressed = layer_balanced_indices(
                compressed_scores[:, task], layer_sizes, fraction
            )
            set_full, set_compressed, offset = set(), set(), 0
            for a, b, size in zip(full, compressed, layer_sizes):
                set_full.update((a + offset).tolist())
                set_compressed.update((b + offset).tolist())
                offset += size
            values.append(len(set_full & set_compressed) / max(1, len(set_full)))
        output[str(fraction)] = float(np.mean(values))
    return output


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    train_transform, test_transform = cifar_transforms()
    train_dataset = datasets.CIFAR100(
        data_root, train=True, download=True, transform=train_transform
    )
    test_dataset_raw = datasets.CIFAR100(
        data_root, train=False, download=True, transform=test_transform
    )
    fine_to_coarse = fine_to_coarse_mapping(data_root)
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    steps = args.steps if args.steps is not None else args.epochs * steps_per_epoch
    sampler = ClassHomogeneousBatchSampler(
        train_dataset.targets, args.batch_size, steps, args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    onehot = torch.eye(100)
    jl32 = normalized_rows(
        torch.randn(100, 32, generator=torch.Generator().manual_seed(717))
    )
    random32 = normalized_rows(
        torch.randn(100, 32, generator=torch.Generator().manual_seed(991))
    )
    representations = {"onehot": onehot, "jl32": jl32, "random32": random32}
    model = CifarResNet18().to(device)
    online, train_metrics = train_model(
        model, train_loader, representations, device, steps, args.tasks_per_update
    )

    test_dataset = cache_dataset(test_dataset_raw)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    reference_indices = balanced_reference_indices(
        test_dataset_raw.targets, args.reference_per_class
    )
    post, amplitudes, post_metrics, tangent_modules, tangent_tasks = posthoc_profiles(
        model,
        test_dataset,
        reference_indices,
        representations,
        device,
        tangent_per_class=1,
    )

    fidelity = {
        variant: {
            name: embedding_fidelity(
                reps[name], post["ief"][name], representations[name]
            )
            for name in representations
        }
        for variant, reps in online.items()
    }
    tangent = tangent_kernel_metrics(tangent_modules, tangent_tasks, representations)
    post_onehot_scores = query_scores(post["ief"]["onehot"], onehot)
    post_jl_scores = query_scores(post["ief"]["jl32"], jl32)
    methods = {
        "post_ief_onehot": post_onehot_scores,
        "post_ief_jl32": post_jl_scores,
        "post_raw_jl32": query_scores(post["raw"]["jl32"], jl32),
        "train_late_raw_jl32": query_scores(online["late_raw"]["jl32"], jl32),
        "activation_jl32": query_scores(post["activation"]["jl32"], jl32),
        "actgrad_jl32": query_scores(post["actgrad"]["jl32"], jl32),
        "weight_norm": model.block_weight_norms()[:, None].repeat(1, 100),
        "random": torch.stack(
            [
                random_layer_balanced_scores(model.layer_sizes, args.seed + task)
                for task in range(100)
            ],
            dim=1,
        ),
    }
    query_tasks = []
    for coarse in range(20):
        query_tasks.append(
            next(i for i, value in enumerate(fine_to_coarse) if value == coarse)
        )
    fractions = (0.05,) if args.smoke else (0.01, 0.05, 0.10)
    if args.smoke:
        query_tasks = query_tasks[:2]
        methods = {
            key: methods[key] for key in ("post_ief_onehot", "post_ief_jl32", "random")
        }
    causal = causal_study(model, test_loader, methods, query_tasks, device, fractions)

    checkpoint = artifact_dir / f"vision_seed{args.seed}.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, checkpoint)
    arrays = {}
    for statistic, reps in post.items():
        for name, value in reps.items():
            arrays[f"post_{statistic}_{name}"] = value.numpy()
    for variant, reps in online.items():
        for name, value in reps.items():
            arrays[f"online_{variant}_{name}"] = value.numpy()
    for name, value in amplitudes.items():
        arrays[f"amplitude_{name}"] = value.numpy()
    np.savez_compressed(
        artifact_dir / f"vision_embeddings_seed{args.seed}.npz", **arrays
    )
    metrics = {
        "setting": "vision_cifar100",
        "seed": args.seed,
        "device": str(device),
        "configuration": vars(args),
        "train": train_metrics,
        "posthoc": post_metrics,
        "proxy_fidelity": fidelity,
        "kernel_fidelity": tangent,
        "causal": causal,
        "compression_topk_overlap": compression_overlap(
            post_onehot_scores, post_jl_scores, model.layer_sizes
        ),
        "task_structure": task_structure_metrics(
            post["ief"]["onehot"], onehot, fine_to_coarse
        ),
        "checkpoint": str(checkpoint),
    }
    save_json(artifact_dir / f"vision_metrics_seed{args.seed}.json", metrics)
    print(f"VISION_RESULT={artifact_dir / f'vision_metrics_seed{args.seed}.json'}")
    print(
        {
            "accuracy": causal["baseline_accuracy_mean"],
            "kernel": tangent,
            "sel_5pct": {
                method: values.get("0.05", {}).get("selective_accuracy_drop_mean")
                for method, values in causal["summary"].items()
            },
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default=str(arc_artifact_dir("01_exploratory", "vision"))
    )
    parser.add_argument("--data-root", default="/home/davwis/main/data/cifar-100")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tasks-per-update", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--reference-per-class", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
