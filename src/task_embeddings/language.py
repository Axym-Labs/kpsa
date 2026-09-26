from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from torch import nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

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


def format_glue_example(task: str, example: dict) -> tuple[str, str]:
    label = int(example["label"])
    if task == "sst2":
        prompt = f"sst2 sentiment; sentence: {example['sentence']}"
        truth = label == 1
    elif task == "mrpc":
        prompt = (
            f"mrpc paraphrase; sentence 1: {example['sentence1']} "
            f"sentence 2: {example['sentence2']}"
        )
        truth = label == 1
    elif task == "rte":
        prompt = (
            f"rte entailment; premise: {example['sentence1']} "
            f"hypothesis: {example['sentence2']}"
        )
        truth = label == 0
    elif task == "qnli":
        prompt = (
            f"qnli entailment; question: {example['question']} "
            f"sentence: {example['sentence']}"
        )
        truth = label == 0
    elif task == "qqp":
        prompt = (
            f"qqp duplicate; question 1: {example['question1']} "
            f"question 2: {example['question2']}"
        )
        truth = label == 1
    elif task == "cola":
        prompt = f"cola acceptable; sentence: {example['sentence']}"
        truth = label == 1
    else:
        raise ValueError(f"unsupported GLUE task: {task}")
    return prompt, "true" if truth else "false"


def sequence_ce_residual_norm_sq(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Squared norm of d(mean token cross entropy)/d(logits)."""
    valid = labels != -100
    count = valid.sum().clamp_min(1)
    probabilities = logits.float().softmax(dim=-1)
    residual = probabilities.clone()
    safe_labels = labels.masked_fill(~valid, 0)
    batch_index, token_index = torch.where(valid)
    residual[batch_index, token_index, safe_labels[valid]] -= 1.0
    residual[~valid] = 0
    residual = residual / count
    return residual.square().sum()


def _input_projections(dense: nn.Module) -> list[nn.Linear]:
    if hasattr(dense, "wi"):
        return [dense.wi]
    return [dense.wi_0, dense.wi_1]


class T5NeuronProbe(nn.Module):
    """Wrap T5 and treat each FFN neuron (incoming rows + outgoing column) as a block."""

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        *,
        every_n: int = 1,
        encoder_only: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.probes: list[tuple[nn.Module, str]] = []
        for index, block in enumerate(model.encoder.block):
            if index % every_n == 0:
                self.probes.append(
                    (block.layer[-1].DenseReluDense, f"encoder.block.{index}.ff")
                )
        if not encoder_only:
            for index, block in enumerate(model.decoder.block):
                if index % every_n == 0:
                    self.probes.append(
                        (block.layer[-1].DenseReluDense, f"decoder.block.{index}.ff")
                    )
        self._masks: list[torch.Tensor | None] = [None] * len(self.probes)
        self._capture = False
        self.activations: list[torch.Tensor | None] = [None] * len(self.probes)
        self._hook_handles = []
        for index, (dense, _) in enumerate(self.probes):
            self._hook_handles.append(
                dense.wo.register_forward_pre_hook(self._make_hook(index))
            )

    def _make_hook(self, index: int):
        def hook(_module, inputs):
            value = inputs[0]
            mask = self._masks[index]
            if mask is not None:
                value = value * mask
            if self._capture:
                value.retain_grad()
                self.activations[index] = value
            return (value, *inputs[1:])

        return hook

    @property
    def layer_sizes(self) -> list[int]:
        return [dense.wo.in_features for dense, _ in self.probes]

    @property
    def n_modules(self) -> int:
        return sum(self.layer_sizes)

    @property
    def module_names(self) -> list[str]:
        return [
            f"{name}.neuron_{neuron}"
            for (dense, name) in self.probes
            for neuron in range(dense.wo.in_features)
        ]

    def set_ablation(self, selected: Sequence[torch.Tensor] | None) -> None:
        if selected is None:
            self._masks = [None] * len(self.probes)
            return
        if len(selected) != len(self.probes):
            raise ValueError("one selected-index tensor is required per probed FFN")
        device = next(self.parameters()).device
        masks = []
        for size, indices in zip(self.layer_sizes, selected):
            mask = torch.ones(size, device=device)
            mask[indices.to(device)] = 0
            masks.append(mask)
        self._masks = masks

    def forward(self, *args, capture: bool = False, **kwargs):
        self._capture = capture
        self.activations = [None] * len(self.probes)
        try:
            return self.model(*args, **kwargs)
        finally:
            self._capture = False

    def block_grad_norms(self) -> torch.Tensor:
        pieces = []
        for dense, _ in self.probes:
            score = sum(
                projection.weight.grad.float().square().sum(dim=1)
                for projection in _input_projections(dense)
            )
            score = score + dense.wo.weight.grad.float().square().sum(dim=0)
            pieces.append(score)
        return torch.cat(pieces)

    def block_weight_norms(self) -> torch.Tensor:
        pieces = []
        for dense, _ in self.probes:
            score = sum(
                projection.weight.detach().float().square().sum(dim=1)
                for projection in _input_projections(dense)
            )
            score = score + dense.wo.weight.detach().float().square().sum(dim=0)
            pieces.append(score.sqrt())
        return torch.cat(pieces).cpu()

    def block_grad_rows(self, global_indices: Sequence[int]) -> list[torch.Tensor]:
        offsets = np.cumsum([0] + self.layer_sizes)
        result = []
        for index in global_indices:
            probe_id = int(np.searchsorted(offsets[1:], index, side="right"))
            local = index - int(offsets[probe_id])
            dense, _ = self.probes[probe_id]
            parts = [
                projection.weight.grad[local].detach().float().flatten()
                for projection in _input_projections(dense)
            ]
            parts.append(dense.wo.weight.grad[:, local].detach().float().flatten())
            result.append(torch.cat(parts))
        return result


TASKS = ("sst2", "mrpc", "rte", "qnli", "qqp", "cola")
TASK_DESCRIPTIONS = (
    "classify whether the sentiment of a sentence is positive",
    "classify whether two sentences are paraphrases",
    "classify whether a premise entails a hypothesis",
    "classify whether a sentence entails the answer to a question",
    "classify whether two questions are duplicates",
    "classify whether a sentence is grammatically acceptable",
)


@dataclass
class EncodedTask:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return len(self.input_ids)


def encode_split(
    dataset,
    task: str,
    tokenizer,
    limit: int,
    seed: int,
    *,
    max_input_length: int = 96,
    max_target_length: int = 4,
) -> EncodedTask:
    selected = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
    prompts, answers = zip(*(format_glue_example(task, row) for row in selected))
    encoded = tokenizer(
        list(prompts),
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    targets = tokenizer(
        text_target=list(answers),
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"]
    targets[targets == tokenizer.pad_token_id] = -100
    return EncodedTask(encoded["input_ids"], encoded["attention_mask"], targets)


def load_glue_mixture(
    tokenizer,
    train_per_task: int,
    validation_per_task: int,
    seed: int,
) -> tuple[list[EncodedTask], list[EncodedTask]]:
    train_tasks, validation_tasks = [], []
    for task_index, task in enumerate(TASKS):
        dataset = load_dataset("nyu-mll/glue", task)
        train_tasks.append(
            encode_split(
                dataset["train"], task, tokenizer, train_per_task, seed + task_index
            )
        )
        validation_tasks.append(
            encode_split(
                dataset["validation"],
                task,
                tokenizer,
                validation_per_task,
                seed + 100 + task_index,
            )
        )
    return train_tasks, validation_tasks


@torch.no_grad()
def semantic_task_representation(
    model: T5ForConditionalGeneration,
    tokenizer,
    device: torch.device,
    dimension: int = 4,
) -> torch.Tensor:
    encoded = tokenizer(
        list(TASK_DESCRIPTIONS),
        padding=True,
        truncation=True,
        max_length=48,
        return_tensors="pt",
    ).to(device)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden = model.encoder(**encoded).last_hidden_state.float()
    mask = encoded.attention_mask.float()[..., None]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    compressed = PCA(n_components=dimension, random_state=0).fit_transform(
        pooled.cpu().numpy()
    )
    return normalized_rows(torch.tensor(compressed, dtype=torch.float32))


def task_batch(
    data: EncodedTask,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    indices = torch.randint(0, len(data), (batch_size,), generator=generator)
    return {
        "input_ids": data.input_ids[indices].to(device),
        "attention_mask": data.attention_mask[indices].to(device),
        "labels": data.labels[indices].to(device),
    }


def train_model(
    model: T5NeuronProbe,
    train_tasks: Sequence[EncodedTask],
    representations: dict[str, torch.Tensor],
    device: torch.device,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[dict, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    accumulator = OnlineAccumulator(
        model.n_modules,
        representations,
        device,
        beta=0.995,
        late_fraction=0.7,
        total_steps=steps,
    )
    generator = torch.Generator().manual_seed(seed + 211)
    schedule = []
    while len(schedule) < steps:
        schedule.extend(torch.randperm(len(TASKS), generator=generator).tolist())
    timer = WallTimer()
    accumulation_seconds = 0.0
    trace = []
    model.train()
    for step, task in enumerate(schedule[:steps]):
        batch = task_batch(train_tasks[task], batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(**batch)
            loss = output.loss
        loss.backward()
        t0 = time.perf_counter()
        residual = sequence_ce_residual_norm_sq(output.logits.detach(), batch["labels"])
        accumulator.update(model.block_grad_norms(), task, residual, step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        accumulation_seconds += time.perf_counter() - t0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == steps - 1:
            prediction = output.logits[:, 0].argmax(dim=-1)
            accuracy = (prediction == batch["labels"][:, 0]).float().mean()
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
        "examples_per_second": steps * batch_size / elapsed,
        "accumulator_seconds_including_sync": accumulation_seconds,
        "accumulator_fraction_upper_bound": accumulation_seconds / elapsed,
        "trace": trace,
    }


def posthoc_profiles(
    model: T5NeuronProbe,
    validation_tasks: Sequence[EncodedTask],
    representations: dict[str, torch.Tensor],
    device: torch.device,
    reference_per_task: int,
    *,
    tangent_per_task: int = 2,
) -> tuple[dict, dict, dict, list[TangentModule], torch.Tensor]:
    accumulator = PosthocAccumulator(model.n_modules, representations, device)
    selected_modules = select_evenly(model.layer_sizes, per_layer=1)
    tangent_rows: dict[int, list[torch.Tensor]] = {idx: [] for idx in selected_modules}
    tangent_tasks: list[int] = []
    encoder_probes = sum(name.startswith("encoder") for _, name in model.probes)
    timer = WallTimer()
    model.eval()
    for task, data in enumerate(validation_tasks):
        for index in range(min(reference_per_task, len(data))):
            batch = {
                "input_ids": data.input_ids[index : index + 1].to(device),
                "attention_mask": data.attention_mask[index : index + 1].to(device),
                "labels": data.labels[index : index + 1].to(device),
            }
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(**batch, capture=True)
                loss = output.loss
            loss.backward()
            residual_norm_sq = sequence_ce_residual_norm_sq(
                output.logits.detach(), batch["labels"]
            )
            raw = model.block_grad_norms()
            ief = raw / residual_norm_sq.clamp_min(EPS)
            activation_parts, actgrad_parts = [], []
            decoder_mask = (batch["labels"] != -100).float()
            for probe_index, value in enumerate(model.activations):
                if value is None:
                    raise RuntimeError("missing captured T5 MLP activation")
                mask = (
                    batch["attention_mask"].float()
                    if probe_index < encoder_probes
                    else decoder_mask
                )
                mask = mask[..., None]
                denominator = mask.sum().clamp_min(1)
                activation_parts.append(
                    (value.detach().float().abs() * mask).sum(dim=(0, 1)) / denominator
                )
                actgrad_parts.append(
                    (
                        (value.detach().float() * value.grad.detach().float()).abs()
                        * mask
                    ).sum(dim=(0, 1))
                    / denominator
                )
            accumulator.update(
                task,
                ief=ief,
                raw=raw,
                activation=torch.cat(activation_parts),
                actgrad=torch.cat(actgrad_parts),
            )
            if index < tangent_per_task:
                scale = residual_norm_sq.sqrt().clamp_min(EPS)
                for module_index, row in zip(
                    selected_modules, model.block_grad_rows(selected_modules)
                ):
                    tangent_rows[module_index].append((row / scale).cpu())
                tangent_tasks.append(task)
    elapsed = timer.elapsed()
    embeddings, amplitudes = accumulator.finalize()
    modules = [
        TangentModule(name=model.module_names[index], gradients=torch.stack(rows))
        for index, rows in tangent_rows.items()
    ]
    sample_count = sum(min(reference_per_task, len(data)) for data in validation_tasks)
    return (
        embeddings,
        amplitudes,
        {
            "wall_seconds": elapsed,
            "samples": sample_count,
            "samples_per_second": sample_count / elapsed,
            "tangent_modules": len(modules),
            "tangent_samples": len(tangent_tasks),
        },
        modules,
        torch.tensor(tangent_tasks),
    )


@torch.no_grad()
def evaluate_tasks(
    model: T5NeuronProbe,
    tasks: Sequence[EncodedTask],
    device: torch.device,
    selected: Sequence[torch.Tensor] | None = None,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.set_ablation(selected)
    losses, accuracies = [], []
    for data in tasks:
        task_loss, task_correct, task_count = 0.0, 0.0, 0
        for start in range(0, len(data), batch_size):
            labels = data.labels[start : start + batch_size].to(device)
            batch = {
                "input_ids": data.input_ids[start : start + batch_size].to(device),
                "attention_mask": data.attention_mask[start : start + batch_size].to(
                    device
                ),
                "labels": labels,
            }
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(**batch)
            token_losses = torch.nn.functional.cross_entropy(
                output.logits.flatten(0, 1).float(),
                labels.flatten(),
                ignore_index=-100,
                reduction="none",
            ).reshape(labels.shape)
            valid = labels != -100
            sample_loss = (token_losses * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1)
            task_loss += float(sample_loss.sum())
            task_correct += float(
                (output.logits[:, 0].argmax(dim=-1) == labels[:, 0]).sum()
            )
            task_count += len(labels)
        losses.append(task_loss / task_count)
        accuracies.append(task_correct / task_count)
    model.set_ablation(None)
    return torch.tensor(losses), torch.tensor(accuracies)


def causal_study(
    model: T5NeuronProbe,
    tasks: Sequence[EncodedTask],
    methods: dict[str, torch.Tensor],
    device: torch.device,
    fractions: Sequence[float] = (0.01, 0.03, 0.05),
) -> dict:
    baseline_loss, baseline_accuracy = evaluate_tasks(model, tasks, device)
    records = []
    for method_name, score_matrix in methods.items():
        for fraction in fractions:
            for task in range(len(TASKS)):
                selected = layer_balanced_indices(
                    score_matrix[:, task], model.layer_sizes, fraction
                )
                loss, accuracy = evaluate_tasks(model, tasks, device, selected)
                delta_accuracy = baseline_accuracy - accuracy
                delta_loss = loss - baseline_loss
                mask = torch.ones(len(TASKS), dtype=torch.bool)
                mask[task] = False
                records.append(
                    {
                        "method": method_name,
                        "fraction": fraction,
                        "task": task,
                        "task_name": TASKS[task],
                        "target_accuracy_drop": float(delta_accuracy[task]),
                        "nontarget_accuracy_drop": float(delta_accuracy[mask].mean()),
                        "selective_accuracy_drop": float(
                            delta_accuracy[task] - delta_accuracy[mask].mean()
                        ),
                        "target_loss_increase": float(delta_loss[task]),
                        "selective_loss_increase": float(
                            delta_loss[task] - delta_loss[mask].mean()
                        ),
                        "cross_task_accuracy_drop": delta_accuracy,
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
        "baseline_task_loss": baseline_loss,
        "baseline_task_accuracy": baseline_accuracy,
        "baseline_accuracy_mean": float(baseline_accuracy.mean()),
        "records": records,
        "summary": summary,
    }


def circuit_relationship_metrics(
    score_matrix: torch.Tensor,
    layer_sizes: Sequence[int],
    causal: dict,
    method_name: str,
    fraction: float = 0.05,
) -> dict:
    circuits = []
    for task in range(len(TASKS)):
        selected = layer_balanced_indices(score_matrix[:, task], layer_sizes, fraction)
        current, offset = set(), 0
        for indices, size in zip(selected, layer_sizes):
            current.update((indices + offset).tolist())
            offset += size
        circuits.append(current)
    rows = {
        row["task"]: row
        for row in causal["records"]
        if row["method"] == method_name and row["fraction"] == fraction
    }
    overlaps, cross_effects, related_flags = [], [], []
    related_pairs = {frozenset((1, 4)), frozenset((2, 3))}
    for i in range(len(TASKS)):
        for j in range(i + 1, len(TASKS)):
            overlaps.append(
                len(circuits[i] & circuits[j]) / len(circuits[i] | circuits[j])
            )
            cross_i_j = rows[i]["cross_task_accuracy_drop"][j]
            cross_j_i = rows[j]["cross_task_accuracy_drop"][i]
            cross_effects.append((cross_i_j + cross_j_i) / 2)
            related_flags.append(frozenset((i, j)) in related_pairs)
    return {
        "overlap_cross_ablation_spearman": float(
            spearmanr(overlaps, cross_effects).statistic
        ),
        "related_pair_overlap_mean": float(
            np.mean([x for x, flag in zip(overlaps, related_flags) if flag])
        ),
        "other_pair_overlap_mean": float(
            np.mean([x for x, flag in zip(overlaps, related_flags) if not flag])
        ),
    }


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    semantic4 = semantic_task_representation(base_model, tokenizer, device)
    onehot = torch.eye(len(TASKS))
    random4 = normalized_rows(
        torch.randn(len(TASKS), 4, generator=torch.Generator().manual_seed(193))
    )
    representations = {"onehot": onehot, "semantic4": semantic4, "random4": random4}
    train_tasks, validation_tasks = load_glue_mixture(
        tokenizer, args.train_per_task, args.validation_per_task, args.seed
    )
    model = T5NeuronProbe(
        base_model, every_n=args.every_n, encoder_only=args.encoder_only
    )
    steps = args.steps
    if steps is None:
        steps = math.ceil(
            args.train_per_task * len(TASKS) * args.epochs / args.batch_size
        )
    online, train_metrics = train_model(
        model, train_tasks, representations, device, steps, args.batch_size, args.seed
    )
    post, amplitudes, post_metrics, tangent_modules, tangent_tasks = posthoc_profiles(
        model,
        validation_tasks,
        representations,
        device,
        args.reference_per_task,
        tangent_per_task=2,
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
    onehot_scores = query_scores(post["ief"]["onehot"], onehot)
    semantic_scores = query_scores(post["ief"]["semantic4"], semantic4)
    methods = {
        "post_ief_onehot": onehot_scores,
        "post_ief_semantic4": semantic_scores,
        "post_raw_semantic4": query_scores(post["raw"]["semantic4"], semantic4),
        "train_late_raw_semantic4": query_scores(
            online["late_raw"]["semantic4"], semantic4
        ),
        "activation_semantic4": query_scores(
            post["activation"]["semantic4"], semantic4
        ),
        "actgrad_semantic4": query_scores(post["actgrad"]["semantic4"], semantic4),
        "weight_norm": model.block_weight_norms()[:, None].repeat(1, len(TASKS)),
        "random": torch.stack(
            [
                random_layer_balanced_scores(model.layer_sizes, args.seed + task)
                for task in range(len(TASKS))
            ],
            dim=1,
        ),
    }
    fractions = (0.03,) if args.smoke else (0.01, 0.03, 0.05)
    if args.smoke:
        methods = {
            key: methods[key]
            for key in ("post_ief_onehot", "post_ief_semantic4", "random")
        }
    causal = causal_study(model, validation_tasks, methods, device, fractions)

    checkpoint = artifact_dir / f"language_seed{args.seed}.pt"
    torch.save(
        {"model": model.model.state_dict(), "semantic4": semantic4, "args": vars(args)},
        checkpoint,
    )
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
        artifact_dir / f"language_embeddings_seed{args.seed}.npz", **arrays
    )

    compression = {}
    for fraction in (0.01, 0.03, 0.05):
        values = []
        for task in range(len(TASKS)):
            full = layer_balanced_indices(
                onehot_scores[:, task], model.layer_sizes, fraction
            )
            compressed = layer_balanced_indices(
                semantic_scores[:, task], model.layer_sizes, fraction
            )
            full_set, compressed_set, offset = set(), set(), 0
            for a, b, size in zip(full, compressed, model.layer_sizes):
                full_set.update((a + offset).tolist())
                compressed_set.update((b + offset).tolist())
                offset += size
            values.append(len(full_set & compressed_set) / max(1, len(full_set)))
        compression[str(fraction)] = float(np.mean(values))
    relationships = {}
    if not args.smoke:
        relationships = circuit_relationship_metrics(
            semantic_scores, model.layer_sizes, causal, "post_ief_semantic4"
        )
    metrics = {
        "setting": "language_glue_t5_small",
        "tasks": TASKS,
        "seed": args.seed,
        "device": str(device),
        "configuration": vars(args),
        "task_representation_semantic4": semantic4,
        "train": train_metrics,
        "posthoc": post_metrics,
        "proxy_fidelity": fidelity,
        "kernel_fidelity": tangent,
        "causal": causal,
        "compression_topk_overlap": compression,
        "circuit_relationships": relationships,
        "checkpoint": str(checkpoint),
    }
    save_json(artifact_dir / f"language_metrics_seed{args.seed}.json", metrics)
    print(f"LANGUAGE_RESULT={artifact_dir / f'language_metrics_seed{args.seed}.json'}")
    print(
        {
            "accuracy": causal["baseline_accuracy_mean"],
            "task_accuracy": causal["baseline_task_accuracy"].tolist(),
            "kernel": tangent,
            "sel_3pct": {
                method: values.get("0.03", {}).get("selective_accuracy_drop_mean")
                for method, values in causal["summary"].items()
            },
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default=str(arc_artifact_dir("01_exploratory", "language"))
    )
    parser.add_argument("--model-name", default="google-t5/t5-small")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-per-task", type=int, default=2000)
    parser.add_argument("--validation-per-task", type=int, default=128)
    parser.add_argument("--reference-per-task", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--encoder-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
