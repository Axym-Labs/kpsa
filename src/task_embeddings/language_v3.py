from __future__ import annotations

import argparse
import math
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sklearn.decomposition import PCA
from torch import nn

from .applications import continual_learning_metrics, selective_effect
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
from .vision import GradientMeanBuffer


def encode_instruction(
    tokenizer, prompt: str, answer: str, max_length: int
) -> dict[str, torch.Tensor]:
    prefix = f"Instruction: {prompt}\nAnswer:"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(f" {answer}", add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[-max_length:]
        prefix_ids = []
    else:
        prefix_ids = prefix_ids[-(max_length - len(answer_ids)) :]
    input_ids = prefix_ids + answer_ids
    padding = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * padding
    attention = [1] * (len(input_ids) - padding) + [0] * padding
    labels = [-100] * len(prefix_ids) + answer_ids + [-100] * padding
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def causal_residual_norm_sq(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Squared norm of the mean causal-LM CE gradient, using valid tokens only."""
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    selected_logits = shifted_logits[valid]
    selected_labels = shifted_labels[valid]
    if not selected_labels.numel():
        return torch.zeros((), device=logits.device)
    residual = selected_logits.softmax(dim=-1)
    residual[
        torch.arange(len(selected_labels), device=logits.device), selected_labels
    ] -= 1.0
    residual = residual / len(selected_labels)
    return residual.square().sum()


class QwenFeatureProbe(nn.Module):
    """Qwen3 wrapper treating each SwiGLU gate/up/down triplet as one module."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._selected: list[torch.Tensor] | None = None
        self._mode: Literal["zero", "mean", "retain"] = "zero"
        self._means: list[torch.Tensor] | None = None
        self._feature_gates: list[torch.Tensor] | None = None
        self._capture = False
        self._captured: list[torch.Tensor | None] = [None] * len(self.mlps)
        self._hook_handles = [
            mlp.down_proj.register_forward_pre_hook(self._make_feature_hook(layer))
            for layer, mlp in enumerate(self.mlps)
        ]

    @property
    def mlps(self) -> list[nn.Module]:
        return [layer.mlp for layer in self.model.model.layers]

    @property
    def layer_sizes(self) -> list[int]:
        return [mlp.down_proj.in_features for mlp in self.mlps]

    @property
    def n_modules(self) -> int:
        return sum(self.layer_sizes)

    def _make_feature_hook(self, layer: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            features = inputs[0]
            if self._feature_gates is not None:
                gate = self._feature_gates[layer].to(
                    device=features.device, dtype=features.dtype
                )
                features = features * gate
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
            return (features, *inputs[1:])

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

    def set_feature_gates(self, gates: Sequence[torch.Tensor] | None) -> None:
        """Apply reversible continuous gates to each SwiGLU feature."""
        if gates is None:
            self._feature_gates = None
            return
        if len(gates) != len(self.layer_sizes):
            raise ValueError("feature gates must provide one vector per layer")
        values = [gate.detach().float().cpu().flatten() for gate in gates]
        if any(gate.numel() != size for gate, size in zip(values, self.layer_sizes)):
            raise ValueError("feature gate sizes must match the SwiGLU layers")
        self._feature_gates = values

    def forward(self, *args, capture: bool = False, **kwargs):
        self._capture = capture
        if capture:
            self._captured = [None] * len(self.mlps)
        try:
            return self.model(*args, **kwargs)
        finally:
            self._capture = False

    def captured_activations(self) -> list[torch.Tensor]:
        if any(value is None for value in self._captured):
            raise RuntimeError("forward(capture=True) is required")
        return [value for value in self._captured if value is not None]

    def clear_capture(self) -> None:
        self._capture = False
        self._captured = [None] * len(self.mlps)

    def module_owned_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for mlp in self.mlps
            for parameter in (
                mlp.gate_proj.weight,
                mlp.up_proj.weight,
                mlp.down_proj.weight,
            )
        ]

    def block_grad_norms(self) -> torch.Tensor:
        scores = []
        for mlp in self.mlps:
            score = mlp.gate_proj.weight.grad.float().square().sum(dim=1)
            score = score + mlp.up_proj.weight.grad.float().square().sum(dim=1)
            score = score + mlp.down_proj.weight.grad.float().square().sum(dim=0)
            scores.append(score)
        return torch.cat(scores)

    @torch.no_grad()
    def block_weight_norms(self) -> torch.Tensor:
        scores = []
        for mlp in self.mlps:
            score = mlp.gate_proj.weight.float().square().sum(dim=1)
            score = score + mlp.up_proj.weight.float().square().sum(dim=1)
            score = score + mlp.down_proj.weight.float().square().sum(dim=0)
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
                device=mlp.gate_proj.weight.device,
                dtype=mlp.gate_proj.weight.grad.dtype,
            )
            mlp.gate_proj.weight.grad.mul_(scale[:, None])
            mlp.up_proj.weight.grad.mul_(scale[:, None])
            mlp.down_proj.weight.grad.mul_(scale[None, :])
            offset += size


TASKS = (
    "sentiment_sst2",
    "entailment_rte",
    "paraphrase_mrpc",
    "syntax_cola",
    "commonsense_qa",
    "arithmetic_gsm8k",
)

TASK_DESCRIPTIONS = (
    "classify the sentiment of a sentence as positive or negative",
    "judge whether a premise entails a hypothesis",
    "judge whether two sentences express the same meaning",
    "judge whether an English sentence is grammatically acceptable",
    "answer a multiple choice commonsense question",
    "solve a short arithmetic word problem and return the numeric answer",
)


@dataclass
class EncodedTask:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class LanguageConfig:
    seed: int = 1
    model_name: str = "Qwen/Qwen3-1.7B"
    max_length: int = 96
    train_per_task: int = 64
    validation_per_task: int = 8
    reference_per_task: int = 2
    train_steps: int = 600
    batch_size: int = 1
    tasks_per_update: int = 6
    intervention_fractions: tuple[float, ...] = (0.005, 0.01, 0.03)
    pruning_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    recovery_steps: int = 3
    cl_steps_per_task: int = 3
    cl_strength: float = 8.0
    smoke: bool = False


def format_task_example(task: str, example: dict) -> tuple[str, str]:
    if task == "sentiment_sst2":
        return (
            f"Classify the sentiment as positive or negative. Sentence: {example['sentence']}",
            "positive" if int(example["label"]) == 1 else "negative",
        )
    if task == "entailment_rte":
        return (
            (
                f"Does the premise entail the hypothesis? Premise: {example['sentence1']} "
                f"Hypothesis: {example['sentence2']}"
            ),
            "yes" if int(example["label"]) == 0 else "no",
        )
    if task == "paraphrase_mrpc":
        return (
            (
                f"Are these sentences paraphrases? Sentence 1: {example['sentence1']} "
                f"Sentence 2: {example['sentence2']}"
            ),
            "yes" if int(example["label"]) == 1 else "no",
        )
    if task == "syntax_cola":
        return (
            f"Is this sentence grammatically acceptable? Sentence: {example['sentence']}",
            "acceptable" if int(example["label"]) == 1 else "unacceptable",
        )
    if task == "commonsense_qa":
        choices = " ".join(
            f"{label}: {text}"
            for label, text in zip(
                example["choices"]["label"], example["choices"]["text"]
            )
        )
        return (
            f"Choose the best commonsense answer. Question: {example['question']} Choices: {choices}",
            str(example["answerKey"]),
        )
    if task == "arithmetic_gsm8k":
        match = re.search(r"####\s*([^\n]+)", example["answer"])
        if match is None:
            raise ValueError("GSM8K example has no final answer marker")
        return (
            f"Solve the arithmetic problem and give only the final answer. {example['question']}",
            match.group(1).replace(",", "").strip(),
        )
    raise ValueError(f"unsupported task: {task}")


def encode_task_dataset(
    dataset,
    task: str,
    tokenizer,
    limit: int,
    seed: int,
    max_length: int,
) -> EncodedTask:
    shuffled = dataset.shuffle(seed=seed)
    candidate_limit = min(len(shuffled), max(limit * 8, limit))
    candidates = [
        format_task_example(task, shuffled[index]) for index in range(candidate_limit)
    ]
    if task != "arithmetic_gsm8k":
        groups: dict[str, list[tuple[str, str]]] = {}
        for item in candidates:
            groups.setdefault(item[1], []).append(item)
        quota = max(1, math.ceil(limit / len(groups)))
        selected = []
        for answer in sorted(groups):
            selected.extend(groups[answer][:quota])
        selected = selected[:limit]
    else:
        selected = candidates[:limit]
    tensors = [
        encode_instruction(tokenizer, prompt, answer, max_length)
        for prompt, answer in selected
    ]
    return EncodedTask(
        input_ids=torch.stack([item["input_ids"] for item in tensors]),
        attention_mask=torch.stack([item["attention_mask"] for item in tensors]),
        labels=torch.stack([item["labels"] for item in tensors]),
    )


def load_task_suite(
    tokenizer,
    config: LanguageConfig,
) -> tuple[list[EncodedTask], list[EncodedTask]]:
    sources = [
        load_dataset("nyu-mll/glue", "sst2"),
        load_dataset("nyu-mll/glue", "rte"),
        load_dataset("nyu-mll/glue", "mrpc"),
        load_dataset("nyu-mll/glue", "cola"),
        load_dataset("tau/commonsense_qa"),
        load_dataset("openai/gsm8k", "main"),
    ]
    validation_splits = [
        "validation",
        "validation",
        "validation",
        "validation",
        "validation",
        "test",
    ]
    train_tasks, validation_tasks = [], []
    for task_index, (task, source, validation_split) in enumerate(
        zip(TASKS, sources, validation_splits)
    ):
        train_tasks.append(
            encode_task_dataset(
                source["train"],
                task,
                tokenizer,
                config.train_per_task,
                config.seed + task_index,
                config.max_length,
            )
        )
        validation_tasks.append(
            encode_task_dataset(
                source[validation_split],
                task,
                tokenizer,
                config.validation_per_task,
                config.seed + 100 + task_index,
                config.max_length,
            )
        )
    return train_tasks, validation_tasks


def semantic_task_representation(dimension: int = 4) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    path = snapshot_download(
        "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    encoder = AutoModel.from_pretrained(path, local_files_only=True).eval()
    encoded = tokenizer(list(TASK_DESCRIPTIONS), padding=True, return_tensors="pt")
    with torch.no_grad():
        hidden = encoder(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    reduced = PCA(n_components=dimension, random_state=0).fit_transform(pooled.numpy())
    return normalized_rows(torch.from_numpy(reduced).float())


def task_batch(
    task: EncodedTask,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    indices = torch.randint(0, len(task), (batch_size,), generator=generator)
    return {
        "input_ids": task.input_ids[indices].to(device),
        "attention_mask": task.attention_mask[indices].to(device),
        "labels": task.labels[indices].to(device),
    }


def load_model_and_tokenizer(config: LanguageConfig, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = snapshot_download(config.model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    if device.type == "cuda":
        base.gradient_checkpointing_enable()
    return QwenFeatureProbe(base.to(device)), tokenizer


def train_model(
    model: QwenFeatureProbe,
    tasks: Sequence[EncodedTask],
    representations: dict[str, torch.Tensor],
    config: LanguageConfig,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict]:
    from transformers.optimization import Adafactor

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-4,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.0,
    )
    buffer = GradientMeanBuffer(model.parameters())
    accumulator = OnlineAccumulator(
        model.n_modules,
        representations,
        device,
        beta=0.995,
        late_fraction=0.7,
        total_steps=config.train_steps,
    )
    generator = torch.Generator().manual_seed(config.seed + 200)
    schedule = []
    while len(schedule) < config.train_steps:
        schedule.extend(torch.randperm(len(tasks), generator=generator).tolist())
    trace = []
    started = time.perf_counter()
    accumulator_seconds = 0.0
    model.train()
    for step, task_id in enumerate(schedule[: config.train_steps]):
        batch = task_batch(tasks[task_id], config.batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(**batch)
        output.loss.backward()
        scoring_started = time.perf_counter()
        accumulator.update(
            model.block_grad_norms(),
            task_id,
            causal_residual_norm_sq(output.logits.detach(), batch["labels"]),
            step,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        accumulator_seconds += time.perf_counter() - scoring_started
        buffer.add()
        if (step + 1) % config.tasks_per_update == 0 or step == config.train_steps - 1:
            buffer.apply_mean()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (
            step % max(1, config.train_steps // 20) == 0
            or step == config.train_steps - 1
        ):
            trace.append({"step": step, "loss": float(output.loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (
        accumulator.finalize(),
        accumulator.amplitudes(),
        {
            "wall_seconds": elapsed,
            "steps": config.train_steps,
            "examples_per_second": config.train_steps
            * config.batch_size
            / max(elapsed, EPS),
            "online_accumulator_seconds_including_sync": accumulator_seconds,
            "online_accumulator_fraction_upper_bound": accumulator_seconds
            / max(elapsed, EPS),
            "optimizer": "full-parameter Adafactor",
            "trace": trace,
        },
    )


def posthoc_profiles(
    model: QwenFeatureProbe,
    tasks: Sequence[EncodedTask],
    reference_per_task: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[torch.Tensor], dict]:
    rows = {name: [] for name in ("ief", "raw", "activation", "actgrad")}
    task_ids = []
    mean_sums = [torch.zeros(size, device=device) for size in model.layer_sizes]
    mean_counts = [0 for _ in model.layer_sizes]
    started = time.perf_counter()
    model.eval()
    model.set_intervention(None)
    for task_id, task in enumerate(tasks):
        for index in range(min(reference_per_task, len(task))):
            batch = {
                "input_ids": task.input_ids[index : index + 1].to(device),
                "attention_mask": task.attention_mask[index : index + 1].to(device),
                "labels": task.labels[index : index + 1].to(device),
            }
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(**batch, capture=True)
            output.loss.backward()
            residual_norm_sq = causal_residual_norm_sq(
                output.logits.detach(), batch["labels"]
            )
            raw = model.block_grad_norms().detach().float().cpu()
            activations = model.captured_activations()
            mask = batch["attention_mask"].float()[..., None]
            denominator = mask.sum().clamp_min(1)
            activation = torch.cat(
                [
                    (value.detach().float().abs() * mask).sum(dim=(0, 1)) / denominator
                    for value in activations
                ]
            ).cpu()
            actgrad = torch.cat(
                [
                    (
                        (value.detach().float() * value.grad.detach().float()).abs()
                        * mask
                    ).sum(dim=(0, 1))
                    / denominator
                    for value in activations
                ]
            ).cpu()
            for layer, value in enumerate(activations):
                mean_sums[layer].add_((value.detach().float() * mask).sum(dim=(0, 1)))
                mean_counts[layer] += int(mask.sum())
            rows["raw"].append(raw)
            rows["ief"].append(raw / residual_norm_sq.cpu().clamp_min(EPS))
            rows["activation"].append(activation)
            rows["actgrad"].append(actgrad)
            task_ids.append(task_id)
    task_tensor = torch.tensor(task_ids)
    atlases, amplitudes = {}, {}
    for name, values in rows.items():
        atlases[name], amplitudes[name] = build_task_atlas(
            torch.stack(values), task_tensor, n_tasks=len(TASKS)
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
def evaluate_tasks(
    model: QwenFeatureProbe,
    tasks: Sequence[EncodedTask],
    device: torch.device,
    batch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    losses, exact_matches = [], []
    for task in tasks:
        task_losses, task_matches = [], []
        for start in range(0, len(task), batch_size):
            labels = task.labels[start : start + batch_size].to(device)
            batch = {
                "input_ids": task.input_ids[start : start + batch_size].to(device),
                "attention_mask": task.attention_mask[start : start + batch_size].to(
                    device
                ),
                "labels": labels,
            }
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(**batch)
            shifted_logits = output.logits[:, :-1].float()
            shifted_labels = labels[:, 1:]
            valid = shifted_labels != -100
            token_losses = torch.nn.functional.cross_entropy(
                shifted_logits.flatten(0, 1),
                shifted_labels.flatten(),
                ignore_index=-100,
                reduction="none",
            ).reshape(shifted_labels.shape)
            sample_loss = (token_losses * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1)
            predictions = shifted_logits.argmax(dim=-1)
            exact = ((predictions == shifted_labels) | ~valid).all(dim=1) & valid.any(
                dim=1
            )
            task_losses.extend(sample_loss.cpu().tolist())
            task_matches.extend(exact.float().cpu().tolist())
        losses.append(float(np.mean(task_losses)))
        exact_matches.append(float(np.mean(task_matches)))
    return torch.tensor(losses), torch.tensor(exact_matches)


def _score_methods(
    model: QwenFeatureProbe,
    atlases: dict[str, torch.Tensor],
    amplitudes: dict[str, torch.Tensor],
    online: dict[str, dict[str, torch.Tensor]],
    online_amplitudes: dict[str, torch.Tensor],
    representations: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, torch.Tensor]:
    methods = {}
    for name in ("onehot", "semantic4", "jl4"):
        representation = representations[name]
        methods[f"post_ief_{name}"] = task_aligned_importance(
            atlases["ief"] @ normalized_rows(representation),
            representation,
            amplitudes["ief"],
        )
    for statistic in ("raw", "activation", "actgrad"):
        representation = representations["semantic4"]
        methods[f"post_{statistic}_semantic4"] = task_aligned_importance(
            atlases[statistic] @ representation, representation, amplitudes[statistic]
        )
    methods["online_late_raw_semantic4"] = task_aligned_importance(
        online["late_raw"]["semantic4"],
        representations["semantic4"],
        online_amplitudes["late_raw"],
    )
    methods["weight_magnitude"] = model.block_weight_norms()[:, None].repeat(
        1, len(TASKS)
    )
    methods["random"] = torch.stack(
        [
            random_layer_balanced_scores(model.layer_sizes, seed + task)
            for task in range(len(TASKS))
        ],
        dim=1,
    )
    return methods


def interpretability_study(
    model: QwenFeatureProbe,
    tasks: Sequence[EncodedTask],
    methods: dict[str, torch.Tensor],
    means: list[torch.Tensor],
    config: LanguageConfig,
    device: torch.device,
) -> dict:
    baseline_loss, baseline_exact = evaluate_tasks(model, tasks, device)
    records = []
    for method_name, scores in methods.items():
        for fraction in config.intervention_fractions:
            for task_id in range(len(TASKS)):
                selected = layer_balanced_indices(
                    scores[:, task_id], model.layer_sizes, fraction
                )
                for mode in ("zero", "mean"):
                    model.set_intervention(
                        selected, mode=mode, means=means if mode == "mean" else None
                    )
                    loss, exact = evaluate_tasks(model, tasks, device)
                    loss_effect = selective_effect(loss - baseline_loss, task_id)
                    accuracy_effect = selective_effect(baseline_exact - exact, task_id)
                    records.append(
                        {
                            "method": method_name,
                            "fraction": fraction,
                            "task": task_id,
                            "task_name": TASKS[task_id],
                            "intervention": mode,
                            "target_loss_increase": loss_effect["target_effect"],
                            "selective_loss_increase": loss_effect["selectivity"],
                            "target_exact_match_drop": accuracy_effect["target_effect"],
                            "selective_exact_match_drop": accuracy_effect[
                                "selectivity"
                            ],
                        }
                    )
    model.set_intervention(None)
    return {
        "baseline_task_loss": baseline_loss,
        "baseline_task_exact_match": baseline_exact,
        "baseline_exact_match_mean": float(baseline_exact.mean()),
        "sanity": {
            "passes_nonzero_exact_match_gate": bool(float(baseline_exact.mean()) > 0.0),
            "mean_teacher_forced_loss": float(baseline_loss.mean()),
        },
        "records": records,
    }


def _restore(model: QwenFeatureProbe, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)
    model.set_intervention(None)
    model.set_feature_gates(None)
    model.clear_capture()


def _recover_target(
    model: QwenFeatureProbe,
    base_state: dict[str, torch.Tensor],
    retained: Sequence[torch.Tensor],
    task_id: int,
    train_tasks: Sequence[EncodedTask],
    validation_tasks: Sequence[EncodedTask],
    config: LanguageConfig,
    device: torch.device,
) -> float:
    _restore(model, base_state)
    model.set_intervention(retained, mode="retain")
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-5)
    generator = torch.Generator().manual_seed(config.seed + 7000 + task_id)
    model.train()
    for _ in range(config.recovery_steps):
        batch = task_batch(train_tasks[task_id], config.batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
    recovered = float(evaluate_tasks(model, validation_tasks, device)[0][task_id])
    _restore(model, base_state)
    return recovered


def pruning_study(
    model: QwenFeatureProbe,
    base_state: dict[str, torch.Tensor],
    train_tasks: Sequence[EncodedTask],
    validation_tasks: Sequence[EncodedTask],
    methods: dict[str, torch.Tensor],
    config: LanguageConfig,
    device: torch.device,
) -> dict:
    baseline_loss, baseline_exact = evaluate_tasks(model, validation_tasks, device)
    records = []
    recovery_methods = {"post_ief_semantic4", "post_activation_semantic4", "random"}
    for method_name, scores in methods.items():
        for fraction in config.pruning_fractions:
            for task_id in range(len(TASKS)):
                retained = layer_balanced_indices(
                    scores[:, task_id], model.layer_sizes, fraction
                )
                model.set_intervention(retained, mode="retain")
                loss, exact = evaluate_tasks(model, validation_tasks, device)
                record = {
                    "method": method_name,
                    "retained_fraction": fraction,
                    "task": task_id,
                    "task_name": TASKS[task_id],
                    "baseline_loss": float(baseline_loss[task_id]),
                    "zero_shot_loss": float(loss[task_id]),
                    "baseline_exact_match": float(baseline_exact[task_id]),
                    "zero_shot_exact_match": float(exact[task_id]),
                    "recovered_loss": float("nan"),
                }
                if (
                    config.recovery_steps > 0
                    and method_name in recovery_methods
                    and abs(fraction - 0.25) < 1e-8
                    and task_id < 2
                ):
                    model.set_intervention(None)
                    record["recovered_loss"] = _recover_target(
                        model,
                        base_state,
                        retained,
                        task_id,
                        train_tasks,
                        validation_tasks,
                        config,
                        device,
                    )
                records.append(record)
    model.set_intervention(None)
    return {"records": records}


def continual_learning_study(
    model: QwenFeatureProbe,
    base_state: dict[str, torch.Tensor],
    train_tasks: Sequence[EncodedTask],
    validation_tasks: Sequence[EncodedTask],
    methods: dict[str, torch.Tensor],
    config: LanguageConfig,
    device: torch.device,
) -> dict:
    selected_methods = {
        "post_ief": methods["post_ief_semantic4"],
        "online_raw_ef": methods["online_late_raw_semantic4"],
        "raw_ef": methods["post_raw_semantic4"],
        "activation": methods["post_activation_semantic4"],
        "random": methods["random"],
        "none": torch.zeros_like(methods["random"]),
    }
    if config.smoke:
        selected_methods = {
            name: selected_methods[name] for name in ("post_ief", "random", "none")
        }
    orders = {
        "related": [1, 2, 0, 3, 4, 5],
        "dissimilar": [0, 5, 1, 4, 2, 3],
    }
    output = {}
    for order_name, order in orders.items():
        output[order_name] = {}
        for method_name, task_scores in selected_methods.items():
            _restore(model, base_state)
            optimizer = torch.optim.SGD(model.parameters(), lr=2e-5)
            generator = torch.Generator().manual_seed(
                config.seed + 9000 + sum(ord(char) for char in order_name + method_name)
            )
            protection = torch.zeros(model.n_modules)
            history = torch.full((len(TASKS), len(TASKS)), float("nan"))
            for stage, task_id in enumerate(order):
                model.train()
                for _ in range(config.cl_steps_per_task):
                    batch = task_batch(
                        train_tasks[task_id], config.batch_size, generator, device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        loss = model(**batch).loss
                    loss.backward()
                    if stage > 0 and method_name != "none":
                        model.apply_feature_gradient_scale(
                            protection, config.cl_strength
                        )
                    optimizer.step()
                losses, _exact = evaluate_tasks(model, validation_tasks, device)
                for seen_stage, seen_task in enumerate(order[: stage + 1]):
                    history[stage, seen_stage] = losses[seen_task]
                importance = task_scores[:, task_id]
                importance = importance / importance.mean().clamp_min(EPS)
                protection = torch.maximum(protection, importance)
            output[order_name][method_name] = {
                "order": order,
                "history": history,
                "summary": continual_learning_metrics(history, higher_is_better=False),
            }
    _restore(model, base_state)
    return output


def run_experiment(
    config: LanguageConfig, *, device: torch.device | None = None
) -> dict:
    seed_everything(config.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_started = time.perf_counter()
    reset_accelerator_peak_memory(device)
    model, tokenizer = load_model_and_tokenizer(config, device)
    train_tasks, validation_tasks = load_task_suite(tokenizer, config)
    semantic = semantic_task_representation(4)
    representations = {
        "onehot": torch.eye(len(TASKS)),
        "semantic4": semantic,
        "jl4": jl_task_representation(len(TASKS), 4, config.seed + 4),
    }
    online, online_amplitudes, train_metrics = train_model(
        model, train_tasks, representations, config, device
    )
    atlases, amplitudes, means, posthoc_metrics = posthoc_profiles(
        model, validation_tasks, config.reference_per_task, device
    )
    fidelity_representations = {
        "onehot": torch.eye(len(TASKS)),
        "semantic4": semantic,
        "semantic2": normalized_rows(semantic[:, :2]),
        "jl2": jl_task_representation(len(TASKS), 2, config.seed + 2),
        "jl4": representations["jl4"],
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
    application_methods = methods
    if config.smoke:
        application_methods = {
            name: methods[name]
            for name in ("post_ief_onehot", "post_ief_semantic4", "random")
        }
    base_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    interpretability = interpretability_study(
        model, validation_tasks, application_methods, means, config, device
    )
    pruning = pruning_study(
        model,
        base_state,
        train_tasks,
        validation_tasks,
        application_methods,
        config,
        device,
    )
    continual = continual_learning_study(
        model,
        base_state,
        train_tasks,
        validation_tasks,
        methods,
        config,
        device,
    )
    _restore(model, base_state)
    baseline_loss, baseline_exact = evaluate_tasks(model, validation_tasks, device)
    return {
        "setting": "language_qwen3_1.7b",
        "tasks": TASKS,
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "model": {
            "name": config.model_name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "n_modules": model.n_modules,
            "module_definition": "SwiGLU gate/up rows plus down column",
            "fine_tuning": "full parameter",
        },
        "train": train_metrics,
        "posthoc": posthoc_metrics,
        "efficiency": {
            "total_wall_seconds": time.perf_counter() - run_started,
            **accelerator_peak_memory(device),
        },
        "baseline_task_loss": baseline_loss,
        "baseline_task_exact_match": baseline_exact,
        "baseline_exact_match_mean": float(baseline_exact.mean()),
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
        "_state_dict": base_state,
    }


def smoke_config(seed: int) -> LanguageConfig:
    return LanguageConfig(
        seed=seed,
        train_per_task=2,
        validation_per_task=1,
        reference_per_task=1,
        train_steps=6,
        batch_size=1,
        tasks_per_update=6,
        intervention_fractions=(0.01,),
        pruning_fractions=(0.25,),
        recovery_steps=0,
        cl_steps_per_task=0,
        smoke=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=arc_artifact_dir("02_exploratory", "language"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()
    config = smoke_config(args.seed) if args.smoke else LanguageConfig(seed=args.seed)
    if args.train_steps is not None:
        config = LanguageConfig(**{**asdict(config), "train_steps": args.train_steps})
    result = run_experiment(config)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    arrays = result.pop("_arrays")
    state_dict = result.pop("_state_dict")
    metrics_path = args.artifact_dir / f"language_v3_metrics_seed{args.seed}.json"
    save_json(metrics_path, result)
    np.savez_compressed(
        args.artifact_dir / f"language_v3_embeddings_seed{args.seed}.npz",
        **{name: value.numpy() for name, value in arrays.items()},
    )
    torch.save(
        {"model": state_dict, "configuration": asdict(config), "tasks": TASKS},
        args.artifact_dir / f"language_v3_seed{args.seed}.pt",
    )
    print(f"LANGUAGE_V3_RESULT={metrics_path}")


if __name__ == "__main__":
    main()
