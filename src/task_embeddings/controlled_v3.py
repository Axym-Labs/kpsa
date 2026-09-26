from __future__ import annotations

import argparse
import copy
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
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


def task_mixtures() -> torch.Tensor:
    """Four primitives, all pairs, all leave-one-out triples, and two skews."""
    rows: list[list[float]] = []
    for first in range(4):
        row = [0.0] * 4
        row[first] = 1.0
        rows.append(row)
    for first in range(4):
        for second in range(first + 1, 4):
            row = [0.0] * 4
            row[first] = row[second] = 0.5
            rows.append(row)
    for omitted in range(4):
        row = [1.0 / 3.0] * 4
        row[omitted] = 0.0
        rows.append(row)
    rows.extend(([0.6, 0.2, 0.2, 0.0], [0.0, 0.2, 0.2, 0.6]))
    return torch.tensor(rows, dtype=torch.float32)


def primitive_targets(tokens: torch.Tensor, token_modulus: int) -> torch.Tensor:
    """Four normalized sequence properties with distinct computational shape."""
    values = tokens.long()
    modular_sum = (values.sum(dim=1) % token_modulus).float()
    modular_sum = 2.0 * modular_sum / max(1, token_modulus - 1) - 1.0
    parity = 2.0 * (values.sum(dim=1) % 2).float() - 1.0
    local_order = torch.sign(values[:, 1:].float() - values[:, :-1].float()).mean(dim=1)
    endpoint_match = 2.0 * ((values[:, 0] % 2) == (values[:, -1] % 2)).float() - 1.0
    return torch.stack((modular_sum, parity, local_order, endpoint_match), dim=1)


class GatedFeedForward(nn.Module):
    """SwiGLU MLP whose gate/up rows and down columns form disjoint features."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self._selected: torch.Tensor | None = None
        self._mode: Literal["zero", "mean", "retain"] = "zero"
        self._means: torch.Tensor | None = None
        self._feature_gate: torch.Tensor | None = None
        self.last_features: torch.Tensor | None = None

    @property
    def n_features(self) -> int:
        return self.gate.out_features

    def set_intervention(
        self,
        selected: torch.Tensor | None,
        *,
        mode: Literal["zero", "mean", "retain"] = "zero",
        means: torch.Tensor | None = None,
    ) -> None:
        self._selected = None if selected is None else selected.detach().long().cpu()
        self._mode = mode
        self._means = None if means is None else means.detach()

    def set_feature_gate(self, gate: torch.Tensor | None) -> None:
        if gate is not None and gate.shape != (self.n_features,):
            raise ValueError("feature gate must contain one value per feature")
        self._feature_gate = gate

    def _intervene(self, features: torch.Tensor) -> torch.Tensor:
        if self._feature_gate is not None:
            gate = self._feature_gate.to(device=features.device, dtype=features.dtype)
            features = features * gate.view(1, 1, -1)
        if self._selected is None:
            return features
        selected = self._selected.to(features.device)
        mask = torch.zeros(
            self.n_features, device=features.device, dtype=features.dtype
        )
        mask[selected] = 1.0
        if self._mode == "retain":
            return features * mask
        keep = 1.0 - mask
        if self._mode == "zero":
            return features * keep
        if self._means is None:
            raise ValueError("mean intervention requires feature means")
        means = self._means.to(device=features.device, dtype=features.dtype)
        return features * keep + means.view(1, 1, -1) * mask

    def forward(self, inputs: torch.Tensor, *, capture: bool = False) -> torch.Tensor:
        features = torch.nn.functional.silu(self.gate(inputs)) * self.up(inputs)
        features = self._intervene(features)
        if capture:
            if features.requires_grad:
                features.retain_grad()
            self.last_features = features
        return self.down(features)

    def feature_grad_norms(self) -> torch.Tensor:
        pieces = (
            self.gate.weight.grad.square().sum(dim=1)
            + self.gate.bias.grad.square()
            + self.up.weight.grad.square().sum(dim=1)
            + self.up.bias.grad.square()
            + self.down.weight.grad.square().sum(dim=0)
        )
        return pieces.float()

    @torch.no_grad()
    def feature_weight_norms(self) -> torch.Tensor:
        return (
            (
                self.gate.weight.square().sum(dim=1)
                + self.gate.bias.square()
                + self.up.weight.square().sum(dim=1)
                + self.up.bias.square()
                + self.down.weight.square().sum(dim=0)
            )
            .sqrt()
            .float()
        )

    @torch.no_grad()
    def scale_feature_gradients(self, scale: torch.Tensor) -> None:
        scale = scale.to(
            device=self.gate.weight.device, dtype=self.gate.weight.grad.dtype
        )
        self.gate.weight.grad.mul_(scale[:, None])
        self.gate.bias.grad.mul_(scale)
        self.up.weight.grad.mul_(scale[:, None])
        self.up.bias.grad.mul_(scale)
        self.down.weight.grad.mul_(scale[None, :])

    def module_owned_parameters(self) -> list[nn.Parameter]:
        """Parameters partitioned into the disjoint hidden-feature modules."""
        return [
            self.gate.weight,
            self.gate.bias,
            self.up.weight,
            self.up.bias,
            self.down.weight,
        ]


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = GatedFeedForward(d_model, d_ff)

    def forward(
        self, inputs: torch.Tensor, causal_mask: torch.Tensor, *, capture: bool
    ) -> torch.Tensor:
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = inputs + attended
        return hidden + self.mlp(self.mlp_norm(hidden), capture=capture)


class ControlledTransformer(nn.Module):
    def __init__(
        self,
        *,
        token_modulus: int = 16,
        sequence_length: int = 16,
        n_tasks: int = 16,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        d_ff: int = 1536,
    ) -> None:
        super().__init__()
        self.token_modulus = token_modulus
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(token_modulus, d_model)
        self.task_embedding = nn.Embedding(n_tasks, d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(sequence_length + 1, d_model) * 0.02
        )
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    @property
    def layer_sizes(self) -> list[int]:
        return [block.mlp.n_features for block in self.blocks]

    @property
    def n_modules(self) -> int:
        return sum(self.layer_sizes)

    def set_intervention(
        self,
        selected: Sequence[torch.Tensor] | None,
        *,
        mode: Literal["zero", "mean", "retain"] = "zero",
        means: Sequence[torch.Tensor] | None = None,
    ) -> None:
        for layer, block in enumerate(self.blocks):
            local = None if selected is None else selected[layer]
            mean = None if means is None else means[layer]
            block.mlp.set_intervention(local, mode=mode, means=mean)

    def set_feature_gates(self, gates: Sequence[torch.Tensor] | None) -> None:
        if gates is not None and len(gates) != len(self.blocks):
            raise ValueError("one feature gate is required per layer")
        for layer, block in enumerate(self.blocks):
            block.mlp.set_feature_gate(None if gates is None else gates[layer])

    def forward(
        self,
        tokens: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        capture: bool = False,
    ) -> torch.Tensor:
        token_state = self.token_embedding(tokens)
        task_state = self.task_embedding(task_ids)[:, None, :]
        hidden = torch.cat((token_state, task_state), dim=1)
        hidden = hidden + self.position_embedding[: hidden.shape[1]]
        length = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, causal_mask, capture=capture)
        return self.output(self.final_norm(hidden[:, -1])).squeeze(-1)

    def block_grad_norms(self) -> torch.Tensor:
        return torch.cat([block.mlp.feature_grad_norms() for block in self.blocks])

    def block_weight_norms(self) -> torch.Tensor:
        return torch.cat(
            [block.mlp.feature_weight_norms() for block in self.blocks]
        ).cpu()

    def captured_activations(self) -> list[torch.Tensor]:
        result = []
        for block in self.blocks:
            if block.mlp.last_features is None:
                raise RuntimeError("forward(capture=True) is required")
            result.append(block.mlp.last_features)
        return result

    def clear_capture(self) -> None:
        """Release autograd-owned feature tensors retained for attribution."""
        for block in self.blocks:
            block.mlp.last_features = None

    def module_owned_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for block in self.blocks
            for parameter in block.mlp.module_owned_parameters()
        ]

    @torch.no_grad()
    def apply_feature_gradient_scale(
        self, importance: torch.Tensor, strength: float
    ) -> None:
        offset = 0
        for size, block in zip(self.layer_sizes, self.blocks):
            local = importance[offset : offset + size].float()
            block.mlp.scale_feature_gradients(1.0 / (1.0 + strength * local))
            offset += size


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True)
class ControlledConfig:
    seed: int = 1
    token_modulus: int = 16
    sequence_length: int = 16
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    train_steps: int = 10000
    batch_size: int = 128
    reference_per_task: int = 8
    test_per_task: int = 64
    intervention_fractions: tuple[float, ...] = (0.01, 0.05, 0.10)
    pruning_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    application_tasks: int = 16
    cl_tasks: int = 16
    cl_steps_per_task: int = 12
    cl_strength: float = 8.0
    recovery_steps: int = 20


def make_task_batch(
    config: ControlledConfig,
    mixtures: torch.Tensor,
    task: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    *,
    noise: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.randint(
        0,
        config.token_modulus,
        (batch_size, config.sequence_length),
        generator=generator,
    )
    primitives = primitive_targets(tokens, config.token_modulus)
    targets = primitives.to(mixtures.device) @ mixtures[task]
    if noise:
        targets = targets + noise * torch.randn(batch_size, generator=generator).to(
            targets.device
        )
    tasks = torch.full((batch_size,), task, dtype=torch.long)
    return tokens.to(device), targets.to(device), tasks.to(device)


def make_balanced_dataset(
    config: ControlledConfig,
    mixtures: torch.Tensor,
    per_task: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tokens, targets, tasks = [], [], []
    for task in range(mixtures.shape[0]):
        batch = make_task_batch(
            config,
            mixtures,
            task,
            per_task,
            generator,
            torch.device("cpu"),
        )
        tokens.append(batch[0])
        targets.append(batch[1])
        tasks.append(batch[2])
    return torch.cat(tokens), torch.cat(targets), torch.cat(tasks)


def _task_representations(mixtures: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    n_tasks = mixtures.shape[0]
    representations = {
        "onehot": torch.eye(n_tasks),
        "structured4": normalized_rows(mixtures),
    }
    for dimension in (2, 4, 8, 12):
        representations[f"jl{dimension}"] = jl_task_representation(
            n_tasks, dimension, seed + 1000 + dimension
        )
    return representations


def train_multitask(
    model: ControlledTransformer,
    config: ControlledConfig,
    mixtures: torch.Tensor,
    representations: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    accumulator = OnlineAccumulator(
        model.n_modules,
        representations,
        device,
        beta=0.99,
        late_fraction=0.7,
        total_steps=config.train_steps,
    )
    generator = torch.Generator().manual_seed(config.seed + 11)
    task_order = torch.randint(
        0, mixtures.shape[0], (config.train_steps,), generator=generator
    )
    loss_trace = []
    started = time.perf_counter()
    accumulator_seconds = 0.0
    model.train()
    for step, task_tensor in enumerate(task_order):
        task = int(task_tensor)
        tokens, targets, task_ids = make_task_batch(
            config, mixtures, task, config.batch_size, generator, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            predictions = model(tokens, task_ids)
            residual = predictions - targets
            loss = 0.5 * residual.square().mean()
        loss.backward()
        score_started = time.perf_counter()
        raw_scores = model.block_grad_norms()
        residual_norm_sq = (
            residual.detach().float().square().sum() / config.batch_size**2
        )
        accumulator.update(raw_scores, task, residual_norm_sq, step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        accumulator_seconds += time.perf_counter() - score_started
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (
            step % max(1, config.train_steps // 20) == 0
            or step == config.train_steps - 1
        ):
            loss_trace.append({"step": step, "loss": float(loss.detach())})
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
            "loss_trace": loss_trace,
        },
    )


@torch.no_grad()
def evaluate_losses(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    model.eval()
    tokens, targets, task_ids = dataset
    n_tasks = int(task_ids.max()) + 1
    sums = torch.zeros(n_tasks, device=device)
    counts = torch.zeros(n_tasks, device=device)
    for start in range(0, len(tokens), batch_size):
        local_tokens = tokens[start : start + batch_size].to(device)
        local_targets = targets[start : start + batch_size].to(device)
        local_tasks = task_ids[start : start + batch_size].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            predictions = model(local_tokens, local_tasks)
        losses = 0.5 * (predictions.float() - local_targets).square()
        sums.scatter_add_(0, local_tasks, losses)
        counts.scatter_add_(0, local_tasks, torch.ones_like(losses))
    return (sums / counts.clamp_min(1)).cpu()


def posthoc_profiles(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[torch.Tensor], dict]:
    all_scores: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("ief", "raw", "activation", "actgrad")
    }
    task_values: list[int] = []
    mean_sums = [torch.zeros(size, device=device) for size in model.layer_sizes]
    mean_counts = [0 for _ in model.layer_sizes]
    tokens, targets, task_ids = dataset
    started = time.perf_counter()
    model.eval()
    model.set_intervention(None)
    for token_row, target, task_tensor in zip(tokens, targets, task_ids):
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            prediction = model(
                token_row[None].to(device), task_tensor[None].to(device), capture=True
            )
            residual = prediction - target.to(device)
            loss = 0.5 * residual.square().sum()
        loss.backward()
        raw = model.block_grad_norms().detach().float().cpu()
        residual_norm_sq = residual.detach().float().square().sum()
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
        all_scores["raw"].append(raw)
        all_scores["ief"].append(raw / residual_norm_sq.cpu().clamp_min(EPS))
        all_scores["activation"].append(activation)
        all_scores["actgrad"].append(actgrad)
        task_values.append(int(task_tensor))
    task_tensor = torch.tensor(task_values)
    atlases: dict[str, torch.Tensor] = {}
    amplitudes: dict[str, torch.Tensor] = {}
    for name, rows in all_scores.items():
        atlases[name], amplitudes[name] = build_task_atlas(
            torch.stack(rows), task_tensor, n_tasks=16
        )
    feature_means = [
        (total / max(1, count)).detach().cpu()
        for total, count in zip(mean_sums, mean_counts)
    ]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.clear_capture()
    return (
        atlases,
        amplitudes,
        feature_means,
        {
            "wall_seconds": elapsed,
            "samples": len(tokens),
            "samples_per_second": len(tokens) / max(elapsed, EPS),
        },
    )


def _score_methods(
    model: ControlledTransformer,
    atlases: dict[str, torch.Tensor],
    amplitudes: dict[str, torch.Tensor],
    online: dict[str, dict[str, torch.Tensor]],
    online_amplitudes: dict[str, torch.Tensor],
    representations: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, torch.Tensor]:
    methods: dict[str, torch.Tensor] = {}
    for representation_name in ("onehot", "structured4", "jl4"):
        representation = representations[representation_name]
        embedding = atlases["ief"] @ normalized_rows(representation)
        methods[f"post_ief_{representation_name}"] = task_aligned_importance(
            embedding, representation, amplitudes["ief"]
        )
    for statistic in ("raw", "activation", "actgrad"):
        representation = representations["structured4"]
        embedding = atlases[statistic] @ normalized_rows(representation)
        methods[f"post_{statistic}_structured4"] = task_aligned_importance(
            embedding, representation, amplitudes[statistic]
        )
    methods["online_late_raw_structured4"] = task_aligned_importance(
        online["late_raw"]["structured4"],
        representations["structured4"],
        online_amplitudes["late_raw"],
    )
    methods["weight_magnitude"] = model.block_weight_norms()[:, None].repeat(1, 16)
    methods["random"] = torch.stack(
        [
            random_layer_balanced_scores(model.layer_sizes, seed + task)
            for task in range(16)
        ],
        dim=1,
    )
    return methods


def interpretability_study(
    model: ControlledTransformer,
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    methods: dict[str, torch.Tensor],
    feature_means: list[torch.Tensor],
    config: ControlledConfig,
    device: torch.device,
) -> dict:
    baseline = evaluate_losses(model, test_data, device)
    records = []
    for method_name, scores in methods.items():
        for fraction in config.intervention_fractions:
            for task in range(min(config.application_tasks, 16)):
                selected = layer_balanced_indices(
                    scores[:, task], model.layer_sizes, fraction
                )
                for mode in ("zero", "mean"):
                    model.set_intervention(
                        selected,
                        mode=mode,
                        means=feature_means if mode == "mean" else None,
                    )
                    losses = evaluate_losses(model, test_data, device)
                    record = selective_effect(losses - baseline, task)
                    record.update(
                        method=method_name,
                        fraction=fraction,
                        task=task,
                        intervention=mode,
                    )
                    records.append(record)
    model.set_intervention(None)
    return {"baseline_task_loss": baseline, "records": records}


def _recover_pruned_target(
    base_model: ControlledTransformer,
    selected: Sequence[torch.Tensor],
    task: int,
    config: ControlledConfig,
    mixtures: torch.Tensor,
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    model = copy.deepcopy(base_model).to(device)
    model.set_intervention(selected, mode="retain")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(config.seed + 7000 + task)
    model.train()
    for _ in range(config.recovery_steps):
        tokens, targets, task_ids = make_task_batch(
            config, mixtures, task, config.batch_size, generator, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss = 0.5 * (model(tokens, task_ids) - targets).square().mean()
        loss.backward()
        optimizer.step()
    return float(evaluate_losses(model, test_data, device)[task])


def pruning_study(
    model: ControlledTransformer,
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    methods: dict[str, torch.Tensor],
    config: ControlledConfig,
    mixtures: torch.Tensor,
    device: torch.device,
) -> dict:
    baseline = evaluate_losses(model, test_data, device)
    records = []
    recover_methods = {"post_ief_structured4", "post_activation_structured4", "random"}
    for method_name, scores in methods.items():
        for fraction in config.pruning_fractions:
            for task in range(min(config.application_tasks, 16)):
                retained = layer_balanced_indices(
                    scores[:, task], model.layer_sizes, fraction
                )
                model.set_intervention(retained, mode="retain")
                pruned_loss = float(evaluate_losses(model, test_data, device)[task])
                record = {
                    "method": method_name,
                    "retained_fraction": fraction,
                    "task": task,
                    "baseline_loss": float(baseline[task]),
                    "zero_shot_loss": pruned_loss,
                    "recovered_loss": float("nan"),
                }
                if (
                    config.recovery_steps > 0
                    and method_name in recover_methods
                    and abs(fraction - 0.25) < 1e-8
                    and task < min(4, config.application_tasks)
                ):
                    model.set_intervention(None)
                    record["recovered_loss"] = _recover_pruned_target(
                        model, retained, task, config, mixtures, test_data, device
                    )
                records.append(record)
    model.set_intervention(None)
    return {"baseline_task_loss": baseline, "records": records}


def _continual_orders(mixtures: torch.Tensor, n_tasks: int) -> dict[str, list[int]]:
    candidates = list(range(n_tasks))
    related = sorted(candidates, key=lambda task: (int(mixtures[task].argmax()), task))
    normalized = normalized_rows(mixtures[:n_tasks])
    dissimilar = [0]
    remaining = set(candidates[1:])
    while remaining:
        last = dissimilar[-1]
        nxt = min(
            remaining, key=lambda task: float(normalized[last] @ normalized[task])
        )
        dissimilar.append(nxt)
        remaining.remove(nxt)
    return {"related": related, "dissimilar": dissimilar}


def continual_learning_study(
    base_model: ControlledTransformer,
    test_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    methods: dict[str, torch.Tensor],
    config: ControlledConfig,
    mixtures: torch.Tensor,
    device: torch.device,
) -> dict:
    selected_methods = {
        "post_ief": methods["post_ief_structured4"],
        "online_raw_ef": methods["online_late_raw_structured4"],
        "raw_ef": methods["post_raw_structured4"],
        "activation": methods["post_activation_structured4"],
        "random": methods["random"],
        "none": torch.zeros_like(methods["random"]),
    }
    n_tasks = min(config.cl_tasks, 16)
    orders = _continual_orders(mixtures, n_tasks)
    output: dict[str, dict[str, dict]] = {}
    for order_name, order in orders.items():
        output[order_name] = {}
        for method_name, task_scores in selected_methods.items():
            model = copy.deepcopy(base_model).to(device)
            model.set_intervention(None)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            generator = torch.Generator().manual_seed(
                config.seed + 9000 + sum(ord(char) for char in order_name + method_name)
            )
            protection = torch.zeros(model.n_modules)
            history = torch.full((n_tasks, n_tasks), float("nan"))
            for stage, task in enumerate(order):
                model.train()
                for _ in range(config.cl_steps_per_task):
                    tokens, targets, task_ids = make_task_batch(
                        config, mixtures, task, config.batch_size, generator, device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        loss = 0.5 * (model(tokens, task_ids) - targets).square().mean()
                    loss.backward()
                    if method_name != "none" and stage > 0:
                        model.apply_feature_gradient_scale(
                            protection, config.cl_strength
                        )
                    optimizer.step()
                losses = evaluate_losses(model, test_data, device)
                for seen_stage, seen_task in enumerate(order[: stage + 1]):
                    history[stage, seen_stage] = losses[seen_task]
                task_importance = task_scores[:, task]
                task_importance = task_importance / task_importance.mean().clamp_min(
                    EPS
                )
                protection = torch.maximum(protection, task_importance)
            output[order_name][method_name] = {
                "order": order,
                "history": history,
                "summary": continual_learning_metrics(history, higher_is_better=False),
            }
    return output


def run_experiment(
    config: ControlledConfig, *, device: torch.device | None = None
) -> dict:
    seed_everything(config.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_started = time.perf_counter()
    reset_accelerator_peak_memory(device)
    mixtures = task_mixtures()
    representations = _task_representations(mixtures, config.seed)
    model = ControlledTransformer(
        token_modulus=config.token_modulus,
        sequence_length=config.sequence_length,
        n_tasks=16,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
    ).to(device)
    online, online_amplitudes, train_metrics = train_multitask(
        model, config, mixtures.to(device), representations, device
    )
    reference = make_balanced_dataset(
        config, mixtures, config.reference_per_task, config.seed + 101
    )
    test_data = make_balanced_dataset(
        config, mixtures, config.test_per_task, config.seed + 202
    )
    atlases, amplitudes, feature_means, posthoc_metrics = posthoc_profiles(
        model, reference, device
    )
    fidelity = {
        name: source_fidelity(atlases["ief"], representation)
        for name, representation in representations.items()
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
    baseline_loss = evaluate_losses(model, test_data, device)
    test_targets = test_data[1]
    test_tasks = test_data[2]
    zero_baseline_loss = float(0.5 * test_targets.square().mean())
    task_means = torch.stack(
        [test_targets[test_tasks == task].mean() for task in range(16)]
    )
    task_mean_baseline_loss = float(
        0.5 * (test_targets - task_means[test_tasks]).square().mean()
    )
    interpretability = interpretability_study(
        model, test_data, methods, feature_means, config, device
    )
    pruning = pruning_study(model, test_data, methods, config, mixtures, device)
    continual = continual_learning_study(
        model, test_data, methods, config, mixtures, device
    )
    result = {
        "setting": "controlled_transformer",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "model": {
            "parameters": parameter_count(model),
            "n_layers": config.n_layers,
            "n_modules": model.n_modules,
            "module_definition": "SwiGLU gate/up rows plus down column",
        },
        "train": train_metrics,
        "posthoc": posthoc_metrics,
        "efficiency": {
            "total_wall_seconds": time.perf_counter() - run_started,
            **accelerator_peak_memory(device),
        },
        "baseline_task_loss": baseline_loss,
        "baseline_loss_mean": float(baseline_loss.mean()),
        "sanity_baselines": {
            "zero_predictor_half_mse": zero_baseline_loss,
            "task_mean_predictor_half_mse": task_mean_baseline_loss,
        },
        "source_fidelity": fidelity,
        "interpretability": interpretability,
        "pruning": pruning,
        "continual_learning": continual,
        "_arrays": {
            **{f"atlas_{name}": value for name, value in atlases.items()},
            **{f"amplitude_{name}": value for name, value in amplitudes.items()},
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
    return result


def _smoke_config(seed: int) -> ControlledConfig:
    return ControlledConfig(
        seed=seed,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        train_steps=8,
        batch_size=8,
        reference_per_task=1,
        test_per_task=2,
        intervention_fractions=(0.10,),
        pruning_fractions=(0.25,),
        application_tasks=2,
        cl_tasks=3,
        cl_steps_per_task=1,
        recovery_steps=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=arc_artifact_dir("02_exploratory", "controlled"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()
    config = (
        _smoke_config(args.seed) if args.smoke else ControlledConfig(seed=args.seed)
    )
    if args.train_steps is not None:
        config = ControlledConfig(**{**asdict(config), "train_steps": args.train_steps})
    result = run_experiment(config)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    arrays = result.pop("_arrays")
    state_dict = result.pop("_state_dict")
    metrics_path = args.artifact_dir / f"controlled_v3_metrics_seed{args.seed}.json"
    save_json(metrics_path, result)
    np.savez_compressed(
        args.artifact_dir / f"controlled_v3_embeddings_seed{args.seed}.npz",
        **{name: value.numpy() for name, value in arrays.items()},
    )
    torch.save(
        {
            "model": state_dict,
            "configuration": asdict(config),
            "mixtures": task_mixtures(),
        },
        args.artifact_dir / f"controlled_v3_seed{args.seed}.pt",
    )
    print(f"CONTROLLED_V3_RESULT={metrics_path}")


if __name__ == "__main__":
    main()
