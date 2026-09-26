from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import ConstantInputWarning, spearmanr

from .applications import causal_assay_gate, faithfulness_scores
from .common import (
    EPS,
    arc_artifact_dir,
    build_task_atlas,
    calibrate_importance_matrices,
    conditional_importance,
    cross_validated_task_query_fidelity,
    jl_task_representation,
    normalized_rows,
    save_json,
    seed_everything,
)
from .controlled_v3 import ControlledTransformer
from .controlled_v4 import (
    HardControlledConfig,
    build_model,
    fraction_to_count,
    hard_task_mixtures,
    make_program_dataset,
)


@dataclass(frozen=True)
class NaturalAssayConfig:
    seed: int
    reference_per_task: int = 32
    calibration_per_task: int = 96
    test_per_task: int = 192
    causal_tasks: int = 6
    circuit_fractions: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
    pruning_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)
    attainable_steps: int = 80
    attainable_restarts: int = 3
    attainable_learning_rate: float = 0.15
    random_masks: int = 32
    jl_outcome_repeats: int = 3
    jl_query_repeats: int = 10
    profile_batch_size: int = 1


def split_global_mask(mask: torch.Tensor, layer_sizes: list[int]) -> list[torch.Tensor]:
    values = mask.detach().bool().cpu().flatten()
    if values.numel() != sum(layer_sizes):
        raise ValueError("layer sizes must cover the global mask")
    result: list[torch.Tensor] = []
    offset = 0
    for size in layer_sizes:
        result.append(torch.where(values[offset : offset + size])[0])
        offset += size
    return result


def _reconstruct_importance(
    importance: torch.Tensor, representation: torch.Tensor
) -> torch.Tensor:
    features = normalized_rows(representation.detach().float().cpu())
    return (importance.detach().float().cpu() @ features @ features.T).clamp_min(0)


def build_method_scores(
    profiles: dict[str, torch.Tensor],
    mixtures: torch.Tensor,
    *,
    seed: int,
    jl_repeats: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    required = {"ief", "raw_ef", "activation", "activation_gradient"}
    if not required <= profiles.keys():
        raise ValueError(f"missing profiles: {sorted(required - profiles.keys())}")
    semantic = normalized_rows(mixtures)
    onehot = torch.eye(mixtures.shape[0])
    permutation = torch.randperm(
        mixtures.shape[0],
        generator=torch.Generator().manual_seed(seed + 1500),
    )
    permuted_semantic = semantic[permutation]
    ief = profiles["ief"]
    methods = {
        "posthoc_ief__onehot": _reconstruct_importance(ief, onehot),
        "posthoc_ief__semantic6": _reconstruct_importance(ief, semantic),
        "posthoc_ief__permuted_semantic6": _reconstruct_importance(
            ief, permuted_semantic
        ),
        "posthoc_raw_ef__semantic6": _reconstruct_importance(
            profiles["raw_ef"], semantic
        ),
        "posthoc_activation__semantic6": _reconstruct_importance(
            profiles["activation"], semantic
        ),
        "posthoc_activation_gradient__semantic6": _reconstruct_importance(
            profiles["activation_gradient"], semantic
        ),
        "task_agnostic_mean": ief.mean(dim=1, keepdim=True).repeat(
            1, mixtures.shape[0]
        ),
    }
    for repeat in range(jl_repeats):
        jl = jl_task_representation(mixtures.shape[0], 6, seed + 1000 + repeat)
        methods[f"posthoc_ief__jl6_seed{repeat}"] = _reconstruct_importance(ief, jl)
    methods["random"] = torch.rand(
        ief.shape, generator=torch.Generator().manual_seed(seed + 2000)
    )
    return calibrate_importance_matrices(methods, reference_key="posthoc_ief__onehot")


def collect_importance_profiles(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    scores: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("ief", "raw_ef", "activation", "activation_gradient")
    }
    observed_tasks: list[int] = []
    tokens, targets, task_ids = dataset
    n_tasks = int(task_ids.max()) + 1
    started = time.perf_counter()
    model.eval()
    model.set_intervention(None)
    model.set_feature_gates(None)
    for token_row, target, task in zip(tokens, targets, task_ids):
        model.zero_grad(set_to_none=True)
        prediction = model(
            token_row[None].to(device), task[None].to(device), capture=True
        )
        residual = prediction - target.to(device)
        (0.5 * residual.square().sum()).backward()
        raw = model.block_grad_norms().detach().float().cpu()
        residual_norm_sq = residual.detach().float().square().sum().cpu()
        activations = model.captured_activations()
        activation = torch.cat(
            [value.detach().float().abs().mean(dim=(0, 1)) for value in activations]
        ).cpu()
        activation_gradient = torch.cat(
            [
                (value.detach().float() * value.grad.detach().float())
                .abs()
                .mean(dim=(0, 1))
                for value in activations
            ]
        ).cpu()
        scores["raw_ef"].append(raw)
        scores["ief"].append(raw / residual_norm_sq.clamp_min(EPS))
        scores["activation"].append(activation)
        scores["activation_gradient"].append(activation_gradient)
        observed_tasks.append(int(task))
    task_tensor = torch.tensor(observed_tasks)
    profiles = {}
    for name, rows in scores.items():
        atlas, amplitude = build_task_atlas(
            torch.stack(rows), task_tensor, n_tasks=n_tasks
        )
        profiles[name] = conditional_importance(atlas, amplitude)
    model.clear_capture()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return profiles, {
        "samples": len(tokens),
        "wall_seconds": time.perf_counter() - started,
        "samples_per_task": len(tokens) // n_tasks,
    }


def reference_reliability(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    correlations = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for task in range(first.shape[1]):
            result = spearmanr(first[:, task].numpy(), second[:, task].numpy())
            correlations.append(float(result.statistic))
    finite = [value for value in correlations if math.isfinite(value)]
    minimum = min(finite) if len(finite) == first.shape[1] else float("nan")
    return {
        "per_task_spearman": correlations,
        "mean_spearman": float(np.mean(finite)) if finite else float("nan"),
        "minimum_spearman": minimum,
        "passed": bool(math.isfinite(minimum) and minimum >= 0.8),
        "threshold": 0.8,
    }


def source_query_results(
    first: torch.Tensor,
    second: torch.Tensor,
    mixtures: torch.Tensor,
    config: NaturalAssayConfig,
) -> dict[str, Any]:
    folds = [[task] for task in range(21, mixtures.shape[0])]
    semantic = normalized_rows(mixtures)
    output = {
        "semantic6": cross_validated_task_query_fidelity(
            first,
            semantic,
            folds=folds,
            top_fraction=0.05,
            target_importance=second,
        ),
        "onehot": cross_validated_task_query_fidelity(
            first,
            torch.eye(mixtures.shape[0]),
            folds=folds,
            top_fraction=0.05,
            target_importance=second,
        ),
        "task_agnostic": cross_validated_task_query_fidelity(
            first,
            torch.ones(mixtures.shape[0], 1),
            folds=folds,
            top_fraction=0.05,
            target_importance=second,
        ),
    }
    for repeat in range(config.jl_query_repeats):
        jl = jl_task_representation(mixtures.shape[0], 6, config.seed + 3000 + repeat)
        output[f"jl6_seed{repeat}"] = cross_validated_task_query_fidelity(
            first,
            jl,
            folds=folds,
            top_fraction=0.05,
            target_importance=second,
        )
        permutation = torch.randperm(
            mixtures.shape[0],
            generator=torch.Generator().manual_seed(config.seed + 4000 + repeat),
        )
        output[f"permuted_semantic6_seed{repeat}"] = (
            cross_validated_task_query_fidelity(
                first,
                semantic[permutation],
                folds=folds,
                top_fraction=0.05,
                target_importance=second,
            )
        )
    return output


@torch.no_grad()
def _predict_task(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    task: int,
    device: torch.device,
    *,
    mask: torch.Tensor | None = None,
    mode: str = "retain",
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, targets, task_ids = dataset
    selected = task_ids == task
    local_tokens = tokens[selected]
    local_targets = targets[selected]
    if mask is None:
        model.set_intervention(None)
    else:
        model.set_intervention(split_global_mask(mask, model.layer_sizes), mode=mode)
    predictions = []
    model.eval()
    for start in range(0, len(local_tokens), batch_size):
        batch = local_tokens[start : start + batch_size].to(device)
        ids = torch.full((len(batch),), task, dtype=torch.long, device=device)
        predictions.append(model(batch, ids).float().cpu())
    model.set_intervention(None)
    return torch.cat(predictions), local_targets.float().cpu()


def task_circuit_result(
    model: ControlledTransformer,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    task: int,
    mask: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.set_feature_gates(None)
    full, targets = _predict_task(model, dataset, task, device)
    empty = torch.zeros(model.n_modules, dtype=torch.bool)
    null, _ = _predict_task(model, dataset, task, device, mask=empty, mode="retain")
    kept, _ = _predict_task(model, dataset, task, device, mask=mask, mode="retain")
    dropped, _ = _predict_task(model, dataset, task, device, mask=mask, mode="zero")
    null_divergence = float((null - full).square().mean())
    kept_divergence = float((kept - full).square().mean())
    dropped_divergence = float((dropped - full).square().mean())
    if null_divergence > EPS:
        normalized = faithfulness_scores(
            kept_divergence=torch.tensor([kept_divergence]),
            dropped_divergence=torch.tensor([dropped_divergence]),
            null_divergence=null_divergence,
        )
        sufficiency = float(normalized["sufficiency"][0])
        necessity = float(normalized["necessity"][0])
    else:
        sufficiency = float("nan")
        necessity = float("nan")
    return {
        "null_divergence": null_divergence,
        "kept_divergence": kept_divergence,
        "dropped_divergence": dropped_divergence,
        "sufficiency": sufficiency,
        "necessity": necessity,
        "full_variance": float(full.var(unbiased=False)),
        "full_half_mse": float(0.5 * (full - targets).square().mean()),
        "kept_half_mse": float(0.5 * (kept - targets).square().mean()),
        "dropped_half_mse": float(0.5 * (dropped - targets).square().mean()),
    }


def _topk_mask(scores: torch.Tensor, count: int) -> torch.Tensor:
    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    if count:
        mask[torch.topk(scores.detach().float().cpu().flatten(), count).indices] = True
    return mask


def _profile_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.float().norm() * right.float().norm()
    if float(denominator) <= EPS:
        return float("nan")
    return float(torch.dot(left.float(), right.float()) / denominator)


def _selected_score_mass(scores: torch.Tensor, mask: torch.Tensor) -> float:
    scores = scores.detach().float().cpu().clamp_min(0)
    denominator = float(scores.sum())
    if denominator <= EPS:
        return float("nan")
    return float(scores[mask.detach().bool().cpu()].sum()) / denominator


def circuit_diagnostic_fields(
    task_basis_scores: torch.Tensor,
    full_atlas_scores: torch.Tensor,
    primitive_mask: torch.Tensor,
    *,
    primitive: int,
    task: int,
    mixtures: torch.Tensor,
) -> dict[str, float | int]:
    """Describe a pure-task circuit's alignment with an evaluation task."""
    count = int(primitive_mask.sum())
    task_basis_task_mask = _topk_mask(task_basis_scores[:, task], count)
    overlap = int((primitive_mask.bool().cpu() & task_basis_task_mask).sum())
    return {
        "task_cardinality": int((mixtures[task] > 0).sum()),
        "task_basis_profile_cosine": _profile_cosine(
            task_basis_scores[:, primitive], task_basis_scores[:, task]
        ),
        "full_atlas_profile_cosine": _profile_cosine(
            full_atlas_scores[:, primitive], full_atlas_scores[:, task]
        ),
        "task_basis_selected_score_mass": _selected_score_mass(
            task_basis_scores[:, task], primitive_mask
        ),
        "full_atlas_selected_score_mass": _selected_score_mass(
            full_atlas_scores[:, task], primitive_mask
        ),
        "task_basis_topk_overlap": overlap / count,
    }


def pure_circuit_overlap_summary(
    task_basis_scores: torch.Tensor,
    task_agnostic_scores: torch.Tensor,
    *,
    count: int,
    n_primitives: int,
) -> dict[str, Any]:
    """Quantify whether pure-task circuits reuse the same generic modules."""
    masks = [
        _topk_mask(task_basis_scores[:, primitive], count)
        for primitive in range(n_primitives)
    ]
    pairwise = [
        float((masks[left] & masks[right]).sum()) / count
        for left in range(n_primitives)
        for right in range(left + 1, n_primitives)
    ]
    agnostic_mask = _topk_mask(task_agnostic_scores, count)
    agnostic = [float((mask & agnostic_mask).sum()) / count for mask in masks]
    union = torch.stack(masks).any(dim=0)
    return {
        "pairwise_overlap_fractions": pairwise,
        "mean_pairwise_overlap_fraction": float(np.mean(pairwise)),
        "task_agnostic_overlap_fractions": agnostic,
        "mean_task_agnostic_overlap_fraction": float(np.mean(agnostic)),
        "union_modules": int(union.sum()),
        "union_fraction_of_modules": float(union.float().mean()),
    }


def optimize_attainable_mask(
    model: ControlledTransformer,
    calibration: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    task: int,
    count: int,
    config: NaturalAssayConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tokens, _, task_ids = calibration
    local_tokens = tokens[task_ids == task].to(device)
    local_ids = torch.full((len(local_tokens),), task, dtype=torch.long, device=device)
    model.set_intervention(None)
    model.set_feature_gates(None)
    model.eval()
    with torch.no_grad():
        full = model(local_tokens, local_ids).detach()
    variance = full.float().var(unbiased=False).clamp_min(1e-6)
    original_requires_grad = [
        parameter.requires_grad for parameter in model.parameters()
    ]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    best_mask: torch.Tensor | None = None
    best_divergence = float("inf")
    restart_records = []
    target_fraction = count / model.n_modules
    initial_logit = math.log(target_fraction / max(EPS, 1.0 - target_fraction))
    for restart in range(config.attainable_restarts):
        generator = torch.Generator(device=device).manual_seed(
            config.seed + 5000 + 100 * task + restart
        )
        logits = torch.nn.Parameter(
            torch.full((model.n_modules,), initial_logit, device=device)
            + 0.05 * torch.randn(model.n_modules, generator=generator, device=device)
        )
        optimizer = torch.optim.Adam([logits], lr=config.attainable_learning_rate)
        final_loss = float("nan")
        for _ in range(config.attainable_steps):
            optimizer.zero_grad(set_to_none=True)
            gates = torch.sigmoid(logits)
            split_gates = []
            offset = 0
            for size in model.layer_sizes:
                split_gates.append(gates[offset : offset + size])
                offset += size
            model.set_feature_gates(split_gates)
            prediction = model(local_tokens, local_ids)
            behavior = (prediction.float() - full.float()).square().mean() / variance
            budget = (gates.mean() - target_fraction).square()
            discreteness = (gates * (1.0 - gates)).mean()
            loss = behavior + 20.0 * budget + 0.01 * discreteness
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
        model.set_feature_gates(None)
        candidate = _topk_mask(logits, count)
        result = task_circuit_result(model, calibration, task, candidate, device)
        restart_records.append(
            {
                "restart": restart,
                "calibration_kept_divergence": result["kept_divergence"],
                "optimization_loss": final_loss,
            }
        )
        if result["kept_divergence"] < best_divergence:
            best_mask = candidate
            best_divergence = result["kept_divergence"]

    for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
        parameter.requires_grad_(requires_grad)
    model.set_feature_gates(None)
    model.zero_grad(set_to_none=True)
    if best_mask is None:
        raise RuntimeError("attainable-mask optimization produced no candidate")
    return best_mask, {
        "selected_restart": int(
            np.argmin([item["calibration_kept_divergence"] for item in restart_records])
        ),
        "calibration_kept_divergence": best_divergence,
        "restarts": restart_records,
    }


def causal_interpretability_study(
    model: ControlledTransformer,
    methods: dict[str, torch.Tensor],
    calibration: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: NaturalAssayConfig,
    device: torch.device,
) -> dict[str, Any]:
    records = []
    premise_records = []
    random_generator = torch.Generator().manual_seed(config.seed + 6000)
    for task in range(config.causal_tasks):
        for fraction in config.circuit_fractions:
            count = fraction_to_count(fraction, model.n_modules)
            attainable_mask, optimization = optimize_attainable_mask(
                model, calibration, task, count, config, device
            )
            attainable = task_circuit_result(model, test, task, attainable_mask, device)
            random_sufficiencies = []
            for _ in range(config.random_masks):
                random_mask = torch.zeros(model.n_modules, dtype=torch.bool)
                random_mask[
                    torch.randperm(model.n_modules, generator=random_generator)[:count]
                ] = True
                random_sufficiencies.append(
                    task_circuit_result(model, test, task, random_mask, device)[
                        "sufficiency"
                    ]
                )
            random_p95 = float(torch.quantile(torch.tensor(random_sufficiencies), 0.95))
            minimum_span = max(1e-6, 0.05 * attainable["full_variance"])
            gate = causal_assay_gate(
                null_divergence=attainable["null_divergence"],
                attainable_sufficiency=attainable["sufficiency"],
                random_sufficiency_p95=random_p95,
                minimum_span=minimum_span,
                minimum_gap=0.20,
            )
            premise_records.append(
                {
                    "task": task,
                    "requested_fraction": fraction,
                    "selected_modules": count,
                    "selected_fraction_actual": count / model.n_modules,
                    "attainable": attainable,
                    "random_sufficiency_p95": random_p95,
                    "minimum_span": minimum_span,
                    "optimization": optimization,
                    "gate": gate,
                }
            )
            for method, scores in methods.items():
                mask = _topk_mask(scores[:, task], count)
                result = task_circuit_result(model, test, task, mask, device)
                result.update(
                    method=method,
                    task=task,
                    requested_fraction=fraction,
                    selected_modules=count,
                    selected_fraction_actual=count / model.n_modules,
                    premise_passed=gate["passed"],
                )
                records.append(result)
    task_passes = [
        any(
            record["gate"]["passed"]
            for record in premise_records
            if record["task"] == task
        )
        for task in range(config.causal_tasks)
    ]
    return {
        "endpoint": "held-out normalized keep/drop behavioral divergence",
        "application_enabled": all(task_passes),
        "premise_records": premise_records,
        "records": records,
    }


def cross_task_and_composition_study(
    model: ControlledTransformer,
    methods: dict[str, torch.Tensor],
    mixtures: torch.Tensor,
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    fraction: float = 0.10,
) -> dict[str, Any]:
    count = fraction_to_count(fraction, model.n_modules)
    task_basis = methods["posthoc_ief__semantic6"]
    full_atlas = methods["posthoc_ief__onehot"]
    cross_task_records = []
    primitive_correlations = []
    for primitive in range(6):
        mask = _topk_mask(task_basis[:, primitive], count)
        necessities = []
        for task in range(mixtures.shape[0]):
            result = task_circuit_result(model, test, task, mask, device)
            necessities.append(result["necessity"])
            diagnostics = circuit_diagnostic_fields(
                task_basis,
                full_atlas,
                mask,
                primitive=primitive,
                task=task,
                mixtures=mixtures,
            )
            cross_task_records.append(
                {
                    "primitive_circuit": primitive,
                    "evaluation_task": task,
                    "mixture_weight": float(mixtures[task, primitive]),
                    **result,
                    **diagnostics,
                }
            )
        correlation = spearmanr(
            mixtures[:, primitive].numpy(), np.asarray(necessities)
        ).statistic
        primitive_correlations.append(float(correlation))

    composition_records = []
    compared_methods = [
        name
        for name in methods
        if name
        not in {
            "random",
            "task_agnostic_mean",
            "posthoc_activation__semantic6",
        }
    ]
    for method in compared_methods:
        scores = methods[method]
        for task in range(6, mixtures.shape[0]):
            direct_mask = _topk_mask(scores[:, task], count)
            composed_scores = scores[:, :6] @ mixtures[task]
            composed_mask = _topk_mask(composed_scores, count)
            direct = task_circuit_result(model, test, task, direct_mask, device)
            composed = task_circuit_result(model, test, task, composed_mask, device)
            composition_records.append(
                {
                    "method": method,
                    "task": task,
                    "direct_sufficiency": direct["sufficiency"],
                    "composed_sufficiency": composed["sufficiency"],
                    "composed_minus_direct": composed["sufficiency"]
                    - direct["sufficiency"],
                }
            )
    return {
        "fraction": fraction,
        "selected_modules": count,
        "cross_task_records": cross_task_records,
        "primitive_weight_necessity_spearman": primitive_correlations,
        "pure_circuit_overlap": pure_circuit_overlap_summary(
            task_basis,
            methods["task_agnostic_mean"][:, 0],
            count=count,
            n_primitives=6,
        ),
        "composition_records": composition_records,
        "status": "descriptive structural diagnostic; no optimized-mask premise on mixtures",
    }


def pruning_study(
    model: ControlledTransformer,
    methods: dict[str, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: NaturalAssayConfig,
    device: torch.device,
) -> dict[str, Any]:
    records = []
    identity_passes = []
    for method, scores in methods.items():
        for task in range(config.causal_tasks):
            for fraction in config.pruning_fractions:
                count = fraction_to_count(fraction, model.n_modules)
                mask = _topk_mask(scores[:, task], count)
                result = task_circuit_result(model, test, task, mask, device)
                records.append(
                    {
                        "method": method,
                        "task": task,
                        "retained_fraction_requested": fraction,
                        "retained_modules": count,
                        "retained_fraction_actual": count / model.n_modules,
                        "sufficiency": result["sufficiency"],
                        "half_mse": result["kept_half_mse"],
                    }
                )
                if fraction == 1.0:
                    identity_passes.append(result["kept_divergence"] < 1e-12)
    return {
        "masking_semantics": "exact hidden-feature contribution masking",
        "retention_identity_passed": all(identity_passes),
        "records": records,
    }


def _model_config_from_checkpoint(payload: dict[str, Any]) -> HardControlledConfig:
    defaults = asdict(HardControlledConfig())
    stored = payload["configuration"]
    values = {
        field.name: stored.get(field.name, defaults[field.name])
        for field in fields(HardControlledConfig)
    }
    return HardControlledConfig(**values)


def run_natural_assay(
    checkpoint_path: Path,
    config: NaturalAssayConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = _model_config_from_checkpoint(payload)
    if model_config.seed != config.seed:
        raise ValueError("assay seed must match the checkpoint seed")
    model = build_model(model_config, device)
    model.load_state_dict(payload["states"]["trained"])
    mixtures = hard_task_mixtures()

    metrics_path = checkpoint_path.with_suffix(".json")
    competence_record = json.loads(metrics_path.read_text())
    competence = competence_record["competence"]
    if not competence["passed"]:
        return {
            "setting": "natural_tagged_compositional_programs",
            "seed": config.seed,
            "configuration": asdict(config),
            "model_competence": competence,
            "comparison_executed": False,
            "status": "inconclusive: model competence gate failed",
        }

    data_roles = {
        "reference_a": config.seed + 4001,
        "reference_b": config.seed + 5001,
        "mask_calibration": config.seed + 6001,
        "final_test": config.seed + 7001,
    }
    reference_a = make_program_dataset(
        model_config,
        mixtures,
        config.reference_per_task,
        data_roles["reference_a"],
    )
    reference_b = make_program_dataset(
        model_config,
        mixtures,
        config.reference_per_task,
        data_roles["reference_b"],
    )
    calibration = make_program_dataset(
        model_config,
        mixtures,
        config.calibration_per_task,
        data_roles["mask_calibration"],
    )
    test = make_program_dataset(
        model_config,
        mixtures,
        config.test_per_task,
        data_roles["final_test"],
    )
    first_profiles, first_profile_timing = collect_importance_profiles(
        model, reference_a, device
    )
    second_profiles, second_profile_timing = collect_importance_profiles(
        model, reference_b, device
    )
    reliability = reference_reliability(first_profiles["ief"], second_profiles["ief"])
    query = source_query_results(
        first_profiles["ief"], second_profiles["ief"], mixtures, config
    )
    result: dict[str, Any] = {
        "setting": "natural_tagged_compositional_programs",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "model_configuration": asdict(model_config),
        "model_competence": competence,
        "data_roles": data_roles,
        "profile_timing": {
            "reference_a": first_profile_timing,
            "reference_b": second_profile_timing,
        },
        "reference_reliability": reliability,
        "cross_validated_task_query": query,
        "comparison_executed": False,
    }
    if not reliability["passed"]:
        result["status"] = "inconclusive: reference reliability gate failed"
        return result

    methods, calibration_metadata = build_method_scores(
        first_profiles,
        mixtures,
        seed=config.seed,
        jl_repeats=config.jl_outcome_repeats,
    )
    result["importance_calibration"] = calibration_metadata
    result["causal_interpretability"] = causal_interpretability_study(
        model, methods, calibration, test, config, device
    )
    result["cross_task_structure"] = cross_task_and_composition_study(
        model, methods, mixtures, test, device
    )
    result["pruning"] = pruning_study(model, methods, test, config, device)
    result["comparison_executed"] = True
    result["status"] = (
        "valid natural assay"
        if result["causal_interpretability"]["application_enabled"]
        else "inconclusive: causal premise failed"
    )
    return result


def _smoke_config(seed: int) -> NaturalAssayConfig:
    return NaturalAssayConfig(
        seed=seed,
        reference_per_task=1,
        calibration_per_task=2,
        test_per_task=2,
        causal_tasks=1,
        circuit_fractions=(0.10,),
        pruning_fractions=(0.10, 1.0),
        attainable_steps=1,
        attainable_restarts=1,
        random_masks=2,
        jl_outcome_repeats=1,
        jl_query_repeats=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=arc_artifact_dir("03_exploratory", "controlled"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = (
        _smoke_config(args.seed) if args.smoke else NaturalAssayConfig(seed=args.seed)
    )
    result = run_natural_assay(args.checkpoint, config)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    output = args.artifact_dir / f"controlled_v4_assay_seed{args.seed}.json"
    save_json(output, result)
    print(f"CONTROLLED_V4_ASSAY_RESULT={output}")


if __name__ == "__main__":
    main()
