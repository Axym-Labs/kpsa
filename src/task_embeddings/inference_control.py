"""Task-conditioned inference control at Transformer MLP-feature granularity."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .common import EPS, jl_task_representation, seed_everything
from .controlled_assay_v4 import (
    _model_config_from_checkpoint,
    collect_importance_profiles,
)
from .controlled_v3 import parameter_count
from .controlled_v4 import (
    build_model,
    evaluate_program_losses,
    hard_task_mixtures,
    make_program_dataset,
)
from .importance import build_regular_score_grid
from .streaming_v5 import tbe_resource_counts


@dataclass(frozen=True)
class InferenceControlConfig:
    seed: int
    reference_per_task: int = 32
    test_per_task: int = 192
    retained_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)
    jl_seed_offset: int = 0


def task_gate_matrix(scores: torch.Tensor, *, retained_fraction: float) -> torch.Tensor:
    """Return one exact-budget binary feature gate per task query."""
    values = scores.detach().float().cpu()
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("scores must have shape [modules, tasks]")
    if not 0 < retained_fraction <= 1:
        raise ValueError("retained_fraction must lie in (0, 1]")
    count = min(values.shape[0], math.ceil(values.shape[0] * retained_fraction))
    gates = torch.zeros_like(values)
    indices = torch.topk(values, count, dim=0).indices
    gates.scatter_(0, indices, 1.0)
    return gates


def split_feature_gate(
    gate: torch.Tensor, layer_sizes: Sequence[int]
) -> list[torch.Tensor]:
    values = gate.detach().float().cpu().flatten()
    if values.numel() != sum(layer_sizes):
        raise ValueError("layer sizes do not cover the feature gate")
    return list(torch.split(values, list(layer_sizes)))


def normalized_control_metrics(
    *,
    baseline_losses: torch.Tensor,
    null_losses: torch.Tensor,
    controlled_losses: torch.Tensor,
    target_task: int,
) -> dict[str, Any]:
    """Normalize retained quality and cross-task damage by the null model."""
    baseline = baseline_losses.detach().float().cpu().flatten()
    null = null_losses.detach().float().cpu().flatten()
    controlled = controlled_losses.detach().float().cpu().flatten()
    if baseline.shape != null.shape or controlled.shape != baseline.shape:
        raise ValueError("loss vectors must have matching shapes")
    if not 0 <= target_task < baseline.numel():
        raise ValueError("target_task lies outside the loss vectors")
    gaps = null - baseline
    target_gap = float(gaps[target_task])
    target_recovery = (
        float((null[target_task] - controlled[target_task]) / target_gap)
        if abs(target_gap) > EPS
        else float("nan")
    )
    other = torch.ones(baseline.numel(), dtype=torch.bool)
    other[target_task] = False
    valid = other & (gaps.abs() > EPS)
    spillover = (
        float(((controlled[valid] - baseline[valid]) / gaps[valid]).mean())
        if bool(valid.any())
        else float("nan")
    )
    return {
        "target_quality_recovery": target_recovery,
        "off_target_spillover": spillover,
        "target_baseline_loss": float(baseline[target_task]),
        "target_controlled_loss": float(controlled[target_task]),
        "target_null_loss": float(null[target_task]),
    }


def run_inference_control(
    checkpoint_path: Path,
    config: InferenceControlConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate oracle-task conditional computation on a frozen Transformer."""
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = _model_config_from_checkpoint(payload)
    if model_config.seed != config.seed:
        raise ValueError("experiment seed must match checkpoint seed")
    model = build_model(model_config, device)
    model.load_state_dict(payload["states"]["trained"])
    features = hard_task_mixtures()
    data_roles = {
        "importance": config.seed + 4001,
        "final_behavior": config.seed + 7001,
    }
    reference = make_program_dataset(
        model_config,
        features,
        config.reference_per_task,
        data_roles["importance"],
    )
    final_test = make_program_dataset(
        model_config,
        features,
        config.test_per_task,
        data_roles["final_behavior"],
    )
    started = time.perf_counter()
    profiles, profile_timing = collect_importance_profiles(model, reference, device)
    atlases = {
        "raw_opg": profiles["raw_ef"],
        "residual_normalized_opg": profiles["ief"],
    }
    jl_features = jl_task_representation(
        features.shape[0],
        features.shape[1],
        config.seed + 3000 + config.jl_seed_offset,
    )
    method_scores = build_regular_score_grid(atlases, features, jl_features=jl_features)
    model.set_feature_gates(None)
    baseline_losses = evaluate_program_losses(model, final_test, device)
    zero_gate = torch.zeros(model.n_modules)
    model.set_feature_gates(split_feature_gate(zero_gate, model.layer_sizes))
    null_losses = evaluate_program_losses(model, final_test, device)
    model.set_feature_gates(None)
    records = []
    identity_errors = []
    for method, scores in method_scores.items():
        for retained_fraction in config.retained_fractions:
            gates = task_gate_matrix(scores, retained_fraction=retained_fraction)
            for task in range(features.shape[0]):
                model.set_feature_gates(
                    split_feature_gate(gates[:, task], model.layer_sizes)
                )
                controlled_losses = evaluate_program_losses(model, final_test, device)
                metrics = normalized_control_metrics(
                    baseline_losses=baseline_losses,
                    null_losses=null_losses,
                    controlled_losses=controlled_losses,
                    target_task=task,
                )
                records.append(
                    {
                        "method": method,
                        "task": task,
                        "retained_fraction": retained_fraction,
                        "retained_modules": int(gates[:, task].sum()),
                        **metrics,
                    }
                )
                if retained_fraction == 1.0:
                    identity_errors.append(
                        float((controlled_losses - baseline_losses).abs().max())
                    )
    model.set_feature_gates(None)
    resources = tbe_resource_counts(
        n_modules=model.n_modules,
        n_tasks=features.shape[0],
        dimension=features.shape[1],
    )
    resources.update(
        {
            "model_parameters": parameter_count(model),
            "oracle_task_id_floats": features.shape[0] * features.shape[1],
        }
    )
    return {
        "setting": "task_conditioned_transformer_activation_control",
        "mechanism": "oracle_task_mlp_feature_gating",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "model_configuration": asdict(model_config),
        "data_roles": data_roles,
        "requires_distinct_task_queries": True,
        "execution": {
            "reference_implementation": "dense_activation_mask",
            "claims_dense_flop_reduction": False,
            "deployment_note": (
                "compute savings require materializing sliced weights or a sparse kernel"
            ),
        },
        "methods": sorted(method_scores),
        "records": records,
        "identity_max_abs_loss_error": max(identity_errors, default=float("nan")),
        "profile_timing": profile_timing,
        "resource_accounting": resources,
        "wall_seconds": time.perf_counter() - started,
    }
