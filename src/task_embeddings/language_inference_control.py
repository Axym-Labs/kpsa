"""Modern Qwen3 activation-control assay using retained OPG profiles."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import (
    conditional_importance,
    jl_task_representation,
    save_json,
    seed_everything,
)
from .importance import build_regular_score_grid
from .inference_control import (
    normalized_control_metrics,
    split_feature_gate,
    task_gate_matrix,
)
from .language_v3 import (
    TASKS,
    EncodedTask,
    LanguageConfig,
    evaluate_tasks,
    load_model_and_tokenizer,
    load_task_suite,
)
from .streaming_v5 import tbe_resource_counts


@dataclass(frozen=True)
class LanguageInferenceControlConfig:
    seed: int
    evaluation_start: int = 2
    evaluation_per_task: int = 6
    retained_fractions: tuple[float, ...] = (0.75, 0.90, 1.0)
    jl_seed_offset: int = 0


def slice_encoded_task(task: EncodedTask, start: int, count: int) -> EncodedTask:
    if start < 0 or count < 1 or start + count > len(task):
        raise ValueError("requested evaluation slice lies outside the encoded task")
    selected = slice(start, start + count)
    return EncodedTask(
        input_ids=task.input_ids[selected],
        attention_mask=task.attention_mask[selected],
        labels=task.labels[selected],
    )


def _load_profiles(path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with np.load(path) as payload:
        required = {
            "atlas_raw",
            "amplitude_raw",
            "atlas_ief",
            "amplitude_ief",
            "semantic_representation",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"profile archive is missing {sorted(missing)}")
        profiles = {
            "raw_opg": conditional_importance(
                torch.from_numpy(payload["atlas_raw"]),
                torch.from_numpy(payload["amplitude_raw"]),
            ),
            "residual_normalized_opg": conditional_importance(
                torch.from_numpy(payload["atlas_ief"]),
                torch.from_numpy(payload["amplitude_ief"]),
            ),
        }
        features = torch.from_numpy(payload["semantic_representation"]).float()
    return profiles, features


def run_language_inference_control(
    checkpoint_path: Path,
    profiles_path: Path,
    config: LanguageInferenceControlConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate task-indexed Qwen3 SwiGLU gates on profile-disjoint examples."""
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = LanguageConfig(**payload["configuration"])
    if model_config.seed != config.seed:
        raise ValueError("experiment seed must match checkpoint seed")
    model, tokenizer = load_model_and_tokenizer(model_config, device)
    model.load_state_dict(payload["model"])
    del payload
    _train_tasks, validation_tasks = load_task_suite(tokenizer, model_config)
    evaluation_tasks = [
        slice_encoded_task(task, config.evaluation_start, config.evaluation_per_task)
        for task in validation_tasks
    ]
    profiles, task_features = _load_profiles(profiles_path)
    if task_features.shape[0] != len(TASKS):
        raise ValueError("task features do not cover the language task suite")
    jl_features = jl_task_representation(
        len(TASKS), task_features.shape[1], config.seed + 3000 + config.jl_seed_offset
    )
    method_scores = build_regular_score_grid(
        profiles, task_features, jl_features=jl_features
    )
    started = time.perf_counter()
    model.set_intervention(None)
    model.set_feature_gates(None)
    baseline_losses, baseline_exact = evaluate_tasks(model, evaluation_tasks, device)
    zero = torch.zeros(model.n_modules)
    model.set_feature_gates(split_feature_gate(zero, model.layer_sizes))
    null_losses, null_exact = evaluate_tasks(model, evaluation_tasks, device)
    one = torch.ones(model.n_modules)
    model.set_feature_gates(split_feature_gate(one, model.layer_sizes))
    identity_losses, identity_exact = evaluate_tasks(model, evaluation_tasks, device)
    identity_error = float((identity_losses - baseline_losses).abs().max())
    identity_exact_error = float((identity_exact - baseline_exact).abs().max())
    records = []
    for method, scores in method_scores.items():
        for retained_fraction in config.retained_fractions:
            if retained_fraction == 1.0:
                continue
            gates = task_gate_matrix(scores, retained_fraction=retained_fraction)
            for task in range(len(TASKS)):
                model.set_feature_gates(
                    split_feature_gate(gates[:, task], model.layer_sizes)
                )
                controlled_losses, controlled_exact = evaluate_tasks(
                    model, evaluation_tasks, device
                )
                metrics = normalized_control_metrics(
                    baseline_losses=baseline_losses,
                    null_losses=null_losses,
                    controlled_losses=controlled_losses,
                    target_task=task,
                )
                other = torch.ones(len(TASKS), dtype=torch.bool)
                other[task] = False
                records.append(
                    {
                        "method": method,
                        "task": task,
                        "task_name": TASKS[task],
                        "retained_fraction": retained_fraction,
                        "retained_modules": int(gates[:, task].sum()),
                        "target_baseline_exact_match": float(baseline_exact[task]),
                        "target_controlled_exact_match": float(controlled_exact[task]),
                        "mean_off_target_exact_match_change": float(
                            (controlled_exact[other] - baseline_exact[other]).mean()
                        ),
                        **metrics,
                    }
                )
    model.set_feature_gates(None)
    resources = tbe_resource_counts(
        n_modules=model.n_modules,
        n_tasks=len(TASKS),
        dimension=task_features.shape[1],
    )
    promotion_blockers = ["single_model_seed"]
    if len(TASKS) <= task_features.shape[1] + 2:
        promotion_blockers.append("too_few_tasks_for_compression")
    return {
        "setting": "qwen3_task_conditioned_activation_control",
        "model_family": "Qwen3",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "source_model_configuration": asdict(model_config),
        "tasks": TASKS,
        "data_roles": {
            "profile_source": "retained posthoc archive",
            "evaluation_indices": list(
                range(
                    config.evaluation_start,
                    config.evaluation_start + config.evaluation_per_task,
                )
            ),
        },
        "methods": sorted(method_scores),
        "requires_distinct_task_queries": True,
        "mechanism": "oracle_task_qwen_swiglu_feature_gating",
        "execution": {
            "reference_implementation": "dense_activation_mask",
            "claims_dense_flop_reduction": False,
        },
        "baseline_task_loss": baseline_losses,
        "baseline_task_exact_match": baseline_exact,
        "null_task_loss": null_losses,
        "null_task_exact_match": null_exact,
        "identity_max_abs_loss_error": identity_error,
        "identity_max_abs_exact_match_error": identity_exact_error,
        "premise_gate": {
            "passed": bool(float(baseline_exact.mean()) > 0 and identity_error < 1e-6),
            "mean_baseline_exact_match": float(baseline_exact.mean()),
            "identity_threshold": 1e-6,
        },
        "promotion_blockers": promotion_blockers,
        "records": records,
        "resource_accounting": resources,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evaluation-per-task", type=int, default=6)
    args = parser.parse_args()
    result = run_language_inference_control(
        args.checkpoint,
        args.profiles,
        LanguageInferenceControlConfig(
            seed=args.seed, evaluation_per_task=args.evaluation_per_task
        ),
    )
    save_json(args.output, result)
    print(f"LANGUAGE_INFERENCE_CONTROL_RESULT={args.output}")


if __name__ == "__main__":
    main()
