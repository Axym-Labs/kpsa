from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

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
)
from .controlled_v4 import hard_task_mixtures


@dataclass(frozen=True)
class PlantedConfig:
    seed: int = 1
    modules_per_primitive: int = 30
    background_modules: int = 120
    reference_per_task: int = 64
    calibration_per_task: int = 256
    test_per_task: int = 512
    observation_noise: float = 0.05
    circuit_fractions: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
    random_masks: int = 100
    jl_repeats: int = 10

    @property
    def n_modules(self) -> int:
        return 6 * self.modules_per_primitive + self.background_modules


def make_planted_data(
    config: PlantedConfig,
    *,
    per_task: int,
    seed: int,
    noise: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mixtures = hard_task_mixtures()
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(mixtures.shape[0] * per_task, 6, generator=generator)
    tasks = torch.arange(mixtures.shape[0]).repeat_interleave(per_task)
    targets = (values * mixtures[tasks]).sum(dim=1)
    noise_scale = config.observation_noise if noise is None else noise
    if noise_scale:
        targets = targets + noise_scale * torch.randn(
            targets.shape, generator=generator
        )
    return values, targets, tasks


class PlantedCircuitBenchmark:
    """A positive control whose causal module partition is exactly known."""

    def __init__(self, config: PlantedConfig) -> None:
        self.config = config
        assigned = torch.arange(6).repeat_interleave(config.modules_per_primitive)
        background = torch.full((config.background_modules,), -1, dtype=torch.long)
        self.module_primitive = torch.cat((assigned, background))
        self.mixtures = hard_task_mixtures()

    def module_contributions(
        self, primitive_values: torch.Tensor, task_ids: torch.Tensor
    ) -> torch.Tensor:
        if primitive_values.ndim != 2 or primitive_values.shape[1] != 6:
            raise ValueError("primitive values must have shape [samples, 6]")
        if task_ids.shape != (primitive_values.shape[0],):
            raise ValueError("one task id is required per sample")
        contributions = torch.zeros(
            primitive_values.shape[0], self.config.n_modules, dtype=torch.float32
        )
        task_weights = self.mixtures[task_ids]
        for primitive in range(6):
            selected = self.module_primitive == primitive
            contribution = (
                primitive_values[:, primitive]
                * task_weights[:, primitive]
                / self.config.modules_per_primitive
            )
            contributions[:, selected] = contribution[:, None]
        return contributions

    def predict(
        self,
        primitive_values: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        keep_mask: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if keep_mask is not None and drop_mask is not None:
            raise ValueError("keep_mask and drop_mask are mutually exclusive")
        active = torch.ones(self.config.n_modules, dtype=torch.bool)
        if keep_mask is not None:
            active = keep_mask.detach().bool().cpu()
        if drop_mask is not None:
            active = ~drop_mask.detach().bool().cpu()
        if active.shape != (self.config.n_modules,):
            raise ValueError("mask must have one value per module")
        return self.module_contributions(primitive_values, task_ids)[:, active].sum(
            dim=1
        )


def _importance_profiles(
    benchmark: PlantedCircuitBenchmark,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    values, targets, tasks = data
    contributions = benchmark.module_contributions(values, tasks)
    predictions = contributions.sum(dim=1)
    residual = predictions - targets
    raw = residual.square()[:, None] * contributions.square()
    ief = contributions.square()
    activation = torch.zeros_like(contributions)
    for primitive in range(6):
        selected = benchmark.module_primitive == primitive
        activation[:, selected] = values[:, primitive, None].square()
    actgrad = residual.abs()[:, None] * contributions.abs()
    profiles = {}
    for name, scores in {
        "ief": ief,
        "raw_ef": raw,
        "activation": activation,
        "activation_gradient": actgrad,
    }.items():
        atlas, amplitude = build_task_atlas(
            scores, tasks, n_tasks=benchmark.mixtures.shape[0]
        )
        profiles[name] = conditional_importance(atlas, amplitude)
    return profiles


def _reconstruct_importance(
    importance: torch.Tensor, representation: torch.Tensor
) -> torch.Tensor:
    features = normalized_rows(representation)
    return (importance @ features @ features.T).clamp_min(0)


def _reference_reliability(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    correlations = []
    for task in range(first.shape[1]):
        result = spearmanr(first[:, task].numpy(), second[:, task].numpy())
        correlations.append(float(result.statistic))
    finite = [value for value in correlations if math.isfinite(value)]
    return {
        "per_task_spearman": correlations,
        "mean_spearman": float(np.mean(finite)) if finite else float("nan"),
        "minimum_spearman": min(finite) if finite else float("nan"),
        "passed": bool(finite) and min(finite) >= 0.8,
    }


def _method_scores(
    benchmark: PlantedCircuitBenchmark,
    profiles: dict[str, torch.Tensor],
    config: PlantedConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    mixtures = benchmark.mixtures
    semantic = normalized_rows(mixtures)
    onehot = torch.eye(mixtures.shape[0])
    jl = jl_task_representation(mixtures.shape[0], 6, config.seed + 1000)
    permuted = semantic[
        torch.randperm(
            mixtures.shape[0],
            generator=torch.Generator().manual_seed(config.seed + 2000),
        )
    ]
    ief = profiles["ief"]
    methods = {
        "posthoc_ief__onehot": _reconstruct_importance(ief, onehot),
        "posthoc_ief__semantic6": _reconstruct_importance(ief, semantic),
        "posthoc_ief__jl6": _reconstruct_importance(ief, jl),
        "posthoc_ief__permuted_semantic6": _reconstruct_importance(ief, permuted),
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
    generator = torch.Generator().manual_seed(config.seed + 3000)
    methods["random"] = torch.rand(ief.shape, generator=generator)
    return calibrate_importance_matrices(methods, reference_key="posthoc_ief__onehot")


def _mask_from_scores(scores: torch.Tensor, count: int) -> torch.Tensor:
    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    if count:
        mask[torch.topk(scores, count).indices] = True
    return mask


def _task_circuit_result(
    benchmark: PlantedCircuitBenchmark,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    task: int,
    mask: torch.Tensor,
) -> dict[str, float]:
    values, targets, tasks = data
    selected = tasks == task
    local_values = values[selected]
    local_targets = targets[selected]
    local_tasks = tasks[selected]
    full = benchmark.predict(local_values, local_tasks)
    null = benchmark.predict(
        local_values,
        local_tasks,
        keep_mask=torch.zeros(benchmark.config.n_modules, dtype=torch.bool),
    )
    kept = benchmark.predict(local_values, local_tasks, keep_mask=mask)
    dropped = benchmark.predict(local_values, local_tasks, drop_mask=mask)
    null_divergence = float((null - full).square().mean())
    normalized = faithfulness_scores(
        kept_divergence=(kept - full).square().mean().view(1),
        dropped_divergence=(dropped - full).square().mean().view(1),
        null_divergence=null_divergence,
    )
    return {
        "null_divergence": null_divergence,
        "sufficiency": float(normalized["sufficiency"][0]),
        "necessity": float(normalized["necessity"][0]),
        "full_half_mse": float(0.5 * (full - local_targets).square().mean()),
        "kept_half_mse": float(0.5 * (kept - local_targets).square().mean()),
        "dropped_half_mse": float(0.5 * (dropped - local_targets).square().mean()),
    }


def _source_query_results(
    benchmark: PlantedCircuitBenchmark,
    first: torch.Tensor,
    second: torch.Tensor,
    config: PlantedConfig,
) -> dict[str, Any]:
    folds = [[task] for task in range(21, benchmark.mixtures.shape[0])]
    semantic = normalized_rows(benchmark.mixtures)
    output: dict[str, Any] = {
        "semantic6": cross_validated_task_query_fidelity(
            first,
            semantic,
            folds=folds,
            top_fraction=0.1,
            target_importance=second,
        ),
        "onehot": cross_validated_task_query_fidelity(
            first,
            torch.eye(benchmark.mixtures.shape[0]),
            folds=folds,
            top_fraction=0.1,
            target_importance=second,
        ),
        "task_agnostic": cross_validated_task_query_fidelity(
            first,
            torch.ones(benchmark.mixtures.shape[0], 1),
            folds=folds,
            top_fraction=0.1,
            target_importance=second,
        ),
    }
    for repeat in range(config.jl_repeats):
        representation = jl_task_representation(
            benchmark.mixtures.shape[0], 6, config.seed + 4000 + repeat
        )
        output[f"jl6_seed{repeat}"] = cross_validated_task_query_fidelity(
            first,
            representation,
            folds=folds,
            top_fraction=0.1,
            target_importance=second,
        )
        permutation = torch.randperm(
            benchmark.mixtures.shape[0],
            generator=torch.Generator().manual_seed(config.seed + 5000 + repeat),
        )
        output[f"permuted_semantic6_seed{repeat}"] = (
            cross_validated_task_query_fidelity(
                first,
                semantic[permutation],
                folds=folds,
                top_fraction=0.1,
                target_importance=second,
            )
        )
    return output


def _causal_interpretability(
    benchmark: PlantedCircuitBenchmark,
    methods: dict[str, torch.Tensor],
    calibration_profiles: dict[str, torch.Tensor],
    calibration: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: PlantedConfig,
) -> dict[str, Any]:
    del calibration
    records = []
    premise_records = []
    random_generator = torch.Generator().manual_seed(config.seed + 6000)
    for task in range(6):
        attainable_scores = calibration_profiles["ief"][:, task]
        for fraction in config.circuit_fractions:
            count = min(
                config.n_modules, max(1, math.ceil(config.n_modules * fraction))
            )
            attainable_mask = _mask_from_scores(attainable_scores, count)
            attainable = _task_circuit_result(benchmark, test, task, attainable_mask)
            random_sufficiencies = []
            for _ in range(config.random_masks):
                mask = torch.zeros(config.n_modules, dtype=torch.bool)
                mask[
                    torch.randperm(config.n_modules, generator=random_generator)[:count]
                ] = True
                random_sufficiencies.append(
                    _task_circuit_result(benchmark, test, task, mask)["sufficiency"]
                )
            random_p95 = float(torch.quantile(torch.tensor(random_sufficiencies), 0.95))
            gate = causal_assay_gate(
                null_divergence=attainable["null_divergence"],
                attainable_sufficiency=attainable["sufficiency"],
                random_sufficiency_p95=random_p95,
                minimum_span=0.05,
                minimum_gap=0.2,
            )
            premise_records.append(
                {
                    "task": task,
                    "requested_fraction": fraction,
                    "selected_modules": count,
                    "selected_fraction_actual": count / config.n_modules,
                    "attainable_sufficiency": attainable["sufficiency"],
                    "random_sufficiency_p95": random_p95,
                    "gate": gate,
                }
            )
            for method, scores in methods.items():
                mask = _mask_from_scores(scores[:, task], count)
                result = _task_circuit_result(benchmark, test, task, mask)
                result.update(
                    method=method,
                    task=task,
                    requested_fraction=fraction,
                    selected_modules=count,
                    selected_fraction_actual=count / config.n_modules,
                    premise_passed=gate["passed"],
                )
                records.append(result)
    task_passes = [
        any(
            record["gate"]["passed"]
            for record in premise_records
            if record["task"] == task
        )
        for task in range(6)
    ]
    return {
        "endpoint": "held-out normalized keep/drop divergence",
        "application_enabled": all(task_passes),
        "premise_records": premise_records,
        "records": records,
    }


def _pruning(
    benchmark: PlantedCircuitBenchmark,
    methods: dict[str, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: PlantedConfig,
) -> dict[str, Any]:
    records = []
    for method, scores in methods.items():
        for task in range(6):
            for fraction in (0.10, 0.25, 0.50, 0.75, 1.0):
                count = math.ceil(config.n_modules * fraction)
                mask = _mask_from_scores(scores[:, task], count)
                result = _task_circuit_result(benchmark, test, task, mask)
                records.append(
                    {
                        "method": method,
                        "task": task,
                        "retained_fraction_requested": fraction,
                        "retained_modules": count,
                        "retained_fraction_actual": count / config.n_modules,
                        "half_mse": result["kept_half_mse"],
                        "sufficiency": result["sufficiency"],
                    }
                )
    identity = [
        abs(record["half_mse"] - record_full["full_half_mse"]) < 1e-7
        for record in records
        if record["retained_fraction_requested"] == 1.0
        for record_full in [
            _task_circuit_result(
                benchmark,
                test,
                record["task"],
                torch.ones(config.n_modules, dtype=torch.bool),
            )
        ]
    ]
    return {
        "masking_semantics": "exact additive module removal",
        "retention_identity_passed": all(identity),
        "records": records,
    }


def run_planted_experiment(config: PlantedConfig) -> dict[str, Any]:
    benchmark = PlantedCircuitBenchmark(config)
    reference_a = make_planted_data(
        config, per_task=config.reference_per_task, seed=config.seed + 101
    )
    reference_b = make_planted_data(
        config, per_task=config.reference_per_task, seed=config.seed + 102
    )
    calibration = make_planted_data(
        config, per_task=config.calibration_per_task, seed=config.seed + 201
    )
    test = make_planted_data(
        config, per_task=config.test_per_task, seed=config.seed + 301
    )
    profiles_a = _importance_profiles(benchmark, reference_a)
    profiles_b = _importance_profiles(benchmark, reference_b)
    calibration_profiles = _importance_profiles(benchmark, calibration)
    reliability = _reference_reliability(profiles_a["ief"], profiles_b["ief"])
    source_query = _source_query_results(
        benchmark, profiles_a["ief"], profiles_b["ief"], config
    )
    methods, calibration_metadata = _method_scores(benchmark, profiles_a, config)
    causal = _causal_interpretability(
        benchmark, methods, calibration_profiles, calibration, test, config
    )
    pruning = _pruning(benchmark, methods, test, config)
    test_values, test_targets, test_tasks = test
    predictions = benchmark.predict(test_values, test_tasks)
    task_losses = torch.stack(
        [
            0.5
            * (predictions[test_tasks == task] - test_targets[test_tasks == task])
            .square()
            .mean()
            for task in range(benchmark.mixtures.shape[0])
        ]
    )
    task_nulls = torch.stack(
        [
            0.5 * test_targets[test_tasks == task].square().mean()
            for task in range(benchmark.mixtures.shape[0])
        ]
    )
    recovered = 1.0 - task_losses / task_nulls.clamp_min(EPS)
    return {
        "setting": "planted_circuit_positive_control",
        "seed": config.seed,
        "configuration": {
            **config.__dict__,
            "n_modules": config.n_modules,
        },
        "data_roles": {
            "reference_a_seed": config.seed + 101,
            "reference_b_seed": config.seed + 102,
            "calibration_seed": config.seed + 201,
            "final_test_seed": config.seed + 301,
        },
        "competence": {
            "minimum_null_gap_recovered": float(recovered.min()),
            "mean_null_gap_recovered": float(recovered.mean()),
            "passed": bool((recovered >= 0.8).all()),
        },
        "reference_reliability": reliability,
        "cross_validated_task_query": source_query,
        "importance_calibration": calibration_metadata,
        "causal_interpretability": causal,
        "pruning": pruning,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=arc_artifact_dir("03_exploratory", "planted"),
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    result = run_planted_experiment(PlantedConfig(seed=args.seed))
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = args.artifact_dir / f"planted_v4_metrics_seed{args.seed}.json"
    save_json(path, result)
    print(f"PLANTED_V4_RESULT={path}")


if __name__ == "__main__":
    main()
