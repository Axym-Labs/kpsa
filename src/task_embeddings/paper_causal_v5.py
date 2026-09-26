from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .common import _safe_corr, jl_task_representation, save_json, seed_everything
from .controlled_assay_v4 import (
    _model_config_from_checkpoint,
    _predict_task,
    collect_importance_profiles,
)
from .controlled_v4 import build_model, hard_task_mixtures, make_program_dataset
from .query_v5 import build_query_methods, select_residual_scale
from .streaming_v5 import tbe_resource_counts


def causal_ranking_records(
    method_scores: dict[str, torch.Tensor],
    baseline_scores: torch.Tensor,
    source_opg: torch.Tensor,
    effects: torch.Tensor,
    *,
    module_indices: torch.Tensor,
    train_tasks: list[int],
    target_tasks: list[int],
) -> list[dict[str, Any]]:
    """Compare score rankings with raw and task-residualized causal effects."""
    indices = module_indices.detach().long().cpu()
    causal = effects.detach().float().cpu()
    baseline = baseline_scores.detach().float().cpu()
    oracle = source_opg.detach().float().cpu()
    if causal.shape != (indices.numel(), oracle.shape[1]):
        raise ValueError("effects must cover sampled modules and every task")
    causal_mean = causal[:, train_tasks].mean(dim=1)
    oracle_mean = oracle[:, train_tasks].mean(dim=1)
    records = []
    for method, scores in method_scores.items():
        values = scores.detach().float().cpu()
        for task in target_tasks:
            raw_scores = values[indices, task]
            if method == "observed_opg_oracle":
                residual_scores = values[indices, task] - oracle_mean[indices]
            else:
                residual_scores = raw_scores - baseline[indices, task]
            raw_effects = causal[:, task]
            residual_effects = raw_effects - causal_mean
            records.append(
                {
                    "method": method,
                    "task": task,
                    "raw_spearman": _safe_corr(
                        raw_scores.numpy(), raw_effects.numpy(), "spearman"
                    ),
                    "residual_spearman": _safe_corr(
                        residual_scores.numpy(),
                        residual_effects.numpy(),
                        "spearman",
                    ),
                    "mean_raw_effect": float(raw_effects.mean()),
                    "mean_absolute_residual_effect": float(
                        residual_effects.abs().mean()
                    ),
                }
            )
    return records


def measure_individual_ablation_effects(
    model,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    module_indices: torch.Tensor,
    device: torch.device,
    *,
    n_tasks: int,
) -> torch.Tensor:
    """Measure exact one-module drop divergence on a common module sample."""
    indices = module_indices.detach().long().cpu()
    effects = torch.empty(indices.numel(), n_tasks)
    for task in range(n_tasks):
        full, _ = _predict_task(model, dataset, task, device)
        for local, module in enumerate(indices):
            mask = torch.zeros(model.n_modules, dtype=torch.bool)
            mask[module] = True
            dropped, _ = _predict_task(
                model, dataset, task, device, mask=mask, mode="zero"
            )
            effects[local, task] = (dropped - full).square().mean()
    return effects


def _scale_label(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def run_causal_ranking(
    checkpoint_path: Path,
    *,
    seed: int,
    module_sample: int = 256,
    repeats: int = 3,
    reference_per_task: int = 32,
    test_per_task: int = 192,
    device: torch.device | None = None,
) -> dict[str, Any]:
    seed_everything(seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _model_config_from_checkpoint(payload)
    if config.seed != seed:
        raise ValueError("experiment seed must match checkpoint seed")
    model = build_model(config, device)
    model.load_state_dict(payload["states"]["trained"])
    features = hard_task_mixtures()
    source_data = make_program_dataset(
        config, features, reference_per_task, seed + 4001
    )
    target_data = make_program_dataset(
        config, features, reference_per_task, seed + 5001
    )
    final_test = make_program_dataset(config, features, test_per_task, seed + 7001)
    source_profiles, _ = collect_importance_profiles(model, source_data, device)
    target_profiles, _ = collect_importance_profiles(model, target_data, device)
    source_opg = source_profiles["ief"]
    target_opg = target_profiles["ief"]
    train_tasks = list(range(21))
    target_tasks = list(range(21, 30))
    folds = [target_tasks]
    residual_scales = (0.25, 0.5, 0.75, 1.0)
    methods = build_query_methods(
        source_opg,
        features,
        folds=folds,
        seed=seed,
        repeats=repeats,
        residual_scales=residual_scales,
    )
    tbe_selection = select_residual_scale(
        source_opg,
        target_opg,
        features,
        task_indices=train_tasks,
        residual_scales=residual_scales,
        top_fraction=0.10,
    )
    chosen: dict[str, torch.Tensor] = {
        "mean_only": methods["mean_only"][0],
        "tbe_kernel": methods["tbe_kernel"][0],
        "observed_opg_oracle": methods["observed_opg_oracle"][0],
        "tbe_linear": methods[
            f"tbe_linear_scale_{_scale_label(tbe_selection['selected_scale'])}"
        ][0],
    }
    selections: dict[str, Any] = {"tbe": tbe_selection}
    for repeat in range(repeats):
        jl_features = jl_task_representation(30, 6, seed + 3000 + repeat)
        jl_selection = select_residual_scale(
            source_opg,
            target_opg,
            jl_features,
            task_indices=train_tasks,
            residual_scales=residual_scales,
            top_fraction=0.10,
        )
        permutation = torch.randperm(
            30,
            generator=torch.Generator().manual_seed(seed + 4000 + repeat),
        )
        permuted_selection = select_residual_scale(
            source_opg,
            target_opg,
            features[permutation],
            task_indices=train_tasks,
            residual_scales=residual_scales,
            top_fraction=0.10,
        )
        jl_label = _scale_label(jl_selection["selected_scale"])
        permuted_label = _scale_label(permuted_selection["selected_scale"])
        chosen[f"jl_linear_seed{repeat}"] = methods[
            f"jl_linear_scale_{jl_label}_seed{repeat}"
        ][0]
        chosen[f"permuted_basis_linear_seed{repeat}"] = methods[
            f"permuted_basis_linear_scale_{permuted_label}_seed{repeat}"
        ][0]
        selections[f"jl_seed{repeat}"] = jl_selection
        selections[f"permuted_basis_seed{repeat}"] = permuted_selection

    if not 1 <= module_sample <= model.n_modules:
        raise ValueError("module_sample must lie within the model module count")
    module_indices = torch.randperm(
        model.n_modules,
        generator=torch.Generator().manual_seed(seed + 9000),
    )[:module_sample]
    started = time.perf_counter()
    effects = measure_individual_ablation_effects(
        model,
        final_test,
        module_indices,
        device,
        n_tasks=features.shape[0],
    )
    records = causal_ranking_records(
        chosen,
        methods["mean_only"][0],
        source_opg,
        effects,
        module_indices=module_indices,
        train_tasks=train_tasks,
        target_tasks=target_tasks,
    )
    return {
        "setting": "controlled_single_module_ablation",
        "seed": seed,
        "device": str(device),
        "model_configuration": asdict(config),
        "module_sample": module_sample,
        "module_sample_seed": seed + 9000,
        "reference_per_task": reference_per_task,
        "test_per_task": test_per_task,
        "train_tasks": train_tasks,
        "target_tasks": target_tasks,
        "scale_selections": selections,
        "records": records,
        "effect_measurement_wall_seconds": time.perf_counter() - started,
        "resource_accounting": tbe_resource_counts(
            n_modules=model.n_modules, n_tasks=30, dimension=6
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_causal_ranking(args.checkpoint, seed=args.seed)
    save_json(args.output, result)
    print(f"PAPER_CAUSAL_V5_RESULT={args.output}")


if __name__ == "__main__":
    main()
