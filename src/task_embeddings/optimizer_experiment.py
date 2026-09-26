"""Task-queryable preconditioner experiments."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .common import jl_task_representation, seed_everything
from .controlled_assay_v4 import (
    _model_config_from_checkpoint,
    collect_importance_profiles,
)
from .controlled_v4 import (
    build_model,
    evaluate_program_losses,
    hard_task_mixtures,
    make_program_batch,
    make_program_dataset,
)
from .experiment_core import Estimator, Representation
from .task_optimizer import (
    LinearTaskScores,
    MatrixTaskScores,
    MeanTaskScores,
    RelativeTaskScores,
    TaskIndexedAdamW,
    optimizer_resource_counts,
    transformer_mlp_group_specs,
)


def _recovery_metrics(
    clean_loss: torch.Tensor,
    corrupted_loss: torch.Tensor,
    final_loss: torch.Tensor,
) -> dict[str, float | None]:
    damage = float(corrupted_loss - clean_loss)
    improvement = float(corrupted_loss - final_loss)
    return {
        "clean_target_half_mse": float(clean_loss),
        "recoverable_damage": damage,
        "damage_recovery_fraction": improvement / damage if damage > 0 else None,
    }


@dataclass(frozen=True)
class OptimizerExperimentConfig:
    seed: int
    reference_per_task: int = 16
    test_per_task: int = 96
    batch_size: int = 64
    steps: int = 8
    task_indices: tuple[int, ...] = (0, 1, 2, 21, 22, 23)
    learning_rate: float = 1e-4
    beta1: float = 0.9
    weight_decay: float = 1e-3
    preconditioner_strength: float = 1.0
    corruption_std: float = 0.005
    jl_seed_offset: int = 0


def _score_provider(
    atlas: torch.Tensor,
    representation: Representation,
    task_features: torch.Tensor,
    jl_features: torch.Tensor,
    *,
    strength: float,
):
    if representation is Representation.FULL_ATLAS:
        base = MatrixTaskScores(atlas)
    elif representation is Representation.TBE:
        base = LinearTaskScores.from_atlas(atlas, task_features)
    elif representation is Representation.JL:
        base = LinearTaskScores.from_atlas(atlas, jl_features)
    elif representation is Representation.MEAN_ONLY:
        base = MeanTaskScores(atlas.mean(dim=1), n_tasks=atlas.shape[1])
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"unsupported representation {representation}")
    return RelativeTaskScores(base, strength=strength)


def _freeze_to_owned_modules(model) -> None:
    owned = {id(parameter) for parameter in model.module_owned_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in owned)


def _corrupted_state(
    model, standard_deviation: float, seed: int
) -> dict[str, torch.Tensor]:
    if standard_deviation < 0:
        raise ValueError("corruption_std must be nonnegative")
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.module_owned_parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.add_(noise.to(parameter.device), alpha=standard_deviation)
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _train_task(
    model,
    optimizer,
    model_config,
    task_features: torch.Tensor,
    task: int,
    config: OptimizerExperimentConfig,
    device: torch.device,
    *,
    indexed: bool,
) -> None:
    generator = torch.Generator().manual_seed(config.seed + 10_000 + task)
    model.train()
    for _ in range(config.steps):
        tokens, targets, task_ids = make_program_batch(
            model_config,
            task_features,
            task,
            config.batch_size,
            generator,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        residual = model(tokens, task_ids) - targets
        (0.5 * residual.square().mean()).backward()
        if indexed:
            optimizer.step(task_ids=task_ids)
        else:
            optimizer.step()


def run_optimizer_experiment(
    checkpoint_path: Path,
    config: OptimizerExperimentConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run a bounded recovery assay with task-homogeneous optimizer steps."""
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = _model_config_from_checkpoint(payload)
    if model_config.seed != config.seed:
        raise ValueError("experiment seed must match checkpoint seed")
    features = hard_task_mixtures()
    if not config.task_indices or any(
        task < 0 or task >= features.shape[0] for task in config.task_indices
    ):
        raise ValueError("task_indices must select valid tasks")
    source_model = build_model(model_config, device)
    source_model.load_state_dict(payload["states"]["trained"])
    reference = make_program_dataset(
        model_config,
        features,
        config.reference_per_task,
        config.seed + 4001,
    )
    final_test = make_program_dataset(
        model_config,
        features,
        config.test_per_task,
        config.seed + 9001,
    )
    clean_losses = evaluate_program_losses(source_model, final_test, device)
    started = time.perf_counter()
    profiles, profile_timing = collect_importance_profiles(
        source_model, reference, device
    )
    atlases = {
        Estimator.RAW_OPG: profiles["raw_ef"],
        Estimator.RESIDUAL_NORMALIZED_OPG: profiles["ief"],
    }
    jl_features = jl_task_representation(
        features.shape[0],
        features.shape[1],
        config.seed + 3000 + config.jl_seed_offset,
    )
    corrupted_model = build_model(model_config, device)
    corrupted_model.load_state_dict(payload["states"]["trained"])
    corrupted_state = _corrupted_state(
        corrupted_model, config.corruption_std, config.seed + 8001
    )
    corrupted_losses = evaluate_program_losses(corrupted_model, final_test, device)
    records = []
    provider_storage: dict[str, int] = {}
    for estimator, atlas in atlases.items():
        for representation in Representation:
            method = f"{estimator.value}/{representation.value}"
            provider = _score_provider(
                atlas,
                representation,
                features,
                jl_features,
                strength=config.preconditioner_strength,
            )
            provider_storage[method] = provider.stored_floats
            for task in config.task_indices:
                model = build_model(model_config, device)
                model.load_state_dict(corrupted_state)
                _freeze_to_owned_modules(model)
                specs = transformer_mlp_group_specs(model)
                optimizer = TaskIndexedAdamW(
                    specs,
                    provider,
                    lr=config.learning_rate,
                    beta1=config.beta1,
                    weight_decay=config.weight_decay,
                )
                _train_task(
                    model,
                    optimizer,
                    model_config,
                    features,
                    task,
                    config,
                    device,
                    indexed=True,
                )
                final_losses = evaluate_program_losses(model, final_test, device)
                other = torch.ones(features.shape[0], dtype=torch.bool)
                other[task] = False
                records.append(
                    {
                        "method": method,
                        "estimator": estimator.value,
                        "representation": representation.value,
                        "task": task,
                        "initial_target_half_mse": float(corrupted_losses[task]),
                        "final_target_half_mse": float(final_losses[task]),
                        "target_improvement": float(
                            corrupted_losses[task] - final_losses[task]
                        ),
                        "off_target_loss_change": float(
                            (final_losses[other] - corrupted_losses[other]).mean()
                        ),
                        "preconditioner_stored_floats": provider.stored_floats,
                        **_recovery_metrics(
                            clean_losses[task],
                            corrupted_losses[task],
                            final_losses[task],
                        ),
                    }
                )
    for task in config.task_indices:
        model = build_model(model_config, device)
        model.load_state_dict(corrupted_state)
        _freeze_to_owned_modules(model)
        parameters = model.module_owned_parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, 0.999),
            weight_decay=config.weight_decay,
        )
        _train_task(
            model,
            optimizer,
            model_config,
            features,
            task,
            config,
            device,
            indexed=False,
        )
        final_losses = evaluate_program_losses(model, final_test, device)
        other = torch.ones(features.shape[0], dtype=torch.bool)
        other[task] = False
        records.append(
            {
                "method": "adamw",
                "estimator": "dynamic_squared_gradient",
                "representation": "parameter_diagonal",
                "task": task,
                "initial_target_half_mse": float(corrupted_losses[task]),
                "final_target_half_mse": float(final_losses[task]),
                "target_improvement": float(
                    corrupted_losses[task] - final_losses[task]
                ),
                "off_target_loss_change": float(
                    (final_losses[other] - corrupted_losses[other]).mean()
                ),
                "preconditioner_stored_floats": sum(
                    parameter.numel() for parameter in parameters
                ),
                **_recovery_metrics(
                    clean_losses[task],
                    corrupted_losses[task],
                    final_losses[task],
                ),
            }
        )
    owned_parameters = sum(
        parameter.numel() for parameter in source_model.module_owned_parameters()
    )
    resources = optimizer_resource_counts(
        n_parameters=owned_parameters,
        n_groups=source_model.n_modules,
        n_tasks=features.shape[0],
        dimension=features.shape[1],
    )
    resources["provider_stored_floats"] = provider_storage
    selected_damage = (
        corrupted_losses[list(config.task_indices)]
        - clean_losses[list(config.task_indices)]
    )
    adamw_records = [record for record in records if record["method"] == "adamw"]
    adamw_improvements = torch.tensor(
        [record["target_improvement"] for record in adamw_records]
    )
    premise_gate = {
        "corruption_damage": {
            "mean_all_tasks": float((corrupted_losses - clean_losses).mean()),
            "mean_selected_tasks": float(selected_damage.mean()),
            "minimum_selected_task": float(selected_damage.min()),
            "positive_selected_fraction": float((selected_damage > 0).float().mean()),
        },
        "adamw_recovery": {
            "mean_target_improvement": float(adamw_improvements.mean()),
            "positive_task_fraction": float((adamw_improvements > 0).float().mean()),
        },
        "passed": bool(
            (selected_damage > 0).all()
            and (adamw_improvements > 0).float().mean() >= 0.5
            and adamw_improvements.mean() > 0
        ),
        "rule": (
            "all selected tasks must be damaged; AdamW must improve at least half "
            "of them and improve them on average"
        ),
    }
    return {
        "setting": "task_conditioned_transformer_recovery_optimizer",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "model_configuration": asdict(model_config),
        "requires_distinct_task_queries": True,
        "task_homogeneous_steps": True,
        "score_granularity": "disjoint_mlp_feature_groups",
        "preconditioner_calibration": (
            "match_weighted_profile_mean_to_current_gradient_second_moment"
        ),
        "profile_timing": profile_timing,
        "premise_gate": premise_gate,
        "records": records,
        "resource_accounting": resources,
        "wall_seconds": time.perf_counter() - started,
    }
