from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .common import jl_task_representation, save_json, seed_everything
from .controlled_assay_v4 import (
    _model_config_from_checkpoint,
    _topk_mask,
    collect_importance_profiles,
    reference_reliability,
    task_circuit_result,
)
from .controlled_v4 import (
    build_model,
    fraction_to_count,
    hard_task_mixtures,
    make_program_dataset,
)
from .query_v5 import (
    build_query_methods,
    query_ranking_metrics,
    select_residual_scale,
)


@dataclass(frozen=True)
class QueryExperimentConfig:
    seed: int
    reference_per_task: int = 32
    test_per_task: int = 192
    repeats: int = 10
    residual_scales: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    ridge: float = 1e-6
    circuit_fraction: float = 0.10
    ranking_top_fraction: float = 0.10


def run_controlled_query(
    checkpoint_path: Path,
    config: QueryExperimentConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate cold-start attribution and interventions on a frozen model."""
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = _model_config_from_checkpoint(payload)
    if model_config.seed != config.seed:
        raise ValueError("experiment seed must match checkpoint seed")
    model = build_model(model_config, device)
    model.load_state_dict(payload["states"]["trained"])
    mixtures = hard_task_mixtures()
    data_roles = {
        "index_source": config.seed + 4001,
        "independent_opg_target": config.seed + 5001,
        "final_behavior": config.seed + 7001,
    }
    source_data = make_program_dataset(
        model_config,
        mixtures,
        config.reference_per_task,
        data_roles["index_source"],
    )
    target_data = make_program_dataset(
        model_config,
        mixtures,
        config.reference_per_task,
        data_roles["independent_opg_target"],
    )
    final_test = make_program_dataset(
        model_config,
        mixtures,
        config.test_per_task,
        data_roles["final_behavior"],
    )
    started = time.perf_counter()
    source_profiles, source_timing = collect_importance_profiles(
        model, source_data, device
    )
    target_profiles, target_timing = collect_importance_profiles(
        model, target_data, device
    )
    # The legacy collector calls this field "ief". It is the module-wise,
    # residual-normalized trace of the per-example OPG used in this study.
    source_opg = source_profiles["ief"]
    target_opg = target_profiles["ief"]
    triple_tasks = list(range(21, mixtures.shape[0]))
    split_folds = {
        "leave_one_triple_out": [[task] for task in triple_tasks],
        "all_triples_out": [triple_tasks],
    }
    scale_selections = {
        "tbe": select_residual_scale(
            source_opg,
            target_opg,
            mixtures,
            task_indices=range(21),
            residual_scales=config.residual_scales,
            top_fraction=config.ranking_top_fraction,
            ridge=config.ridge,
        )
    }
    for repeat in range(config.repeats):
        jl_features = jl_task_representation(
            mixtures.shape[0], mixtures.shape[1], config.seed + 3000 + repeat
        )
        scale_selections[f"jl_seed{repeat}"] = select_residual_scale(
            source_opg,
            target_opg,
            jl_features,
            task_indices=range(21),
            residual_scales=config.residual_scales,
            top_fraction=config.ranking_top_fraction,
            ridge=config.ridge,
        )
        permutation = torch.randperm(
            mixtures.shape[0],
            generator=torch.Generator().manual_seed(config.seed + 4000 + repeat),
        )
        scale_selections[f"permuted_basis_seed{repeat}"] = select_residual_scale(
            source_opg,
            target_opg,
            mixtures[permutation],
            task_indices=range(21),
            residual_scales=config.residual_scales,
            top_fraction=config.ranking_top_fraction,
            ridge=config.ridge,
        )
    splits: dict[str, Any] = {}
    for split_name, folds in split_folds.items():
        methods = build_query_methods(
            source_opg,
            mixtures,
            folds=folds,
            seed=config.seed,
            repeats=config.repeats,
            residual_scales=config.residual_scales,
            ridge=config.ridge,
        )
        ranking = {
            method: query_ranking_metrics(
                scores,
                source_opg,
                target_opg,
                folds=folds,
                queryable=queryable,
                top_fraction=config.ranking_top_fraction,
            )
            for method, (scores, queryable) in methods.items()
        }
        causal_records = []
        if split_name == "all_triples_out":
            count = fraction_to_count(config.circuit_fraction, model.n_modules)
            for method, (scores, queryable) in methods.items():
                for task in triple_tasks:
                    if not bool(queryable[task]):
                        continue
                    result = task_circuit_result(
                        model,
                        final_test,
                        task,
                        _topk_mask(scores[:, task], count),
                        device,
                    )
                    causal_records.append(
                        {
                            "method": method,
                            "task": task,
                            "selected_modules": count,
                            "selected_fraction_actual": count / model.n_modules,
                            **result,
                        }
                    )
        splits[split_name] = {
            "folds": folds,
            "ranking": ranking,
            "causal_records": causal_records,
            "scale_selections": (
                scale_selections if split_name == "all_triples_out" else None
            ),
        }

    modules, tasks = source_opg.shape
    dimension = mixtures.shape[1]
    full_floats = modules * tasks
    compact_floats = modules * (dimension + 1) + tasks * dimension
    return {
        "setting": "controlled_tagged_compositional_programs",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "model_configuration": asdict(model_config),
        "data_roles": data_roles,
        "importance_statistic": "residual_normalized_opg_trace",
        "reference_reliability": reference_reliability(source_opg, target_opg),
        "profile_timing": {
            "index_source": source_timing,
            "independent_opg_target": target_timing,
        },
        "resource_accounting": {
            "modules": modules,
            "tasks": tasks,
            "task_dimension": dimension,
            "full_atlas_stored_floats": full_floats,
            "tbe_stored_floats": compact_floats,
            "storage_reduction_fraction": 1.0 - compact_floats / full_floats,
            "construction_mode": "posthoc_from_full_atlas",
        },
        "splits": splits,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "task-embeddings-paper-internal"
            / "01_empirical"
            / "artifacts"
            / "query"
        ),
    )
    args = parser.parse_args()
    result = run_controlled_query(
        args.checkpoint, QueryExperimentConfig(seed=args.seed)
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    output = args.artifact_dir / f"query_v5_seed{args.seed}.json"
    save_json(output, result)
    print(f"PAPER_QUERY_V5_RESULT={output}")


if __name__ == "__main__":
    main()
