"""Unified command-line entry point for active experiments.

Legacy module CLIs remain reproducible entry points. New iterations should use
this module so the run tier and output boundary are explicit in every command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .common import PROJECT_ROOT, save_json, seed_everything
from .continual_v4 import ContinualConfig
from .domain_train import OPTIMIZER_METHODS, DomainConfig
from .experiment_core import run_profile
from .inference_control import InferenceControlConfig, run_inference_control
from .language_inference_control import LanguageInferenceControlConfig
from .optimizer_experiment import (
    OptimizerExperimentConfig,
    run_optimizer_experiment,
)
from .paper_causal_v5 import run_causal_ranking
from .paper_continual_v5 import run_continual_curves
from .paper_query_v5 import QueryExperimentConfig, run_controlled_query


@dataclass(frozen=True)
class ExperimentPlan:
    experiment: str
    profile: str
    seed: int
    configuration: Any
    checkpoint: Path | None
    profiles: Path | None
    output: Path
    order: str | None
    strengths: tuple[float, ...]
    control_repeats: int
    claim_ready: bool
    data: Path | None = None
    feature_file: str = "task_features.pt"


def _output_path(experiment: str, profile: str, seed: int, order: str | None) -> Path:
    suffix = f"_{order}" if order is not None else ""
    if profile == "paper":
        root = (
            PROJECT_ROOT.parent
            / "kpsa-paper-internal"
            / "01_empirical"
            / "artifacts"
        )
    else:
        root = (
            PROJECT_ROOT.parent
            / "kpsa-internal"
            / "04_queryable_mechanisms"
            / "artifacts"
        )
    return root / experiment / f"{experiment}_{profile}_seed{seed}{suffix}.json"


def _continual_configuration(profile: str, seed: int) -> ContinualConfig:
    if profile == "paper":
        return ContinualConfig(
            seed=seed,
            n_tasks=30,
            n_modules=512,
            steps_per_task=80,
            reference_per_task=128,
            test_per_task=512,
        )
    return ContinualConfig(
        seed=seed,
        n_tasks=10,
        n_modules=128,
        batch_size=64,
        steps_per_task=40,
        reference_per_task=32,
        test_per_task=128,
    )


def _query_configuration(profile: str, seed: int) -> QueryExperimentConfig:
    if profile == "paper":
        return QueryExperimentConfig(seed=seed)
    return QueryExperimentConfig(
        seed=seed,
        reference_per_task=2,
        test_per_task=12,
        repeats=1,
        residual_scales=(0.5, 1.0),
    )


def build_experiment_plan(
    experiment: str,
    profile: str,
    *,
    seed: int,
    checkpoint: Path | None = None,
    profiles: Path | None = None,
    order: str | None = None,
    data: Path | None = None,
    method: str = "adamw",
    partition: str = "row",
    estimator: str = "raw",
    task_mode: str = "multi",
    score_link: str = "linear",
    role: str = "validation",
    lr: float = 0.001,
    feature_file: str = "task_features.pt",
    include_fisher: bool = False,
    skip_pruning: bool = False,
) -> ExperimentPlan:
    policy = run_profile(profile)
    if experiment == "optimizer":
        if data is None:
            raise ValueError("scratch optimizer runs require --data")
        if checkpoint is not None:
            raise ValueError(
                "scratch optimizer must not load a checkpoint; archival assay is optimizer_recovery"
            )
        configuration = DomainConfig(
            seed=seed,
            steps=12000 if profile == "paper" else 2000,
            hidden=512 if profile == "paper" else 384,
            layers=12 if profile == "paper" else 8,
            intermediate=1536 if profile == "paper" else 1024,
            heads=8 if profile == "paper" else 6,
            batch_size=32 if profile == "paper" else 16,
            eval_blocks=128 if profile == "paper" else 32,
            eval_every=2000,
            method=method,
            partition=partition,
            estimator=estimator,
            task_mode=task_mode,
            score_link=score_link,
            role=role,
            lr=lr,
        )
        strengths = ()
    elif experiment == "domain_applications":
        if data is None or checkpoint is None:
            raise ValueError("domain applications require --data and --checkpoint")
        configuration = {
            "samples": 256 if profile == "paper" else 64,
            "eval_blocks": 128 if profile == "paper" else 32,
            "causal_groups": 128 if profile == "paper" else 48,
            "role": role,
            "score_link": score_link,
            "feature_file": feature_file,
            "include_fisher": include_fisher,
            "pruning_fractions": () if skip_pruning else (0.05, 0.15),
        }
        strengths = ()
    elif experiment == "continual":
        if order not in {"forward", "reverse"}:
            raise ValueError("continual runs require forward or reverse order")
        configuration: Any = _continual_configuration(profile, seed)
        strengths = (2.0, 8.0, 32.0, 128.0) if profile == "paper" else (2.0, 8.0)
    elif experiment == "query":
        if checkpoint is None:
            raise ValueError("query runs require a checkpoint")
        configuration = _query_configuration(profile, seed)
        strengths = ()
    elif experiment in {"causal", "streaming"}:
        if checkpoint is None:
            raise ValueError(f"{experiment} runs require a checkpoint")
        configuration = {
            "reference_per_task": 32 if profile == "paper" else 2,
            "test_per_task": 192 if profile == "paper" else 12,
            "module_sample": 256 if profile == "paper" else 16,
            "repeats": policy.control_repeats,
        }
        strengths = ()
    elif experiment == "inference_control":
        if checkpoint is None:
            raise ValueError("inference_control runs require a checkpoint")
        configuration = InferenceControlConfig(
            seed=seed,
            reference_per_task=32 if profile == "paper" else 2,
            test_per_task=192 if profile == "paper" else 12,
            retained_fractions=(0.10, 0.25, 0.50, 0.75, 1.0)
            if profile == "paper"
            else (0.25, 0.50, 1.0),
        )
        strengths = ()
    elif experiment == "optimizer_recovery":
        if checkpoint is None:
            raise ValueError("optimizer runs require a checkpoint")
        configuration = OptimizerExperimentConfig(
            seed=seed,
            reference_per_task=32 if profile == "paper" else 4,
            test_per_task=192 if profile == "paper" else 48,
            batch_size=64 if profile == "paper" else 32,
            steps=20 if profile == "paper" else 8,
            task_indices=(0, 1, 2, 21, 22, 23) if profile == "paper" else (0, 21),
            corruption_std=0.05,
        )
        strengths = ()
    elif experiment == "language_inference_control":
        if checkpoint is None or profiles is None:
            raise ValueError(
                "language_inference_control runs require a checkpoint and profiles"
            )
        configuration = LanguageInferenceControlConfig(
            seed=seed,
            evaluation_per_task=6 if profile == "paper" else 2,
            retained_fractions=(0.50, 0.75, 0.90, 1.0)
            if profile == "paper"
            else (0.75, 0.90, 1.0),
        )
        strengths = ()
    else:
        raise ValueError(f"unknown experiment {experiment!r}")
    output = _output_path(experiment, profile, seed, order)
    if experiment == "optimizer":
        recipe = {
            "configuration": asdict(configuration),
            "data": str(data.resolve()),
            "feature_file": feature_file,
        }
        digest = hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode()
        ).hexdigest()[:12]
        output = output.with_stem(f"{output.stem}_{digest}")
    return ExperimentPlan(
        experiment=experiment,
        profile=profile,
        seed=seed,
        configuration=configuration,
        checkpoint=checkpoint,
        profiles=profiles,
        output=output,
        order=order,
        strengths=strengths,
        control_repeats=policy.control_repeats,
        claim_ready=(policy.claim_ready and experiment != "language_inference_control"),
        data=data,
        feature_file=feature_file,
    )


def execute_experiment_plan(
    plan: ExperimentPlan, *, device: torch.device | None = None
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if plan.experiment == "optimizer":
        from .domain_train import train

        assert plan.data is not None
        result = train(
            plan.data / "corpus.pt",
            plan.data / plan.feature_file,
            plan.output,
            plan.configuration,
            checkpoint=True,
        )
    elif plan.experiment == "domain_applications":
        from .domain_applications import run_applications

        assert plan.data is not None and plan.checkpoint is not None
        result = run_applications(
            plan.checkpoint,
            plan.data,
            plan.output,
            profile_cache=plan.profiles,
            control_seed=plan.seed,
            **plan.configuration,
        )
    elif plan.experiment == "continual":
        order = list(range(plan.configuration.n_tasks))
        if plan.order == "reverse":
            order.reverse()
        result = run_continual_curves(
            plan.configuration,
            order=order,
            strengths=plan.strengths,
            control_repeats=plan.control_repeats,
            device=device,
        )
    elif plan.experiment == "query":
        assert plan.checkpoint is not None
        result = run_controlled_query(
            plan.checkpoint, plan.configuration, device=device
        )
    elif plan.experiment == "causal":
        assert plan.checkpoint is not None
        result = run_causal_ranking(
            plan.checkpoint,
            seed=plan.seed,
            module_sample=plan.configuration["module_sample"],
            repeats=plan.configuration["repeats"],
            reference_per_task=plan.configuration["reference_per_task"],
            test_per_task=plan.configuration["test_per_task"],
            device=device,
        )
    elif plan.experiment == "streaming":
        assert plan.checkpoint is not None
        from .controlled_assay_v4 import _model_config_from_checkpoint
        from .controlled_v4 import build_model, hard_task_mixtures, make_program_dataset
        from .paper_streaming_v5 import compare_construction

        seed_everything(plan.seed, deterministic=True)
        payload = torch.load(plan.checkpoint, map_location="cpu", weights_only=False)
        model_config = _model_config_from_checkpoint(payload)
        model = build_model(model_config, device)
        model.load_state_dict(payload["states"]["trained"])
        features = hard_task_mixtures()
        dataset = make_program_dataset(
            model_config,
            features,
            plan.configuration["reference_per_task"],
            plan.seed + 4001,
        )
        result = {
            "setting": "streaming_tbe_construction",
            "seed": plan.seed,
            **compare_construction(model, dataset, features, device, ridge=1e-6),
        }
    elif plan.experiment == "inference_control":
        assert plan.checkpoint is not None
        result = run_inference_control(
            plan.checkpoint, plan.configuration, device=device
        )
    elif plan.experiment == "optimizer_recovery":
        assert plan.checkpoint is not None
        result = run_optimizer_experiment(
            plan.checkpoint, plan.configuration, device=device
        )
    elif plan.experiment == "language_inference_control":
        assert plan.checkpoint is not None and plan.profiles is not None
        from .language_inference_control import run_language_inference_control

        result = run_language_inference_control(
            plan.checkpoint,
            plan.profiles,
            plan.configuration,
            device=device,
        )
    else:  # pragma: no cover - plans validate this earlier
        raise ValueError(f"unknown experiment {plan.experiment!r}")
    payload = {
        **result,
        "run_profile": plan.profile,
        "claim_ready": plan.claim_ready,
    }
    save_json(plan.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        choices=(
            "query",
            "causal",
            "streaming",
            "continual",
            "inference_control",
            "optimizer",
            "optimizer_recovery",
            "domain_applications",
            "language_inference_control",
        ),
    )
    parser.add_argument("--profile", choices=("explore", "paper"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--method",
        default="adamw",
        choices=OPTIMIZER_METHODS,
    )
    parser.add_argument(
        "--partition", default="row", choices=("row", "tensor", "swiglu")
    )
    parser.add_argument("--estimator", default="raw", choices=("raw", "normalized"))
    parser.add_argument("--task-mode", default="multi", choices=("single", "multi"))
    parser.add_argument("--score-link", default="linear", choices=("linear", "log"))
    parser.add_argument("--role", default="validation", choices=("validation", "test"))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--feature-file", default="task_features.pt")
    parser.add_argument("--include-fisher", action="store_true")
    parser.add_argument("--skip-pruning", action="store_true")
    args = parser.parse_args()
    plan = build_experiment_plan(
        args.experiment,
        args.profile,
        seed=args.seed,
        checkpoint=args.checkpoint,
        profiles=args.profiles,
        order=args.order,
        data=args.data,
        method=args.method,
        partition=args.partition,
        estimator=args.estimator,
        task_mode=args.task_mode,
        score_link=args.score_link,
        role=args.role,
        lr=args.lr,
        feature_file=args.feature_file,
        include_fisher=args.include_fisher,
        skip_pruning=args.skip_pruning,
    )
    if args.output is not None:
        plan = replace(plan, output=args.output)
    execute_experiment_plan(plan)
    print(f"EXPERIMENT_RESULT={plan.output}")


if __name__ == "__main__":
    main()
