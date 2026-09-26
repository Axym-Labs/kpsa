from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .applications import continual_learning_assay_gate, continual_learning_metrics
from .common import (
    EPS,
    arc_artifact_dir,
    calibrate_importance_matrices,
    jl_task_representation,
    normalized_rows,
    save_json,
    seed_everything,
)
from .controlled_v4 import hard_task_mixtures
from .importance import build_regular_score_grid
from .streaming_v5 import fit_linear_tbe, query_linear_tbe


@dataclass(frozen=True)
class ContinualConfig:
    seed: int = 1
    input_dim: int = 8
    n_tasks: int = 10
    n_modules: int = 256
    batch_size: int = 128
    steps_per_task: int = 120
    reference_per_task: int = 128
    test_per_task: int = 512
    learning_rate: float = 1e-3
    strength: float = 8.0
    replay_per_task: int = 32


class ModularRegressor(nn.Module):
    """One hidden layer whose neurons form a complete parameter partition."""

    def __init__(self, input_dim: int, n_tasks: int, n_modules: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_tasks = n_tasks
        self.n_modules = n_modules
        self.hidden = nn.Linear(input_dim + n_tasks, n_modules)
        self.output = nn.Linear(n_modules, 1, bias=False)
        self.last_hidden: torch.Tensor | None = None

    def forward(
        self, inputs: torch.Tensor, task_ids: torch.Tensor, *, capture: bool = False
    ) -> torch.Tensor:
        task_features = torch.nn.functional.one_hot(
            task_ids, num_classes=self.n_tasks
        ).to(dtype=inputs.dtype)
        hidden = torch.nn.functional.silu(
            self.hidden(torch.cat((inputs, task_features), dim=1))
        )
        if capture:
            self.last_hidden = hidden
        return self.output(hidden).squeeze(1)

    def module_owned_parameters(self) -> list[nn.Parameter]:
        return [self.hidden.weight, self.hidden.bias, self.output.weight]

    def module_gradient_scores(self) -> torch.Tensor:
        return (
            self.hidden.weight.grad.square().sum(dim=1)
            + self.hidden.bias.grad.square()
            + self.output.weight.grad.square().squeeze(0)
        ).float()

    @torch.no_grad()
    def apply_module_gradient_scale(
        self, importance: torch.Tensor, strength: float
    ) -> None:
        scale = (1.0 / (1.0 + strength * importance)).to(
            device=self.hidden.weight.device, dtype=self.hidden.weight.grad.dtype
        )
        self.hidden.weight.grad.mul_(scale[:, None])
        self.hidden.bias.grad.mul_(scale)
        self.output.weight.grad.mul_(scale[None, :])


def _primitive_targets(inputs: torch.Tensor) -> torch.Tensor:
    n = inputs.shape[1]

    def column(index: int) -> torch.Tensor:
        return inputs[:, index % n]

    return torch.stack(
        (
            torch.sin(math.pi * column(0)),
            torch.tanh(2.0 * column(1) * column(2)),
            2.0 * column(3).square() - 2.0 / 3.0,
            torch.tanh(3.0 * (column(4) - column(5))),
            torch.sin(2.0 * (column(6) + column(7))),
            torch.tanh(inputs.sum(dim=1) / math.sqrt(n)),
        ),
        dim=1,
    )


def _make_task_examples(
    config: ContinualConfig,
    task: int,
    count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = 2.0 * torch.rand(count, config.input_dim, generator=generator) - 1.0
    mixture = hard_task_mixtures()[task]
    targets = _primitive_targets(inputs) @ mixture
    targets = targets + 0.02 * torch.randn(count, generator=generator)
    task_ids = torch.full((count,), task, dtype=torch.long)
    return inputs, targets, task_ids


def make_experiment_data(
    config: ContinualConfig, *, order: list[int] | None = None
) -> dict[str, Any]:
    if config.n_tasks > hard_task_mixtures().shape[0]:
        raise ValueError("n_tasks exceeds the task mixture table")
    train_generator = torch.Generator().manual_seed(config.seed + 101)
    reference_generator = torch.Generator().manual_seed(config.seed + 201)
    validation_generator = torch.Generator().manual_seed(config.seed + 251)
    test_generator = torch.Generator().manual_seed(config.seed + 301)
    task_order = list(range(config.n_tasks)) if order is None else list(order)
    if sorted(task_order) != list(range(config.n_tasks)):
        raise ValueError("order must be a permutation of every task")
    train_by_task = []
    references = []
    validation = []
    tests = []
    for task in range(config.n_tasks):
        train_by_task.append(
            [
                _make_task_examples(config, task, config.batch_size, train_generator)
                for _ in range(config.steps_per_task)
            ]
        )
        references.append(
            _make_task_examples(
                config, task, config.reference_per_task, reference_generator
            )
        )
        validation.append(
            _make_task_examples(
                config, task, config.test_per_task, validation_generator
            )
        )
        tests.append(
            _make_task_examples(config, task, config.test_per_task, test_generator)
        )
    return {
        "order": task_order,
        "train_batches": [train_by_task[task] for task in task_order],
        "references": references,
        "validation": validation,
        "tests": tests,
        "seeds": {
            "train_plan": config.seed + 101,
            "reference": config.seed + 201,
            "validation": config.seed + 251,
            "test": config.seed + 301,
        },
    }


@torch.no_grad()
def _evaluate_task(
    model: ModularRegressor,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    inputs, targets, tasks = (value.to(device) for value in data)
    predictions = model(inputs, tasks)
    return float(0.5 * (predictions - targets).square().mean())


def _profile_task(
    model: ModularRegressor,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    model.zero_grad(set_to_none=True)
    inputs, targets, tasks = (value.to(device) for value in data)
    predictions = model(inputs, tasks, capture=True)
    residual = predictions - targets
    loss = 0.5 * residual.square().mean()
    loss.backward()
    raw = model.module_gradient_scores().detach().float().cpu()
    residual_norm_sq = residual.detach().float().square().sum() / residual.numel() ** 2
    if model.last_hidden is None:
        raise RuntimeError("captured hidden features are unavailable")
    activation = model.last_hidden.detach().float().square().mean(dim=0).cpu()
    return {
        "ief": raw / residual_norm_sq.cpu().clamp_min(EPS),
        "raw_ef": raw,
        "activation": activation,
    }


def _reconstruct(columns: torch.Tensor, representation: torch.Tensor) -> torch.Tensor:
    features = normalized_rows(representation)
    return (columns @ features @ features.T).clamp_min(0)


def _protection_from_seen(
    method: str,
    ief_columns: list[torch.Tensor],
    raw_columns: list[torch.Tensor],
    activation_columns: list[torch.Tensor],
    online_columns: list[torch.Tensor],
    seen_tasks: list[int],
    semantic: torch.Tensor,
    jl: torch.Tensor,
    permuted: torch.Tensor,
    random_scores: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ief = torch.stack(ief_columns, dim=1)
    raw = torch.stack(raw_columns, dim=1)
    activation = torch.stack(activation_columns, dim=1)
    online = torch.stack(online_columns, dim=1)
    semantic_seen = semantic[seen_tasks]
    jl_seen = jl[seen_tasks]
    permuted_seen = permuted[seen_tasks]
    regular = build_regular_score_grid(
        {
            "raw_opg": raw,
            "residual_normalized_opg": ief,
        },
        semantic_seen,
        jl_features=jl_seen,
    )
    permuted_linear = query_linear_tbe(
        fit_linear_tbe(ief, permuted_seen), permuted_seen
    ).clamp_min(0)
    candidates = {
        **regular,
        # Read-compatible aliases for retained v4 artifacts and callers.
        "full_atlas": regular["residual_normalized_opg/full_atlas"],
        "tbe_linear": regular["residual_normalized_opg/tbe"],
        "jl_linear": regular["residual_normalized_opg/jl"],
        "permuted_basis_linear": permuted_linear,
        "mean_only": regular["residual_normalized_opg/mean_only"],
        "posthoc_ief__onehot": ief,
        "posthoc_ief__semantic6": _reconstruct(ief, semantic_seen),
        "posthoc_ief__jl6": _reconstruct(ief, jl_seen),
        "posthoc_raw_ef__semantic6": _reconstruct(raw, semantic_seen),
        "online_raw_ef__semantic6": _reconstruct(online, semantic_seen),
        "posthoc_activation__semantic6": _reconstruct(activation, semantic_seen),
        "random": random_scores[:, seen_tasks],
    }
    calibrated, metadata = calibrate_importance_matrices(
        candidates, reference_key="posthoc_ief__onehot"
    )
    reference_rms = metadata["posthoc_ief__onehot"]["calibrated_rms"]
    for name in calibrated:
        calibrated[name] = calibrated[name] / max(float(reference_rms), EPS)
        metadata[name]["application_rms"] = float(
            calibrated[name].square().mean().sqrt()
        )
    if method not in calibrated:
        return torch.zeros(ief.shape[0]), metadata
    return calibrated[method].amax(dim=1), metadata


def _parameter_fisher(
    model: ModularRegressor,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    inputs, targets, tasks = (value.to(device) for value in data)
    loss = 0.5 * (model(inputs, tasks) - targets).square().mean()
    loss.backward()
    fisher = [
        parameter.grad.detach().float().square().cpu()
        for parameter in model.module_owned_parameters()
    ]
    mean = torch.cat([value.flatten() for value in fisher]).mean().clamp_min(EPS)
    return [value / mean for value in fisher]


def _ewc_penalty(
    model: ModularRegressor,
    terms: list[tuple[list[torch.Tensor], list[torch.Tensor]]],
) -> torch.Tensor:
    parameters = model.module_owned_parameters()
    total = sum(parameter.numel() for parameter in parameters)
    penalty = torch.zeros((), device=parameters[0].device)
    for centers, fisher in terms:
        for parameter, center, weight in zip(parameters, centers, fisher):
            penalty = (
                penalty
                + (
                    weight.to(parameter.device)
                    * (parameter - center.to(parameter.device)).square()
                ).sum()
                / total
            )
    return 0.5 * penalty


def _module_penalty(
    model: ModularRegressor,
    centers: list[torch.Tensor],
    importance: torch.Tensor,
) -> torch.Tensor:
    hidden_weight, hidden_bias, output_weight = model.module_owned_parameters()
    weight = importance.to(hidden_weight.device)
    total = sum(parameter.numel() for parameter in model.module_owned_parameters())
    penalty = (
        (hidden_weight - centers[0].to(hidden_weight.device)).square() * weight[:, None]
    ).sum()
    penalty = (
        penalty
        + ((hidden_bias - centers[1].to(hidden_bias.device)).square() * weight).sum()
    )
    penalty = (
        penalty
        + (
            (output_weight - centers[2].to(output_weight.device)).square()
            * weight[None, :]
        ).sum()
    )
    return 0.5 * penalty / total


def run_cl_method(
    config: ContinualConfig,
    method: str,
    initial_state: dict[str, torch.Tensor],
    data: dict[str, Any],
    *,
    strength: float,
    device: torch.device,
    evaluation_key: str = "tests",
    jl_seed_offset: int = 0,
) -> dict[str, Any]:
    attenuation_methods = {
        "raw_opg/full_atlas",
        "raw_opg/tbe",
        "raw_opg/jl",
        "raw_opg/mean_only",
        "residual_normalized_opg/full_atlas",
        "residual_normalized_opg/tbe",
        "residual_normalized_opg/jl",
        "residual_normalized_opg/mean_only",
        "full_atlas",
        "tbe_linear",
        "jl_linear",
        "permuted_basis_linear",
        "mean_only",
        "posthoc_ief__semantic6",
        "posthoc_ief__onehot",
        "posthoc_ief__jl6",
        "posthoc_raw_ef__semantic6",
        "online_raw_ef__semantic6",
        "posthoc_activation__semantic6",
        "random",
    }
    special_methods = {"none", "ewc", "module_iewc__semantic6", "replay"}
    if method not in attenuation_methods | special_methods:
        raise ValueError(f"unknown continual-learning method {method!r}")
    model = ModularRegressor(config.input_dim, config.n_tasks, config.n_modules).to(
        device
    )
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(
        model.module_owned_parameters(), lr=config.learning_rate
    )
    mixtures = hard_task_mixtures()[: config.n_tasks]
    semantic = normalized_rows(mixtures)
    jl = jl_task_representation(config.n_tasks, 6, config.seed + 1001 + jl_seed_offset)
    permutation = torch.randperm(
        config.n_tasks,
        generator=torch.Generator().manual_seed(config.seed + 3001 + jl_seed_offset),
    )
    permuted = semantic[permutation]
    random_scores = torch.rand(
        config.n_modules,
        config.n_tasks,
        generator=torch.Generator().manual_seed(config.seed + 2001),
    )
    order = data["order"]
    history = torch.full((config.n_tasks, config.n_tasks), float("nan"))
    pre_update = torch.empty(config.n_tasks)
    protection = torch.zeros(config.n_modules)
    ief_columns: list[torch.Tensor] = []
    raw_columns: list[torch.Tensor] = []
    activation_columns: list[torch.Tensor] = []
    online_columns: list[torch.Tensor] = []
    calibration_history = []
    ewc_terms: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
    iewc_state: tuple[list[torch.Tensor], torch.Tensor] | None = None
    replay_inputs = torch.empty(0, config.input_dim)
    replay_targets = torch.empty(0)
    replay_tasks = torch.empty(0, dtype=torch.long)
    for stage, task in enumerate(order):
        pre_update[stage] = _evaluate_task(model, data[evaluation_key][task], device)
        online_sum = torch.zeros(config.n_modules)
        model.train()
        for step, (inputs, targets, tasks) in enumerate(data["train_batches"][stage]):
            if method == "replay" and replay_inputs.numel():
                replay_count = min(config.batch_size // 2, replay_inputs.shape[0])
                start = (step * replay_count) % replay_inputs.shape[0]
                indices = (
                    torch.arange(start, start + replay_count) % replay_inputs.shape[0]
                )
                inputs = torch.cat((inputs, replay_inputs[indices]), dim=0)
                targets = torch.cat((targets, replay_targets[indices]), dim=0)
                tasks = torch.cat((tasks, replay_tasks[indices]), dim=0)
            inputs, targets, tasks = (
                value.to(device) for value in (inputs, targets, tasks)
            )
            optimizer.zero_grad(set_to_none=True)
            residual = model(inputs, tasks) - targets
            loss = 0.5 * residual.square().mean()
            if method == "ewc" and ewc_terms:
                loss = loss + strength * _ewc_penalty(model, ewc_terms)
            if method == "module_iewc__semantic6" and iewc_state is not None:
                loss = loss + strength * _module_penalty(model, *iewc_state)
            loss.backward()
            online_sum.add_(model.module_gradient_scores().detach().float().cpu())
            if stage > 0 and method in attenuation_methods:
                model.apply_module_gradient_scale(protection, strength)
            optimizer.step()
        for previous_stage, seen_task in enumerate(order[: stage + 1]):
            history[stage, previous_stage] = _evaluate_task(
                model, data[evaluation_key][seen_task], device
            )
        profile = _profile_task(model, data["references"][task], device)
        ief_columns.append(profile["ief"])
        raw_columns.append(profile["raw_ef"])
        activation_columns.append(profile["activation"])
        online_columns.append(online_sum / config.steps_per_task)
        protection_method = (
            "posthoc_ief__semantic6" if method == "module_iewc__semantic6" else method
        )
        protection, metadata = _protection_from_seen(
            protection_method,
            ief_columns,
            raw_columns,
            activation_columns,
            online_columns,
            order[: stage + 1],
            semantic,
            jl,
            permuted,
            random_scores,
        )
        calibration_history.append(metadata)
        if method == "ewc":
            ewc_terms.append(
                (
                    [
                        parameter.detach().cpu().clone()
                        for parameter in model.module_owned_parameters()
                    ],
                    _parameter_fisher(model, data["references"][task], device),
                )
            )
        if method == "module_iewc__semantic6":
            iewc_state = (
                [
                    parameter.detach().cpu().clone()
                    for parameter in model.module_owned_parameters()
                ],
                protection.detach().cpu(),
            )
        if method == "replay":
            reference_inputs, reference_targets, reference_tasks = data["references"][
                task
            ]
            take = min(config.replay_per_task, reference_inputs.shape[0])
            replay_inputs = torch.cat((replay_inputs, reference_inputs[:take]), dim=0)
            replay_targets = torch.cat(
                (replay_targets, reference_targets[:take]), dim=0
            )
            replay_tasks = torch.cat((replay_tasks, reference_tasks[:take]), dim=0)
    return {
        "method": method,
        "history": history,
        "pre_update_per_task": pre_update,
        "summary": continual_learning_metrics(
            history,
            higher_is_better=False,
            pre_update_per_task=pre_update,
        ),
        "calibration_history": calibration_history,
        "update_scope": {
            "owned_trainable_parameters": sum(
                parameter.numel() for parameter in model.module_owned_parameters()
            ),
            "all_trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    }


def run_no_protection_pilot(
    config: ContinualConfig, *, device: torch.device | None = None
) -> dict[str, Any]:
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_experiment_data(config)
    model = ModularRegressor(config.input_dim, config.n_tasks, config.n_modules)
    initial = copy.deepcopy(model.state_dict())
    result = run_cl_method(config, "none", initial, data, strength=0.0, device=device)
    gate = continual_learning_assay_gate(
        no_protection_forgetting=result["summary"]["average_forgetting"],
        no_protection_acquisition_gain=result["summary"]["average_acquisition_gain"],
        minimum_forgetting=0.02,
        minimum_acquisition_gain=0.02,
    )
    return {
        "setting": "task_neutral_modular_regressor",
        "seed": config.seed,
        "configuration": asdict(config),
        "data_seeds": data["seeds"],
        "no_protection": result,
        "premise_gate": gate,
        "initial_state": initial,
    }


def run_continual_experiment(
    config: ContinualConfig,
    *,
    strength_candidates: tuple[float, ...] = (
        0.5,
        2.0,
        8.0,
        32.0,
        128.0,
        512.0,
        2048.0,
    ),
    replay_candidates: tuple[int, ...] = (4, 16, 64, 128),
    device: torch.device | None = None,
    force_comparison: bool = False,
    task_order: list[int] | None = None,
) -> dict[str, Any]:
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_experiment_data(config, order=task_order)
    initial_model = ModularRegressor(config.input_dim, config.n_tasks, config.n_modules)
    initial = copy.deepcopy(initial_model.state_dict())
    none = run_cl_method(
        config,
        "none",
        initial,
        data,
        strength=0.0,
        device=device,
        evaluation_key="tests",
    )
    none_validation = run_cl_method(
        config,
        "none",
        initial,
        data,
        strength=0.0,
        device=device,
        evaluation_key="validation",
    )
    minimum_acquisition = 0.9 * none_validation["summary"]["average_acquisition_gain"]
    gate = continual_learning_assay_gate(
        no_protection_forgetting=none["summary"]["average_forgetting"],
        no_protection_acquisition_gain=none["summary"]["average_acquisition_gain"],
        minimum_forgetting=0.02,
        minimum_acquisition_gain=0.02,
    )
    methods: dict[str, Any] = {"none": none}
    selection: dict[str, Any] = {}
    if gate["passed"] or force_comparison:
        tuned_methods = (
            "posthoc_ief__semantic6",
            "posthoc_ief__onehot",
            "posthoc_ief__jl6",
            "posthoc_raw_ef__semantic6",
            "online_raw_ef__semantic6",
            "posthoc_activation__semantic6",
            "random",
            "ewc",
            "module_iewc__semantic6",
        )
        for method in tuned_methods:
            candidate_records = []
            for strength in strength_candidates:
                candidate = run_cl_method(
                    config,
                    method,
                    initial,
                    data,
                    strength=strength,
                    device=device,
                    evaluation_key="validation",
                )
                candidate_records.append(
                    {
                        "strength": strength,
                        "eligible": candidate["summary"]["average_acquisition_gain"]
                        >= minimum_acquisition,
                        "acquisition_ratio_to_no_protection": candidate["summary"][
                            "average_acquisition_gain"
                        ]
                        / max(
                            none_validation["summary"]["average_acquisition_gain"],
                            EPS,
                        ),
                        "summary": candidate["summary"],
                    }
                )
            eligible = [record for record in candidate_records if record["eligible"]]
            selected = min(
                eligible if eligible else candidate_records,
                key=lambda record: record["summary"]["final_average"],
            )
            selection[method] = {
                "selected_strength": selected["strength"],
                "criterion": "lowest validation final half-MSE subject to >=90% of no-protection acquisition",
                "minimum_acquisition_gain": minimum_acquisition,
                "selection_constraint_satisfied": bool(eligible),
                "candidates": candidate_records,
            }
            methods[method] = run_cl_method(
                config,
                method,
                initial,
                data,
                strength=selected["strength"],
                device=device,
                evaluation_key="tests",
            )
            if method == "posthoc_ief__jl6":
                for projection_repeat in (1, 2):
                    label = f"posthoc_ief__jl6_seed{projection_repeat}"
                    methods[label] = run_cl_method(
                        config,
                        method,
                        initial,
                        data,
                        strength=selected["strength"],
                        device=device,
                        evaluation_key="tests",
                        jl_seed_offset=projection_repeat,
                    )
                    methods[label]["method"] = label
                    selection[label] = {
                        "selected_strength": selected["strength"],
                        "criterion": "shared with primary JL projection",
                        "projection_seed_offset": projection_repeat,
                    }
        replay_records = []
        for memory in replay_candidates:
            replay_config = replace(config, replay_per_task=memory)
            candidate = run_cl_method(
                replay_config,
                "replay",
                initial,
                data,
                strength=0.0,
                device=device,
                evaluation_key="validation",
            )
            replay_records.append(
                {
                    "replay_examples_per_task": memory,
                    "eligible": candidate["summary"]["average_acquisition_gain"]
                    >= minimum_acquisition,
                    "acquisition_ratio_to_no_protection": candidate["summary"][
                        "average_acquisition_gain"
                    ]
                    / max(
                        none_validation["summary"]["average_acquisition_gain"],
                        EPS,
                    ),
                    "summary": candidate["summary"],
                }
            )
        replay_eligible = [record for record in replay_records if record["eligible"]]
        selected_replay = min(
            replay_eligible if replay_eligible else replay_records,
            key=lambda record: record["summary"]["final_average"],
        )
        selected_replay_config = replace(
            config,
            replay_per_task=selected_replay["replay_examples_per_task"],
        )
        methods["replay"] = run_cl_method(
            selected_replay_config,
            "replay",
            initial,
            data,
            strength=0.0,
            device=device,
            evaluation_key="tests",
        )
        selection["replay"] = {
            "selected_replay_examples_per_task": selected_replay[
                "replay_examples_per_task"
            ],
            "criterion": "lowest validation final half-MSE subject to >=90% of no-protection acquisition",
            "minimum_acquisition_gain": minimum_acquisition,
            "selection_constraint_satisfied": bool(replay_eligible),
            "candidates": replay_records,
        }
    return {
        "setting": "task_neutral_modular_regressor",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "data_roles": data["seeds"],
        "task_order": data["order"],
        "premise_gate": gate,
        "strength_selection_on_validation": selection,
        "methods": methods,
        "comparison_executed": bool(gate["passed"] or force_comparison),
        "forced_despite_failed_gate": bool(force_comparison and not gate["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=arc_artifact_dir("03_exploratory", "continual"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    config = ContinualConfig(seed=args.seed)
    if args.smoke:
        config = ContinualConfig(
            seed=args.seed,
            input_dim=4,
            n_tasks=3,
            n_modules=12,
            batch_size=8,
            steps_per_task=2,
            reference_per_task=8,
            test_per_task=8,
        )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.full:
        task_order = (
            list(range(config.n_tasks))
            if args.order == "forward"
            else list(reversed(range(config.n_tasks)))
        )
        result = run_continual_experiment(config, task_order=task_order)
        path = args.artifact_dir / (
            f"continual_v4_metrics_{args.order}_seed{args.seed}.json"
        )
        save_json(path, result)
        print(f"CONTINUAL_V4_RESULT={path}")
    else:
        result = run_no_protection_pilot(config)
        initial = result.pop("initial_state")
        path = args.artifact_dir / f"continual_v4_premise_seed{args.seed}.json"
        save_json(path, result)
        torch.save(
            {"initial_state": initial, "configuration": asdict(config)},
            args.artifact_dir / f"continual_v4_initial_seed{args.seed}.pt",
        )
        print(f"CONTINUAL_V4_PREMISE_RESULT={path}")


if __name__ == "__main__":
    main()
