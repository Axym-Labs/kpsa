"""Task-queryable optimizer preconditioners.

This is an AdamW-like first-moment optimizer with an externally estimated,
task-conditioned diagonal at parameter-group resolution. It does not claim
that squared gradients are the model Fisher; the score provider determines
whether raw or residual-normalized OPG statistics are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.optim import Optimizer

from .streaming_v5 import LinearTBEIndex, fit_linear_tbe, query_linear_tbe


class TaskScoreProvider(Protocol):
    @property
    def n_groups(self) -> int: ...

    @property
    def stored_floats(self) -> int: ...

    def query(self, task: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class MatrixTaskScores:
    scores: torch.Tensor

    def __post_init__(self) -> None:
        values = self.scores.detach().float().cpu()
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("scores must have shape [groups, tasks]")
        object.__setattr__(self, "scores", values)

    @property
    def n_groups(self) -> int:
        return self.scores.shape[0]

    @property
    def stored_floats(self) -> int:
        return self.scores.numel()

    def query(self, task: int) -> torch.Tensor:
        if not 0 <= task < self.scores.shape[1]:
            raise ValueError(f"task {task} is outside the score matrix")
        return self.scores[:, task]


@dataclass(frozen=True)
class LinearTaskScores:
    index: LinearTBEIndex
    task_features: torch.Tensor

    def __post_init__(self) -> None:
        features = self.task_features.detach().float().cpu()
        if features.ndim != 2 or features.shape[1] != self.index.feature_mean.numel():
            raise ValueError("task features do not match the TBE index")
        object.__setattr__(self, "task_features", features)

    @classmethod
    def from_atlas(
        cls,
        atlas: torch.Tensor,
        task_features: torch.Tensor,
        *,
        ridge: float = 1e-6,
    ) -> LinearTaskScores:
        return cls(fit_linear_tbe(atlas, task_features, ridge=ridge), task_features)

    @property
    def n_groups(self) -> int:
        return self.index.module_mean.numel()

    @property
    def stored_floats(self) -> int:
        return (
            self.index.module_mean.numel()
            + self.index.feature_mean.numel()
            + self.index.coefficients.numel()
            + self.task_features.numel()
        )

    def query(self, task: int) -> torch.Tensor:
        if not 0 <= task < self.task_features.shape[0]:
            raise ValueError(f"task {task} is outside the task feature table")
        return query_linear_tbe(
            self.index, self.task_features[task : task + 1]
        ).squeeze(1)


@dataclass(frozen=True)
class MeanTaskScores:
    scores: torch.Tensor
    n_tasks: int

    def __post_init__(self) -> None:
        values = self.scores.detach().float().cpu().flatten()
        if values.numel() < 1 or self.n_tasks < 1:
            raise ValueError("mean scores and n_tasks must be nonempty")
        object.__setattr__(self, "scores", values)

    @property
    def n_groups(self) -> int:
        return self.scores.numel()

    @property
    def stored_floats(self) -> int:
        return self.scores.numel()

    def query(self, task: int) -> torch.Tensor:
        if not 0 <= task < self.n_tasks:
            raise ValueError(f"task {task} is outside the configured tasks")
        return self.scores


@dataclass(frozen=True)
class RelativeTaskScores:
    """Convert nonnegative importance into a stable relative denominator."""

    base: TaskScoreProvider
    strength: float = 1.0
    floor: float = 1.0

    def __post_init__(self) -> None:
        if self.strength < 0 or self.floor <= 0:
            raise ValueError("strength must be nonnegative and floor must be positive")

    @property
    def n_groups(self) -> int:
        return self.base.n_groups

    @property
    def stored_floats(self) -> int:
        return self.base.stored_floats

    def query(self, task: int) -> torch.Tensor:
        values = self.base.query(task).clamp_min(0)
        mean = values.mean()
        if float(mean) <= 0:
            return torch.full_like(values, self.floor)
        return self.floor + self.strength * values / mean


@dataclass(frozen=True)
class ParameterGroupSpec:
    parameter: torch.nn.Parameter
    group_ids: torch.Tensor

    def __post_init__(self) -> None:
        ids = self.group_ids.detach().long().cpu()
        if ids.shape != self.parameter.shape:
            raise ValueError("group_ids must match the parameter shape")
        if ids.numel() and int(ids.min()) < 0:
            raise ValueError("group ids must be nonnegative")
        object.__setattr__(self, "group_ids", ids)


def transformer_mlp_group_specs(model) -> list[ParameterGroupSpec]:
    """Map controlled or Qwen SwiGLU parameters to disjoint feature groups."""
    specs: list[ParameterGroupSpec] = []
    offset = 0
    if hasattr(model, "blocks"):
        mlps = [block.mlp for block in model.blocks]
        projection_names = ("gate", "up", "down")
    elif hasattr(model, "mlps"):
        mlps = model.mlps
        projection_names = ("gate_proj", "up_proj", "down_proj")
    else:
        raise TypeError("model does not expose supported Transformer MLP blocks")
    for mlp in mlps:
        gate = getattr(mlp, projection_names[0])
        up = getattr(mlp, projection_names[1])
        down = getattr(mlp, projection_names[2])
        n_features = gate.out_features
        local = torch.arange(offset, offset + n_features)
        specs.append(
            ParameterGroupSpec(gate.weight, local[:, None].expand_as(gate.weight))
        )
        if gate.bias is not None:
            specs.append(ParameterGroupSpec(gate.bias, local))
        specs.append(ParameterGroupSpec(up.weight, local[:, None].expand_as(up.weight)))
        if up.bias is not None:
            specs.append(ParameterGroupSpec(up.bias, local))
        specs.append(
            ParameterGroupSpec(down.weight, local[None, :].expand_as(down.weight))
        )
        offset += n_features
    if offset != model.n_modules:
        raise RuntimeError("group mapping does not cover the model modules")
    return specs


def controlled_transformer_group_specs(model) -> list[ParameterGroupSpec]:
    """Compatibility alias for the original controlled-only helper."""
    return transformer_mlp_group_specs(model)


def _homogeneous_task(task_ids: int | torch.Tensor) -> int:
    if isinstance(task_ids, int):
        return task_ids
    values = task_ids.detach().long().flatten().cpu().unique()
    if values.numel() != 1:
        raise ValueError(
            "TaskIndexedAdamW requires a task-homogeneous gradient step; "
            "mixed-task gradients contain unrecoverable cross-task terms"
        )
    return int(values.item())


class TaskIndexedAdamW(Optimizer):
    """AdamW first moment with a queried group-wise OPG preconditioner."""

    def __init__(
        self,
        specs: list[ParameterGroupSpec],
        score_provider: TaskScoreProvider,
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if not specs:
            raise ValueError("at least one parameter specification is required")
        if lr < 0 or not 0 <= beta1 < 1 or eps < 0 or weight_decay < 0:
            raise ValueError("invalid optimizer hyperparameters")
        parameters = [spec.parameter for spec in specs]
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("each parameter must have exactly one group mapping")
        for spec in specs:
            if (
                spec.group_ids.numel()
                and int(spec.group_ids.max()) >= score_provider.n_groups
            ):
                raise ValueError("group mapping exceeds the score provider")
        super().__init__(
            parameters,
            {"lr": lr, "beta1": beta1, "eps": eps, "weight_decay": weight_decay},
        )
        self._specs = specs
        self._spec_group_counts = [
            torch.bincount(
                spec.group_ids.flatten(), minlength=score_provider.n_groups
            ).float()
            for spec in specs
        ]
        self.score_provider = score_provider

    @torch.no_grad()
    def step(
        self,
        closure=None,
        *,
        task_ids: int | torch.Tensor,
    ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        task = _homogeneous_task(task_ids)
        group_scores = self.score_provider.query(task).clamp_min(0)
        options = self.param_groups[0]
        active = [
            (spec, counts)
            for spec, counts in zip(self._specs, self._spec_group_counts)
            if spec.parameter.grad is not None
        ]
        if not active:
            return loss
        element_counts = sum(
            (counts for _, counts in active), torch.zeros_like(group_scores)
        )
        n_elements = element_counts.sum()
        score_mean = (group_scores * element_counts).sum() / n_elements
        if float(score_mean) <= 0:
            raise ValueError("queried preconditioner must have positive mean")
        live_second_moment = sum(
            spec.parameter.grad.detach().float().square().sum() for spec, _ in active
        ) / n_elements.to(active[0][0].parameter.device)
        calibrated_scores = group_scores / score_mean * live_second_moment.cpu()
        for spec, _ in active:
            parameter = spec.parameter
            gradient = parameter.grad
            if gradient.is_sparse:
                raise RuntimeError("TaskIndexedAdamW does not support sparse gradients")
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            beta1 = options["beta1"]
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            bias_correction = 1.0 - beta1 ** state["step"]
            if options["weight_decay"]:
                parameter.mul_(1.0 - options["lr"] * options["weight_decay"])
            ids = spec.group_ids.to(parameter.device)
            denominator = calibrated_scores.to(parameter.device)[ids].sqrt()
            denominator = denominator.add(options["eps"])
            if bool((denominator <= 0).any()):
                raise ValueError(
                    "queried preconditioner must be positive after epsilon"
                )
            parameter.addcdiv_(
                exp_avg,
                denominator,
                value=-options["lr"] / bias_correction,
            )
        return loss


def optimizer_resource_counts(
    *, n_parameters: int, n_groups: int, n_tasks: int, dimension: int
) -> dict[str, int | float]:
    if min(n_parameters, n_groups, n_tasks, dimension) < 1:
        raise ValueError("all resource dimensions must be positive")
    tbe = n_groups * (dimension + 1) + n_tasks * dimension + dimension
    consolidated = n_parameters
    group_atlas = n_groups * n_tasks
    adamw_total = 2 * n_parameters
    tbe_total = n_parameters + tbe
    group_atlas_total = n_parameters + group_atlas
    return {
        "consolidated_diagonal_preconditioner": consolidated,
        "parameter_task_atlas_preconditioner": n_parameters * n_tasks,
        "group_task_atlas_preconditioner": group_atlas,
        "tbe_preconditioner": tbe,
        "tbe_vs_consolidated_reduction": 1.0 - tbe / consolidated,
        "tbe_vs_parameter_task_atlas_reduction": 1.0 - tbe / (n_parameters * n_tasks),
        "tbe_vs_full_group_atlas_ratio": group_atlas / tbe,
        "adamw_total_state": adamw_total,
        "task_indexed_tbe_total_state": tbe_total,
        "task_indexed_parameter_atlas_total_state": n_parameters
        + n_parameters * n_tasks,
        "task_indexed_tbe_vs_adamw_ratio": adamw_total / tbe_total,
        "task_indexed_tbe_vs_full_group_atlas_ratio": group_atlas_total / tbe_total,
    }
