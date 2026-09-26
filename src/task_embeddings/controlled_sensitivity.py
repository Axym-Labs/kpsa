"""Controlled empirical checks for representation-conditioned sensitivity."""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .controlled_v4 import (
    HardControlledConfig,
    build_model,
    hard_task_mixtures,
    make_program_dataset,
)
from .domain_optimizer import ParameterPartition
from .representation_sensitivity import (
    SensitivityAtlas,
    SensitivityAtlasAccumulator,
    coarsen_atlas,
    group_relative_shares,
    partition_gradient_energy,
    profile_ranking_metrics,
    query_atlas,
)


@dataclass(frozen=True)
class ParameterHierarchy:
    partition: ParameterPartition
    assignments: dict[str, torch.Tensor]


@dataclass(frozen=True)
class ControlledAtlasResult:
    partition: ParameterPartition
    atlases: dict[str, SensitivityAtlas]
    task_profiles: torch.Tensor
    examples_per_task: torch.Tensor
    wall_seconds: float
    max_partition_relative_error: float


def _assignment(labels: list[str]) -> torch.Tensor:
    indices: dict[str, int] = {}
    values = []
    for label in labels:
        if label not in indices:
            indices[label] = len(indices)
        values.append(indices[label])
    return torch.tensor(values, dtype=torch.long)


def parameter_hierarchy(model, *, bundle_width: int = 16) -> ParameterHierarchy:
    """Construct nested coarse labels over a complete coupled-SwiGLU partition."""
    if bundle_width < 1:
        raise ValueError("bundle width must be positive")
    partition = ParameterPartition(model, "swiglu")
    bundle_labels: list[str | None] = [None] * partition.n_groups
    block_labels: list[str | None] = [None] * partition.n_groups
    for spec in partition.slices:
        mlp_prefix = spec.name.split(".mlp.")[0] if ".mlp." in spec.name else None
        match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", spec.name)
        block = f"block.{match.group(1)}" if match else spec.name.split(".")[0]
        for local in range(spec.count):
            group = spec.offset + local
            bundle = (
                f"{mlp_prefix}.mlp.bundle.{local // bundle_width}"
                if mlp_prefix is not None and spec.axis is not None
                else spec.name
            )
            if bundle_labels[group] not in {None, bundle}:
                raise RuntimeError("shared group received inconsistent bundle labels")
            if block_labels[group] not in {None, block}:
                raise RuntimeError("shared group received inconsistent block labels")
            bundle_labels[group] = bundle
            block_labels[group] = block
    if any(label is None for label in bundle_labels + block_labels):
        raise RuntimeError("parameter hierarchy did not cover every fine group")
    return ParameterHierarchy(
        partition=partition,
        assignments={
            "fine": torch.arange(partition.n_groups),
            "bundle": _assignment([str(label) for label in bundle_labels]),
            "block": _assignment([str(label) for label in block_labels]),
            "global": torch.zeros(partition.n_groups, dtype=torch.long),
        },
    )


def measure_controlled_atlas(
    model,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    representations: dict[str, torch.Tensor],
    *,
    functional: str,
) -> ControlledAtlasResult:
    """Measure canonical per-example shares and representation moments."""
    if functional not in {"output", "loss"}:
        raise ValueError("functional must be 'output' or 'loss'")
    tokens, targets, task_ids = dataset
    if not len(tokens) or len(tokens) != len(targets) or len(tokens) != len(task_ids):
        raise ValueError("controlled dataset tensors must be nonempty and aligned")
    n_tasks = int(task_ids.max()) + 1
    for name, features in representations.items():
        if features.ndim != 2 or len(features) < n_tasks:
            raise ValueError(f"representation {name!r} does not cover all tasks")
    hierarchy = parameter_hierarchy(model)
    partition = hierarchy.partition
    accumulators = {
        name: SensitivityAtlasAccumulator(partition.n_groups, features.shape[1])
        for name, features in representations.items()
    }
    task_sums = torch.zeros(partition.n_groups, n_tasks, dtype=torch.float64)
    task_counts = torch.zeros(n_tasks, dtype=torch.float64)
    device = next(model.parameters()).device
    max_relative_error = 0.0
    started = time.perf_counter()
    model.eval()
    for token, target, task in zip(tokens, targets, task_ids):
        model.zero_grad(set_to_none=True)
        prediction = model(token[None].to(device), task[None].to(device)).float().sum()
        objective = (
            prediction
            if functional == "output"
            else 0.5 * (prediction - target.to(device).float()).square()
        )
        objective.backward()
        energy = partition_gradient_energy(partition)
        direct = sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        relative_error = float((energy.sum() - direct).abs() / direct.clamp_min(1e-30))
        max_relative_error = max(max_relative_error, relative_error)
        local_energy = energy.detach().cpu()[None]
        share = group_relative_shares(local_energy.double())[0]
        task_index = int(task)
        task_sums[:, task_index].add_(share)
        task_counts[task_index] += 1
        for name, accumulator in accumulators.items():
            accumulator.update(
                local_energy,
                representations[name][task_index : task_index + 1],
            )
    model.zero_grad(set_to_none=True)
    if (task_counts == 0).any():
        raise ValueError("controlled profiling requires at least one example per task")
    return ControlledAtlasResult(
        partition=partition,
        atlases={name: value.compute() for name, value in accumulators.items()},
        task_profiles=task_sums / task_counts[None],
        examples_per_task=task_counts,
        wall_seconds=time.perf_counter() - started,
        max_partition_relative_error=max_relative_error,
    )


def _only_primitive_tasks(dataset):
    selected = dataset[2] < 6
    return tuple(value[selected] for value in dataset)


def _aggregate_profiles(values: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    groups = int(assignment.max()) + 1
    result = torch.zeros(groups, values.shape[1], dtype=values.dtype)
    result.index_add_(0, assignment, values)
    return result


def run_controlled_study(
    checkpoint: Path,
    *,
    device: torch.device | None = None,
    reference_sizes: tuple[int, ...] = (2, 4, 8, 12),
    target_per_task: int = 8,
) -> tuple[dict, dict[str, object]]:
    """Run the controlled normalization, stability, resolution, and cold-query study."""
    sizes = tuple(sorted(set(reference_sizes)))
    if not sizes or sizes[0] < 1 or target_per_task < 1:
        raise ValueError("reference sizes and target examples must be positive")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = HardControlledConfig(**payload["configuration"])
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    model.load_state_dict(payload["states"]["trained"])
    mixtures = payload.get("mixtures", hard_task_mixtures()).float()
    semantic_queries = F.normalize(mixtures, dim=1)
    primitive_semantic = torch.eye(6)
    generator = torch.Generator().manual_seed(config.seed + 9100)
    primitive_jl = F.normalize(torch.randn(6, 3, generator=generator), dim=1)
    permutation = torch.randperm(6, generator=generator)
    primitive_representations = {
        "semantic6": primitive_semantic,
        "jl3": primitive_jl,
        "permuted_semantic6": primitive_semantic[permutation],
    }

    stability = {}
    largest_source = None
    largest_dataset = None
    fidelity_errors = []
    for size in sizes:
        dataset_a = _only_primitive_tasks(
            make_program_dataset(
                config, mixtures, size, config.seed + 10_000 + size, noise=0
            )
        )
        dataset_b = _only_primitive_tasks(
            make_program_dataset(
                config, mixtures, size, config.seed + 20_000 + size, noise=0
            )
        )
        source = measure_controlled_atlas(
            model,
            dataset_a,
            primitive_representations,
            functional="output",
        )
        replicate = measure_controlled_atlas(
            model,
            dataset_b,
            {"semantic6": primitive_semantic},
            functional="output",
        )
        fidelity_errors.extend(
            [source.max_partition_relative_error, replicate.max_partition_relative_error]
        )
        stability[str(size)] = profile_ranking_metrics(
            query_atlas(source.atlases["semantic6"].joint_moment, primitive_semantic),
            replicate.task_profiles,
            top_fraction=0.05,
        )
        if size == sizes[-1]:
            largest_source, largest_dataset = source, dataset_a

    target_dataset = make_program_dataset(
        config,
        mixtures,
        target_per_task,
        config.seed + 30_000,
        noise=0,
    )
    target = measure_controlled_atlas(
        model,
        target_dataset,
        {"semantic6": semantic_queries},
        functional="output",
    )
    loss_source = measure_controlled_atlas(
        model,
        largest_dataset,
        {"semantic6": primitive_semantic},
        functional="loss",
    )
    fidelity_errors.extend(
        [target.max_partition_relative_error, loss_source.max_partition_relative_error]
    )
    functional_difference = float(
        (largest_source.task_profiles - loss_source.task_profiles).abs().max()
    )

    heldout = slice(6, len(mixtures))
    target_profiles = target.task_profiles[:, heldout]
    prediction = {
        "semantic6": query_atlas(
            largest_source.atlases["semantic6"].joint_moment,
            semantic_queries[heldout],
        ),
        "jl3": query_atlas(
            largest_source.atlases["jl3"].joint_moment,
            F.normalize(mixtures @ primitive_jl, dim=1)[heldout],
        ),
        "permuted_semantic6": query_atlas(
            largest_source.atlases["permuted_semantic6"].joint_moment,
            semantic_queries[heldout],
        ),
        "scalar_mass": largest_source.atlases["semantic6"].mass[:, None].expand(
            -1, target_profiles.shape[1]
        ),
        "random": torch.rand(
            target_profiles.shape,
            generator=torch.Generator().manual_seed(config.seed + 9200),
            dtype=target_profiles.dtype,
        ),
        "direct_gradient_oracle": target_profiles,
    }
    hierarchy = parameter_hierarchy(model)
    cold_query = {}
    resolution = {}
    for name, assignment in hierarchy.assignments.items():
        measured = _aggregate_profiles(target_profiles, assignment)
        cold_query[name] = (
            {
                method: profile_ranking_metrics(
                    _aggregate_profiles(scores, assignment),
                    measured,
                    top_fraction=0.05,
                )
                for method, scores in prediction.items()
            }
            if len(measured) >= 2
            else {"status": "unavailable: ranking requires at least two groups"}
        )
        coarse = coarsen_atlas(
            largest_source.atlases["semantic6"].mass,
            largest_source.atlases["semantic6"].joint_moment,
            assignment,
        )
        resolution[name] = {
            "groups": len(coarse.mass),
            "mass_sum": float(coarse.mass.sum()),
            "storage_bytes_float32": int(
                (coarse.mass.numel() + coarse.joint_moment.numel()) * 4
            ),
        }

    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "setting": "controlled_compositional_transformer",
        "checkpoint": str(checkpoint.resolve()),
        "configuration": asdict(config),
        "protocol": {
            "reference_tasks": list(range(6)),
            "heldout_query_tasks": list(range(6, len(mixtures))),
            "reference_sizes_per_task": list(sizes),
            "target_examples_per_task": target_per_task,
            "functional": "scalar model output; squared-loss equivalence checked",
            "partition": "complete coupled-SwiGLU feature/row partition",
        },
        "model": {
            "parameters": model_parameters,
            "fine_groups": largest_source.partition.n_groups,
        },
        "fidelity": {
            "max_partition_relative_error": max(fidelity_errors),
            "fine_mass_sum": float(largest_source.atlases["semantic6"].mass.sum()),
            "target_profile_mass_sum_min": float(target.task_profiles.sum(0).min()),
            "target_profile_mass_sum_max": float(target.task_profiles.sum(0).max()),
            "loss_vs_output_max_share_difference": functional_difference,
        },
        "reference_stability": stability,
        "cold_query": cold_query,
        "resolution": resolution,
        "timing": {
            "largest_source_seconds": largest_source.wall_seconds,
            "target_seconds": target.wall_seconds,
        },
    }
    tensors = {
        "mixtures": mixtures,
        "target_profiles": target_profiles,
        "prediction": prediction,
        "mass": largest_source.atlases["semantic6"].mass,
        "joint_moment": {
            name: atlas.joint_moment for name, atlas in largest_source.atlases.items()
        },
        "hierarchy": hierarchy.assignments,
        "primitive_jl": primitive_jl,
        "primitive_permutation": permutation,
    }
    return result, tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-sizes", type=int, nargs="+", default=(2, 4, 8, 12))
    parser.add_argument("--target-per-task", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    result, tensors = run_controlled_study(
        args.checkpoint,
        device=torch.device("cpu") if args.cpu else None,
        reference_sizes=tuple(args.reference_sizes),
        target_per_task=args.target_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.with_suffix(".json"), result)
    torch.save(tensors, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
