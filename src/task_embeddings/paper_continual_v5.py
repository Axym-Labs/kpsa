from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .applications import continual_learning_assay_gate
from .common import save_json, seed_everything
from .continual_v4 import (
    ContinualConfig,
    ModularRegressor,
    make_experiment_data,
    run_cl_method,
)
from .streaming_v5 import tbe_resource_counts


def run_continual_curves(
    config: ContinualConfig,
    *,
    order: list[int],
    strengths: tuple[float, ...],
    control_repeats: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate a predeclared stability--plasticity curve without test tuning."""
    if not strengths or any(strength <= 0 for strength in strengths):
        raise ValueError("strengths must be positive")
    if control_repeats < 1:
        raise ValueError("control_repeats must be positive")
    seed_everything(config.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_experiment_data(config, order=order)
    initial_model = ModularRegressor(config.input_dim, config.n_tasks, config.n_modules)
    initial = copy.deepcopy(initial_model.state_dict())
    started = time.perf_counter()
    none = run_cl_method(
        config,
        "none",
        initial,
        data,
        strength=0.0,
        device=device,
    )
    records = [{"method": "none", "strength": 0.0, **none["summary"]}]

    for estimator in ("raw_opg", "residual_normalized_opg"):
        for representation in ("full_atlas", "tbe", "mean_only"):
            method = f"{estimator}/{representation}"
            for strength in strengths:
                result = run_cl_method(
                    config,
                    method,
                    initial,
                    data,
                    strength=strength,
                    device=device,
                )
                records.append(
                    {"method": method, "strength": strength, **result["summary"]}
                )

    for repeat in range(control_repeats):
        for estimator in ("raw_opg", "residual_normalized_opg"):
            method = f"{estimator}/jl"
            for strength in strengths:
                result = run_cl_method(
                    config,
                    method,
                    initial,
                    data,
                    strength=strength,
                    device=device,
                    jl_seed_offset=repeat,
                )
                records.append(
                    {
                        "method": f"{method}_seed{repeat}",
                        "strength": strength,
                        **result["summary"],
                    }
                )
        for strength in strengths:
            result = run_cl_method(
                config,
                "permuted_basis_linear",
                initial,
                data,
                strength=strength,
                device=device,
                jl_seed_offset=repeat,
            )
            records.append(
                {
                    "method": f"permuted_basis_linear_seed{repeat}",
                    "strength": strength,
                    **result["summary"],
                }
            )

    no_protection = none["summary"]
    premise = continual_learning_assay_gate(
        no_protection_forgetting=no_protection["average_forgetting"],
        no_protection_acquisition_gain=no_protection["average_acquisition_gain"],
        minimum_forgetting=0.02,
        minimum_acquisition_gain=0.02,
    )
    resource_accounting = tbe_resource_counts(
        n_modules=config.n_modules,
        n_tasks=config.n_tasks,
        dimension=6,
    )
    consolidated_diagonal = sum(
        parameter.numel() for parameter in initial_model.module_owned_parameters()
    )
    resource_accounting.update(
        {
            "consolidated_parameter_diagonal_floats": consolidated_diagonal,
            "tbe_vs_consolidated_diagonal_ratio": consolidated_diagonal
            / resource_accounting["tbe_stored_floats"],
            "tbe_vs_consolidated_diagonal_reduction": 1.0
            - resource_accounting["tbe_stored_floats"] / consolidated_diagonal,
        }
    )
    return {
        "setting": "thirty_task_compositional_regression_stream",
        "seed": config.seed,
        "device": str(device),
        "configuration": asdict(config),
        "data_seeds": data["seeds"],
        "order": order,
        "strengths": list(strengths),
        "control_repeats": control_repeats,
        "premise_gate": premise,
        "records": records,
        "resource_accounting": resource_accounting,
        "selection_policy": "none; report the predeclared test-set curve",
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = ContinualConfig(
        seed=args.seed,
        n_tasks=30,
        n_modules=512,
        steps_per_task=80,
        reference_per_task=128,
        test_per_task=512,
    )
    order = list(range(config.n_tasks))
    if args.order == "reverse":
        order.reverse()
    result = run_continual_curves(
        config,
        order=order,
        strengths=(2.0, 8.0, 32.0, 128.0),
        control_repeats=3,
    )
    save_json(args.output, result)
    print(f"PAPER_CONTINUAL_V5_RESULT={args.output}")


if __name__ == "__main__":
    main()
