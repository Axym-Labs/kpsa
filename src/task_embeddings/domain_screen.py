"""Bounded validation screen. Each run is independently resumable and hashed."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .common import save_json
from .domain_train import DomainConfig, experiment_id, train


def run_screen(data: Path, output: Path, steps=600, seed=11):
    output.mkdir(parents=True, exist_ok=True)
    configs = []
    for lr in (3e-4, 1e-3):
        for method in ("adamw", "adafactor"):
            configs.append(
                DomainConfig(
                    seed=seed, steps=steps, lr=lr, method=method, eval_every=steps
                )
            )
        for partition in ("row", "tensor", "swiglu"):
            for method in ("full", "tbe", "jl", "mean"):
                for estimator in ("raw", "normalized"):
                    configs.append(
                        DomainConfig(
                            seed=seed,
                            steps=steps,
                            lr=lr,
                            method=method,
                            partition=partition,
                            estimator=estimator,
                            eval_every=steps,
                        )
                    )
    save_json(
        output / "protocol.json",
        {
            "selection_role": "validation",
            "configs": [asdict(c) for c in configs],
            "budget_runs": len(configs),
            "steps_per_run": steps,
        },
    )
    for index, config in enumerate(configs):
        name = f"{config.method}_{config.partition}_{config.estimator}_{config.lr:g}_{experiment_id(config)}.json"
        target = output / name
        if target.exists():
            prior = json.loads(target.read_text())
            if prior.get("experiment_id") != experiment_id(config):
                raise RuntimeError("configuration mismatch")
            continue
        print(f"SCREEN {index + 1}/{len(configs)} {name}", flush=True)
        train(data / "corpus.pt", data / "task_features.pt", target, config)
        gc.collect()
        torch.cuda.empty_cache()
    records = [
        json.loads(p.read_text())
        for p in output.glob("*.json")
        if p.name != "protocol.json" and p.name != "summary.json"
    ]
    selected = {}
    for r in records:
        c = r["configuration"]
        key = f"{c['method']}/{c['partition']}/{c['estimator']}"
        if (
            key not in selected
            or r["final"]["macro_nll"] < selected[key]["final"]["macro_nll"]
        ):
            selected[key] = r
    save_json(
        output / "summary.json",
        {"selected": selected, "selection_role": "validation", "runs": len(records)},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    a = p.parse_args()
    run_screen(a.data, a.output, a.steps)
