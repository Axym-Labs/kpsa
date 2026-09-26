"""Resumable development/confirmation jobs with recorded, finite run budgets."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .common import save_json
from .domain_train import DomainConfig, experiment_id, train


def execute(
    configs,
    data,
    output,
    checkpoint=False,
    feature_file="task_features.pt",
    *,
    allow_divergence=False,
):
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous.get("feature_file", "task_features.pt") != feature_file:
            raise ValueError(
                "different task features require a separate campaign directory"
            )
    save_json(
        protocol_path,
        {
            "runs": [asdict(c) for c in configs],
            "checkpoint": checkpoint,
            "feature_file": feature_file,
            "allow_divergence": allow_divergence,
        },
    )
    destinations = []
    for i, c in enumerate(configs):
        destination = (
            output
            / f"{c.method}_{c.partition}_{c.score_link}_{c.seed}_{experiment_id(c)}.json"
        )
        destinations.append(destination)
        if destination.exists():
            continue
        print(f"CAMPAIGN {i + 1}/{len(configs)} {destination.name}", flush=True)
        try:
            train(
                data / "corpus.pt",
                data / feature_file,
                destination,
                c,
                checkpoint=checkpoint,
            )
        except FloatingPointError as error:
            if not allow_divergence:
                raise
            save_json(
                destination,
                {
                    "configuration": asdict(c),
                    "status": "diverged",
                    "failure": str(error),
                    "final": {"macro_nll": None},
                    "claim_ready": False,
                },
            )
            print(f"DIVERGED {destination.name}: {error}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    rows = [json.loads(p.read_text()) for p in destinations]
    save_json(output / "summary.json", {"runs": rows})


def validation_winner(records):
    if not records or any(
        r["configuration"]["role"] != "validation"
        or not math.isfinite(r["final"]["macro_nll"])
        for r in records
    ):
        raise ValueError("nonempty finite validation-only results required")
    return min(records, key=lambda r: r["final"]["macro_nll"])


def optimizer_baseline_confirmation(data, output):
    """Complete the grouped oracle repeats and a tuned quantized-Adam baseline."""
    full = next(c for c in medium_confirmation() if c.method == "full")
    execute(
        [replace(full, seed=s) for s in (22, 23)],
        data,
        output / "optimizer_full_repeats",
        checkpoint=True,
    )
    optimizer_external_confirmation(data, output, "adamw8bit")


def optimizer_external_confirmation(data, output, method):
    """Identical validation-only LR selection and three-seed test for baselines."""
    label = {"adamw8bit": "8bit", "adammini": "adammini"}[method]
    tuning = [
        DomainConfig(method=method, lr=lr, eval_blocks=128, eval_every=2000)
        for lr in (0.0003, 0.001, 0.003)
    ]
    directory = output / f"optimizer_{label}_tuning"
    execute(tuning, data, directory)
    rows = json.loads((directory / "summary.json").read_text())["runs"]
    winner = validation_winner(rows)
    save_json(
        directory / "selection.json",
        {
            "criterion": "minimum development-validation macro NLL",
            "winner": winner["configuration"],
            "final_test_used": False,
        },
    )
    base = next(c for c in medium_confirmation() if c.method == "adamw")
    execute(
        [
            replace(base, method=method, lr=winner["configuration"]["lr"], seed=s)
            for s in (21, 22, 23)
        ],
        data,
        output / f"optimizer_{label}_confirmation",
        checkpoint=True,
    )


def optimizer_revision():
    # Positivity is the single changed mechanism; retain matched linear runs.
    jobs = []
    for kind in ("row", "tensor", "swiglu"):
        for method in ("tbe", "jl"):
            for link in ("linear", "log"):
                jobs.append(
                    DomainConfig(
                        steps=2000,
                        eval_every=2000,
                        eval_blocks=128,
                        method=method,
                        partition=kind,
                        score_link=link,
                        lr=0.001,
                    )
                )
        jobs.append(
            DomainConfig(
                steps=2000,
                eval_every=2000,
                eval_blocks=128,
                method="tbe",
                partition=kind,
                task_mode="single",
                lr=0.001,
            )
        )
    for lr in (0.01, 0.03, 0.1):
        jobs.append(
            DomainConfig(
                steps=2000, eval_every=2000, eval_blocks=128, method="adafactor", lr=lr
            )
        )
    for lr in (0.0003, 0.001, 0.003):
        jobs.append(
            DomainConfig(
                steps=2000, eval_every=2000, eval_blocks=128, method="adamw", lr=lr
            )
        )
    return jobs


def optimizer_adafactor_kernel_screen():
    """Matched momentum-free kernel; row variance is the sole approximation.

    A bounded validation diagnosis, not automatic medium confirmation. Keep the
    original zero-momentum Adam-like campaign intact as a separate intervention.
    """
    return [
        DomainConfig(
            method=method,
            task_mode="single" if method == "mean_adafactor" else "multi",
            partition="row",
            estimator="raw",
            score_link="linear",
            lr=lr,
            eval_blocks=128,
            eval_every=2000,
        )
        for method in (
            "adafactor",
            "mean_adafactor",
            "full_adafactor",
            "tbe_adafactor",
            "jl_adafactor",
        )
        for lr in (0.01, 0.03, 0.1)
    ]


def optimizer_momentum_free_screen():
    """Finite factorial screen; arithmetic second moments, never log moments."""
    return [
        DomainConfig(
            method=method,
            partition=partition,
            estimator=estimator,
            task_mode="single" if method == "mean_nomomentum" else "multi",
            score_link="linear",
            lr=lr,
            eval_blocks=128,
            eval_every=2000,
        )
        for partition in ("row", "tensor", "swiglu")
        for estimator in ("raw", "normalized")
        for method in (
            "mean_nomomentum",
            "full_nomomentum",
            "tbe_nomomentum",
            "jl_nomomentum",
        )
        for lr in (0.0001, 0.0003, 0.001, 0.003, 0.01)
    ]


def optimizer_momentum_free_confirmation(data, output):
    """Same validation search budget per representation, then frozen test seeds."""
    directory = output / "optimizer_momentum_free_tuning"
    execute(
        optimizer_momentum_free_screen(),
        data,
        directory,
        feature_file="token_features.pt",
        allow_divergence=True,
    )
    rows = json.loads((directory / "summary.json").read_text())["runs"]
    finite_rows = [r for r in rows if r.get("status") != "diverged"]
    winners = [
        validation_winner(
            [r for r in finite_rows if r["configuration"]["method"] == method]
        )
        for method in (
            "mean_nomomentum",
            "full_nomomentum",
            "tbe_nomomentum",
            "jl_nomomentum",
        )
        if any(r["configuration"]["method"] == method for r in finite_rows)
    ]
    save_json(
        directory / "selection.json",
        {
            "criterion": "minimum validation macro NLL within each representation",
            "winners": [r["configuration"] for r in winners],
            "final_test_used": False,
            "searches_per_representation": 30,
            "diverged_configurations": [
                r["configuration"] for r in rows if r.get("status") == "diverged"
            ],
            "caveat": "Exploratory partition/estimator/LR selection; Adafactor has only its three LR choices. Report fixed-partition development comparisons too.",
        },
    )
    base = next(c for c in medium_confirmation() if c.method == "adamw")
    jobs = [
        replace(
            base,
            **{
                k: r["configuration"][k]
                for k in (
                    "method",
                    "partition",
                    "estimator",
                    "task_mode",
                    "score_link",
                    "lr",
                )
            },
            seed=seed,
        )
        for r in winners
        for seed in (21, 22, 23)
    ]
    execute(
        jobs,
        data,
        output / "optimizer_momentum_free_confirmation",
        checkpoint=True,
        feature_file="token_features.pt",
        allow_divergence=True,
    )


def application_screen(checkpoint, data, output):
    from .domain_applications import run_applications

    output.mkdir(parents=True, exist_ok=True)
    cache = None
    for features in ("task_features.pt", "token_features.pt"):
        for link in ("linear", "log"):
            destination = output / f"{Path(features).stem}_{link}.json"
            if not destination.exists():
                run_applications(
                    checkpoint,
                    data,
                    destination,
                    samples=256,
                    eval_blocks=128,
                    causal_groups=96,
                    feature_file=features,
                    score_link=link,
                    profile_cache=cache,
                )
            if cache is None:
                cache = destination.with_suffix(".pt")
            gc.collect()
            torch.cuda.empty_cache()


def optimizer_final_tuning():
    jobs = []
    for kind in ("row", "tensor", "swiglu"):
        for method in ("tbe", "jl"):
            for lr in (0.0003, 0.003):
                jobs.append(
                    DomainConfig(
                        steps=2000,
                        eval_every=2000,
                        eval_blocks=128,
                        method=method,
                        partition=kind,
                        score_link="log",
                        lr=lr,
                    )
                )
    for lr in (0.0003, 0.001, 0.003):
        jobs.append(
            DomainConfig(
                steps=2000, eval_every=2000, eval_blocks=128, method="muon", lr=lr
            )
        )
    return jobs


def medium_confirmation():
    """Frozen validation-selected recipes; no test-set hyperparameter search."""
    common = {
        "steps": 12000,
        "batch_size": 32,
        "hidden": 512,
        "layers": 12,
        "intermediate": 1536,
        "heads": 8,
        "kv_heads": 2,
        "eval_every": 12000,
        "eval_blocks": 256,
        "role": "test",
        "partition": "swiglu",
    }
    jobs = [
        DomainConfig(**common, seed=seed, method="adamw", lr=0.001)
        for seed in (21, 22, 23)
    ]
    for method, lr, mode, link in (
        ("adafactor", 0.03, "multi", "linear"),
        ("muon", 0.003, "multi", "linear"),
        ("tbe", 0.001, "multi", "log"),
        ("jl", 0.001, "multi", "log"),
        ("full", 0.001, "multi", "log"),
        ("tbe", 0.001, "single", "log"),
        ("tbe", 0.001, "single", "linear"),
    ):
        jobs.append(
            DomainConfig(
                **common, seed=21, method=method, lr=lr, task_mode=mode, score_link=link
            )
        )
    return jobs


def application_confirmation(checkpoints, data, output):
    from .domain_applications import run_applications

    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(checkpoints.glob("adamw_*.pt"))
    if len(paths) != 3:
        raise ValueError("three frozen independently trained checkpoints required")
    for checkpoint in paths:
        destination = output / f"{checkpoint.stem}_cold_log.json"
        if destination.exists():
            continue
        run_applications(
            checkpoint,
            data,
            destination,
            samples=256,
            eval_blocks=256,
            causal_groups=0,
            role="test",
            score_link="log",
            feature_file="token_features.pt",
            all_tasks=True,
            query_mode="cold",
            methods=("full", "tbe", "jl", "mean", "permuted"),
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=(
            "optimizer",
            "optimizer-final-tuning",
            "medium-confirmation",
            "application-confirmation",
            "applications",
        ),
        default="optimizer",
    )
    p.add_argument("--checkpoint", type=Path)
    a = p.parse_args()
    if a.mode == "optimizer":
        execute(optimizer_revision(), a.data, a.output)
    elif a.mode == "optimizer-final-tuning":
        execute(optimizer_final_tuning(), a.data, a.output)
    elif a.mode == "medium-confirmation":
        execute(medium_confirmation(), a.data, a.output, checkpoint=True)
    elif a.mode == "application-confirmation":
        application_confirmation(a.checkpoint, a.data, a.output)
    else:
        application_screen(a.checkpoint, a.data, a.output)
