"""Reuse canonical profiles for a bounded nonlinear precision search."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .qwen_sensitivity import (
    cold_fold_predictions,
    evaluate_precision_allocations,
    nonlinear_fold_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-blocks", type=int, default=4)
    parser.add_argument("--role", choices=("validation", "test"), default="validation")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1])
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    seed_everything(17_071)

    from transformers import AutoModelForCausalLM

    profile_payload = torch.load(
        args.profiles, map_location="cpu", weights_only=False, mmap=True
    )
    feature_payload = torch.load(
        args.features, map_location="cpu", weights_only=False, mmap=True
    )
    features = feature_payload["features"]
    source = profile_payload["source_profiles"].double()
    target = profile_payload["target_profiles"].double()
    base = cold_fold_predictions(source, features, seed=17_071, folds=4)
    predictions = {
        name: base[name]
        for name in (
            "semantic",
            "affine_semantic",
            "rbf_semantic",
            "scalar_mass",
            "jl",
            "permuted",
            "nearest",
        )
    }
    predictions.update(nonlinear_fold_predictions(source, features, folds=4))
    del base
    if args.methods is not None:
        missing = set(args.methods) - set(predictions)
        if missing:
            raise ValueError(f"unknown requested methods: {sorted(missing)}")
        predictions = {name: predictions[name] for name in args.methods}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda()
    model.config.use_cache = False
    corpus = torch.load(args.data / "corpus.pt", weights_only=False, mmap=True)
    partition = ParameterPartition(model, "swiglu")
    precision = evaluate_precision_allocations(
        model,
        partition,
        corpus[args.role],
        predictions,
        direct_profiles=target,
        source_profiles=source,
        eval_blocks=args.eval_blocks,
        fractions=tuple(args.fractions),
    )
    result = {
        "model": args.model,
        "role": args.role,
        "domains": list(corpus["domains"]),
        "profiles": str(args.profiles),
        "features": {
            "source": str(args.features),
            "model": feature_payload.get("model"),
            "definition": feature_payload.get("pooling", feature_payload.get("definition")),
        },
        "search": {
            "rbf_scales": [0.25, 0.5, 1.0, 2.0, 4.0],
            "cosine_powers": [2, 4, 8],
            "cluster_counts": [4, 6, 8],
            "fractions": args.fractions,
            "methods": list(predictions),
            "selection_role": "development validation only",
        },
        "precision": precision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
