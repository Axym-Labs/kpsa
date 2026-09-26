"""Fixed prototype-space and normalization strengthening for the MMLU atlas."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .kernel_sensitivity import PrototypeResponseMap, fit_prototype_response_map
from .language_mcq_premise_v8 import (
    answer_margin,
    answer_token_ids,
    calibrate_feature_means,
    encode_examples,
)
from .language_mcq_space_search_v8 import (
    _cold_summary,
    _kernel,
    _median_squared_distance,
    _paired,
    causal_degradations,
    choose_candidate,
    encode_target_hidden,
    prepare_queries,
    round_robin_indices,
    select_two_correct_per_subject,
    subject_prediction,
)
from .refined_analysis import paired_t_summary
from .representation_sensitivity import partition_gradient_energy
from .vision_causal_refined import mlp_feature_layout, mlp_feature_scope


def profile_dual(model, tokenizer, examples, partition, scope):
    """Compute raw and normalized sensitivity from each backward pass."""
    device = next(model.parameters()).device
    candidate_ids = answer_token_ids(tokenizer, device)
    groups = int(scope.sum())
    normalized = torch.empty(groups, len(examples), dtype=torch.float32)
    raw = torch.empty_like(normalized)
    margins = []
    max_error = 0.0
    started = time.perf_counter()
    local_scope = scope.to(device)
    for column, example in enumerate(examples):
        batch = encode_examples(tokenizer, [example], device)
        answer = torch.tensor([int(example["answer"])], device=device)
        model.zero_grad(set_to_none=True)
        margin = answer_margin(model, batch, answer, candidate_ids)[0]
        margin.backward()
        energy = partition_gradient_energy(partition)
        direct = sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        max_error = max(
            max_error,
            float((energy.sum() - direct).abs() / direct.clamp_min(1e-30)),
        )
        local = energy.detach()[local_scope].float().cpu()
        raw[:, column] = local
        normalized[:, column] = local / float(energy.sum())
        margins.append(float(margin.detach()))
    model.zero_grad(set_to_none=True)
    return {
        "normalized": normalized,
        "raw": raw,
        "margins": margins,
        "max_error": max_error,
        "seconds": time.perf_counter() - started,
    }


def make_prediction(profiles, similarity):
    return (profiles.cuda() @ similarity.cuda() / profiles.shape[1]).cpu()


def evaluate_family(
    model,
    prepared,
    candidate_ids,
    prediction,
    permuted,
    scalar,
    subject,
    partition,
    scope,
    sizes,
    layout,
    means,
    fraction,
):
    degradation, _ = causal_degradations(
        model,
        prepared,
        candidate_ids,
        prediction,
        partition,
        scope,
        sizes,
        layout,
        means,
        fraction,
    )
    permuted_degradation, _ = causal_degradations(
        model,
        prepared,
        candidate_ids,
        permuted,
        partition,
        scope,
        sizes,
        layout,
        means,
        fraction,
    )
    scalar_degradation, _ = causal_degradations(
        model,
        prepared,
        candidate_ids,
        scalar,
        partition,
        scope,
        sizes,
        layout,
        means,
        fraction,
    )
    subject_degradation, _ = causal_degradations(
        model,
        prepared,
        candidate_ids,
        subject,
        partition,
        scope,
        sizes,
        layout,
        means,
        fraction,
    )
    return {
        "vs_scalar": _paired(degradation, scalar_degradation),
        "vs_permutation": _paired(degradation, permuted_degradation),
        "vs_subject_onehot": _paired(degradation, subject_degradation),
    }


def run(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = load_dataset("cais/mmlu", "all", split="dev")
    validation = load_dataset("cais/mmlu", "all", split="validation")
    test = load_dataset("cais/mmlu", "all", split="test")
    construction_data = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda().eval()
    model.config.use_cache = False

    development_indices, confirmation_indices, selection_competence = (
        select_two_correct_per_subject(model, tokenizer, test)
    )
    source_order = round_robin_indices(validation)
    source_examples = [validation[index] for index in source_order]
    development_examples = [test[index] for index in development_indices[: args.dev_queries]]
    confirmation_examples = [test[index] for index in confirmation_indices]
    generator = torch.Generator().manual_seed(args.seed)
    construction_indices = torch.randperm(
        len(construction_data), generator=generator
    )[: args.construction_examples]
    construction_examples = [
        construction_data[int(index)] for index in construction_indices
    ]

    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout).cpu()
    sizes = partition.sizes.detach().double().cpu()
    calibration_by_subject = defaultdict(list)
    for example in dev:
        calibration_by_subject[example["subject"]].append(example)
    calibration = [
        example
        for values in calibration_by_subject.values()
        for example in values[:2]
    ]
    means = calibrate_feature_means(model, tokenizer, calibration, layout)
    source = profile_dual(model, tokenizer, source_examples, partition, scope)
    development = profile_dual(
        model, tokenizer, development_examples, partition, scope
    )
    confirmation = profile_dual(
        model, tokenizer, confirmation_examples, partition, scope
    )
    print("profiled raw and normalized language sensitivities", flush=True)

    all_examples = (
        source_examples
        + development_examples
        + confirmation_examples
        + construction_examples
    )
    features = encode_target_hidden(
        model,
        tokenizer,
        all_examples,
        answer_conditioned=False,
        batch_size=16,
    )
    source_end = len(source_examples)
    development_end = source_end + len(development_examples)
    confirmation_end = development_end + len(confirmation_examples)
    source_features = features[:source_end]
    development_features = features[source_end:development_end]
    confirmation_features = features[development_end:confirmation_end]
    construction_features = features[confirmation_end:]
    correct = torch.tensor(
        [index for index, margin in enumerate(source["margins"]) if margin > 0]
    )
    source_features = source_features[correct]
    source_subjects = [source_examples[int(index)]["subject"] for index in correct]
    source_median = _median_squared_distance(source_features)
    construction_median = _median_squared_distance(construction_features)
    permutation = torch.randperm(
        len(correct), generator=torch.Generator().manual_seed(args.seed + len(correct))
    )

    development_prepared, candidate_ids = prepare_queries(
        model, tokenizer, development_examples
    )
    confirmation_prepared, _ = prepare_queries(
        model, tokenizer, confirmation_examples
    )
    development_subjects = [item["subject"] for item in development_examples]
    confirmation_subjects = [item["subject"] for item in confirmation_examples]

    development_candidates = []
    codebook_maps = {}
    for count in args.prototype_counts:
        codebook_maps[count] = fit_prototype_response_map(
            construction_features,
            count,
            bandwidth_squared=construction_median * args.prototype_scales[0] ** 2,
            seed=args.seed + count,
        )
    for mode in ("normalized", "raw"):
        profiles = source[mode][:, correct]
        scalar = profiles.mean(1, keepdim=True).expand(
            -1, len(development_examples)
        )
        subject = subject_prediction(
            profiles, source_subjects, development_subjects
        )
        for count, base_map in codebook_maps.items():
            for scale in args.prototype_scales:
                feature_map = PrototypeResponseMap(
                    base_map.prototypes,
                    construction_median * scale**2,
                )
                source_response = feature_map.transform(source_features).float()
                target_response = feature_map.transform(development_features).float()
                similarity = source_response @ target_response.T
                permuted_similarity = source_response[permutation] @ target_response.T
                prediction = make_prediction(profiles, similarity)
                permuted = make_prediction(profiles, permuted_similarity)
                causal = evaluate_family(
                    model,
                    development_prepared,
                    candidate_ids,
                    prediction,
                    permuted,
                    scalar,
                    subject,
                    partition,
                    scope,
                    sizes,
                    layout,
                    means,
                    args.search_fraction,
                )
                development_candidates.append(
                    {
                        "space": f"prototype_{count}",
                        "weight_mode": mode,
                        "prototype_count": count,
                        "scale": scale,
                        "source_examples": len(correct),
                        "cold_query": _cold_summary(
                            prediction, development[mode]
                        ),
                        "causal_vs_scalar": causal["vs_scalar"],
                        "causal_vs_permutation": causal["vs_permutation"],
                        "causal_vs_subject_onehot": causal[
                            "vs_subject_onehot"
                        ],
                    }
                )
                del prediction, permuted
        print(f"searched prototype grid for {mode}", flush=True)

    selected_by_mode = {}
    for mode in ("normalized", "raw"):
        selected_by_mode[mode], gated = choose_candidate(
            [item for item in development_candidates if item["weight_mode"] == mode]
        )
        selected_by_mode[mode] = {
            **selected_by_mode[mode],
            "at_least_one_candidate_passed_gate": gated,
        }

    confirmation_results = {}
    for mode in ("normalized", "raw"):
        profiles = source[mode][:, correct]
        scalar = profiles.mean(1, keepdim=True).expand(
            -1, len(confirmation_examples)
        )
        subject = subject_prediction(
            profiles, source_subjects, confirmation_subjects
        )
        selected = selected_by_mode[mode]
        base_map = codebook_maps[selected["prototype_count"]]
        feature_map = PrototypeResponseMap(
            base_map.prototypes,
            construction_median * selected["scale"] ** 2,
        )
        source_response = feature_map.transform(source_features).float()
        target_response = feature_map.transform(confirmation_features).float()
        prototype = make_prediction(
            profiles, source_response @ target_response.T
        )
        prototype_permuted = make_prediction(
            profiles, source_response[permutation] @ target_response.T
        )
        exact_kernel = _kernel(
            source_features,
            confirmation_features,
            source_median * 0.1**2,
        )
        exact_permuted_kernel = _kernel(
            source_features[permutation],
            confirmation_features,
            source_median * 0.1**2,
        )
        exact = make_prediction(profiles, exact_kernel)
        exact_permuted = make_prediction(profiles, exact_permuted_kernel)
        methods = {
            "prototype": prototype,
            "prototype_permuted": prototype_permuted,
            "exact_rbf": exact,
            "exact_rbf_permuted": exact_permuted,
            "scalar": scalar,
            "subject_onehot": subject,
            "direct_gradient": confirmation[mode],
        }
        current = {
            "cold_query": {
                name: _cold_summary(scores, confirmation[mode])
                for name, scores in methods.items()
                if not name.endswith("permuted")
            },
            "budgets": {},
        }
        for fraction in args.confirmation_fractions:
            effects = {
                name: causal_degradations(
                    model,
                    confirmation_prepared,
                    candidate_ids,
                    scores,
                    partition,
                    scope,
                    sizes,
                    layout,
                    means,
                    fraction,
                    with_off_target=name in ("prototype", "exact_rbf"),
                )
                for name, scores in methods.items()
            }
            current["budgets"][f"{fraction:g}"] = {
                family: {
                    "vs_scalar": _paired(effects[family][0], effects["scalar"][0]),
                    "vs_permutation": _paired(
                        effects[family][0], effects[family + "_permuted"][0]
                    ),
                    "vs_subject_onehot": _paired(
                        effects[family][0], effects["subject_onehot"][0]
                    ),
                    "target_minus_off_target": paired_t_summary(
                        effects[family][1]
                    ),
                }
                for family in ("prototype", "exact_rbf")
            }
            current["budgets"][f"{fraction:g}"]["direct_gradient_vs_scalar"] = (
                _paired(effects["direct_gradient"][0], effects["scalar"][0])
            )
        confirmation_results[mode] = current

    result = {
        "setting": "qwen3_1_7b_mmlu_fixed_prototype_strengthening",
        "model": {
            "name": args.model,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
        },
        "protocol": {
            "source_split": "MMLU validation",
            "source_selection": "positive correct-answer margin only",
            "source_examples": len(correct),
            "development_indices": development_indices[: args.dev_queries],
            "confirmation_indices": confirmation_indices,
            "construction_split": "MMLU auxiliary_train",
            "construction_indices": construction_indices.tolist(),
            "construction_examples": len(construction_examples),
            "representation": "Qwen3-1.7B final hidden state for question context",
            "functional": "correct-answer letter logit minus strongest alternative",
            "prototype_counts": args.prototype_counts,
            "prototype_scales": args.prototype_scales,
            "exact_scale": 0.1,
            "search_fraction": args.search_fraction,
            "confirmation_fractions": args.confirmation_fractions,
            "selection_rule": (
                "separately for raw and normalized sensitivity, require positive "
                "95% intervals versus scalar and shuffled pairing, then maximize "
                "the smaller mean gain"
            ),
        },
        "competence": {
            "test_selection_accuracy": sum(
                item["correct"] for item in selection_competence
            )
            / len(selection_competence),
            "source_positive_margins": len(correct),
            "source_examples": len(source_examples),
        },
        "geometry": {
            "source_median_squared_distance": source_median,
            "construction_median_squared_distance": construction_median,
        },
        "fidelity": {
            "source_max_partition_relative_error": source["max_error"],
            "development_max_partition_relative_error": development["max_error"],
            "confirmation_max_partition_relative_error": confirmation["max_error"],
        },
        "timing": {
            "source_profile_seconds": source["seconds"],
            "development_profile_seconds": development["seconds"],
            "confirmation_profile_seconds": confirmation["seconds"],
        },
        "development_candidates": development_candidates,
        "selection": selected_by_mode,
        "confirmation": confirmation_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dev-queries", type=int, default=24)
    parser.add_argument("--construction-examples", type=int, default=800)
    parser.add_argument(
        "--prototype-counts", type=int, nargs="+", default=[128, 256, 512, 800]
    )
    parser.add_argument(
        "--prototype-scales", type=float, nargs="+", default=[0.025, 0.05, 0.1, 0.25]
    )
    parser.add_argument("--search-fraction", type=float, default=0.001)
    parser.add_argument(
        "--confirmation-fractions", type=float, nargs="+", default=[0.0002, 0.001]
    )
    parser.add_argument("--seed", type=int, default=83_021)
    args = parser.parse_args()
    seed_everything(args.seed)
    run(args)


if __name__ == "__main__":
    main()
