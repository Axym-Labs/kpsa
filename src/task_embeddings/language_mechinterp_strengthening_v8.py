"""Fresh MMLU replication against established activation-attribution controls."""

from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .kernel_sensitivity import fit_prototype_response_map
from .language_mcq_final_confirmations_v8 import predict, profile_functional
from .language_mcq_premise_v8 import (
    answer_token_ids,
    calibrate_feature_means,
    evaluate_competence,
    query_scores,
)
from .language_mcq_space_search_v8 import (
    _cold_summary,
    _kernel,
    _median_squared_distance,
    _paired,
    causal_degradations,
    encode_external_space,
    encode_target_hidden,
    round_robin_indices,
    subject_prediction,
)
from .refined_analysis import paired_t_summary
from .vision_causal_refined import mlp_feature_layout, mlp_feature_scope


def select_correct_rank_block(records, *, start_rank, count):
    """Select a fixed rank block of correct examples within every subject."""
    by_subject = defaultdict(list)
    subject_order = []
    seen_subjects = set()
    for record in records:
        subject = record["subject"]
        if subject not in seen_subjects:
            subject_order.append(subject)
            seen_subjects.add(subject)
        if record["correct"]:
            by_subject[subject].append(int(record["index"]))
    selected = []
    for subject in subject_order:
        block = by_subject[subject][start_rank : start_rank + count]
        if len(block) != count:
            raise RuntimeError(f"{subject} has only {len(by_subject[subject])} correct")
        selected.extend(block)
    return selected


def nearest_prediction(profiles, similarity):
    indices = similarity.argmax(0)
    return profiles[:, indices]


def load_target(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda().eval()
    model.config.use_cache = False
    return model, tokenizer


def summarize_confirmation(
    model,
    prepared,
    candidate_ids,
    methods,
    partition,
    scope,
    sizes,
    layout,
    means,
    fractions,
    ablation,
):
    results = {"budgets": {}}
    for fraction in fractions:
        effects = {
            name: causal_degradations(
                model,
                prepared,
                candidate_ids,
                scores,
                partition,
                scope,
                sizes,
                layout,
                means,
                fraction,
                ablation=ablation,
            )[0]
            for name, scores in methods.items()
        }
        current = {
            "mean_degradation": {
                name: paired_t_summary(values) for name, values in effects.items()
            },
            "reference_comparisons": {
                "activation_taylor_vs_scalar": _paired(
                    effects["activation_taylor"], effects["scalar"]
                ),
                "direct_parameter_vs_scalar": _paired(
                    effects["direct_parameter"], effects["scalar"]
                ),
                "activation_x_gradient_vs_scalar": _paired(
                    effects["activation_x_gradient"], effects["scalar"]
                ),
                "activation_magnitude_vs_scalar": _paired(
                    effects["activation_magnitude"], effects["scalar"]
                ),
            },
            "atlas_comparisons": {},
        }
        families = {
            "target_hidden_exact": "target_hidden_exact_permuted",
            "target_hidden_prototype": "target_hidden_prototype_permuted",
            "qwen_embedding_exact": "qwen_embedding_exact_permuted",
        }
        for family, permuted in families.items():
            nearest = (
                "qwen_embedding_nearest"
                if family == "qwen_embedding_exact"
                else "target_hidden_nearest"
            )
            comparisons = {
                "vs_scalar": _paired(effects[family], effects["scalar"]),
                "vs_shuffled_pairing": _paired(
                    effects[family], effects[permuted]
                ),
                "vs_subject_onehot": _paired(
                    effects[family], effects["subject_onehot"]
                ),
                "vs_nearest": _paired(effects[family], effects[nearest]),
                "activation_taylor_vs_atlas": _paired(
                    effects["activation_taylor"], effects[family]
                ),
            }
            numerator = comparisons["vs_scalar"]["mean"]
            denominator = current["reference_comparisons"][
                "activation_taylor_vs_scalar"
            ]["mean"]
            comparisons["activation_taylor_gap_recovered"] = (
                numerator / denominator if denominator > 0 else None
            )
            parameter_denominator = current["reference_comparisons"][
                "direct_parameter_vs_scalar"
            ]["mean"]
            comparisons["direct_parameter_gap_recovered"] = (
                numerator / parameter_denominator
                if parameter_denominator > 0
                else None
            )
            current["atlas_comparisons"][family] = comparisons
        results["budgets"][f"{fraction:g}"] = current
        print(f"evaluated {fraction * 100:g}% parameter budget", flush=True)
    return results


def run(args):
    from datasets import load_dataset

    dev = load_dataset("cais/mmlu", "all", split="dev")
    validation = load_dataset("cais/mmlu", "all", split="validation")
    test = load_dataset("cais/mmlu", "all", split="test")
    construction_data = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    source_order = round_robin_indices(validation)
    source_examples = [validation[index] for index in source_order]

    # Select a new rank block using competence only, then release the target
    # while the larger independent encoder is resident on the GPU.
    model, tokenizer = load_target(args.model)
    competence_started = time.perf_counter()
    test_competence = evaluate_competence(
        model, tokenizer, test, batch_size=args.competence_batch_size
    )
    competence_seconds = time.perf_counter() - competence_started
    query_indices = select_correct_rank_block(
        test_competence,
        start_rank=args.query_start_rank,
        count=args.queries_per_subject,
    )
    query_examples = [test[index] for index in query_indices]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    external_started = time.perf_counter()
    external = encode_external_space(
        args.encoder,
        source_examples + query_examples,
        answer_conditioned=True,
    )
    external_seconds = time.perf_counter() - external_started
    external_source_all = external[: len(source_examples)]
    external_target = external[len(source_examples) :]
    del external

    model, tokenizer = load_target(args.model)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout).cpu()
    sizes = partition.sizes.detach().double().cpu()
    candidate_ids = answer_token_ids(tokenizer, next(model.parameters()).device)

    by_subject = defaultdict(list)
    for item in dev:
        by_subject[item["subject"]].append(item)
    calibration = [item for values in by_subject.values() for item in values[:2]]
    means = calibrate_feature_means(model, tokenizer, calibration, layout)

    source_profiles, source_margins, source_error, source_seconds = (
        profile_functional(
            model,
            tokenizer,
            source_examples,
            partition,
            scope,
            "margin",
        )
    )
    correct = torch.tensor(
        [index for index, margin in enumerate(source_margins) if margin > 0]
    )
    profiles = source_profiles[:, correct]
    correct_examples = [source_examples[int(index)] for index in correct]
    source_subjects = [example["subject"] for example in correct_examples]
    target_subjects = [example["subject"] for example in query_examples]
    external_source = external_source_all[correct]
    del external_source_all
    print(
        f"profiled {len(source_examples)} sources; {len(correct)} correct",
        flush=True,
    )

    query_details = [
        query_scores(
            model,
            tokenizer,
            example,
            partition,
            layout,
            means,
            candidate_ids,
        )
        for example in query_examples
    ]
    target_profiles = torch.stack(
        [item["parameter_profile"][scope] for item in query_details], dim=1
    ).float()
    prepared = [
        {
            "batch": item["batch"],
            "answer": item["answer"],
            "margin": item["margin"],
            "subject": example["subject"],
        }
        for item, example in zip(query_details, query_examples)
    ]
    taylor_key = "zero_taylor" if args.ablation == "zero" else "taylor"
    activation_taylor = torch.stack(
        [item[taylor_key][scope].clamp_min(0) for item in query_details], dim=1
    ).float()
    activation_x_gradient = torch.stack(
        [item["activation_x_gradient"][scope] for item in query_details], dim=1
    ).float()
    activation_magnitude = torch.stack(
        [item["activation_magnitude"][scope] for item in query_details], dim=1
    ).float()
    target_error = max(item["partition_error"] for item in query_details)
    print(f"profiled {len(query_examples)} fresh query attributions", flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    construction_indices = torch.randperm(
        len(construction_data), generator=generator
    )[: args.construction_examples]
    construction_examples = [
        construction_data[int(index)] for index in construction_indices
    ]
    hidden = encode_target_hidden(
        model,
        tokenizer,
        correct_examples + query_examples + construction_examples,
        answer_conditioned=False,
    )
    source_end = len(correct_examples)
    target_end = source_end + len(query_examples)
    hidden_source = hidden[:source_end]
    hidden_target = hidden[source_end:target_end]
    hidden_construction = hidden[target_end:]
    del hidden

    permutation = torch.randperm(
        len(correct), generator=torch.Generator().manual_seed(args.seed + len(correct))
    )
    hidden_median = _median_squared_distance(hidden_source)
    hidden_similarity = _kernel(
        hidden_source,
        hidden_target,
        hidden_median * args.hidden_scale**2,
    )
    hidden_permuted = _kernel(
        hidden_source[permutation],
        hidden_target,
        hidden_median * args.hidden_scale**2,
    )
    construction_median = _median_squared_distance(hidden_construction)
    prototype_map = fit_prototype_response_map(
        hidden_construction,
        args.prototype_count,
        bandwidth_squared=construction_median * args.prototype_scale**2,
        seed=args.seed + args.prototype_count,
    )
    source_response = prototype_map.transform(hidden_source).float()
    target_response = prototype_map.transform(hidden_target).float()
    prototype_similarity = source_response @ target_response.T
    prototype_permuted = source_response[permutation] @ target_response.T

    external_median = _median_squared_distance(external_source)
    external_similarity = _kernel(
        external_source,
        external_target,
        external_median * args.encoder_scale**2,
    )
    external_permuted = _kernel(
        external_source[permutation],
        external_target,
        external_median * args.encoder_scale**2,
    )
    scalar = profiles.mean(1, keepdim=True).expand(-1, len(query_examples))
    methods = {
        "target_hidden_exact": predict(profiles, hidden_similarity),
        "target_hidden_exact_permuted": predict(profiles, hidden_permuted),
        "target_hidden_prototype": predict(profiles, prototype_similarity),
        "target_hidden_prototype_permuted": predict(
            profiles, prototype_permuted
        ),
        "target_hidden_nearest": nearest_prediction(profiles, hidden_similarity),
        "qwen_embedding_exact": predict(profiles, external_similarity),
        "qwen_embedding_exact_permuted": predict(profiles, external_permuted),
        "qwen_embedding_nearest": nearest_prediction(profiles, external_similarity),
        "scalar": scalar,
        "subject_onehot": subject_prediction(
            profiles, source_subjects, target_subjects
        ),
        "direct_parameter": target_profiles,
        "activation_taylor": activation_taylor,
        "activation_x_gradient": activation_x_gradient,
        "activation_magnitude": activation_magnitude,
    }
    result = {
        "setting": f"mmlu_language_mechinterp_strengthening_{args.ablation}_ablation",
        "model": {
            "name": args.model,
            "parameters": parameters,
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
        },
        "protocol": {
            "source_split": "MMLU validation",
            "source_examples": len(source_examples),
            "correct_source_examples": len(correct),
            "query_split": "MMLU test",
            "query_correct_rank_block": [
                args.query_start_rank,
                args.query_start_rank + args.queries_per_subject,
            ],
            "queries_per_subject": args.queries_per_subject,
            "subjects": len(set(target_subjects)),
            "query_examples": len(query_examples),
            "query_indices": query_indices,
            "encoder": args.encoder,
            "hidden_scale": args.hidden_scale,
            "encoder_scale": args.encoder_scale,
            "prototype_count": args.prototype_count,
            "prototype_scale": args.prototype_scale,
            "fractions": args.fractions,
            "ablation": args.ablation,
            "activation_attribution": (
                "gradient * activation"
                if args.ablation == "zero"
                else "gradient * (activation - calibration_mean)"
            ),
            "selection": "all choices frozen from prior development",
        },
        "competence": {
            "test_accuracy": sum(item["correct"] for item in test_competence)
            / len(test_competence),
            "source_correct": len(correct),
        },
        "fidelity": {
            "source_partition_error": source_error,
            "target_partition_error": target_error,
        },
        "timing": {
            "competence_seconds": competence_seconds,
            "independent_encoder_seconds": external_seconds,
            "source_profile_seconds": source_seconds,
        },
        "cold_query": {
            name: _cold_summary(scores, target_profiles)
            for name, scores in methods.items()
            if not name.endswith("_permuted")
        },
        **summarize_confirmation(
            model,
            prepared,
            candidate_ids,
            methods,
            partition,
            scope,
            sizes,
            layout,
            means,
            args.fractions,
            args.ablation,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--encoder", default="/home/davwis/main/data/models/qwen3-embedding-4b"
    )
    parser.add_argument("--query-start-rank", type=int, default=2)
    parser.add_argument("--queries-per-subject", type=int, default=4)
    parser.add_argument("--competence-batch-size", type=int, default=8)
    parser.add_argument("--construction-examples", type=int, default=800)
    parser.add_argument("--prototype-count", type=int, default=800)
    parser.add_argument("--hidden-scale", type=float, default=0.1)
    parser.add_argument("--prototype-scale", type=float, default=0.05)
    parser.add_argument("--encoder-scale", type=float, default=0.25)
    parser.add_argument("--ablation", choices=("mean", "zero"), default="mean")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0002, 0.001])
    parser.add_argument("--seed", type=int, default=83_021)
    args = parser.parse_args()
    seed_everything(args.seed)
    run(args)


if __name__ == "__main__":
    main()
