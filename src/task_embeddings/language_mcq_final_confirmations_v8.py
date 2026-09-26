"""Independent-encoder and functional confirmations for the MMLU atlas."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .kernel_sensitivity import fit_prototype_response_map
from .language_mcq_premise_v8 import (
    answer_margin_from_logits,
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
    encode_external_space,
    encode_target_hidden,
    prepare_queries,
    round_robin_indices,
    select_two_correct_per_subject,
    subject_prediction,
)
from .refined_analysis import paired_t_summary
from .representation_sensitivity import partition_gradient_energy
from .vision_causal_refined import mlp_feature_layout, mlp_feature_scope


def profile_functional(
    model,
    tokenizer,
    examples,
    partition,
    scope,
    functional,
):
    """Profile normalized scoped energy for one answer behavior."""
    device = next(model.parameters()).device
    candidate_ids = answer_token_ids(tokenizer, device)
    profiles = torch.empty(int(scope.sum()), len(examples), dtype=torch.float32)
    margins = []
    max_error = 0.0
    local_scope = scope.to(device)
    started = time.perf_counter()
    for column, example in enumerate(examples):
        batch = encode_examples(tokenizer, [example], device)
        answer = torch.tensor([int(example["answer"])], device=device)
        model.zero_grad(set_to_none=True)
        logits = model(**batch, use_cache=False).logits
        candidates = logits[:, -1].float()[:, candidate_ids]
        margin = answer_margin_from_logits(logits, answer, candidate_ids)[0]
        if functional == "margin":
            objective = margin
        elif functional == "correct_logit":
            objective = candidates.gather(1, answer[:, None]).squeeze(1)[0]
        elif functional == "answer_loss":
            objective = F.cross_entropy(candidates, answer)
        else:
            raise ValueError(functional)
        objective.backward()
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
        profiles[:, column] = (
            energy.detach()[local_scope].float().cpu()
            / float(energy.sum().clamp_min(1e-30))
        )
        margins.append(float(margin.detach()))
    model.zero_grad(set_to_none=True)
    return profiles, margins, max_error, time.perf_counter() - started


def predict(profiles, similarity):
    return (profiles.cuda() @ similarity.cuda() / profiles.shape[1]).cpu()


def confirmation_summary(
    model,
    prepared,
    candidate_ids,
    methods,
    target_profiles,
    partition,
    scope,
    sizes,
    layout,
    means,
    fractions,
    families,
):
    output = {
        "cold_query": {
            name: _cold_summary(scores, target_profiles)
            for name, scores in methods.items()
            if not name.endswith("_permuted")
        },
        "budgets": {},
    }
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
                with_off_target=name in families,
            )
            for name, scores in methods.items()
        }
        current = {}
        for family in families:
            current[family] = {
                "vs_scalar": _paired(effects[family][0], effects["scalar"][0]),
                "vs_permutation": _paired(
                    effects[family][0], effects[family + "_permuted"][0]
                ),
                "vs_subject_onehot": _paired(
                    effects[family][0], effects["subject_onehot"][0]
                ),
                "target_minus_off_target": paired_t_summary(effects[family][1]),
            }
        current["direct_gradient_vs_scalar"] = _paired(
            effects["direct_gradient"][0], effects["scalar"][0]
        )
        output["budgets"][f"{fraction:g}"] = current
    return output


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
    development_indices, confirmation_indices, competence = (
        select_two_correct_per_subject(model, tokenizer, test)
    )
    del development_indices
    source_order = round_robin_indices(validation)
    source_examples = [validation[index] for index in source_order]
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
    by_subject = defaultdict(list)
    for item in dev:
        by_subject[item["subject"]].append(item)
    calibration = [item for values in by_subject.values() for item in values[:2]]
    means = calibrate_feature_means(model, tokenizer, calibration, layout)
    prepared, candidate_ids = prepare_queries(model, tokenizer, confirmation_examples)

    margin_source, source_margins, margin_source_error, margin_source_seconds = (
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
    correct_examples = [source_examples[int(index)] for index in correct]
    profiles = {"margin": margin_source[:, correct]}
    target_profiles = {}
    fidelity = {"margin_source": margin_source_error}
    timing = {"margin_source_seconds": margin_source_seconds}
    for functional in ("margin", "correct_logit", "answer_loss"):
        if functional != "margin":
            values, _, error, seconds = profile_functional(
                model,
                tokenizer,
                correct_examples,
                partition,
                scope,
                functional,
            )
            profiles[functional] = values
            fidelity[functional + "_source"] = error
            timing[functional + "_source_seconds"] = seconds
        values, _, error, seconds = profile_functional(
            model,
            tokenizer,
            confirmation_examples,
            partition,
            scope,
            functional,
        )
        target_profiles[functional] = values
        fidelity[functional + "_target"] = error
        timing[functional + "_target_seconds"] = seconds
    print("profiled three functionals", flush=True)

    hidden_examples = correct_examples + confirmation_examples + construction_examples
    hidden_features = encode_target_hidden(
        model,
        tokenizer,
        hidden_examples,
        answer_conditioned=False,
    )
    source_end = len(correct_examples)
    target_end = source_end + len(confirmation_examples)
    hidden_source = hidden_features[:source_end]
    hidden_target = hidden_features[source_end:target_end]
    hidden_construction = hidden_features[target_end:]
    hidden_source_median = _median_squared_distance(hidden_source)
    hidden_construction_median = _median_squared_distance(hidden_construction)
    prototype_map = fit_prototype_response_map(
        hidden_construction,
        800,
        bandwidth_squared=hidden_construction_median * 0.05**2,
        seed=args.seed + 800,
    )
    hidden_source_response = prototype_map.transform(hidden_source).float()
    hidden_target_response = prototype_map.transform(hidden_target).float()
    hidden_permutation = torch.randperm(
        len(correct), generator=torch.Generator().manual_seed(args.seed + len(correct))
    )
    exact_similarity = _kernel(
        hidden_source, hidden_target, hidden_source_median * 0.1**2
    )
    exact_permuted_similarity = _kernel(
        hidden_source[hidden_permutation],
        hidden_target,
        hidden_source_median * 0.1**2,
    )
    prototype_similarity = hidden_source_response @ hidden_target_response.T
    prototype_permuted_similarity = (
        hidden_source_response[hidden_permutation] @ hidden_target_response.T
    )

    functional_confirmation = {}
    correct_subjects = [example["subject"] for example in correct_examples]
    confirmation_subjects = [example["subject"] for example in confirmation_examples]
    for functional in ("margin", "correct_logit", "answer_loss"):
        source_profile = profiles[functional]
        scalar = source_profile.mean(1, keepdim=True).expand(
            -1, len(confirmation_examples)
        )
        subject = subject_prediction(
            source_profile, correct_subjects, confirmation_subjects
        )
        methods = {
            "exact_rbf": predict(source_profile, exact_similarity),
            "exact_rbf_permuted": predict(
                source_profile, exact_permuted_similarity
            ),
            "prototype": predict(source_profile, prototype_similarity),
            "prototype_permuted": predict(
                source_profile, prototype_permuted_similarity
            ),
            "scalar": scalar,
            "subject_onehot": subject,
            "direct_gradient": target_profiles[functional],
        }
        functional_confirmation[functional] = confirmation_summary(
            model,
            prepared,
            candidate_ids,
            methods,
            target_profiles[functional],
            partition,
            scope,
            sizes,
            layout,
            means,
            args.fractions,
            ("exact_rbf", "prototype"),
        )
        print(f"confirmed functional {functional}", flush=True)

    independent_specs = {
        "qwen3_embedding_4b_answer": {
            "model": "/home/davwis/main/data/models/qwen3-embedding-4b",
            "answer_conditioned": True,
            "indices": correct,
            "scale": 0.25,
            "development_choice": "correct_only, scale .25",
        },
        "modernbert_answer": {
            "model": "answerdotai/ModernBERT-base",
            "answer_conditioned": True,
            "indices": torch.arange(456),
            "scale": 0.1,
            "development_choice": "balanced 456, scale .1",
        },
    }
    independent_confirmation = {}
    external_examples = source_examples + confirmation_examples
    for name, spec in independent_specs.items():
        features = encode_external_space(
            spec["model"],
            external_examples,
            answer_conditioned=spec["answer_conditioned"],
        )
        source_features = features[: len(source_examples)][spec["indices"]]
        target_features = features[len(source_examples) :]
        source_profile = margin_source[:, spec["indices"]]
        median = _median_squared_distance(source_features)
        # Reproduce the original search's exact space-name seed.
        original_name = (
            "qwen_embed_4b_answer" if name.startswith("qwen3") else "modernbert_answer"
        )
        permutation = torch.randperm(
            len(spec["indices"]),
            generator=torch.Generator().manual_seed(
                args.seed + len(spec["indices"]) + sum(map(ord, original_name))
            ),
        )
        similarity = _kernel(
            source_features,
            target_features,
            median * spec["scale"] ** 2,
        )
        permuted_similarity = _kernel(
            source_features[permutation],
            target_features,
            median * spec["scale"] ** 2,
        )
        scalar = margin_source.mean(1, keepdim=True).expand(
            -1, len(confirmation_examples)
        )
        all_source_subjects = [item["subject"] for item in source_examples]
        subject = subject_prediction(
            margin_source, all_source_subjects, confirmation_subjects
        )
        methods = {
            "encoder_rbf": predict(source_profile, similarity),
            "encoder_rbf_permuted": predict(source_profile, permuted_similarity),
            "scalar": scalar,
            "subject_onehot": subject,
            "direct_gradient": target_profiles["margin"],
        }
        independent_confirmation[name] = {
            "development_choice": spec["development_choice"],
            "source_examples": len(spec["indices"]),
            "scale": spec["scale"],
            "median_squared_distance": median,
            **confirmation_summary(
                model,
                prepared,
                candidate_ids,
                methods,
                target_profiles["margin"],
                partition,
                scope,
                sizes,
                layout,
                means,
                args.fractions,
                ("encoder_rbf",),
            ),
        }
        print(f"confirmed independent space {name}", flush=True)

    result = {
        "setting": "qwen3_1_7b_mmlu_final_language_confirmations",
        "model": {
            "name": args.model,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
        },
        "protocol": {
            "source_split": "MMLU validation",
            "source_examples_all": len(source_examples),
            "source_examples_correct": len(correct),
            "confirmation_split": "MMLU test",
            "confirmation_indices": confirmation_indices,
            "confirmation_examples": len(confirmation_examples),
            "functional_variants": ["margin", "correct_logit", "answer_loss"],
            "functional_geometry": (
                "frozen target-hidden exact scale .1 and 800-prototype scale .05"
            ),
            "independent_encoder_variants": {
                name: spec["development_choice"]
                for name, spec in independent_specs.items()
            },
            "fractions": args.fractions,
        },
        "competence": {
            "test_selection_accuracy": sum(item["correct"] for item in competence)
            / len(competence),
            "source_correct": len(correct),
        },
        "fidelity": fidelity,
        "timing": timing,
        "functional_confirmation": functional_confirmation,
        "independent_encoder_confirmation": independent_confirmation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--construction-examples", type=int, default=800)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0002, 0.001])
    parser.add_argument("--seed", type=int, default=83_021)
    args = parser.parse_args()
    seed_everything(args.seed)
    run(args)


if __name__ == "__main__":
    main()
