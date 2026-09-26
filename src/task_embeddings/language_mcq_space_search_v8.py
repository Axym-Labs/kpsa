"""Modality-specific representation and atlas search for MMLU sensitivity."""

from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .language_mcq_premise_v8 import (
    LETTERS,
    answer_margin,
    answer_token_ids,
    calibrate_feature_means,
    encode_examples,
    evaluate_competence,
    format_mmlu_prompt,
    margin_value,
)
from .refined_analysis import paired_t_summary
from .representation_sensitivity import (
    partition_gradient_energy,
    profile_ranking_metrics,
)
from .task_description_features import last_token_pool
from .vision_causal_refined import (
    mean_ablate_mlp_activations,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
    zero_ablate_mlp_activations,
)


def round_robin_indices(dataset) -> list[int]:
    """Interleave MMLU subjects so every prefix is as balanced as possible."""
    by_subject = defaultdict(list)
    subject_order = []
    for index, subject in enumerate(dataset["subject"]):
        if subject not in by_subject:
            subject_order.append(subject)
        by_subject[subject].append(index)
    ordered = []
    for rank in range(max(map(len, by_subject.values()))):
        for subject in subject_order:
            if rank < len(by_subject[subject]):
                ordered.append(by_subject[subject][rank])
    return ordered


def representation_text(example: dict, *, answer_conditioned: bool) -> str:
    base = format_mmlu_prompt(example)
    if not answer_conditioned:
        return base
    answer = int(example["answer"])
    return (
        base
        + "\nRequested output direction: the correct answer is "
        + f"{LETTERS[answer]} ({example['choices'][answer]})."
    )


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    examples,
    *,
    answer_conditioned: bool,
    pooling: str,
    instruction: bool,
    batch_size: int,
):
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(examples), batch_size):
        texts = [
            representation_text(item, answer_conditioned=answer_conditioned)
            for item in examples[start : start + batch_size]
        ]
        if instruction:
            direction = (
                "question and requested answer direction"
                if answer_conditioned
                else "multiple-choice question"
            )
            texts = [
                "Instruct: Given a "
                + direction
                + ", retrieve examples with similar model parameter sensitivity.\nQuery: "
                + text
                for text in texts
            ]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        output = model(**batch)
        hidden = output.last_hidden_state
        pooled = (
            last_token_pool(hidden, batch["attention_mask"])
            if pooling == "last"
            else _mean_pool(hidden, batch["attention_mask"])
        )
        values.append(F.normalize(pooled.float(), dim=1).cpu())
    return torch.cat(values)


@torch.no_grad()
def encode_target_hidden(model, tokenizer, examples, *, answer_conditioned, batch_size=16):
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(examples), batch_size):
        prompts = []
        for item in examples[start : start + batch_size]:
            content = representation_text(item, answer_conditioned=answer_conditioned)
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        output = model.model(**batch, use_cache=False)
        pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"])
        values.append(F.normalize(pooled.float(), dim=1).cpu())
    return torch.cat(values)


def select_two_correct_per_subject(model, tokenizer, dataset, *, candidates=100):
    by_subject = defaultdict(list)
    for index, example in enumerate(dataset):
        if len(by_subject[example["subject"]]) < candidates:
            by_subject[example["subject"]].append(index)
    candidate_indices = [index for values in by_subject.values() for index in values]
    competence = evaluate_competence(
        model,
        tokenizer,
        [dataset[index] for index in candidate_indices],
        batch_size=32,
    )
    correct = defaultdict(list)
    for local, row in enumerate(competence):
        if row["correct"]:
            correct[row["subject"]].append(candidate_indices[local])
    missing = [subject for subject, values in by_subject.items() if len(correct[subject]) < 2]
    if missing:
        raise RuntimeError(f"fewer than two correct test examples for: {missing}")
    development = [correct[subject][0] for subject in by_subject]
    confirmation = [correct[subject][1] for subject in by_subject]
    return development, confirmation, competence


def profile_scoped_sensitivity(
    model,
    tokenizer,
    examples,
    partition,
    scope,
    *,
    weight_mode="normalized",
):
    device = next(model.parameters()).device
    candidate_ids = answer_token_ids(tokenizer, device)
    profiles = torch.empty(int(scope.sum()), len(examples), dtype=torch.float32)
    margins = []
    max_error = 0.0
    started = time.perf_counter()
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
        if weight_mode == "normalized":
            energy = energy / energy.sum().clamp_min(1e-30)
        elif weight_mode != "raw":
            raise ValueError("weight_mode must be normalized or raw")
        profiles[:, column] = energy.detach()[scope.to(energy.device)].float().cpu()
        margins.append(float(margin.detach()))
    model.zero_grad(set_to_none=True)
    return profiles, margins, max_error, time.perf_counter() - started


def _kernel(left, right, bandwidth_squared):
    left = F.normalize(left.double(), dim=1)
    right = F.normalize(right.double(), dim=1)
    log_weight = -torch.cdist(left, right).square() / (2 * bandwidth_squared)
    # A target-wise constant does not change group ranks or selected groups.
    log_weight -= log_weight.max(0, keepdim=True).values
    return torch.exp(log_weight).float()


def _median_squared_distance(values):
    distances = torch.pdist(F.normalize(values.double(), dim=1)).square()
    return float(distances[distances > 0].median())


def _expand_scores(scoped_scores, scope, groups):
    result = torch.zeros(groups, dtype=torch.float64)
    result[scope] = scoped_scores.double()
    return result


def prepare_queries(model, tokenizer, examples):
    device = next(model.parameters()).device
    candidate_ids = answer_token_ids(tokenizer, device)
    prepared = []
    for example in examples:
        batch = encode_examples(tokenizer, [example], device)
        answer = torch.tensor([int(example["answer"])], device=device)
        prepared.append(
            {
                "batch": {key: value.detach() for key, value in batch.items()},
                "answer": answer,
                "margin": margin_value(model, batch, answer, candidate_ids),
                "subject": example["subject"],
            }
        )
    return prepared, candidate_ids


def causal_degradations(
    model,
    prepared,
    candidate_ids,
    scoped_scores,
    partition,
    scope,
    sizes,
    layout,
    means,
    fraction,
    *,
    with_off_target=False,
    ablation="mean",
):
    degradations, selectivities = [], []
    for query, current in enumerate(prepared):
        scores = _expand_scores(scoped_scores[:, query], scope, partition.n_groups)
        selected, _actual = select_parameter_budget(
            scores.clamp_min(0), sizes, scope, fraction
        )
        off = prepared[(query + len(prepared) // 2) % len(prepared)]
        if ablation == "mean":
            intervention = mean_ablate_mlp_activations(layout, selected, means)
        elif ablation == "zero":
            intervention = zero_ablate_mlp_activations(layout, selected)
        else:
            raise ValueError("ablation must be mean or zero")
        with intervention:
            intervened = margin_value(
                model, current["batch"], current["answer"], candidate_ids
            )
            if with_off_target:
                off_intervened = margin_value(
                    model, off["batch"], off["answer"], candidate_ids
                )
        degradation = current["margin"] - intervened
        degradations.append(degradation)
        if with_off_target:
            off_degradation = off["margin"] - off_intervened
            selectivities.append(degradation - off_degradation)
    return degradations, selectivities


def _paired(left, right):
    return paired_t_summary([a - b for a, b in zip(left, right)])


def _cold_summary(predicted, target):
    result = profile_ranking_metrics(predicted.double(), target.double())
    return {
        key: result[key]
        for key in (
            "groups",
            "queries",
            "top_count",
            "mean_spearman",
            "mean_topk_recall",
            "mean_ndcg",
            "mean_cosine",
        )
    }


def subject_prediction(source_profiles, source_subjects, target_subjects):
    columns = defaultdict(list)
    for index, subject in enumerate(source_subjects):
        columns[subject].append(index)
    fallback = source_profiles.mean(1)
    return torch.stack(
        [
            source_profiles[:, columns[subject]].mean(1)
            if columns[subject]
            else fallback
            for subject in target_subjects
        ],
        dim=1,
    )


def encode_external_space(model_name, examples, *, answer_conditioned):
    from transformers import AutoModel, AutoTokenizer

    is_qwen = "qwen3-embedding" in str(model_name).lower()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        padding_side="left" if is_qwen else "right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"local_files_only": True, "dtype": torch.bfloat16}
    if is_qwen:
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModel.from_pretrained(model_name, **kwargs).cuda().eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    features = encode_texts(
        model,
        tokenizer,
        examples,
        answer_conditioned=answer_conditioned,
        pooling="last" if is_qwen else "mean",
        instruction=is_qwen,
        batch_size=8 if "4b" in str(model_name).lower() else 32,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return features


def choose_candidate(candidates):
    gated = [
        item
        for item in candidates
        if item["causal_vs_scalar"]["lower_95"] > 0
        and item["causal_vs_permutation"]["lower_95"] > 0
    ]
    pool = gated or candidates
    selected = max(
        pool,
        key=lambda item: min(
            item["causal_vs_scalar"]["mean"],
            item["causal_vs_permutation"]["mean"],
        ),
    )
    return selected, bool(gated)


def run(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = load_dataset("cais/mmlu", "all", split="dev")
    validation = load_dataset("cais/mmlu", "all", split="validation")
    test = load_dataset("cais/mmlu", "all", split="test")

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
    cache_path = args.output.with_name(args.output.stem + "_intermediate.pt")
    cached = (
        torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_path.exists()
        else None
    )
    if cached is None:
        development_indices, confirmation_indices, selection_competence = (
            select_two_correct_per_subject(model, tokenizer, test)
        )
        source_order = round_robin_indices(validation)
        selection_count = len(selection_competence)
        selection_correct = sum(item["correct"] for item in selection_competence)
    else:
        development_indices = cached["development_indices"].tolist()
        confirmation_indices = cached["confirmation_indices"].tolist()
        source_order = cached["source_order"].tolist()
        selection_count = int(cached["selection_count"])
        selection_correct = int(cached["selection_correct"])
    source_examples = [validation[index] for index in source_order]
    development_examples = [test[index] for index in development_indices[: args.dev_queries]]
    confirmation_examples = [test[index] for index in confirmation_indices]
    calibration = []
    by_subject = defaultdict(list)
    for item in dev:
        by_subject[item["subject"]].append(item)
    for values in by_subject.values():
        calibration.extend(values[:2])

    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout).cpu()
    sizes = partition.sizes.detach().double().cpu()
    means = calibrate_feature_means(model, tokenizer, calibration, layout)
    if cached is None:
        source_profiles, source_margins, source_error, source_seconds = (
            profile_scoped_sensitivity(
                model,
                tokenizer,
                source_examples,
                partition,
                scope,
                weight_mode=args.weight_mode,
            )
        )
        development_profiles, _, development_error, development_seconds = (
            profile_scoped_sensitivity(
                model,
                tokenizer,
                development_examples,
                partition,
                scope,
                weight_mode=args.weight_mode,
            )
        )
        confirmation_profiles, _, confirmation_error, confirmation_seconds = (
            profile_scoped_sensitivity(
                model,
                tokenizer,
                confirmation_examples,
                partition,
                scope,
                weight_mode=args.weight_mode,
            )
        )
    else:
        source_profiles = cached["source_profiles"]
        source_margins = cached["source_margins"].tolist()
        development_profiles = cached["development_profiles"]
        confirmation_profiles = cached["confirmation_profiles"]
        source_error = float(cached["source_error"])
        development_error = float(cached["development_error"])
        confirmation_error = float(cached["confirmation_error"])
        source_seconds = float(cached["source_seconds"])
        development_seconds = float(cached["development_seconds"])
        confirmation_seconds = float(cached["confirmation_seconds"])
    print(
        f"profiled {len(source_examples)} source and "
        f"{len(development_examples) + len(confirmation_examples)} target examples",
        flush=True,
    )

    all_examples = source_examples + development_examples + confirmation_examples
    feature_spaces = {}
    if cached is None:
        for mode in (False, True):
            suffix = "answer" if mode else "context"
            feature_spaces[f"target_hidden_{suffix}"] = encode_target_hidden(
                model, tokenizer, all_examples, answer_conditioned=mode
            )
        torch.save(
            {
                "development_indices": torch.tensor(development_indices),
                "confirmation_indices": torch.tensor(confirmation_indices),
                "source_order": torch.tensor(source_order),
                "selection_count": selection_count,
                "selection_correct": selection_correct,
                "source_profiles": source_profiles,
                "source_margins": torch.tensor(source_margins),
                "development_profiles": development_profiles,
                "confirmation_profiles": confirmation_profiles,
                "source_error": source_error,
                "development_error": development_error,
                "confirmation_error": confirmation_error,
                "source_seconds": source_seconds,
                "development_seconds": development_seconds,
                "confirmation_seconds": confirmation_seconds,
                "target_hidden_context": feature_spaces["target_hidden_context"],
                "target_hidden_answer": feature_spaces["target_hidden_answer"],
            },
            cache_path,
        )
    else:
        feature_spaces["target_hidden_context"] = cached["target_hidden_context"]
        feature_spaces["target_hidden_answer"] = cached["target_hidden_answer"]
    external = {
        "qwen_embed_0_6b": "Qwen/Qwen3-Embedding-0.6B",
        "qwen_embed_4b": "/home/davwis/main/data/models/qwen3-embedding-4b",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "modernbert": "answerdotai/ModernBERT-base",
    }
    for short_name, model_name in external.items():
        for mode in (False, True):
            suffix = "answer" if mode else "context"
            print(f"encoding {short_name}_{suffix}", flush=True)
            feature_spaces[f"{short_name}_{suffix}"] = encode_external_space(
                model_name, all_examples, answer_conditioned=mode
            )

    development_prepared, candidate_ids = prepare_queries(
        model, tokenizer, development_examples
    )
    confirmation_prepared, _ = prepare_queries(model, tokenizer, confirmation_examples)
    source_subjects = [item["subject"] for item in source_examples]
    development_subjects = [item["subject"] for item in development_examples]
    confirmation_subjects = [item["subject"] for item in confirmation_examples]
    source_configs = {
        f"balanced_{count}": torch.arange(min(count, len(source_examples)))
        for count in args.source_counts
    }
    source_configs["all"] = torch.arange(len(source_examples))
    correct_indices = torch.tensor(
        [index for index, margin in enumerate(source_margins) if margin > 0]
    )
    source_configs["correct_only"] = correct_indices

    source_gpu = source_profiles.cuda()
    scalar_development = source_profiles.mean(1, keepdim=True).expand(
        -1, len(development_examples)
    )
    subject_development = subject_prediction(
        source_profiles, source_subjects, development_subjects
    )
    scalar_degradation, _ = causal_degradations(
        model,
        development_prepared,
        candidate_ids,
        scalar_development,
        partition,
        scope,
        sizes,
        layout,
        means,
        args.search_fraction,
    )
    subject_degradation, _ = causal_degradations(
        model,
        development_prepared,
        candidate_ids,
        subject_development,
        partition,
        scope,
        sizes,
        layout,
        means,
        args.search_fraction,
    )
    direct_degradation, _ = causal_degradations(
        model,
        development_prepared,
        candidate_ids,
        development_profiles,
        partition,
        scope,
        sizes,
        layout,
        means,
        args.search_fraction,
    )

    candidates = []
    n_source = len(source_examples)
    n_development = len(development_examples)
    for space_name, all_features in feature_spaces.items():
        source_features = all_features[:n_source]
        target_features = all_features[n_source : n_source + n_development]
        for config_name, indices in source_configs.items():
            local_features = source_features[indices]
            local_profiles = source_gpu.index_select(1, indices.cuda())
            median = _median_squared_distance(local_features)
            generator = torch.Generator().manual_seed(
                args.seed + len(indices) + sum(map(ord, space_name))
            )
            permutation = torch.randperm(len(indices), generator=generator)
            for scale in args.rbf_scales:
                kernel = _kernel(
                    local_features,
                    target_features,
                    median * scale**2,
                )
                permuted_kernel = _kernel(
                    local_features[permutation],
                    target_features,
                    median * scale**2,
                )
                prediction = (local_profiles @ kernel.cuda() / len(indices)).cpu()
                permuted = (
                    local_profiles @ permuted_kernel.cuda() / len(indices)
                ).cpu()
                degradation, _ = causal_degradations(
                    model,
                    development_prepared,
                    candidate_ids,
                    prediction,
                    partition,
                    scope,
                    sizes,
                    layout,
                    means,
                    args.search_fraction,
                )
                permuted_degradation, _ = causal_degradations(
                    model,
                    development_prepared,
                    candidate_ids,
                    permuted,
                    partition,
                    scope,
                    sizes,
                    layout,
                    means,
                    args.search_fraction,
                )
                candidates.append(
                    {
                        "space": space_name,
                        "source_config": config_name,
                        "source_examples": len(indices),
                        "scale": scale,
                        "median_squared_distance": median,
                        "cold_query": _cold_summary(
                            prediction, development_profiles
                        ),
                        "causal_vs_scalar": _paired(
                            degradation, scalar_degradation
                        ),
                        "causal_vs_permutation": _paired(
                            degradation, permuted_degradation
                        ),
                        "causal_vs_subject_onehot": _paired(
                            degradation, subject_degradation
                        ),
                    }
                )
                del prediction, permuted, kernel, permuted_kernel
            del local_profiles
            gc.collect()
            torch.cuda.empty_cache()
        best_space = max(
            [item for item in candidates if item["space"] == space_name],
            key=lambda item: item["causal_vs_permutation"]["mean"],
        )
        print(
            f"searched {space_name}; best pairing gain "
            f"{best_space['causal_vs_permutation']['mean']:.3f}",
            flush=True,
        )

    selected, any_gate = choose_candidate(candidates)
    print(f"selected {selected}", flush=True)
    selected_features = feature_spaces[selected["space"]]
    source_features = selected_features[:n_source]
    confirmation_features = selected_features[n_source + n_development :]
    indices = source_configs[selected["source_config"]]
    local_features = source_features[indices]
    local_profiles = source_gpu.index_select(1, indices.cuda())
    generator = torch.Generator().manual_seed(
        args.seed + len(indices) + sum(map(ord, selected["space"]))
    )
    permutation = torch.randperm(len(indices), generator=generator)
    bandwidth = selected["median_squared_distance"] * selected["scale"] ** 2
    kernel = _kernel(local_features, confirmation_features, bandwidth)
    permuted_kernel = _kernel(
        local_features[permutation], confirmation_features, bandwidth
    )
    prediction = (local_profiles @ kernel.cuda() / len(indices)).cpu()
    permuted = (local_profiles @ permuted_kernel.cuda() / len(indices)).cpu()
    scalar_confirmation = source_profiles.mean(1, keepdim=True).expand(
        -1, len(confirmation_examples)
    )
    subject_confirmation = subject_prediction(
        source_profiles, source_subjects, confirmation_subjects
    )
    confirmation = {
        "source_examples": len(indices),
        "cold_query": {
            "selected": _cold_summary(prediction, confirmation_profiles),
            "scalar": _cold_summary(scalar_confirmation, confirmation_profiles),
            "subject_onehot": _cold_summary(
                subject_confirmation, confirmation_profiles
            ),
        },
        "budgets": {},
    }
    for fraction in args.confirmation_fractions:
        methods = {}
        for name, scores in (
            ("selected", prediction),
            ("permuted", permuted),
            ("scalar", scalar_confirmation),
            ("subject_onehot", subject_confirmation),
            ("direct_gradient", confirmation_profiles),
        ):
            methods[name] = causal_degradations(
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
                with_off_target=name == "selected",
            )
        confirmation["budgets"][f"{fraction:g}"] = {
            "selected_vs_scalar": _paired(methods["selected"][0], methods["scalar"][0]),
            "selected_vs_permutation": _paired(
                methods["selected"][0], methods["permuted"][0]
            ),
            "selected_vs_subject_onehot": _paired(
                methods["selected"][0], methods["subject_onehot"][0]
            ),
            "direct_gradient_vs_scalar": _paired(
                methods["direct_gradient"][0], methods["scalar"][0]
            ),
            "selected_target_minus_off_target": paired_t_summary(
                methods["selected"][1]
            ),
        }

    result = {
        "setting": "qwen3_1_7b_mmlu_modality_specific_space_search",
        "model": {
            "name": args.model,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
        },
        "protocol": {
            "source_split": "MMLU validation",
            "source_order": "round-robin by subject",
            "development_target": "first correct MMLU-test item per subject",
            "confirmation_target": "second correct MMLU-test item per subject",
            "development_queries": len(development_examples),
            "confirmation_queries": len(confirmation_examples),
            "calibration": "first two MMLU-dev examples per subject",
            "functional": "correct-answer letter logit minus strongest alternative",
            "weight_mode": args.weight_mode,
            "search_fraction": args.search_fraction,
            "confirmation_fractions": args.confirmation_fractions,
            "rbf_scales": args.rbf_scales,
            "source_counts": args.source_counts,
            "spaces": sorted(feature_spaces),
            "selection_rule": (
                "among candidates with positive 95% paired intervals versus scalar "
                "and shuffled feature/profile pairing, maximize the smaller mean gain; "
                "if none pass, use the same ranking over all candidates"
            ),
        },
        "competence": {
            "selection_candidates": selection_count,
            "selection_accuracy": selection_correct / selection_count,
            "source_positive_margins": sum(value > 0 for value in source_margins),
            "source_examples": len(source_margins),
        },
        "fidelity": {
            "source_max_partition_relative_error": source_error,
            "development_max_partition_relative_error": development_error,
            "confirmation_max_partition_relative_error": confirmation_error,
        },
        "timing": {
            "source_profile_seconds": source_seconds,
            "development_profile_seconds": development_seconds,
            "confirmation_profile_seconds": confirmation_seconds,
        },
        "development_controls": {
            "subject_onehot_vs_scalar": _paired(
                subject_degradation, scalar_degradation
            ),
            "direct_gradient_vs_scalar": _paired(
                direct_degradation, scalar_degradation
            ),
        },
        "development_candidates": candidates,
        "selection": {
            **{key: value for key, value in selected.items() if key != "cold_query"},
            "at_least_one_candidate_passed_gate": any_gate,
            "cold_query": selected["cold_query"],
        },
        "confirmation": confirmation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    cache_path.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dev-queries", type=int, default=24)
    parser.add_argument("--source-counts", type=int, nargs="+", default=[171, 456])
    parser.add_argument(
        "--rbf-scales", type=float, nargs="+", default=[0.025, 0.05, 0.1, 0.25]
    )
    parser.add_argument("--search-fraction", type=float, default=0.001)
    parser.add_argument(
        "--confirmation-fractions", type=float, nargs="+", default=[0.0002, 0.001]
    )
    parser.add_argument("--weight-mode", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--seed", type=int, default=83_021)
    args = parser.parse_args()
    seed_everything(args.seed)
    run(args)


if __name__ == "__main__":
    main()
