"""Development search for a sample-conditioned MMLU sensitivity atlas."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .kernel_sensitivity import (
    fit_empirical_rbf_index,
    median_squared_distance,
)
from .language_mcq_premise_v8 import (
    answer_margin,
    answer_token_ids,
    calibrate_feature_means,
    encode_examples,
    evaluate_competence,
    format_mmlu_prompt,
    margin_value,
    query_scores,
)
from .refined_analysis import paired_t_summary
from .representation_sensitivity import (
    group_relative_shares,
    partition_gradient_energy,
    profile_ranking_metrics,
)
from .task_description_features import last_token_pool
from .vision_causal_refined import (
    mean_ablate_mlp_activations,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)


@torch.no_grad()
def encode_mmlu_examples(model, tokenizer, examples, *, batch_size: int = 16):
    """Encode question and choices with a frozen, independent text encoder."""
    device = next(model.parameters()).device
    encoded = []
    for start in range(0, len(examples), batch_size):
        prompts = [format_mmlu_prompt(item) for item in examples[start : start + batch_size]]
        batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(**batch, use_cache=False)
        pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"])
        encoded.append(F.normalize(pooled.float(), dim=1).cpu())
    return torch.cat(encoded)


def fit_codebook(representations: torch.Tensor, prototypes: int, *, seed: int):
    """Fit shared, sensitivity-independent prototypes in encoder space."""
    from sklearn.cluster import KMeans

    if not 1 < prototypes <= len(representations):
        raise ValueError("prototype count must fit the construction set")
    estimator = KMeans(
        n_clusters=prototypes,
        init="k-means++",
        n_init=1,
        max_iter=100,
        random_state=seed,
    ).fit(F.normalize(representations.float(), dim=1).numpy())
    return F.normalize(
        torch.from_numpy(estimator.cluster_centers_).to(torch.float64), dim=1
    )


def prototype_responses(
    representations: torch.Tensor,
    codebook: torch.Tensor,
    *,
    bandwidth_squared: float,
) -> torch.Tensor:
    """Map encoder vectors to normalized local responses to shared prototypes."""
    if bandwidth_squared <= 0:
        raise ValueError("bandwidth_squared must be positive")
    values = F.normalize(representations.detach().double().cpu(), dim=1)
    centers = F.normalize(codebook.detach().double().cpu(), dim=1)
    log_response = -torch.cdist(values, centers).square() / (2 * bandwidth_squared)
    # Rowwise rescaling prevents numerical underflow and vanishes under L2
    # normalization, so it does not alter the represented direction.
    response = torch.exp(log_response - log_response.max(dim=1, keepdim=True).values)
    return F.normalize(response, dim=1)


def source_examples_from_dev(development):
    by_subject = defaultdict(list)
    for example in development:
        by_subject[example["subject"]].append(example)
    calibration = [item for values in by_subject.values() for item in values[:2]]
    source = [item for values in by_subject.values() for item in values[2:]]
    subjects = [item["subject"] for item in source]
    return calibration, source, subjects


def premise_query_indices(payload: dict, count: int) -> list[int]:
    by_query = {}
    for row in payload["records"]:
        by_query.setdefault(int(row["query"]), int(row["dataset_index"]))
    ordered = [by_query[index] for index in sorted(by_query)]
    if len(ordered) < count:
        raise ValueError("premise artifact contains too few distinct queries")
    return ordered[:count]


def first_correct_test_indices(
    model,
    tokenizer,
    dataset,
    *,
    count: int,
    candidates_per_subject: int = 8,
):
    """Choose the first correct item among fixed early candidates per subject."""
    by_subject = defaultdict(list)
    for index, example in enumerate(dataset):
        if len(by_subject[example["subject"]]) < candidates_per_subject:
            by_subject[example["subject"]].append(index)
    candidate_indices = [index for values in by_subject.values() for index in values]
    candidate_examples = [dataset[index] for index in candidate_indices]
    competence = evaluate_competence(model, tokenizer, candidate_examples)
    selected = []
    seen = set()
    for row in competence:
        subject = row["subject"]
        if row["correct"] and subject not in seen:
            selected.append(candidate_indices[int(row["index"])])
            seen.add(subject)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"only {len(selected)} subjects had a correct early test candidate"
        )
    return selected, competence


def profile_parameter_sensitivity(model, tokenizer, examples, partition):
    device = next(model.parameters()).device
    candidate_ids = answer_token_ids(tokenizer, device)
    profiles = torch.empty(partition.n_groups, len(examples), dtype=torch.float32)
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
        profiles[:, column] = group_relative_shares(
            energy.detach().double().cpu()[None]
        )[0].float()
        margins.append(float(margin.detach()))
    model.zero_grad(set_to_none=True)
    return profiles, margins, max_error, time.perf_counter() - started


def atlas_predictions(
    source_profiles,
    source_features,
    target_features,
    source_subjects,
    target_subjects,
    construction_features,
    *,
    rbf_scales=(0.05, 0.1, 0.25, 0.5),
    prototype_counts=(64, 128, 200),
    prototype_scales=(0.05, 0.1, 0.25, 0.5),
    seed=83_021,
):
    source = source_profiles.detach().double().cpu()
    source_features = F.normalize(source_features.detach().double().cpu(), dim=1)
    target_features = F.normalize(target_features.detach().double().cpu(), dim=1)
    construction = F.normalize(construction_features.detach().double().cpu(), dim=1)
    median = median_squared_distance(construction)
    cosine = source_features @ target_features.T
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(source_features), generator=generator)
    permuted_cosine = source_features[permutation] @ target_features.T
    predictions = {
        "linear_encoder": source @ cosine / len(source_features),
        "nearest_encoder": source[:, cosine.argmax(0)],
        "nearest_encoder_permuted": source[:, permuted_cosine.argmax(0)],
        "scalar_mass": source.mean(1, keepdim=True).expand(-1, len(target_features)),
    }
    for scale in rbf_scales:
        index = fit_empirical_rbf_index(
            source,
            source_features,
            bandwidth_squared=median * scale**2,
        )
        predictions[f"exact_rbf_scale_{scale:g}"] = index.query(
            target_features, mass_weighted=True
        )
        permuted_index = fit_empirical_rbf_index(
            source,
            source_features[permutation],
            bandwidth_squared=median * scale**2,
        )
        predictions[f"exact_rbf_scale_{scale:g}_permuted"] = permuted_index.query(
            target_features, mass_weighted=True
        )

    projection = torch.randn(
        source_features.shape[1],
        max(prototype_counts),
        generator=generator,
        dtype=torch.float64,
    ) / max(prototype_counts) ** 0.5
    source_jl = F.normalize(source_features @ projection, dim=1)
    target_jl = F.normalize(target_features @ projection, dim=1)
    predictions["jl_encoder_200"] = (
        source @ (source_jl @ target_jl.T) / len(source_features)
    )
    predictions["permuted_encoder"] = (
        source
        @ (source_features[permutation] @ target_features.T)
        / len(source_features)
    )

    subject_columns = defaultdict(list)
    for column, subject in enumerate(source_subjects):
        subject_columns[subject].append(column)
    predictions["subject_onehot"] = torch.stack(
        [
            source[:, subject_columns[subject]].mean(1)
            if subject in subject_columns
            else source.mean(1)
            for subject in target_subjects
        ],
        dim=1,
    )

    codebooks = {}
    for count in prototype_counts:
        codebook = fit_codebook(construction, count, seed=seed + count)
        codebooks[count] = codebook
        for scale in prototype_scales:
            bandwidth = median * scale**2
            source_response = prototype_responses(
                source_features, codebook, bandwidth_squared=bandwidth
            )
            target_response = prototype_responses(
                target_features, codebook, bandwidth_squared=bandwidth
            )
            predictions[f"prototype_{count}_scale_{scale:g}"] = (
                source
                @ (source_response @ target_response.T)
                / len(source_features)
            )
            predictions[f"prototype_{count}_scale_{scale:g}_permuted"] = (
                source
                @ (source_response[permutation] @ target_response.T)
                / len(source_features)
            )
    return predictions, codebooks, median


def paired_against(records, method, baseline):
    left = {
        int(row["query"]): float(row["degradation"])
        for row in records
        if row["method"] == method
    }
    right = {
        int(row["query"]): float(row["degradation"])
        for row in records
        if row["method"] == baseline
    }
    return paired_t_summary([left[key] - right[key] for key in left.keys() & right.keys()])


def run_study(
    model,
    tokenizer,
    dev,
    validation,
    premise,
    source_features,
    target_features,
    construction_features,
    *,
    queries=24,
    target_split="validation",
    phase="development",
    fraction=0.001,
    rbf_scales=(0.05, 0.1, 0.25, 0.5),
    prototype_counts=(64, 128, 200),
    prototype_scales=(0.05, 0.1, 0.25, 0.5),
    seed=83_021,
):
    calibration, source_examples, source_subjects = source_examples_from_dev(dev)
    target_indices = premise_query_indices(premise, queries)
    target_examples = [validation[index] for index in target_indices]
    target_subjects = [item["subject"] for item in target_examples]
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    means = calibrate_feature_means(model, tokenizer, calibration, layout)

    source_profiles, source_margins, source_error, source_seconds = (
        profile_parameter_sensitivity(model, tokenizer, source_examples, partition)
    )
    target_profiles, target_margins, target_error, target_seconds = (
        profile_parameter_sensitivity(model, tokenizer, target_examples, partition)
    )
    predictions, _codebooks, median = atlas_predictions(
        source_profiles,
        source_features,
        target_features,
        source_subjects,
        target_subjects,
        construction_features,
        rbf_scales=rbf_scales,
        prototype_counts=prototype_counts,
        prototype_scales=prototype_scales,
        seed=seed,
    )
    scoped_target = target_profiles[scope].double()
    cold_query = {
        method: profile_ranking_metrics(values[scope], scoped_target)
        for method, values in predictions.items()
    }
    exact_methods = [
        name
        for name in predictions
        if name.startswith("exact_rbf_") and not name.endswith("_permuted")
    ]
    prototype_methods = [
        name
        for name in predictions
        if name.startswith("prototype_") and not name.endswith("_permuted")
    ]
    matched_permutation_methods = [
        name
        for name in predictions
        if name.endswith("_permuted")
        and name.startswith(("exact_rbf_", "prototype_"))
    ]
    cold_selected_exact = max(
        exact_methods, key=lambda name: cold_query[name]["mean_spearman"]
    )
    cold_selected_prototype = max(
        prototype_methods, key=lambda name: cold_query[name]["mean_spearman"]
    )

    source_normalized = F.normalize(source_features.double(), dim=1)
    target_normalized = F.normalize(target_features.double(), dim=1)
    distance_squared = torch.cdist(source_normalized, target_normalized).square()
    locality = {}
    for scale in rbf_scales:
        weights = torch.exp(-distance_squared / (2 * median * scale**2))
        normalized = weights / weights.sum(0, keepdim=True).clamp_min(1e-300)
        same_subject = torch.tensor(
            [
                [source == target for target in target_subjects]
                for source in source_subjects
            ],
            dtype=torch.float64,
        )
        locality[f"exact_rbf_scale_{scale:g}"] = {
            "mean_effective_neighbors": float(
                (1 / normalized.square().sum(0).clamp_min(1e-300)).mean()
            ),
            "mean_max_weight": float(normalized.max(0).values.mean()),
            "mean_same_subject_mass": float((normalized * same_subject).sum(0).mean()),
        }

    device = next(model.parameters()).device
    candidate_ids = answer_token_ids(tokenizer, device)
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
        for example in target_examples
    ]
    weight = partition.weight_energy().detach().double().cpu() * sizes
    random_generator = torch.Generator().manual_seed(seed)
    causal_methods = {
        **{name: predictions[name] for name in exact_methods},
        **{name: predictions[name] for name in prototype_methods},
        **{name: predictions[name] for name in matched_permutation_methods},
        "linear_encoder": predictions["linear_encoder"],
        "nearest_encoder": predictions["nearest_encoder"],
        "nearest_encoder_permuted": predictions["nearest_encoder_permuted"],
        "subject_onehot": predictions["subject_onehot"],
        "jl_encoder_200": predictions["jl_encoder_200"],
        "permuted_encoder": predictions["permuted_encoder"],
        "scalar_mass": predictions["scalar_mass"],
        "direct_parameter_gradient": target_profiles.double(),
        "direct_coordinate_taylor": torch.stack(
            [item["taylor"].clamp_min(0) for item in query_details], dim=1
        ),
        "activation_magnitude": torch.stack(
            [item["activation_magnitude"] for item in query_details], dim=1
        ),
        "weight_magnitude": weight[:, None].expand(-1, queries),
        "random": torch.rand(
            partition.n_groups,
            queries,
            generator=random_generator,
            dtype=torch.float64,
        ),
    }
    records = []
    for query, current in enumerate(query_details):
        off_query = (query + len(query_details) // 2) % len(query_details)
        off = query_details[off_query]
        for method, scores in causal_methods.items():
            selected, actual = select_parameter_budget(
                scores[:, query].clamp_min(0), sizes, scope, fraction
            )
            with mean_ablate_mlp_activations(layout, selected, means):
                intervened = margin_value(
                    model, current["batch"], current["answer"], candidate_ids
                )
                off_intervened = margin_value(
                    model, off["batch"], off["answer"], candidate_ids
                )
            degradation = current["margin"] - intervened
            off_degradation = off["margin"] - off_intervened
            records.append(
                {
                    "query": query,
                    "dataset_index": target_indices[query],
                    "subject": target_subjects[query],
                    "method": method,
                    "fraction": fraction,
                    "actual_scoped_parameter_fraction": actual,
                    "selected_groups": int(selected.sum()),
                    "baseline_margin": current["margin"],
                    "degradation": degradation,
                    "off_query": off_query,
                    "off_subject": target_subjects[off_query],
                    "off_degradation": off_degradation,
                    "selectivity": degradation - off_degradation,
                }
            )
    comparisons = {
        method: paired_against(records, method, "scalar_mass")
        for method in causal_methods
        if method != "scalar_mass"
    }
    direct_gap = comparisons["direct_parameter_gradient"]["mean"]
    causal_selected_exact = max(
        exact_methods, key=lambda name: comparisons[name]["mean"]
    )
    causal_selected_prototype = max(
        prototype_methods, key=lambda name: comparisons[name]["mean"]
    )

    def closure(method):
        return (
            comparisons[method]["mean"] / direct_gap
            if direct_gap > 0
            else float("nan")
        )

    best_exact_gate = (
        comparisons[causal_selected_exact]["lower_95"] > 0
        and closure(causal_selected_exact) >= 0.25
        and paired_against(
            records,
            causal_selected_exact,
            causal_selected_exact + "_permuted",
        )["lower_95"]
        > 0
    )
    best_prototype_gate = (
        comparisons[causal_selected_prototype]["lower_95"] > 0
        and closure(causal_selected_prototype) >= 0.25
        and paired_against(
            records,
            causal_selected_prototype,
            causal_selected_prototype + "_permuted",
        )["lower_95"]
        > 0
    )
    control_comparisons = {
        family: {
            control: paired_against(records, family, control)
            for control in (
                "nearest_encoder",
                "subject_onehot",
                "linear_encoder",
                family + "_permuted",
            )
            if control != family
        }
        for family in (causal_selected_exact, causal_selected_prototype)
    }
    return {
        "setting": "qwen3_1_7b_mmlu_sample_conditioned_sensitivity_atlas",
        "model": {
            "name": model.config._name_or_path,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
        },
        "protocol": {
            "calibration": "first two MMLU-dev examples per subject",
            "source": "remaining three MMLU-dev examples per subject",
            "target_split": target_split,
            "phase": phase,
            "target_queries": (
                "premise-screened MMLU-validation examples"
                if phase == "development"
                else "first correct among the first eight MMLU-test examples per subject"
            ),
            "representation_construction": "800 seeded MMLU auxiliary-train examples",
            "source_examples": len(source_examples),
            "queries": len(target_examples),
            "functional": "correct-answer letter logit minus strongest alternative",
            "intervention": "calibration-mean ablation at coupled MLP down-projection inputs",
            "fraction": fraction,
            "rbf_scales": list(rbf_scales),
            "prototype_counts": list(prototype_counts),
            "prototype_scales": list(prototype_scales),
            "selection_rule": (
                "bounded development grid; cold and causal winners are both "
                "recorded, causal winner is frozen for confirmation"
                if phase == "development"
                else "frozen from development: exact/prototype scale 0.05 and 128 prototypes"
            ),
            "cold_selected_exact": cold_selected_exact,
            "cold_selected_prototype": cold_selected_prototype,
            "causal_selected_exact": causal_selected_exact,
            "causal_selected_prototype": causal_selected_prototype,
            "encoder_median_squared_distance": median,
        },
        "competence": {
            "source_positive_margins": sum(value > 0 for value in source_margins),
            "source_examples": len(source_margins),
            "target_positive_margins": sum(value > 0 for value in target_margins),
            "target_examples": len(target_margins),
        },
        "fidelity": {
            "max_partition_relative_error": max(
                source_error,
                target_error,
                max(item["partition_error"] for item in query_details),
            ),
            "source_profile_mass_error": float(
                (source_profiles.double().sum(0) - 1).abs().max()
            ),
            "target_profile_mass_error": float(
                (target_profiles.double().sum(0) - 1).abs().max()
            ),
        },
        "timing": {
            "source_profile_seconds": source_seconds,
            "target_profile_seconds": target_seconds,
        },
        "cold_query_scoped": cold_query,
        "kernel_locality": locality,
        "paired_causal_vs_scalar": comparisons,
        "paired_selected_families_vs_controls": control_comparisons,
        "fraction_of_direct_parameter_gap": {
            causal_selected_exact: closure(causal_selected_exact),
            causal_selected_prototype: closure(causal_selected_prototype),
        },
        "development_gate": {
            "exact_sample_kernel_passed": best_exact_gate,
            "fixed_prototype_representation_passed": best_prototype_gate,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--premise", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--representation-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--construction-examples", type=int, default=800)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    seed_everything(83_021)
    from datasets import load_dataset
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    dev = load_dataset("cais/mmlu", "all", split="dev")
    target_data = load_dataset("cais/mmlu", "all", split=args.split)
    construction_data = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    if args.split == "validation":
        if args.premise is None:
            parser.error("--premise is required for the validation development run")
        premise = json.loads(args.premise.read_text())
        target_indices = premise_query_indices(premise, args.queries)
        selection_competence = None
    else:
        selection_tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, padding_side="left"
        )
        if selection_tokenizer.pad_token_id is None:
            selection_tokenizer.pad_token = selection_tokenizer.eos_token
        selection_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).cuda().eval()
        selection_model.config.use_cache = False
        target_indices, selection_competence = first_correct_test_indices(
            selection_model,
            selection_tokenizer,
            target_data,
            count=args.queries,
        )
        premise = {
            "records": [
                {"query": query, "dataset_index": index}
                for query, index in enumerate(target_indices)
            ]
        }
        del selection_model
        gc.collect()
        torch.cuda.empty_cache()
    calibration, source_examples, _ = source_examples_from_dev(dev)
    del calibration
    generator = torch.Generator().manual_seed(83_021)
    construction_indices = torch.randperm(
        len(construction_data), generator=generator
    )[: args.construction_examples]
    construction_examples = [construction_data[int(index)] for index in construction_indices]
    target_examples = [target_data[index] for index in target_indices]

    representation_tokenizer = AutoTokenizer.from_pretrained(
        args.representation_model, local_files_only=True, padding_side="left"
    )
    if representation_tokenizer.pad_token_id is None:
        representation_tokenizer.pad_token = representation_tokenizer.eos_token
    representation_model = AutoModel.from_pretrained(
        args.representation_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda().eval()
    source_features = encode_mmlu_examples(
        representation_model, representation_tokenizer, source_examples
    )
    target_features = encode_mmlu_examples(
        representation_model, representation_tokenizer, target_examples
    )
    construction_features = encode_mmlu_examples(
        representation_model, representation_tokenizer, construction_examples
    )
    del representation_model
    gc.collect()
    torch.cuda.empty_cache()

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
    result = run_study(
        model,
        tokenizer,
        dev,
        target_data,
        premise,
        source_features,
        target_features,
        construction_features,
        queries=args.queries,
        target_split=args.split,
        phase="development" if args.split == "validation" else "confirmation",
        rbf_scales=(0.05,) if args.split == "test" else (0.05, 0.1, 0.25, 0.5),
        prototype_counts=(128,) if args.split == "test" else (64, 128, 200),
        prototype_scales=(0.05,) if args.split == "test" else (0.05, 0.1, 0.25, 0.5),
    )
    result["representation_model"] = {
        "name": args.representation_model,
        "dimension": source_features.shape[1],
    }
    if selection_competence is not None:
        result["test_query_selection"] = {
            "candidates_evaluated": len(selection_competence),
            "candidate_accuracy": sum(row["correct"] for row in selection_competence)
            / len(selection_competence),
            "selected_indices": target_indices,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
