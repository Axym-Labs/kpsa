"""Coordinate-matched causal premise gate for a sample-conditioned MMLU atlas."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .refined_analysis import paired_t_summary
from .representation_sensitivity import group_relative_shares, partition_gradient_energy
from .vision_causal_refined import (
    mean_ablate_mlp_activations,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)

LETTERS = "ABCD"


def format_mmlu_prompt(example: dict) -> str:
    choices = "\n".join(
        f"{LETTERS[index]}. {choice}"
        for index, choice in enumerate(example["choices"])
    )
    return (
        f"Question: {example['question']}\n{choices}\n"
        "Answer with only the letter."
    )


def render_prompt(tokenizer, example: dict) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": format_mmlu_prompt(example)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def answer_token_ids(tokenizer, device: torch.device) -> torch.Tensor:
    encoded = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(tokens) != 1 for tokens in encoded):
        raise ValueError("each answer label must be represented by one token")
    return torch.tensor([tokens[0] for tokens in encoded], device=device)


def encode_examples(tokenizer, examples: list[dict], device: torch.device):
    prompts = [render_prompt(tokenizer, example) for example in examples]
    return tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)


def answer_margin_from_logits(
    logits: torch.Tensor,
    answer: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    candidates = logits[:, -1].float()[:, candidate_ids]
    correct = candidates.gather(1, answer[:, None]).squeeze(1)
    alternatives = candidates.masked_fill(
        torch.nn.functional.one_hot(answer, len(candidate_ids)).bool(), -torch.inf
    ).max(1).values
    return correct - alternatives


def answer_margin(model, batch, answers, candidate_ids):
    output = model(**batch, use_cache=False)
    return answer_margin_from_logits(output.logits, answers, candidate_ids)


@torch.no_grad()
def evaluate_competence(model, tokenizer, dataset, *, batch_size: int = 32):
    device = next(model.parameters()).device
    candidate_ids = answer_token_ids(tokenizer, device)
    records = []
    for start in range(0, len(dataset), batch_size):
        examples = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        batch = encode_examples(tokenizer, examples, device)
        answers = torch.tensor([example["answer"] for example in examples], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(**batch, use_cache=False)
        candidates = output.logits[:, -1].float()[:, candidate_ids]
        margins = answer_margin_from_logits(output.logits, answers, candidate_ids)
        predictions = candidates.argmax(1)
        for offset, (example, margin, prediction) in enumerate(
            zip(examples, margins.cpu().tolist(), predictions.cpu().tolist())
        ):
            records.append(
                {
                    "index": start + offset,
                    "subject": example["subject"],
                    "answer": int(example["answer"]),
                    "prediction": int(prediction),
                    "margin": float(margin),
                    "correct": int(prediction) == int(example["answer"]),
                }
            )
    return records


@torch.no_grad()
def calibrate_feature_means(model, tokenizer, examples, layout, *, batch_size=8):
    device = next(model.parameters()).device
    totals = {item["name"]: torch.zeros(item["count"], device=device) for item in layout}
    counts = {item["name"]: 0 for item in layout}
    mask_holder = {}
    handles = []
    for item in layout:
        name = item["name"]

        def capture(_module, inputs, name=name):
            value = inputs[0].detach().float()
            valid = mask_holder["mask"].to(value.device)
            totals[name].add_(value[valid].sum(0))
            counts[name] += int(valid.sum())

        handles.append(item["module"].register_forward_pre_hook(capture))
    try:
        for start in range(0, len(examples), batch_size):
            batch = encode_examples(tokenizer, examples[start : start + batch_size], device)
            mask_holder["mask"] = batch["attention_mask"].bool()
            model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {name: total / max(1, counts[name]) for name, total in totals.items()}


def query_scores(
    model,
    tokenizer,
    example,
    partition,
    layout,
    means,
    candidate_ids,
):
    device = next(model.parameters()).device
    batch = encode_examples(tokenizer, [example], device)
    answer = torch.tensor([int(example["answer"])], device=device)
    captured = {}
    handles = []
    for item in layout:
        name = item["name"]

        def capture(_module, inputs, name=name):
            captured[name] = inputs[0]
            captured[name].retain_grad()

        handles.append(item["module"].register_forward_pre_hook(capture))
    try:
        model.zero_grad(set_to_none=True)
        margin = answer_margin(model, batch, answer, candidate_ids)[0]
        margin.backward()
        energy = partition_gradient_energy(partition).detach().double().cpu()
        direct = sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        partition_error = float(
            (energy.sum() - direct).abs() / direct.clamp_min(1e-30)
        )
        taylor = torch.zeros(partition.n_groups, dtype=torch.float64)
        zero_taylor = torch.zeros_like(taylor)
        magnitude = torch.zeros_like(taylor)
        activation_gradient = torch.zeros_like(taylor)
        valid = batch["attention_mask"].bool()[0]
        for item in layout:
            value = captured[item["name"]][0, valid].detach().float()
            gradient = captured[item["name"]].grad[0, valid].detach().float()
            center = means[item["name"]].float()
            local = slice(item["offset"], item["offset"] + item["count"])
            taylor[local] = (gradient * (value - center)).sum(0).double().cpu()
            zero_taylor[local] = (gradient * value).sum(0).double().cpu()
            magnitude[local] = value.square().mean(0).double().cpu()
            activation_gradient[local] = (
                value * gradient
            ).square().mean(0).double().cpu()
        return {
            "margin": float(margin.detach()),
            "parameter_profile": group_relative_shares(energy[None])[0],
            "taylor": taylor,
            "zero_taylor": zero_taylor,
            "activation_magnitude": magnitude,
            "activation_x_gradient": activation_gradient,
            "partition_error": partition_error,
            "batch": {key: value.detach() for key, value in batch.items()},
            "answer": answer,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def margin_value(model, batch, answer, candidate_ids) -> float:
    return float(answer_margin(model, batch, answer, candidate_ids)[0])


def _paired(records, method, baseline, fraction):
    left = {
        row["query"]: row["degradation"]
        for row in records
        if row["method"] == method and row["fraction"] == fraction
    }
    right = {
        row["query"]: row["degradation"]
        for row in records
        if row["method"] == baseline and row["fraction"] == fraction
    }
    return paired_t_summary([left[key] - right[key] for key in left.keys() & right.keys()])


def run_premise(
    model,
    tokenizer,
    development,
    calibration,
    *,
    queries: int = 24,
    fractions=(0.0002, 0.001),
    seed: int = 83_021,
):
    device = next(model.parameters()).device
    candidate_ids = answer_token_ids(tokenizer, device)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    calibration_by_subject = defaultdict(list)
    for example in calibration:
        calibration_by_subject[example["subject"]].append(example)
    calibration_examples = [
        values[index]
        for values in calibration_by_subject.values()
        for index in range(min(2, len(values)))
    ]
    means = calibrate_feature_means(model, tokenizer, calibration_examples, layout)

    source_scores = []
    source_errors = []
    for values in calibration_by_subject.values():
        # Keep sensitivity sources disjoint from the examples used to estimate
        # the intervention reference activation.
        if len(values) < 3:
            continue
        current = query_scores(
            model,
            tokenizer,
            values[2],
            partition,
            layout,
            means,
            candidate_ids,
        )
        source_scores.append(current)
        source_errors.append(current["partition_error"])
    pooled_taylor = torch.stack(
        [item["taylor"].clamp_min(0) for item in source_scores]
    ).mean(0)
    scalar_mass = torch.stack(
        [item["parameter_profile"] for item in source_scores]
    ).mean(0)

    competence = evaluate_competence(model, tokenizer, development)
    selected = []
    seen_subjects = set()
    # Correct-query filtering ensures a meaningful behavioral margin. Dataset
    # order avoids selecting unusually high-margin examples.
    for row in competence:
        if row["correct"] and row["subject"] not in seen_subjects:
            selected.append(row)
            seen_subjects.add(row["subject"])
        if len(selected) == queries:
            break
    if len(selected) < max(8, queries // 2):
        raise RuntimeError("too few correct, subject-distinct development queries")

    weight = partition.weight_energy().detach().double().cpu() * sizes
    random_generator = torch.Generator().manual_seed(seed)
    records = []
    max_error = max(source_errors)
    started = time.perf_counter()
    for query, row in enumerate(selected):
        example = development[int(row["index"])]
        current = query_scores(
            model,
            tokenizer,
            example,
            partition,
            layout,
            means,
            candidate_ids,
        )
        max_error = max(max_error, current["partition_error"])
        methods = {
            "direct_coordinate_taylor": current["taylor"].clamp_min(0),
            "pooled_coordinate_taylor": pooled_taylor,
            "direct_parameter_gradient": current["parameter_profile"],
            "scalar_mass": scalar_mass,
            "activation_magnitude": current["activation_magnitude"],
            "activation_x_gradient": current["activation_x_gradient"],
            "weight_magnitude": weight,
            "random": torch.rand(
                partition.n_groups, generator=random_generator, dtype=torch.float64
            ),
        }
        baseline = current["margin"]
        for method, scores in methods.items():
            for fraction in fractions:
                chosen, actual = select_parameter_budget(
                    scores, sizes, scope, fraction
                )
                with mean_ablate_mlp_activations(layout, chosen, means):
                    intervened = margin_value(
                        model,
                        current["batch"],
                        current["answer"],
                        candidate_ids,
                    )
                records.append(
                    {
                        "query": query,
                        "dataset_index": int(row["index"]),
                        "subject": row["subject"],
                        "method": method,
                        "fraction": float(fraction),
                        "actual_scoped_parameter_fraction": actual,
                        "selected_groups": int(chosen.sum()),
                        "baseline_margin": baseline,
                        "intervened_margin": intervened,
                        "degradation": baseline - intervened,
                    }
                )
    paired = defaultdict(dict)
    for method in (
        "direct_coordinate_taylor",
        "pooled_coordinate_taylor",
        "direct_parameter_gradient",
        "activation_magnitude",
        "activation_x_gradient",
        "weight_magnitude",
        "random",
    ):
        for fraction in fractions:
            paired[method][f"{fraction:g}"] = _paired(
                records, method, "scalar_mass", float(fraction)
            )
    direct_vs_pooled = {
        f"{fraction:g}": _paired(
            records,
            "direct_coordinate_taylor",
            "pooled_coordinate_taylor",
            float(fraction),
        )
        for fraction in fractions
    }
    gate_by_fraction = {
        f"{fraction:g}": (
            paired["direct_coordinate_taylor"][f"{fraction:g}"]["lower_95"] > 0
            and direct_vs_pooled[f"{fraction:g}"]["lower_95"] > 0
            and paired["direct_parameter_gradient"][f"{fraction:g}"]["lower_95"]
            > 0
        )
        for fraction in fractions
    }
    by_subject = defaultdict(lambda: [0, 0])
    for row in competence:
        by_subject[row["subject"]][0] += int(row["correct"])
        by_subject[row["subject"]][1] += 1
    return {
        "setting": "qwen3_1_7b_mmlu_parameter_sensitivity_premise",
        "model": {
            "name": model.config._name_or_path,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "groups": partition.n_groups,
        },
        "protocol": {
            "development_split": "MMLU validation",
            "calibration_split": "MMLU dev",
            "functional": "correct-answer letter logit minus strongest alternative",
            "intervention": "calibration-mean ablation at coupled MLP down-projection inputs",
            "queries": len(selected),
            "fractions": list(fractions),
            "query_selection": (
                "first correct query in dataset order, at most one per subject"
            ),
            "source_selection": (
                "third MMLU-dev example per subject, disjoint from the first two "
                "calibration examples"
            ),
        },
        "competence": {
            "accuracy": sum(row["correct"] for row in competence) / len(competence),
            "correct": sum(row["correct"] for row in competence),
            "examples": len(competence),
            "subject_accuracy": {
                subject: correct / count
                for subject, (correct, count) in sorted(by_subject.items())
            },
        },
        "fidelity": {"max_partition_relative_error": max_error},
        "paired_vs_scalar": dict(paired),
        "direct_taylor_vs_pooled_taylor": direct_vs_pooled,
        "premise_gate_by_fraction": gate_by_fraction,
        "premise_gate_passed_all_fractions": all(gate_by_fraction.values()),
        "premise_gate_passed_any_fraction": any(gate_by_fraction.values()),
        "causal_loop_seconds": time.perf_counter() - started,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0002, 0.001])
    args = parser.parse_args()
    seed_everything(83_021)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    development = load_dataset("cais/mmlu", "all", split="validation")
    calibration = load_dataset("cais/mmlu", "all", split="dev")
    result = run_premise(
        model,
        tokenizer,
        development,
        calibration,
        queries=args.queries,
        fractions=tuple(args.fractions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
