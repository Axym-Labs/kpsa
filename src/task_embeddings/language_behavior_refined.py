"""Behavior-specific multilingual circuit experiment on a modern decoder."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .qwen_sensitivity import cold_fold_predictions
from .representation_sensitivity import (
    group_relative_shares,
    partition_gradient_energy,
    profile_ranking_metrics,
)
from .vision_causal_refined import (
    mean_ablate_mlp_activations,
    mlp_feature_layout,
    mlp_feature_scope,
    select_parameter_budget,
)


def behavior_margin_from_logits(
    logits: torch.Tensor,
    correct: torch.Tensor,
    alternatives: torch.Tensor,
) -> torch.Tensor:
    """Correct-token logit margin against position-matched other languages."""
    if alternatives.ndim != 2 or logits.ndim != 3:
        raise ValueError("expected [batch, time, vocab] logits and [alternatives, time]")
    valid = correct.ge(0) & alternatives.ge(0).all(0)
    if not bool(valid.any()):
        raise ValueError("no positions have aligned valid target tokens")
    local_logits = logits[0, valid]
    correct_logits = local_logits.gather(1, correct[valid, None]).squeeze(1)
    wrong_logits = torch.stack(
        [
            local_logits.gather(1, candidate[valid, None]).squeeze(1)
            for candidate in alternatives
        ]
    ).mean(0)
    return (correct_logits - wrong_logits).mean()


def behavior_margin(model, block: torch.Tensor, alternatives: torch.Tensor):
    device = next(model.parameters()).device
    inputs = block[:-1].long()
    correct = block[1:].long().to(device)
    alternative_targets = alternatives[:, 1:].long().to(device)
    attention_mask = inputs.ge(0)
    pad = getattr(model.config, "pad_token_id", None)
    if pad is None:
        pad = 0
    inputs = inputs.masked_fill(~attention_mask, int(pad)).to(device)
    output = model(
        input_ids=inputs[None],
        attention_mask=attention_mask[None].to(device),
        use_cache=False,
    )
    return behavior_margin_from_logits(output.logits.float(), correct, alternative_targets)


def alternative_blocks(domains, task: int, index: int) -> torch.Tensor:
    tasks = len(domains)
    offsets = (1, max(2, tasks // 3), max(3, 2 * tasks // 3))
    return torch.stack([domains[(task + offset) % tasks][index] for offset in offsets])


def profile_behavior(model, domains, partition, indices: torch.Tensor):
    profiles = torch.zeros(partition.n_groups, len(domains), dtype=torch.float64)
    max_error = 0.0
    margins = []
    started = time.perf_counter()
    for task, data in enumerate(domains):
        task_margins = []
        for index in indices.tolist():
            model.zero_grad(set_to_none=True)
            value = behavior_margin(
                model, data[index], alternative_blocks(domains, task, index)
            )
            value.backward()
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
            profiles[:, task].add_(
                group_relative_shares(energy.detach().double().cpu()[None])[0]
            )
            task_margins.append(float(value.detach()))
        margins.append(sum(task_margins) / len(task_margins))
    model.zero_grad(set_to_none=True)
    return {
        "profiles": profiles / len(indices),
        "mean_margins": margins,
        "max_partition_relative_error": max_error,
        "wall_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def calibrate_language_means(model, domains, indices: torch.Tensor, layout):
    device = next(model.parameters()).device
    totals = {item["name"]: torch.zeros(item["count"], device=device) for item in layout}
    counts = {item["name"]: 0 for item in layout}
    mask_holder = {}
    handles = []
    for item in layout:
        name = item["name"]

        def capture(_module, inputs, name=name):
            value = inputs[0].detach().float()[0]
            valid = mask_holder["mask"].to(value.device)
            totals[name].add_(value[valid].sum(0))
            counts[name] += int(valid.sum())

        handles.append(item["module"].register_forward_pre_hook(capture))
    try:
        for data in domains:
            for index in indices.tolist():
                block = data[index].long()
                mask = block[:-1].ge(0)
                mask_holder["mask"] = mask
                pad = getattr(model.config, "pad_token_id", None) or 0
                inputs = block[:-1].masked_fill(~mask, int(pad))[None].to(device)
                model(input_ids=inputs, attention_mask=mask[None].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {name: total / counts[name] for name, total in totals.items()}


def activation_behavior_scores(model, block, alternatives, partition, layout):
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
        value = behavior_margin(model, block, alternatives)
        value.backward()
        magnitude = torch.zeros(partition.n_groups, dtype=torch.float64)
        act_grad = torch.zeros_like(magnitude)
        for item in layout:
            activation = captured[item["name"]]
            dims = tuple(range(activation.ndim - 1))
            sl = slice(item["offset"], item["offset"] + item["count"])
            magnitude[sl] = activation.detach().float().square().mean(dim=dims).double().cpu()
            act_grad[sl] = (
                activation.detach().float() * activation.grad.detach().float()
            ).square().mean(dim=dims).double().cpu()
        return magnitude, act_grad
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def margin_value(model, domains, task: int, index: int) -> float:
    return float(
        behavior_margin(
            model,
            domains[task][index],
            alternative_blocks(domains, task, index),
        )
    )


def run_study(
    model,
    corpus,
    representations: dict[str, torch.Tensor],
    *,
    samples: int = 4,
    queries: int = 20,
    fractions: tuple[float, ...] = (0.0002, 0.001),
    folds: int = 4,
):
    domains = corpus["train"]
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    order = torch.randperm(len(domains[0]), generator=torch.Generator().manual_seed(71_071))
    source_indices = order[:samples]
    target_indices = order[samples : 2 * samples]
    calibration_indices = order[2 * samples : 2 * samples + 2]
    source = profile_behavior(model, domains, partition, source_indices)
    target = profile_behavior(model, domains, partition, target_indices)
    predictions = {}
    cold_metrics = {}
    for name, features in representations.items():
        local = cold_fold_predictions(source["profiles"], features, seed=71_071, folds=folds)
        for method, values in local.items():
            key = f"{name}_{method}"
            predictions[key] = values
            cold_metrics[key] = profile_ranking_metrics(values, target["profiles"])
    cold_metrics["source_onehot_reference"] = profile_ranking_metrics(
        source["profiles"], target["profiles"]
    )
    cold_metrics["direct_gradient_oracle"] = profile_ranking_metrics(
        target["profiles"], target["profiles"]
    )
    means = calibrate_language_means(model, domains, calibration_indices, layout)
    weight_magnitude = partition.weight_energy().detach().double().cpu() * sizes
    generator = torch.Generator().manual_seed(71_071)
    records = []
    query_tasks = torch.linspace(0, len(domains) - 1, min(queries, len(domains))).round().long().unique()
    validation_index = 0
    for task in query_tasks.tolist():
        block = domains[task][int(source_indices[0])]
        source_activation, source_act_grad = activation_behavior_scores(
            model,
            block,
            alternative_blocks(domains, task, int(source_indices[0])),
            partition,
            layout,
        )
        score_grid = {
            **{name: values[:, task] for name, values in predictions.items()},
            "source_onehot_reference": source["profiles"][:, task],
            "direct_gradient_oracle": target["profiles"][:, task],
            "source_activation_magnitude": source_activation,
            "source_activation_x_gradient": source_act_grad,
            "weight_magnitude": weight_magnitude,
            "random": torch.rand(partition.n_groups, generator=generator, dtype=torch.float64),
        }
        baseline = margin_value(model, corpus["validation"], task, validation_index)
        off_task = (task + 1) % len(domains)
        off_baseline = margin_value(
            model, corpus["validation"], off_task, validation_index
        )
        for method, scores in score_grid.items():
            for fraction in fractions:
                selected, actual = select_parameter_budget(
                    scores.clamp_min(0), sizes, scope, fraction
                )
                with mean_ablate_mlp_activations(layout, selected, means):
                    intervened = margin_value(
                        model, corpus["validation"], task, validation_index
                    )
                    off_intervened = margin_value(
                        model, corpus["validation"], off_task, validation_index
                    )
                degradation = baseline - intervened
                off_degradation = off_baseline - off_intervened
                records.append(
                    {
                        "task": task,
                        "domain": corpus["domains"][task],
                        "method": method,
                        "requested_parameter_fraction": fraction,
                        "actual_scoped_parameter_fraction": actual,
                        "selected_groups": int(selected.sum()),
                        "baseline_margin": baseline,
                        "intervened_margin": intervened,
                        "degradation": degradation,
                        "off_target_degradation": off_degradation,
                        "selectivity": degradation - off_degradation,
                    }
                )
    return (
        {
            "setting": "qwen3_multilingual_correct_token_margin",
            "model": {
                "source": model.config._name_or_path,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "groups": partition.n_groups,
            },
            "domains": list(corpus["domains"]),
            "protocol": {
                "functional": "mean correct-token logit minus three position-matched wrong-language token logits",
                "samples_per_language_per_profile": samples,
                "source_indices": source_indices.tolist(),
                "target_indices": target_indices.tolist(),
                "calibration_indices": calibration_indices.tolist(),
                "query_role": "validation",
                "query_index": validation_index,
                "intervention": "calibration-mean ablation at coupled MLP down-projection inputs",
                "fractions": list(fractions),
            },
            "competence": {
                "source_mean_margin": float(torch.tensor(source["mean_margins"]).mean()),
                "target_mean_margin": float(torch.tensor(target["mean_margins"]).mean()),
                "positive_source_languages": sum(value > 0 for value in source["mean_margins"]),
                "positive_target_languages": sum(value > 0 for value in target["mean_margins"]),
            },
            "fidelity": {
                "source_mass_error": float((source["profiles"].sum(0) - 1).abs().max()),
                "target_mass_error": float((target["profiles"].sum(0) - 1).abs().max()),
                "max_partition_relative_error": max(
                    source["max_partition_relative_error"],
                    target["max_partition_relative_error"],
                ),
            },
            "timing": {
                "source_profile_seconds": source["wall_seconds"],
                "target_profile_seconds": target["wall_seconds"],
            },
            "cold_query": cold_metrics,
            "records": records,
        },
        {
            "source_profiles": source["profiles"].float(),
            "target_profiles": target["profiles"].float(),
            "representations": {name: value.float() for name, value in representations.items()},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--description-features", type=Path, required=True)
    parser.add_argument("--input-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()
    seed_everything(71_071)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda().eval()
    model.config.use_cache = False
    corpus = torch.load(args.data / "corpus.pt", weights_only=False, mmap=True)
    representations = {
        "description": torch.load(
            args.description_features, weights_only=False, mmap=True
        )["features"],
        "input": torch.load(args.input_features, weights_only=False, mmap=True)["features"],
    }
    result, tensors = run_study(
        model,
        corpus,
        representations,
        samples=args.samples,
        queries=args.queries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.with_suffix(".json"), result)
    torch.save(tensors, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
