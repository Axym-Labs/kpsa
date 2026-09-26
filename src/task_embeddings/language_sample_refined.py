"""Sample-level multilingual behavior atlas for a modern decoder model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .language_behavior_refined import (
    activation_behavior_scores,
    alternative_blocks,
    behavior_margin,
    calibrate_language_means,
    margin_value,
)
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
from .vision_sample_refined import sample_predictions


def profile_samples(model, domains, pairs, partition):
    profiles = torch.empty(partition.n_groups, len(pairs), dtype=torch.float32)
    margins = []
    max_error = 0.0
    started = time.perf_counter()
    for column, (task, index) in enumerate(pairs):
        model.zero_grad(set_to_none=True)
        value = behavior_margin(
            model, domains[task][index], alternative_blocks(domains, task, index)
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
        profiles[:, column] = group_relative_shares(
            energy.detach().double().cpu()[None]
        )[0].float()
        margins.append(float(value.detach()))
    model.zero_grad(set_to_none=True)
    return profiles, margins, max_error, time.perf_counter() - started


@torch.no_grad()
def encode_samples(model, domains, pairs, *, pad_token_id, batch_size=16):
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(pairs), batch_size):
        blocks = torch.stack(
            [domains[task][index].long() for task, index in pairs[start : start + batch_size]]
        )
        mask = blocks.ge(0)
        input_ids = blocks.masked_fill(~mask, pad_token_id).to(device)
        mask = mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        pooled = last_token_pool(output.last_hidden_state, mask)
        values.append(F.normalize(pooled.float(), dim=1).cpu())
    return torch.cat(values)


def run_study(
    model,
    representation_model,
    corpus,
    *,
    split="validation",
    source_samples=8,
    target_samples=4,
    causal_samples_per_task=2,
    fractions=(0.0002, 0.001),
    rbf_scales=(0.025, 0.05, 0.1, 0.25, 0.5),
    seed=71_071,
):
    source_domains = corpus["train"]
    target_domains = corpus[split]
    tasks = len(source_domains)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    order = torch.randperm(
        len(source_domains[0]), generator=torch.Generator().manual_seed(seed)
    )
    source_indices = order[:source_samples].tolist()
    calibration_indices = order[source_samples : source_samples + 2]
    source_pairs = [
        (task, index) for task in range(tasks) for index in source_indices
    ]
    target_pairs = [
        (task, index)
        for task in range(tasks)
        for index in range(target_samples)
    ]
    source_profiles, source_margins, source_error, source_seconds = profile_samples(
        model, source_domains, source_pairs, partition
    )
    target_profiles, target_margins, target_error, target_seconds = profile_samples(
        model, target_domains, target_pairs, partition
    )
    pad = getattr(representation_model.config, "pad_token_id", None) or 0
    source_features = encode_samples(
        representation_model,
        source_domains,
        source_pairs,
        pad_token_id=pad,
    )
    target_features = encode_samples(
        representation_model,
        target_domains,
        target_pairs,
        pad_token_id=pad,
    )
    predictions = sample_predictions(
        source_profiles,
        source_features,
        target_features,
        rbf_scales=rbf_scales,
        seed=seed,
    )
    task_reference = source_profiles.reshape(
        partition.n_groups, tasks, source_samples
    ).mean(2)
    predictions["task_source_onehot_reference"] = task_reference.repeat_interleave(
        target_samples, dim=1
    )
    predictions["direct_gradient_oracle"] = target_profiles.double()
    cold_query = {
        method: profile_ranking_metrics(values, target_profiles.double())
        for method, values in predictions.items()
    }

    means = calibrate_language_means(
        model, source_domains, calibration_indices, layout
    )
    weight = partition.weight_energy().detach().double().cpu() * sizes
    predictions["weight_magnitude"] = weight[:, None].expand(
        -1, len(target_pairs)
    )
    predictions["random"] = torch.rand(
        target_profiles.shape,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    query_columns = [
        task * target_samples + local
        for task in range(tasks)
        for local in range(min(causal_samples_per_task, target_samples))
    ]
    records = []
    for column in query_columns:
        task, target_index = target_pairs[column]
        source_index = source_indices[target_index % source_samples]
        source_activation, source_act_grad = activation_behavior_scores(
            model,
            source_domains[task][source_index],
            alternative_blocks(source_domains, task, source_index),
            partition,
            layout,
        )
        target_activation, target_act_grad = activation_behavior_scores(
            model,
            target_domains[task][target_index],
            alternative_blocks(target_domains, task, target_index),
            partition,
            layout,
        )
        score_grid = {
            **{method: values[:, column] for method, values in predictions.items()},
            "source_activation_magnitude": source_activation,
            "source_activation_x_gradient": source_act_grad,
            "target_activation_magnitude": target_activation,
            "target_activation_x_gradient": target_act_grad,
        }
        baseline = margin_value(model, target_domains, task, target_index)
        off_task = (task + tasks // 2) % tasks
        off_baseline = margin_value(model, target_domains, off_task, target_index)
        for method, scores in score_grid.items():
            for fraction in fractions:
                selected, actual = select_parameter_budget(
                    scores.clamp_min(0), sizes, scope, fraction
                )
                with mean_ablate_mlp_activations(layout, selected, means):
                    intervened = margin_value(
                        model, target_domains, task, target_index
                    )
                    off_intervened = margin_value(
                        model, target_domains, off_task, target_index
                    )
                degradation = baseline - intervened
                off_degradation = off_baseline - off_intervened
                records.append(
                    {
                        "query_column": column,
                        "task": task,
                        "domain": corpus["domains"][task],
                        "sample_index": target_index,
                        "method": method,
                        "requested_parameter_fraction": fraction,
                        "actual_scoped_parameter_fraction": actual,
                        "selected_groups": int(selected.sum()),
                        "baseline_margin": baseline,
                        "intervened_margin": intervened,
                        "degradation": degradation,
                        "off_target_task": off_task,
                        "off_target_degradation": off_degradation,
                        "selectivity": degradation - off_degradation,
                    }
                )
    return (
        {
            "setting": "qwen3_multilingual_sample_level_margin",
            "model": {
                "name": model.config._name_or_path,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "groups": partition.n_groups,
            },
            "domains": list(corpus["domains"]),
            "protocol": {
                "source_split": "train",
                "target_split": split,
                "source_samples_per_language": source_samples,
                "target_samples_per_language": target_samples,
                "causal_samples_per_language": causal_samples_per_task,
                "source_indices": source_indices,
                "calibration_indices": calibration_indices.tolist(),
                "representation": "full normalized frozen Qwen3-Embedding-0.6B last-token state per sample",
                "functional": "correct-token margin against three position-matched wrong-language alternatives",
                "intervention": "calibration-mean ablation at coupled MLP down-projection inputs",
                "fractions": list(fractions),
                "rbf_scales": list(rbf_scales),
            },
            "competence": {
                "source_mean_margin": sum(source_margins) / len(source_margins),
                "target_mean_margin": sum(target_margins) / len(target_margins),
                "positive_target_samples": sum(value > 0 for value in target_margins),
                "target_samples": len(target_margins),
            },
            "fidelity": {
                "source_mass_error": float(
                    (source_profiles.sum(0).double() - 1).abs().max()
                ),
                "target_mass_error": float(
                    (target_profiles.sum(0).double() - 1).abs().max()
                ),
                "max_partition_relative_error": max(source_error, target_error),
            },
            "timing": {
                "source_profile_seconds": source_seconds,
                "target_profile_seconds": target_seconds,
            },
            "cold_query": cold_query,
            "records": records,
        },
        {
            "source_profiles": source_profiles,
            "target_profiles": target_profiles,
            "source_features": source_features,
            "target_features": target_features,
            "source_pairs": torch.tensor(source_pairs),
            "target_pairs": torch.tensor(target_pairs),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--representation-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--source-samples", type=int, default=8)
    parser.add_argument("--target-samples", type=int, default=4)
    parser.add_argument("--causal-samples-per-task", type=int, default=2)
    parser.add_argument("--rbf-scales", type=float, nargs="+", default=[0.1])
    parser.add_argument("--no-tensors", action="store_true")
    args = parser.parse_args()
    seed_everything(71_071)
    from transformers import AutoModel, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda().eval()
    model.config.use_cache = False
    representation_model = AutoModel.from_pretrained(
        args.representation_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda().eval()
    corpus = torch.load(args.data / "corpus.pt", weights_only=False, mmap=True)
    result, tensors = run_study(
        model,
        representation_model,
        corpus,
        split=args.split,
        source_samples=args.source_samples,
        target_samples=args.target_samples,
        causal_samples_per_task=args.causal_samples_per_task,
        rbf_scales=tuple(args.rbf_scales),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.with_suffix(".json"), result)
    if not args.no_tensors:
        torch.save(tensors, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
