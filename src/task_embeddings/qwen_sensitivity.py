"""Modern decoder-only LM experiments for representation-conditioned sensitivity."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_applications import (
    apply_precision,
    precision_candidates,
    precision_scope,
    scoped_budget_mask,
)
from .domain_optimizer import ParameterPartition
from .domain_train import batch_loss, evaluate_details
from .representation_sensitivity import (
    group_relative_shares,
    partition_gradient_energy,
    profile_ranking_metrics,
)


@dataclass(frozen=True)
class RelativeSensitivityProfiles:
    task_profiles: torch.Tensor
    examples_per_task: torch.Tensor
    wall_seconds: float
    max_partition_relative_error: float


def cold_fold_predictions(
    task_profiles: torch.Tensor,
    features: torch.Tensor,
    *,
    seed: int,
    folds: int = 4,
) -> dict[str, torch.Tensor]:
    """Build each query profile using only the other folds' gradient profiles."""
    if task_profiles.ndim != 2 or features.ndim != 2:
        raise ValueError("profiles and features must be matrices")
    if task_profiles.shape[1] != features.shape[0] or folds < 2:
        raise ValueError("task axes must match and at least two folds are required")
    profiles = task_profiles.detach().double().cpu()
    semantic = F.normalize(features.detach().double().cpu(), dim=1)
    generator = torch.Generator().manual_seed(seed)
    affine_semantic = F.normalize(
        torch.cat((torch.ones(len(semantic), 1), semantic), dim=1), dim=1
    )
    distance_sq = torch.cdist(semantic, semantic).square()
    positive = distance_sq[distance_sq > 0]
    bandwidth_sq = positive.median() if len(positive) else torch.tensor(1.0)
    kernel = torch.exp(-distance_sq / (2 * bandwidth_sq.clamp_min(1e-12)))
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    rbf_semantic = eigenvectors * eigenvalues.clamp_min(0).sqrt()[None]
    jl = F.normalize(
        torch.randn(features.shape, generator=generator, dtype=torch.float64), dim=1
    )
    permuted = semantic[torch.randperm(len(semantic), generator=generator)]
    methods = (
        "semantic",
        "affine_semantic",
        "rbf_semantic",
        "scalar_mass",
        "jl",
        "permuted",
        "nearest",
        "random",
    )
    result = {name: torch.empty_like(profiles) for name in methods}
    result["random"] = torch.rand(
        profiles.shape, generator=generator, dtype=torch.float64
    )
    for fold in range(folds):
        queries = list(range(fold, profiles.shape[1], folds))
        observed = [task for task in range(profiles.shape[1]) if task not in queries]
        for name, representation in (
            ("semantic", semantic),
            ("affine_semantic", affine_semantic),
            ("rbf_semantic", rbf_semantic),
            ("jl", jl),
            ("permuted", permuted),
        ):
            joint = profiles[:, observed] @ representation[observed] / len(observed)
            result[name][:, queries] = joint @ representation[queries].T
        result["scalar_mass"][:, queries] = profiles[:, observed].mean(1, keepdim=True)
        similarity = semantic[queries] @ semantic[observed].T
        nearest = torch.as_tensor(observed)[similarity.argmax(1)]
        result["nearest"][:, queries] = profiles[:, nearest]
    return result


def nonlinear_fold_predictions(
    task_profiles: torch.Tensor,
    features: torch.Tensor,
    *,
    folds: int = 4,
    rbf_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    cosine_powers: tuple[int, ...] = (2, 4, 8),
    cluster_counts: tuple[int, ...] = (4, 6, 8),
    seed: int = 17_071,
) -> dict[str, torch.Tensor]:
    """Bounded nonlinear cold-query grid with no held-out gradient access."""
    from sklearn.cluster import KMeans

    if task_profiles.ndim != 2 or features.ndim != 2:
        raise ValueError("profiles and features must be matrices")
    if task_profiles.shape[1] != features.shape[0] or folds < 2:
        raise ValueError("task axes must match and at least two folds are required")
    if any(scale <= 0 for scale in rbf_scales):
        raise ValueError("RBF scales must be positive")
    if any(power < 1 for power in cosine_powers):
        raise ValueError("cosine powers must be positive")
    profiles = task_profiles.detach().double().cpu()
    semantic = F.normalize(features.detach().double().cpu(), dim=1)
    generator = torch.Generator().manual_seed(seed)
    distance_sq = torch.cdist(semantic, semantic).square()
    positive = distance_sq[distance_sq > 0]
    median = positive.median() if len(positive) else torch.tensor(1.0)
    cosine = semantic @ semantic.T
    kernels = {
        **{
            f"rbf_scale_{scale:g}": torch.exp(
                -distance_sq / (2 * median.clamp_min(1e-12) * scale**2)
            )
            for scale in rbf_scales
        },
        **{
            f"cosine_power_{power}": ((cosine + 1) / 2).clamp(0, 1) ** power
            for power in cosine_powers
        },
    }
    cluster_labels = {
        count: torch.from_numpy(
            KMeans(n_clusters=count, random_state=seed, n_init=20)
            .fit_predict(semantic.numpy())
        )
        for count in cluster_counts
        if 1 < count < len(semantic)
    }
    random_cluster_labels = {
        count: labels[torch.randperm(len(labels), generator=generator)]
        for count, labels in cluster_labels.items()
    }
    result = {
        name: torch.empty_like(profiles)
        for name in (
            *kernels,
            *(f"cluster_{count}" for count in cluster_labels),
            *(f"cluster_{count}_permuted" for count in cluster_labels),
        )
    }
    for fold in range(folds):
        queries = list(range(fold, profiles.shape[1], folds))
        observed = [task for task in range(profiles.shape[1]) if task not in queries]
        for name, kernel in kernels.items():
            result[name][:, queries] = profiles[:, observed] @ kernel[
                observed
            ][:, queries]
        for count, labels in cluster_labels.items():
            for suffix, assignments in (
                ("", labels),
                ("_permuted", random_cluster_labels[count]),
            ):
                for query in queries:
                    members = [
                        task
                        for task in observed
                        if assignments[task] == assignments[query]
                    ]
                    values = profiles[:, members] if members else profiles[:, observed]
                    result[f"cluster_{count}{suffix}"][:, query] = values.mean(1)
    return result


def profile_relative_sensitivity(
    model,
    domains: list[torch.Tensor],
    partition,
    *,
    samples: int,
    start: int = 0,
) -> RelativeSensitivityProfiles:
    """Measure mean per-example gradient-energy shares for every domain."""
    if samples < 1 or start < 0 or not domains:
        raise ValueError("positive samples, nonnegative start, and domains are required")
    profiles = torch.zeros(partition.n_groups, len(domains), dtype=torch.float64)
    counts = torch.zeros(len(domains), dtype=torch.float64)
    max_relative_error = 0.0
    device = next(model.parameters()).device
    started = time.perf_counter()
    model.eval()
    for task, data in enumerate(domains):
        indices = torch.randperm(
            len(data), generator=torch.Generator().manual_seed(31_415 + task)
        )[start : start + samples]
        if len(indices) != samples:
            raise ValueError("insufficient disjoint profiling examples")
        for block in data[indices]:
            model.zero_grad(set_to_none=True)
            local = block[None].to(device).long()
            loss, _ = batch_loss(model, local)
            loss.backward()
            energy = partition_gradient_energy(partition)
            direct = sum(
                parameter.grad.detach().float().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            relative_error = float(
                (energy.sum() - direct).abs() / direct.clamp_min(1e-30)
            )
            max_relative_error = max(max_relative_error, relative_error)
            profiles[:, task].add_(
                group_relative_shares(energy.detach().double().cpu()[None])[0]
            )
            counts[task] += 1
    model.zero_grad(set_to_none=True)
    return RelativeSensitivityProfiles(
        task_profiles=profiles / counts[None],
        examples_per_task=counts,
        wall_seconds=time.perf_counter() - started,
        max_partition_relative_error=max_relative_error,
    )


def evaluate_precision_allocations(
    model,
    partition,
    domains: list[torch.Tensor],
    predictions: dict[str, torch.Tensor],
    *,
    direct_profiles: torch.Tensor,
    source_profiles: torch.Tensor | None = None,
    eval_blocks: int,
    fractions: tuple[float, ...] = (0.1, 0.3),
    low_bits: int = 4,
    high_bits: int = 8,
    quant_group_size: int = 128,
) -> dict[str, object]:
    """Evaluate cold mixed-precision rankings at matched parameter budgets."""
    if direct_profiles.shape != (partition.n_groups, len(domains)):
        raise ValueError("direct profiles must contain one column per evaluation task")
    if source_profiles is not None and source_profiles.shape != direct_profiles.shape:
        raise ValueError("source and direct profiles must share shape")
    if any(values.shape != direct_profiles.shape for values in predictions.values()):
        raise ValueError("all predicted profiles must match the direct profiles")
    if not fractions or any(not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("precision fractions must lie strictly between zero and one")
    original = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    baseline, baseline_details = evaluate_details(model, domains, eval_blocks)
    sizes = partition.sizes.detach().cpu()
    quantizable = precision_scope(partition, "transformer")
    quantizable_parameters = int(sizes[quantizable].double().sum())
    weight_energy = partition.weight_energy().detach().cpu().double()
    low, high, quant_gain, scale_count = precision_candidates(
        partition,
        low_bits=low_bits,
        high_bits=high_bits,
        group_size=quant_group_size,
        scope="transformer",
    )
    quant_gain = quant_gain.detach().cpu().double()
    score_grid = {
        **{name: value.detach().double().cpu() for name, value in predictions.items()},
        "direct_gradient_oracle": direct_profiles.detach().double().cpu(),
        "quant_error_only": torch.ones_like(direct_profiles, dtype=torch.float64),
        "weight_magnitude": weight_energy[:, None].expand(-1, len(domains)),
    }
    if source_profiles is not None:
        score_grid["source_onehot_reference"] = (
            source_profiles.detach().double().cpu()
        )
    unquantized_bits = 0
    for spec in partition.slices:
        in_scope = spec.parameter.ndim >= 2 and ".layers." in spec.name
        if not in_scope:
            unquantized_bits += (
                spec.parameter.numel() * spec.parameter.element_size() * 8
            )
    records = []
    uniform = []
    device = next(model.parameters()).device

    def average_bits(actual_high_fraction: float, selectors: bool = True) -> float:
        bits = (
            low_bits * quantizable_parameters
            + (high_bits - low_bits)
            * actual_high_fraction
            * quantizable_parameters
            + unquantized_bits
            + 32 * scale_count
            + (int(quantizable.sum()) if selectors else 0)
        )
        return bits / partition.n_parameters

    try:
        for label, high_mask, high_fraction in (
            (f"uniform_{low_bits}bit", torch.zeros(partition.n_groups), 0.0),
            (f"uniform_{high_bits}bit", torch.ones(partition.n_groups), 1.0),
        ):
            apply_precision(partition, low, high, high_mask.to(device))
            losses, details = evaluate_details(model, domains, eval_blocks)
            for task, (loss, detail) in enumerate(zip(losses, details)):
                uniform.append(
                    {
                        "method": label,
                        "task": task,
                        "baseline_nll": float(baseline[task]),
                        "nll": float(loss),
                        "nll_increase": float(loss - baseline[task]),
                        "high_precision_scope_fraction": high_fraction,
                        "ideal_packed_bits_per_parameter": average_bits(
                            high_fraction, selectors=False
                        ),
                        "evaluation": detail,
                    }
                )
        for method, scores in score_grid.items():
            clipped_fraction = float((scores < 0).double().mean())
            scores = scores.clamp_min(0)
            for fraction in fractions:
                for task in range(len(domains)):
                    benefit = scores[:, task] * quant_gain
                    low_mask, actual = scoped_budget_mask(
                        -benefit, sizes, fraction, quantizable
                    )
                    apply_precision(partition, low, high, (1 - low_mask).to(device))
                    losses, details = evaluate_details(
                        model, [domains[task]], eval_blocks
                    )
                    loss = losses[0]
                    records.append(
                        {
                            "method": method,
                            "task": task,
                            "requested_high_precision_scope_fraction": fraction,
                            "high_precision_scope_fraction": actual,
                            "high_precision_parameter_fraction": actual
                            * quantizable_parameters
                            / partition.n_parameters,
                            "ideal_packed_bits_per_parameter": average_bits(actual),
                            "baseline_nll": float(baseline[task]),
                            "nll": float(loss),
                            "nll_increase": float(loss - baseline[task]),
                            "clipped_prediction_fraction": clipped_fraction,
                            "evaluation": details[0],
                        }
                    )
    finally:
        model.load_state_dict(original)
    return {
        "baseline_nll": baseline.tolist(),
        "baseline_evaluation": baseline_details,
        "uniform": uniform,
        "records": records,
        "resources": {
            "groups": partition.n_groups,
            "parameters": partition.n_parameters,
            "quantizable_groups": int(quantizable.sum()),
            "quantizable_parameters": quantizable_parameters,
            "quantization_scale_count": scale_count,
            "low_bits": low_bits,
            "high_bits": high_bits,
            "quant_group_size": quant_group_size,
        },
    }


def run_qwen_study(
    model,
    corpus: dict,
    features: torch.Tensor,
    *,
    model_source: str,
    representation_label: str = "normalized training-input token-frequency PCA",
    samples: int = 4,
    folds: int = 4,
    evaluation_role: str | None = "validation",
    eval_blocks: int = 4,
    fractions: tuple[float, ...] = (0.1, 0.3),
) -> tuple[dict, dict[str, object]]:
    """Measure disjoint profiles, cold retrieval, and optional precision utility."""
    if len(corpus["train"]) != len(features):
        raise ValueError("corpus domains and representation rows must match")
    partition = ParameterPartition(model, "swiglu")
    source = profile_relative_sensitivity(
        model, corpus["train"], partition, samples=samples, start=0
    )
    target = profile_relative_sensitivity(
        model, corpus["train"], partition, samples=samples, start=samples
    )
    predictions = cold_fold_predictions(
        source.task_profiles, features, seed=17_071, folds=folds
    )
    cold_query = {
        method: profile_ranking_metrics(
            estimate, target.task_profiles, top_fraction=0.05
        )
        for method, estimate in predictions.items()
    }
    cold_query["direct_gradient_oracle"] = profile_ranking_metrics(
        target.task_profiles, target.task_profiles, top_fraction=0.05
    )
    precision = None
    if evaluation_role is not None:
        if evaluation_role not in corpus:
            raise ValueError(f"corpus does not contain role {evaluation_role!r}")
        precision = evaluate_precision_allocations(
            model,
            partition,
            corpus[evaluation_role],
            predictions,
            direct_profiles=target.task_profiles,
            source_profiles=source.task_profiles,
            eval_blocks=eval_blocks,
            fractions=fractions,
        )
    configuration = model.config.to_dict() if hasattr(model, "config") else {}
    result = {
        "setting": "modern_decoder_review_domains",
        "model_source": model_source,
        "model": {
            "architecture": type(model).__name__,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "configuration": configuration,
            "groups": partition.n_groups,
            "partition": "complete coupled-SwiGLU feature/row partition",
        },
        "domains": list(corpus.get("domains", range(len(features)))),
        "protocol": {
            "source_start": 0,
            "target_start": samples,
            "samples_per_domain_per_profile": samples,
            "folds": folds,
            "evaluation_role": evaluation_role,
            "eval_blocks_per_domain": eval_blocks if evaluation_role else None,
            "functional": "mean next-token cross-entropy loss",
            "representation": representation_label,
        },
        "fidelity": {
            "source_mass_sum_min": float(source.task_profiles.sum(0).min()),
            "source_mass_sum_max": float(source.task_profiles.sum(0).max()),
            "target_mass_sum_min": float(target.task_profiles.sum(0).min()),
            "target_mass_sum_max": float(target.task_profiles.sum(0).max()),
            "max_partition_relative_error": max(
                source.max_partition_relative_error,
                target.max_partition_relative_error,
            ),
        },
        "reference_replicate": profile_ranking_metrics(
            source.task_profiles, target.task_profiles, top_fraction=0.05
        ),
        "cold_query": cold_query,
        "precision": precision,
        "timing": {
            "source_profile_seconds": source.wall_seconds,
            "target_profile_seconds": target.wall_seconds,
        },
    }
    tensors = {
        "features": features.detach().float().cpu(),
        "source_profiles": source.task_profiles.float(),
        "target_profiles": target.task_profiles.float(),
    }
    return result, tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", default="token_features.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--eval-blocks", type=int, default=4)
    parser.add_argument("--role", choices=("validation", "test"), default="validation")
    parser.add_argument("--profiles-only", action="store_true")
    args = parser.parse_args()
    seed_everything(17_071)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda()
    model.config.use_cache = False
    corpus = torch.load(args.data / "corpus.pt", weights_only=False, mmap=True)
    feature_payload = torch.load(
        args.data / args.features, weights_only=False, mmap=True
    )
    features = feature_payload["features"]
    if "model" in feature_payload and "descriptions" in feature_payload:
        representation_label = (
            f"frozen task-description embeddings from {feature_payload['model']} "
            f"({feature_payload.get('pooling', 'normalized pooling')})"
        )
    else:
        representation_label = "normalized training-input token-frequency PCA"
    result, tensors = run_qwen_study(
        model,
        corpus,
        features,
        model_source=args.model,
        representation_label=representation_label,
        samples=args.samples,
        folds=args.folds,
        evaluation_role=None if args.profiles_only else args.role,
        eval_blocks=args.eval_blocks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.with_suffix(".json"), result)
    torch.save(tensors, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
