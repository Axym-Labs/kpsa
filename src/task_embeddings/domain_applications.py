"""Real-domain attribution and intervention screen with explicit data roles."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from scipy.stats import spearmanr

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .domain_train import (
    DomainConfig,
    batch_loss,
    evaluate_details,
    logit_residual_norm,
    make_model,
)
from .streaming_v5 import fit_linear_tbe, query_linear_tbe


def load_checkpoint(path):
    if path.is_dir():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda()
        model.config.use_cache = False
        cfg = model.config
        return model, DomainConfig(
            steps=0,
            method="pretrained",
            hidden=cfg.hidden_size,
            layers=cfg.num_hidden_layers,
            intermediate=cfg.intermediate_size,
            heads=cfg.num_attention_heads,
            kv_heads=cfg.num_key_value_heads,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = DomainConfig(**payload["configuration"])
    model = make_model(config, payload["vocab_size"])
    model.load_state_dict(payload["model"])
    return model, config


def profile(
    model,
    domains,
    partitions,
    samples=16,
    start=0,
    parallel=False,
    include_fisher=False,
):
    """One sequence per backward: per-example OPG (example = packed sequence)."""
    model.eval()
    output = {
        kind: {
            stat: torch.zeros(part.n_groups, len(domains))
            for stat in (
                ("raw", "normalized", "fisher")
                if include_fisher
                else ("raw", "normalized")
            )
        }
        for kind, part in partitions.items()
    }
    for task, data in enumerate(domains):
        indices = torch.randperm(
            len(data),
            generator=torch.Generator().manual_seed(31415 + (0 if parallel else task)),
        )[start : start + samples]
        if len(indices) != samples:
            raise ValueError("insufficient disjoint profiling examples")
        for block in data[indices]:
            model.zero_grad(set_to_none=True)
            blocks = block[None].cuda().long()
            loss, logits = batch_loss(model, blocks)
            norm = logit_residual_norm(logits.detach(), blocks[:, 1:])
            loss.backward(retain_graph=include_fisher)
            for kind, part in partitions.items():
                scores = part.gradient_scores().cpu()
                output[kind]["raw"][:, task] += scores / samples
                output[kind]["normalized"][:, task] += (
                    scores / float(norm.clamp_min(1e-12)) / samples
                )
            if include_fisher:
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    probabilities = logits.detach().float().softmax(-1)
                    labels = torch.multinomial(probabilities.flatten(0, 1), 1).reshape(
                        blocks[:, 1:].shape
                    )
                    labels[blocks[:, 1:] < 0] = -1
                sampled_loss = torch.nn.functional.cross_entropy(
                    logits.float().flatten(0, 1), labels.flatten(), ignore_index=-1
                )
                sampled_loss.backward()
                # Mean conditional-output Fisher aligns with per-token output KL.
                count = int((labels >= 0).sum())
                for kind, part in partitions.items():
                    output[kind]["fisher"][:, task] += (
                        part.gradient_scores().cpu() * count / samples
                    )
    model.zero_grad(set_to_none=True)
    return output


def query_scores(
    atlas,
    features,
    method,
    seed,
    observed=None,
    score_link="linear",
    residual_scale=1.0,
):
    tasks = atlas.shape[1]
    if observed is None:
        observed = list(range(tasks))
    values = atlas[:, observed]
    if method == "full":
        return atlas
    if method == "wrong_task":
        return atlas.roll(1, 1)
    if method == "mean":
        return values.mean(1, keepdim=True).expand(-1, tasks)
    if method == "cluster":
        from sklearn.cluster import KMeans

        inputs = features.detach().cpu().numpy()
        count = min(
            features.shape[1] + 1,
            len(observed),
            len(torch.unique(features[observed], dim=0)),
        )
        clustering = KMeans(n_clusters=count, n_init=10, random_state=seed).fit(
            inputs[observed]
        )
        labels = torch.as_tensor(clustering.labels_, device=values.device)
        prototypes = torch.stack(
            [values[:, labels == k].mean(1) for k in range(count)], 1
        )
        assignment = torch.as_tensor(clustering.predict(inputs), device=values.device)
        mean = values.mean(1, keepdim=True)
        return (mean + residual_scale * (prototypes[:, assignment] - mean)).clamp_min(0)
    generator = torch.Generator().manual_seed(seed)
    if method == "random":
        return torch.rand(atlas.shape, generator=generator)
    if method == "jl":
        features = torch.randn(features.shape, generator=generator)
    elif method == "permuted":
        features = features[torch.randperm(tasks, generator=generator)]
    elif method != "tbe":
        raise ValueError(method)
    if score_link == "log":
        values = values.clamp_min(values.mean(1, keepdim=True) * 1e-4 + 1e-30).log()
    index = fit_linear_tbe(values, features[observed], ridge=0.01)
    result = query_linear_tbe(index, features, residual_scale=residual_scale)
    return result.clamp(-69, 40).exp() if score_link == "log" else result.clamp_min(0)


def correlation(x, y):
    if x.numel() < 2 or y.numel() < 2:
        return None
    if x.numel() >= 32768 and torch.cuda.is_available():
        a, b = tensor_ranks(x.cuda()), tensor_ranks(y.cuda())
        a, b = a - a.mean(), b - b.mean()
        denominator = a.square().sum().sqrt() * b.square().sum().sqrt()
        value = float((a * b).sum() / denominator)
        return value if math.isfinite(value) else None
    if x.std() == 0 or y.std() == 0:
        return None
    value = float(spearmanr(x.cpu().numpy(), y.cpu().numpy()).statistic)
    return value if math.isfinite(value) else None


def tensor_ranks(value):
    """Zero-based average-tie ranks, including on GPU for million-group atlases."""
    ordered, indices = value.sort()
    _, inverse, counts = torch.unique_consecutive(
        ordered, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(0)
    averages = (ends + ends - counts - 1).float() / 2
    result = torch.empty_like(value, dtype=torch.float32)
    result.scatter_(0, indices, averages[inverse])
    return result


def application_scores(
    atlas,
    features,
    method,
    seed,
    query_mode="seen",
    score_link="linear",
    residual_scale=1.0,
):
    if query_mode == "seen":
        return query_scores(
            atlas,
            features,
            method,
            seed,
            score_link=score_link,
            residual_scale=residual_scale,
        )
    if query_mode != "cold":
        raise ValueError(query_mode)
    result = torch.empty_like(atlas)
    for fold in range(4):
        queries = list(range(fold, atlas.shape[1], 4))
        observed = [t for t in range(atlas.shape[1]) if t not in queries]
        result[:, queries] = query_scores(
            atlas,
            features,
            method,
            seed,
            observed=observed,
            score_link=score_link,
            residual_scale=residual_scale,
        )[:, queries]
    return result


def budget_mask(scores, sizes, removed_fraction):
    """Ranked first-fit allocation; skip groups too large for the remaining cap."""
    if not 0 <= removed_fraction <= 1 or scores.shape != sizes.shape:
        raise ValueError("matched scores/sizes and a fraction in [0, 1] required")
    if not bool(torch.isfinite(sizes).all() and (sizes > 0).all()):
        raise ValueError("finite positive parameter counts required")
    order = torch.argsort(scores, stable=True)
    ordered_sizes = sizes[order].double()
    total = float(ordered_sizes.sum())
    budget = total * removed_fraction
    cumulative = ordered_sizes.cumsum(0)
    prefix = int((cumulative <= budget).sum())
    selected = order[:prefix]
    remaining = budget - (float(cumulative[prefix - 1]) if prefix else 0.0)
    # Most fine-grained partitions fill the cap with the prefix. Whole tensors
    # can encounter a large embedding first, but smaller ranked groups still fit.
    tail = order[prefix:]
    tail_sizes = ordered_sizes[prefix:]
    feasible = tail_sizes <= remaining
    extra = []
    for group, size in zip(tail[feasible].tolist(), tail_sizes[feasible].tolist()):
        if size <= remaining:
            extra.append(group)
            remaining -= size
    if extra:
        selected = torch.cat((selected, order.new_tensor(extra)))
    scales = torch.ones_like(scores)
    scales[selected] = 0
    return scales, float(sizes[selected].double().sum()) / total


def scoped_budget_mask(scores, sizes, fraction, allowed):
    selected, actual = budget_mask(scores[allowed], sizes[allowed], fraction)
    result = torch.ones_like(scores)
    result[allowed] = selected
    return result, actual


@torch.no_grad()
def quantize_weight(value, bits, group_size=128):
    """Symmetric small-group fake quantization; zero group size means per row."""
    if bits >= 16:
        return value.clone()
    if bits < 2:
        raise ValueError("symmetric signed quantization needs at least two bits")
    width = value.shape[-1] if value.ndim else 1
    group = width if not group_size else min(group_size, width)
    matrix = value.float().reshape(-1, width)
    padding = (-width) % group
    if padding:
        matrix = torch.nn.functional.pad(matrix, (0, padding))
    chunks = matrix.reshape(matrix.shape[0], -1, group)
    bound = chunks.abs().amax(-1, keepdim=True)
    scale = bound.clamp_min(1e-12) / (2 ** (bits - 1) - 1)
    quantized = (chunks / scale).round_().mul_(scale).reshape(matrix.shape)
    return quantized[:, :width].reshape(value.shape).to(value.dtype)


@torch.no_grad()
def precision_scope(partition, scope="all"):
    """CPU mask of groups whose matrix weights can actually change precision."""
    if scope not in {"all", "transformer"}:
        raise ValueError(scope)
    allowed = torch.zeros(partition.n_groups, dtype=torch.bool)
    for spec in partition.slices:
        if spec.parameter.ndim >= 2 and (scope == "all" or ".layers." in spec.name):
            allowed[spec.offset : spec.offset + spec.count] = True
    return allowed


@torch.no_grad()
def precision_candidates(
    partition, low_bits=4, high_bits=8, group_size=128, scope="all"
):
    low = {}
    high = {}
    gain = torch.zeros_like(partition.sizes)
    scale_count = 0
    for spec in partition.slices:
        p = spec.parameter
        if p.ndim < 2 or (scope == "transformer" and ".layers." not in spec.name):
            low[spec.name] = p.clone()
            high[spec.name] = p.clone()
            continue
        low[spec.name] = quantize_weight(p, low_bits, group_size)
        high[spec.name] = quantize_weight(p, high_bits, group_size)
        gain[spec.offset : spec.offset + spec.count] += spec.reduce(
            p - low[spec.name]
        ) - spec.reduce(p - high[spec.name])
        width = p.shape[-1] if p.ndim else 1
        group = width if not group_size else min(group_size, width)
        scale_count += (p.numel() // width) * math.ceil(width / group)
    return low, high, gain / partition.sizes, scale_count


@torch.no_grad()
def apply_precision(partition, low, high, high_mask):
    for spec in partition.slices:
        spec.parameter.copy_(
            low[spec.name] + spec.expand(high_mask) * (high[spec.name] - low[spec.name])
        )


@torch.no_grad()
def perturb_group(partition, group, generator, sigma=0.02):
    originals = []
    for spec in partition.slices:
        if spec.offset <= group < spec.offset + spec.count:
            index = [slice(None)] * spec.parameter.ndim
            if spec.axis is not None:
                index[spec.axis] = group - spec.offset
            target = spec.parameter[tuple(index)]
            originals.append((target, target.clone()))
            noise = torch.randn(
                target.shape,
                generator=generator,
                device=target.device,
                dtype=target.dtype,
            )
            target.add_(noise, alpha=sigma)
    return originals


@torch.no_grad()
def restore(originals):
    for target, source in originals:
        target.copy_(source)


@torch.no_grad()
def causal_effects(
    model, partition, domains, groups, seed, blocks_per_task=2, sigma=0.002
):
    model.eval()
    if len(groups) == 0:
        return torch.empty(0, len(domains))
    clean = []
    for data in domains:
        blocks = data[:blocks_per_task].cuda().long()
        _, logits = batch_loss(model, blocks)
        clean.append(logits.float().log_softmax(-1).cpu())
    effects = torch.zeros(len(groups), len(domains))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for index, group in enumerate(groups):
        originals = perturb_group(partition, int(group), generator, sigma=sigma)
        for task, data in enumerate(domains):
            blocks = data[:blocks_per_task].cuda().long()
            _, logits = batch_loss(model, blocks)
            target = clean[task].cuda()
            valid = blocks[:, 1:] >= 0
            kl = (
                (target.exp() * (target - logits.float().log_softmax(-1))).sum(-1)
                * valid
            ).sum() / valid.sum()
            effects[index, task] = kl.clamp_min(0) / partition.sizes[group]
        restore(originals)
    return effects


def run_applications(
    checkpoint,
    data,
    output,
    samples=16,
    eval_blocks=16,
    causal_groups=48,
    role="validation",
    score_link="linear",
    feature_file="task_features.pt",
    profile_cache=None,
    include_fisher=False,
    causal_sigma=0.002,
    control_seed=None,
    low_bits=4,
    quant_group_size=128,
    all_tasks=False,
    query_mode="seen",
    partition_kinds=("row", "tensor", "swiglu"),
    methods=("full", "tbe", "jl", "mean", "permuted", "random", "wrong_task"),
    action_oracle=None,
    residual_scale=1.0,
    estimators=None,
    intervention_scope="all",
    method_scales=None,
    importance_coordinate="absolute",
    action_target=None,
    pruning_fractions=(0.05, 0.15),
    precision_fractions=(0.1, 0.3),
):
    if intervention_scope not in {"all", "transformer"}:
        raise ValueError(intervention_scope)
    model, config = load_checkpoint(checkpoint)
    seed_everything(config.seed)
    control_seed = config.seed if control_seed is None else control_seed
    torch.set_num_threads(4)
    corpus = torch.load(data / "corpus.pt", weights_only=False)
    features = torch.load(data / feature_file, weights_only=False)["features"]
    action_scores = {}
    action_truth = {}
    if action_oracle is not None:
        payload = torch.load(action_oracle, weights_only=False)
        expected = {
            "checkpoint": str(checkpoint.resolve()),
            "data": str(data.resolve()),
            "samples": samples,
            "low_bits": low_bits,
            "quant_group_size": quant_group_size,
        }
        if any(payload.get(k) != v for k, v in expected.items()):
            raise ValueError("diagonal action oracle provenance mismatch")
        action_scores = payload["scores"]
        if action_target is not None:
            independent = torch.load(action_target, weights_only=False)
            if (
                any(independent.get(k) != v for k, v in expected.items())
                or independent.get("start") != samples
            ):
                raise ValueError(
                    "independent diagonal-action reference provenance mismatch"
                )
            action_truth = independent["scores"]
    if importance_coordinate not in {"absolute", "relative"}:
        raise ValueError(importance_coordinate)
    if importance_coordinate == "relative" and (
        not action_scores or not action_truth or causal_groups
    ):
        raise ValueError(
            "relative OPG requires disjoint weighted references and no absolute-noise causal assay"
        )
    partitions = {kind: ParameterPartition(model, kind) for kind in partition_kinds}
    started = time.perf_counter()
    signature = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "corpus": str((data / "corpus.pt").resolve()),
        "samples": samples,
        "sampling_seed": 31415,
        "include_fisher": include_fisher,
    }
    coordinate_key = "pruning" if importance_coordinate == "relative" else "absolute"
    if (
        action_scores
        and action_truth
        and all(
            coordinate_key in action_scores[kind][stat]
            and coordinate_key in action_truth[kind][stat]
            for kind in partition_kinds
            for stat in action_scores[kind]
        )
    ):
        source = {
            kind: {
                stat: values[coordinate_key]
                for stat, values in action_scores[kind].items()
            }
            for kind in partition_kinds
        }
        target = {
            kind: {
                stat: values[coordinate_key]
                for stat, values in action_truth[kind].items()
            }
            for kind in partition_kinds
        }
    elif profile_cache and profile_cache.exists():
        cached = torch.load(profile_cache, weights_only=False)
        if cached.get("signature") != signature:
            raise ValueError("profile cache provenance mismatch")
        source, target = cached["source"], cached["target"]
    else:
        source = profile(
            model,
            corpus["train"],
            partitions,
            samples,
            0,
            corpus.get("parallel_examples", False),
            include_fisher,
        )
        target = profile(
            model,
            corpus["train"],
            partitions,
            samples,
            samples,
            corpus.get("parallel_examples", False),
            include_fisher,
        )
        torch.save(
            {"source": source, "target": target, "signature": signature},
            output.with_suffix(".pt"),
        )
    baseline, baseline_details = evaluate_details(model, corpus[role], eval_blocks)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    heldout = (
        list(range(len(features))) if all_tasks else list(range(3, len(features), 4))
    )
    ranking = []
    pruning = []
    causal = []
    quantization = []
    resources = {}
    uniform_quantization = []
    histogram_path = data / "token_features.pt"
    histograms = (
        torch.load(histogram_path, weights_only=False).get("full_histograms")
        if histogram_path.exists()
        else None
    )
    for kind, part in partitions.items():
        # Core transformer groups, uniformly sampled; embeddings excluded explicitly.
        eligible = set()
        for spec in part.slices:
            if ".layers." in spec.name:
                eligible.update(range(spec.offset, spec.offset + spec.count))
        eligible = torch.tensor(sorted(eligible))
        generator = torch.Generator().manual_seed(control_seed + 500)
        groups = eligible[
            torch.randperm(len(eligible), generator=generator)[:causal_groups]
        ]
        effects = causal_effects(
            model, part, corpus[role], groups, control_seed + 900, sigma=causal_sigma
        )
        sizes = part.sizes.cpu()
        allowed = (
            torch.ones(part.n_groups, dtype=torch.bool)
            if intervention_scope == "all"
            else torch.zeros(part.n_groups, dtype=torch.bool)
        )
        if intervention_scope == "transformer":
            allowed[eligible] = True
        scope_fraction = float(sizes[allowed].sum() / sizes.sum())
        quantizable = precision_scope(part, intervention_scope)
        quantizable_fraction = float(
            sizes[quantizable].double().sum() / part.n_parameters
        )
        energy = part.weight_energy().cpu()
        pruning_energy = (
            torch.ones_like(energy) if importance_coordinate == "relative" else energy
        )
        low, high, quant_gain, scale_count = precision_candidates(
            part,
            low_bits=low_bits,
            group_size=quant_group_size,
            scope=intervention_scope,
        )
        quant_gain = quant_gain.cpu()
        if importance_coordinate == "relative":
            quant_gain = quant_gain / energy.clamp_min(1e-30)
        unquantized = [
            s.parameter
            for s in part.slices
            if s.parameter.ndim < 2
            or (intervention_scope == "transformer" and ".layers." not in s.name)
        ]
        unquantized_parameters = sum(p.numel() for p in unquantized)
        unquantized_bits = sum(p.numel() * p.element_size() * 8 for p in unquantized)
        if not uniform_quantization:
            for bits, values in ((low_bits, low), (8, high)):
                for spec in part.slices:
                    with torch.no_grad():
                        spec.parameter.copy_(values[spec.name])
                losses, loss_details = evaluate_details(
                    model, [corpus[role][t] for t in heldout], eval_blocks
                )
                for task, loss, detail in zip(heldout, losses, loss_details):
                    uniform_quantization.append(
                        {
                            "bits": bits,
                            "evaluation": detail,
                            "task": task,
                            "nll_increase": float(loss - baseline[task]),
                            "ideal_packed_bits_per_parameter": bits
                            + (
                                32 * scale_count
                                + unquantized_bits
                                - bits * unquantized_parameters
                            )
                            / part.n_parameters,
                        }
                    )
            model.load_state_dict(original)
        resources[kind] = {
            "groups": part.n_groups,
            "parameters": part.n_parameters,
            "full_atlas_floats": part.n_groups * len(features),
            "tbe_floats": part.n_groups * (features.shape[1] + 1)
            + features.numel()
            + features.shape[1],
            "cluster_max_floats": part.n_groups * (features.shape[1] + 1)
            + (features.shape[1] + 1) * features.shape[1]
            + features.numel(),
            "consolidated_diagonal_floats": part.n_parameters,
            "intervention_scope_parameters": int(sizes[allowed].double().sum()),
            "intervention_scope_groups": int(allowed.sum()),
            "quantization_scope_parameters": int(sizes[quantizable].double().sum()),
            "quantization_scope_groups": int(quantizable.sum()),
            "observed_atlas_floats_per_cold_fold": part.n_groups
            * (len(features) - math.ceil(len(features) / 4)),
        }
        for estimator in source[kind]:
            if estimators is not None and estimator not in estimators:
                continue
            atlas = source[kind][estimator]
            truth = target[kind][estimator]
            method_grid = tuple(methods) + (
                ("diagonal",) if kind in action_scores else ()
            )
            for method in method_grid:
                scale = (method_scales or {}).get(method, residual_scale)
                for fold in range(4) if method != "diagonal" else ():
                    query_tasks = list(range(fold, len(features), 4))
                    training_tasks = [
                        t for t in range(len(features)) if t not in query_tasks
                    ]
                    scores = query_scores(
                        atlas,
                        features,
                        method,
                        control_seed + 800,
                        observed=training_tasks,
                        score_link=score_link,
                        residual_scale=scale,
                    )
                    residual = scores - atlas[:, training_tasks].mean(1, keepdim=True)
                    truth_residual = truth - truth[:, training_tasks].mean(
                        1, keepdim=True
                    )
                    residual_effects = effects - effects[:, training_tasks].mean(
                        1, keepdim=True
                    )
                    for task in query_tasks:
                        ranking.append(
                            {
                                "partition": kind,
                                "estimator": estimator,
                                "method": method,
                                "task": task,
                                "fold": fold,
                                "residual_opg_spearman": correlation(
                                    residual[:, task], truth_residual[:, task]
                                ),
                                "transformer_residual_opg_spearman": correlation(
                                    residual[eligible, task],
                                    truth_residual[eligible, task],
                                ),
                                "raw_opg_spearman": correlation(
                                    scores[:, task], truth[:, task]
                                ),
                                "oracle": method in {"full", "wrong_task"},
                            }
                        )
                        causal.append(
                            {
                                "partition": kind,
                                "estimator": estimator,
                                "method": method,
                                "task": task,
                                "fold": fold,
                                "residual_causal_spearman": correlation(
                                    residual[groups, task], residual_effects[:, task]
                                ),
                                "oracle": method in {"full", "wrong_task"},
                                "sampled_groups": groups.tolist(),
                            }
                        )
                # Per-task weight sparsification as an application screen.
                # It promises no speedup; actual NLL/perplexity cost is the endpoint.
                all_scores = (
                    None
                    if method == "diagonal"
                    else application_scores(
                        atlas,
                        features,
                        method,
                        control_seed + 800,
                        query_mode=query_mode,
                        score_link=score_link,
                        residual_scale=scale,
                    )
                )
                for fraction in pruning_fractions:
                    for task in heldout:
                        model.load_state_dict(original)
                        saliency = (
                            (
                                all_scores[:, task]
                                if method == "random"
                                else all_scores[:, task] * pruning_energy
                            )
                            if method != "diagonal"
                            else action_scores[kind][estimator]["pruning"][:, task]
                        )
                        mask, actual = scoped_budget_mask(
                            saliency, sizes, fraction, allowed
                        )
                        part.scale_parameters(mask.cuda())
                        losses, detail = evaluate_details(
                            model, [corpus[role][task]], eval_blocks
                        )
                        loss = losses[0]
                        delta = float(loss - baseline[task])
                        pruning.append(
                            {
                                "partition": kind,
                                "evaluation": detail[0],
                                "estimator": estimator,
                                "method": method,
                                "task": task,
                                "removed_parameter_fraction": actual * scope_fraction,
                                "removed_scope_fraction": actual,
                                "oracle": query_mode == "cold"
                                and method in {"full", "diagonal", "wrong_task"},
                                "requested_removed_fraction": fraction,
                                "baseline_nll": float(baseline[task]),
                                "nll_increase": delta,
                                "perplexity_ratio": math.exp(min(delta, 80)),
                            }
                        )
                for high_fraction in precision_fractions:
                    for task in heldout:
                        benefit = (
                            (
                                all_scores[:, task]
                                if method == "random"
                                else all_scores[:, task] * quant_gain
                            )
                            if method != "diagonal"
                            else action_scores[kind][estimator]["quantization"][:, task]
                        )
                        low_mask, actual = scoped_budget_mask(
                            -benefit, sizes, high_fraction, quantizable
                        )
                        apply_precision(part, low, high, 1 - low_mask.cuda())
                        losses, detail = evaluate_details(
                            model, [corpus[role][task]], eval_blocks
                        )
                        loss = losses[0]
                        delta = float(loss - baseline[task])
                        quantization.append(
                            {
                                "partition": kind,
                                "evaluation": detail[0],
                                "estimator": estimator,
                                "method": method,
                                "task": task,
                                "requested_high_precision_fraction": high_fraction,
                                "high_precision_fraction": actual
                                * quantizable_fraction,
                                "high_precision_scope_fraction": actual,
                                "oracle": query_mode == "cold"
                                and method in {"full", "diagonal", "wrong_task"},
                                "ideal_packed_bits_per_parameter": low_bits
                                + (8 - low_bits) * actual * quantizable_fraction
                                + (32 * scale_count + int(quantizable.sum()))
                                / part.n_parameters
                                + (unquantized_bits - low_bits * unquantized_parameters)
                                / part.n_parameters,
                                "baseline_nll": float(baseline[task]),
                                "nll_increase": delta,
                                "perplexity_ratio": math.exp(min(delta, 80)),
                            }
                        )
        for fraction in pruning_fractions:
            for task in heldout:
                model.load_state_dict(original)
                mask, actual = scoped_budget_mask(energy, sizes, fraction, allowed)
                part.scale_parameters(mask.cuda())
                losses, detail = evaluate_details(
                    model, [corpus[role][task]], eval_blocks
                )
                loss = losses[0]
                delta = float(loss - baseline[task])
                pruning.append(
                    {
                        "partition": kind,
                        "evaluation": detail[0],
                        "estimator": "none",
                        "method": "magnitude",
                        "task": task,
                        "removed_parameter_fraction": actual * scope_fraction,
                        "removed_scope_fraction": actual,
                        "requested_removed_fraction": fraction,
                        "baseline_nll": float(baseline[task]),
                        "nll_increase": delta,
                        "perplexity_ratio": math.exp(min(delta, 80)),
                    }
                )
        # A cheaper, gradient-free vocabulary-specialization control.
        embedding = next(
            (s for s in part.slices if "embed_tokens.weight" in s.name and s.axis == 0),
            None,
        )
        if (
            histograms is not None
            and embedding is not None
            and intervention_scope == "all"
        ):
            frequency = histograms.square().T
            frequency = torch.nn.functional.pad(
                frequency, (0, 0, 0, embedding.count - len(frequency))
            )
            frequency = frequency + 0.001 * frequency.mean(1, keepdim=True)
            embedding_indices = torch.arange(
                embedding.offset, embedding.offset + embedding.count
            )
            for fraction in pruning_fractions:
                if float(sizes[embedding_indices].sum() / sizes.sum()) < fraction:
                    continue
                for task in heldout:
                    model.load_state_dict(original)
                    score = torch.full((part.n_groups,), float("inf"))
                    score[embedding_indices] = frequency[:, task]
                    mask, actual = budget_mask(score, sizes, fraction)
                    part.scale_parameters(mask.cuda())
                    losses, detail = evaluate_details(
                        model, [corpus[role][task]], eval_blocks
                    )
                    loss = losses[0]
                    delta = float(loss - baseline[task])
                    pruning.append(
                        {
                            "partition": kind,
                            "evaluation": detail[0],
                            "estimator": "none",
                            "method": "token_frequency",
                            "task": task,
                            "requested_removed_fraction": fraction,
                            "removed_parameter_fraction": actual,
                            "baseline_nll": float(baseline[task]),
                            "nll_increase": delta,
                            "perplexity_ratio": math.exp(min(delta, 80)),
                        }
                    )
        model.load_state_dict(original)
        print(f"APPLICATIONS {kind} finished", flush=True)
    result = {
        "seed": config.seed,
        "control_seed": control_seed,
        "training_seed": None if checkpoint.is_dir() else config.seed,
        "model_configuration": config.__dict__,
        "data_role": role,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_source": str(checkpoint.resolve()),
        "data_source": str(data.resolve()),
        "initialization": "pretrained"
        if checkpoint.is_dir()
        else "trained_from_scratch",
        "baseline_nll": baseline.tolist(),
        "baseline_evaluation": baseline_details,
        "heldout_attribution_folds": [
            list(range(f, len(features), 4)) for f in range(4)
        ],
        "intervention_screen_tasks": heldout,
        "cold_start_definition": "query-domain importance columns are withheld; model parameters remain fixed; no claim that domains were absent from model pretraining",
        "reference_samples_per_domain": samples,
        "profile_sampling": "one corpus row per example: packed block or padded segment",
        "ranking": ranking,
        "causal": causal,
        "pruning": pruning,
        "quantization": quantization,
        "uniform_quantization": uniform_quantization,
        "resources": resources,
        "wall_seconds": time.perf_counter() - started,
        "claim_ready": False,
        "protocol_version": 10,
        "precision_budget_definition": "fraction of quantizable matrix parameters only; unchanged vectors and out-of-scope weights excluded",
        "pruning_fractions": list(pruning_fractions),
        "precision_fractions": list(precision_fractions),
        "budget_selection": "ranked first-fit; skip groups exceeding remaining parameter budget",
        "intervention_scope": intervention_scope,
        "method_scales": method_scales,
        "importance_coordinate": importance_coordinate,
        "action_target": str(action_target) if action_target else None,
        "query_mode": query_mode,
        "partition_kinds": list(partition_kinds),
        "methods": list(methods),
        "action_oracle": str(action_oracle) if action_oracle else None,
        "residual_scale": residual_scale,
        "estimators": list(estimators) if estimators is not None else None,
        "low_bits": low_bits,
        "quant_group_size": quant_group_size,
        "causal_sigma": causal_sigma,
        "include_fisher": include_fisher,
        "score_link": score_link,
        "feature_file": feature_file,
        "pruning_score": "group mean importance times group mean squared weight; random and magnitude controls",
        "quantization_contract": "matrix weights in intervention scope only: base low_bits, selectively 8-bit groups; normalization/bias and excluded weights preserved at original precision; symmetric quant_group_size simulated quantization; analytical packed bits including FP32 scales and group selectors, no measured deployment saving or speedup",
        "evaluation_sampling": "fixed random blocks, seed 90210",
        "profile_sampling_seed": 31415,
    }
    save_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--causal-groups", type=int, default=48)
    p.add_argument("--eval-blocks", type=int, default=16)
    p.add_argument("--role", choices=("validation", "test"), default="validation")
    p.add_argument("--score-link", choices=("linear", "log"), default="linear")
    p.add_argument("--feature-file", default="task_features.pt")
    p.add_argument("--profile-cache", type=Path)
    p.add_argument("--include-fisher", action="store_true")
    p.add_argument("--causal-sigma", type=float, default=0.002)
    p.add_argument("--low-bits", type=int, default=4)
    p.add_argument("--quant-group-size", type=int, default=128)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--skip-pruning", action="store_true")
    p.add_argument("--query-mode", choices=("seen", "cold"), default="seen")
    p.add_argument("--action-oracle", type=Path)
    p.add_argument(
        "--partitions",
        nargs="+",
        choices=("row", "tensor", "swiglu"),
        default=("row", "tensor", "swiglu"),
    )
    a = p.parse_args()
    run_applications(
        a.checkpoint,
        a.data,
        a.output,
        a.samples,
        pruning_fractions=() if a.skip_pruning else (0.05, 0.15),
        eval_blocks=a.eval_blocks,
        causal_groups=a.causal_groups,
        role=a.role,
        score_link=a.score_link,
        feature_file=a.feature_file,
        profile_cache=a.profile_cache,
        include_fisher=a.include_fisher,
        action_oracle=a.action_oracle,
        all_tasks=a.all_tasks,
        query_mode=a.query_mode,
        partition_kinds=a.partitions,
        causal_sigma=a.causal_sigma,
        low_bits=a.low_bits,
        quant_group_size=a.quant_group_size,
    )
