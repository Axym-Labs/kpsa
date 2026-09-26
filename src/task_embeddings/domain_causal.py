"""Bounded numerical-resolution/repeatability audit, not a tuned headline assay."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import save_json
from .domain_analysis import normalize_profile_shape
from .domain_applications import correlation, load_checkpoint, query_scores, restore
from .domain_optimizer import ParameterPartition


def stable_kl(log_p, log_q):
    delta = log_q - log_p
    return (log_p.exp() * (torch.expm1(delta) - delta)).sum(-1)


def mlp_bundles(partition, width):
    """Contiguous coupled-feature bundles; no attention or embedding weights."""
    if width < 1 or partition.kind != "swiglu":
        raise ValueError("positive width and coupled SwiGLU partition required")
    families = sorted(
        {
            (s.offset, s.count, s.name.split(".mlp.")[0])
            for s in partition.slices
            if ".mlp." in s.name
        }
    )
    bundles = []
    for offset, count, layer in families:
        for start in range(offset, offset + count, width):
            stop = min(start + width, offset + count)
            bundles.append(
                {
                    "index": len(bundles),
                    "layer": layer,
                    "start": start,
                    "stop": stop,
                    "parameters": int(partition.sizes[start:stop].double().sum()),
                }
            )
    return bundles


def bundle_profiles(atlas, sizes, bundles):
    return torch.stack(
        [
            (atlas[b["start"] : b["stop"]] * sizes[b["start"] : b["stop"], None]).sum(0)
            / sizes[b["start"] : b["stop"]].sum()
            for b in bundles
        ]
    )


@torch.no_grad()
def sampled_log_probs(model, blocks, positions=8):
    """Midpoints of equal position strata, restricted to valid prediction tokens."""
    if positions < 1:
        raise ValueError("at least one output position required")
    blocks = blocks.to(next(model.parameters()).device).long()
    inputs = blocks[:, :-1]
    logits = model(
        input_ids=inputs.clamp_min(0), attention_mask=inputs >= 0, use_cache=False
    ).logits
    choices = []
    for valid in blocks[:, 1:] >= 0:
        candidates = valid.nonzero().flatten()
        count = min(positions, len(candidates))
        if not count:
            raise ValueError("each segment must contain a valid prediction target")
        which = (
            (torch.arange(count, device=blocks.device) + 0.5) * len(candidates) / count
        ).long()
        choices.append(candidates[which])
    width = max(map(len, choices))
    indices = torch.zeros(len(blocks), width, device=blocks.device, dtype=torch.long)
    mask = torch.zeros_like(indices, dtype=torch.bool)
    for i, selected in enumerate(choices):
        indices[i, : len(selected)] = selected
        mask[i, : len(selected)] = True
    selected = logits[torch.arange(len(blocks), device=blocks.device)[:, None], indices]
    return selected.float().log_softmax(-1), mask


@torch.no_grad()
def measure_bundles(model, partition, domains, bundles, *, amplitude, directions, seed):
    """Paired +/- relative Rademacher perturbations; returns per-segment output KL.

    The perturbations are probes, not a task-conditioned inference modification.
    Only eight output positions per segment are cached, keeping host RAM bounded.
    """
    if not 0 <= amplitude < 1 or directions < 1:
        raise ValueError("amplitude in [0,1) and positive direction count required")
    if len({len(d) for d in domains}) != 1:
        raise ValueError("aligned language segments required")
    model.eval()
    reference = []
    for data in domains:
        chunks = []
        for batch in data.split(2):
            log_p, mask = sampled_log_probs(model, batch)
            chunks.append((batch, log_p.cpu(), mask.cpu()))
        reference.append(chunks)
    values = torch.zeros(
        len(bundles), len(domains), len(domains[0]), dtype=torch.float64
    )
    device = next(model.parameters()).device
    for row, bundle in enumerate(bundles):
        for direction in range(directions):
            rng = torch.Generator(device=device).manual_seed(
                seed + 100 * bundle["index"] + direction
            )
            targets = []
            for spec in partition.slices:
                start, stop = (
                    bundle["start"] - spec.offset,
                    bundle["stop"] - spec.offset,
                )
                if not 0 <= start < stop <= spec.count:
                    continue
                index = [slice(None)] * spec.parameter.ndim
                index[spec.axis] = slice(start, stop)
                target = spec.parameter[tuple(index)]
                noise = torch.randint(
                    2, target.shape, device=device, generator=rng
                ).float()
                targets.append((target, target.clone(), noise * 2 - 1))
            if len(targets) != 3:
                raise ValueError("a coupled MLP bundle must own gate/up/down slices")
            try:
                for sign in (-1, 1):
                    for target, original, noise in targets:
                        target.copy_(original * (1 + sign * amplitude * noise))
                    for task, chunks in enumerate(reference):
                        cursor = 0
                        for batch, log_p, valid in chunks:
                            log_q, mask = sampled_log_probs(model, batch)
                            if not torch.equal(mask.cpu(), valid):
                                raise RuntimeError(
                                    "output positions changed during a probe"
                                )
                            kl = (stable_kl(log_p.to(device), log_q) * mask).sum(1)
                            kl = (kl / mask.sum(1)).cpu().double()
                            values[row, task, cursor : cursor + len(batch)] += kl / (
                                2 * directions
                            )
                            cursor += len(batch)
            finally:
                for target, original, _ in targets:
                    target.copy_(original)
            for target, original, _ in targets:
                if not torch.equal(target, original):
                    raise RuntimeError("probe failed to restore model weights")
        if (row + 1) % 8 == 0 or row + 1 == len(bundles):
            print(f"CAUSAL BUNDLES {row + 1}/{len(bundles)}", flush=True)
    return values


@torch.no_grad()
def multilingual_bundle_audit(
    checkpoint,
    data,
    source_path,
    target_path,
    output,
    groups_count=64,
    width=64,
    contexts=32,
    directions=2,
    amplitude=0.1,
    seed=717,
):
    """Validation-only repeatability gate before any larger causal confirmation."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load_checkpoint(checkpoint)
    model.float().eval()
    part = ParameterPartition(model, "swiglu")
    corpus = torch.load(data / "corpus.pt", weights_only=False)
    if not corpus.get("parallel_examples"):
        raise ValueError("this audit requires aligned multilingual source segments")
    source = torch.load(source_path, weights_only=False, mmap=True)
    target = torch.load(target_path, weights_only=False, mmap=True)
    for payload in (source, target):
        if payload["checkpoint"] != str(checkpoint.resolve()) or payload["data"] != str(
            data.resolve()
        ):
            raise ValueError("profile provenance mismatch")
    if target.get("start") != source["samples"]:
        raise ValueError("source and reference profiles must be disjoint")
    features = torch.load(data / "token_features.pt", weights_only=False)["features"]
    bundles = mlp_bundles(part, width)
    rng = torch.Generator().manual_seed(seed)
    sampled = torch.randperm(len(bundles), generator=rng)[:groups_count].tolist()
    selected = [bundles[i] for i in sampled]
    tasks = list(range(3, len(features), 4))
    available = min(len(corpus["validation"][t]) for t in tasks)
    if available < 2 * contexts:
        raise ValueError("insufficient independent validation contexts")
    indices = torch.randperm(available, generator=rng)[: 2 * contexts]
    data_a = [corpus["validation"][t][indices[:contexts]] for t in tasks]
    data_b = [corpus["validation"][t][indices[contexts:]] for t in tasks]
    kwargs = {"directions": directions, "amplitude": amplitude}
    identity = measure_bundles(
        model,
        part,
        [d[:1] for d in data_a],
        selected[:1],
        amplitude=0.0,
        directions=1,
        seed=seed + 1000,
    )
    a = measure_bundles(model, part, data_a, selected, seed=seed + 1000, **kwargs)
    b = measure_bundles(model, part, data_b, selected, seed=seed + 2000, **kwargs)
    half = measure_bundles(
        model,
        part,
        data_a,
        selected[:8],
        directions=directions,
        amplitude=amplitude / 2,
        seed=seed + 1000,
    )
    counts = torch.tensor([b["parameters"] for b in selected], dtype=torch.float64)
    effect_a, effect_b = [v.mean(-1) / counts[:, None] for v in (a, b)]

    def centered(v):
        return v - v.mean(1, keepdim=True)

    def correlations(x, y):
        return [correlation(x[:, t], y[:, t]) for t in range(len(tasks))]

    repeatability = correlations(centered(effect_a), centered(effect_b))
    shape_a, shape_b = [
        centered(normalize_profile_shape(v, counts)) for v in (effect_a, effect_b)
    ]
    shape_repeatability = correlations(shape_a, shape_b)
    ratio = a[: len(half)].mean(-1) / (4 * half.mean(-1)).clamp_min(1e-30)
    median_ratio = float(ratio.median())
    finite = [r for r in repeatability if r is not None]
    shape_finite = [r for r in shape_repeatability if r is not None]
    gate = (
        len(finite) == len(tasks)
        and sum(finite) / len(finite) >= 0.5
        and sum(r < 0 for r in finite) <= 1
        and len(shape_finite) == len(tasks)
        and sum(shape_finite) / len(shape_finite) >= 0.5
        and sum(r < 0 for r in shape_finite) <= 1
        and 0.8 <= median_ratio <= 1.2
        and float(identity.abs().max()) == 0
    )
    rows = []
    observed = [t for t in range(len(features)) if t not in tasks]
    bundle_sizes = torch.tensor([b["parameters"] for b in bundles], dtype=torch.float64)
    effect = (effect_a + effect_b) / 2
    shape_effect = centered(normalize_profile_shape(effect, counts))
    for estimator, payload in source["scores"]["swiglu"].items():
        atlas = bundle_profiles(payload["pruning"], part.sizes.cpu(), bundles)
        independent = bundle_profiles(
            target["scores"]["swiglu"][estimator]["pruning"], part.sizes.cpu(), bundles
        )
        energy = (atlas.double() * bundle_sizes[:, None]).sum(
            0, keepdim=True
        ) / bundle_sizes.sum()
        gain = query_scores(
            energy,
            features,
            "tbe",
            811,
            observed=observed,
            score_link="log",
            residual_scale=0.5,
        )
        gain = gain / energy[:, observed].mean().clamp_min(1e-30)
        for method in ("full", "tbe", "jl", "cluster", "mean", "permuted", "gain_only"):
            if method == "gain_only":
                predicted = atlas[:, observed].double().mean(1, keepdim=True) * gain
            else:
                predicted = query_scores(
                    atlas,
                    features,
                    method,
                    811,
                    observed=observed,
                    score_link="linear",
                    residual_scale=0.5,
                )
            scores = predicted[sampled][:, tasks].double()
            rows.append(
                {
                    "estimator": estimator,
                    "method": method,
                    "oracle": method == "full",
                    "raw_causal_spearman": correlations(scores, effect),
                    "task_residual_causal_spearman": [None] * len(tasks)
                    if method == "mean"
                    else correlations(centered(scores), centered(effect)),
                    "shape_residual_causal_spearman": [None] * len(tasks)
                    if method in ("mean", "gain_only")
                    else correlations(
                        centered(normalize_profile_shape(scores, counts)), shape_effect
                    ),
                    "profile_repeatability": correlations(
                        centered(atlas[sampled][:, tasks].double()),
                        centered(independent[sampled][:, tasks].double()),
                    ),
                }
            )
    result = {
        "model_source": str(checkpoint.resolve()),
        "data_source": str(data.resolve()),
        "source_profile": str(source_path.resolve()),
        "reference_profile": str(target_path.resolve()),
        "data_role": "validation",
        "tasks": tasks,
        "languages": [corpus["domains"][t] for t in tasks],
        "groups": selected,
        "bundle_width": width,
        "contexts_per_repeat": contexts,
        "directions": directions,
        "amplitude": amplitude,
        "seed": seed,
        "context_indices": indices.tolist(),
        "identity_max_kl": float(identity.abs().max()),
        "quadratic_scale_ratio_median": median_ratio,
        "effect_repeatability": repeatability,
        "shape_effect_repeatability": shape_repeatability,
        "measurement_gate_passed": gate,
        "per_segment_kl_A": a.tolist(),
        "per_segment_kl_B": b.tolist(),
        "results": rows,
        "claim_ready": False,
        "scope": "FP32 model; relative antithetic Rademacher parameter perturbations, never selected by importance. Eight midpoint-stratified output positions per segment; equal-segment mean KL. Disjoint aligned validation examples and independent directions in A/B. Residuals center within these five query languages: a measurement pilot, not the final twenty-language cold-query endpoint. No task-conditioned inference-scaling utility claim.",
    }
    save_json(output, result)
    return result


@torch.no_grad()
def audit(checkpoint, data, profile_cache, output, groups_count=32):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load_checkpoint(checkpoint)
    model.eval()
    corpus = torch.load(data / "corpus.pt", weights_only=False)
    cache = torch.load(profile_cache, weights_only=False)
    part = ParameterPartition(model, "swiglu")
    eligible = sorted(
        {
            g
            for s in part.slices
            if ".layers." in s.name
            for g in range(s.offset, s.offset + s.count)
        }
    )
    generator = torch.Generator().manual_seed(717)
    groups = torch.tensor(eligible)[
        torch.randperm(len(eligible), generator=generator)[:groups_count]
    ]
    tasks = list(range(3, len(corpus["domains"]), 4))
    indices = torch.randperm(
        min(len(corpus["validation"][t]) for t in tasks), generator=generator
    )[:16]

    def logs(blocks):
        inputs = blocks[:, :-1].cuda().long()
        logits = model(
            input_ids=inputs.clamp_min(0), attention_mask=inputs >= 0, use_cache=False
        ).logits.float()
        return logits.log_softmax(-1)

    def measure(context_indices, directions, seed):
        batches = [corpus["validation"][t][context_indices] for t in tasks]
        clean = [
            [logs(batch.cuda().long()).cpu() for batch in blocks.split(2)]
            for blocks in batches
        ]
        values = torch.zeros(len(groups), len(tasks))
        noise_rng = torch.Generator(device="cuda").manual_seed(seed)
        for row, group in enumerate(groups):
            for _ in range(directions):
                originals = []
                for spec in part.slices:
                    if spec.offset <= group < spec.offset + spec.count:
                        index = [slice(None)] * spec.parameter.ndim
                        index[spec.axis] = int(group) - spec.offset
                        target = spec.parameter[tuple(index)]
                        originals.append((target, target.clone()))
                        noise = torch.randn(
                            target.shape,
                            generator=noise_rng,
                            device="cuda",
                            dtype=torch.float32,
                        )
                        target.add_(noise.to(target.dtype), alpha=0.002)
                for t, blocks in enumerate(batches):
                    total, count = 0.0, 0
                    for reference, batch in zip(clean[t], blocks.split(2)):
                        batch = batch.cuda().long()
                        valid = batch[:, 1:] >= 0
                        total += float(
                            (stable_kl(reference.cuda(), logs(batch)) * valid).sum()
                        )
                        count += int(valid.sum())
                    values[row, t] += (
                        total / count / float(part.sizes[group]) / directions
                    )
                restore(originals)
        return values

    stages = {}
    model.bfloat16()
    stages["bf16_two_contexts_one_direction"] = measure(indices[:2], 1, 812)
    model.float()
    stages["fp32_two_contexts_one_direction"] = measure(indices[:2], 1, 812)
    stages["fp32_eight_contexts_four_directions_A"] = measure(indices[:8], 4, 812)
    stages["fp32_eight_contexts_four_directions_B"] = measure(indices[8:16], 4, 915)

    def residual(matrix):
        return matrix - matrix.mean(1, keepdim=True)

    def correlations(a, b):
        return [correlation(a[:, t], b[:, t]) for t in range(len(tasks))]

    rows = []
    for estimator, atlas in cache["source"]["swiglu"].items():
        source = atlas[groups][:, tasks]
        independent = cache["target"]["swiglu"][estimator][groups][:, tasks]
        for stage, effects in stages.items():
            rows.append(
                {
                    "estimator": estimator,
                    "stage": stage,
                    "raw_correlation": correlations(source, effects),
                    "task_residual_correlation": correlations(
                        residual(source), residual(effects)
                    ),
                    "reference_repeatability": correlations(
                        residual(source), residual(independent)
                    ),
                }
            )
    a = stages["fp32_eight_contexts_four_directions_A"]
    b = stages["fp32_eight_contexts_four_directions_B"]
    save_json(
        output,
        {
            "model_source": str(checkpoint.resolve()),
            "groups": groups.tolist(),
            "tasks": tasks,
            "context_indices": indices.tolist(),
            "results": rows,
            "effects": {k: v.tolist() for k, v in stages.items()},
            "effect_repeatability": correlations(residual(a), residual(b)),
            "scope": "validation-only numerical audit; task residual centers within these five languages/domains; no cold-start performance claim",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "data", "profile-cache", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--groups", type=int, default=32)
    p.add_argument("--multilingual-bundles", action="store_true")
    p.add_argument("--target-cache", type=Path)
    a = p.parse_args()
    if a.multilingual_bundles:
        if a.target_cache is None:
            p.error("--target-cache is required for the multilingual audit")
        multilingual_bundle_audit(
            a.checkpoint,
            a.data,
            a.profile_cache,
            a.target_cache,
            a.output,
            groups_count=a.groups,
        )
    else:
        audit(a.checkpoint, a.data, a.profile_cache, a.output, a.groups)
