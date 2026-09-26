"""Amortized feature localization in a pretrained time-series transformer.

This is the forecasting counterpart of the vision neuron-localization assay.
The stored KPSA object is unchanged: normalized squared parameter-gradient
profiles are indexed by a kernel over frozen query representations.  The
downstream action is instead exact zero-deactivation of the corresponding FF
hidden features.  A query-specific ``|gradient * activation|`` ranking is the
action-matched reference.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .kernel_sensitivity import median_squared_distance
from .timeseries_parameter_influence_v8 import (
    DEFAULT_FRACTIONS,
    DEFAULT_MODEL,
    ChronosSensitivity,
    _load_values,
    _paired_summary,
    _temporal_coordinates,
)


class ChronosForecastLossSensitivity(ChronosSensitivity):
    """Chronos sensitivity for realized negative median-forecast MSE."""

    def window(self, end: int) -> torch.Tensor:
        return self.values[
            end - self.context_length : end + self.horizon
        ]

    def context(self, window: torch.Tensor) -> torch.Tensor:
        return window[: self.context_length]

    def _forward_components(self, window: torch.Tensor):
        context = self.context(window)[None].to(self.device)
        target = window[
            self.context_length : self.context_length + self.horizon
        ].to(self.device)
        hidden, loc_scale, input_embeds, attention_mask = self.model.encode(context)
        decoded = self.model.decode(input_embeds, attention_mask, hidden)
        normalized = self.model.output_patch_embedding(decoded).view(
            1,
            self.model.num_quantiles,
            self.model.chronos_config.prediction_length,
        )
        prediction = self.model.instance_norm.inverse(
            normalized.view(1, -1), loc_scale
        ).view_as(normalized)
        median_index = int((self.model.quantiles.float() - 0.5).abs().argmin())
        median = prediction[0, median_index, : self.horizon]
        functional = -(median - target).square().mean()
        return functional, hidden[0, -1]


def _make_experiment(args, values: torch.Tensor) -> ChronosSensitivity:
    cls = (
        ChronosSensitivity
        if args.functional == "trend"
        else ChronosForecastLossSensitivity
    )
    return cls(
        args.model,
        values,
        device=args.device,
        context_length=args.context_length,
        horizon=args.horizon,
    )


def parameter_profile(
    experiment: ChronosSensitivity,
    x: torch.Tensor,
    components: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Profile a configurable but fixed part of each FF parameter group."""

    if components not in {"coupled", "incoming", "outgoing"}:
        raise ValueError(f"unsupported group components: {components}")
    experiment.model.zero_grad(set_to_none=True)
    functional, feature = experiment._forward_components(x)
    functional.backward()
    energies = []
    for _name, incoming, outgoing in experiment.ff_pairs:
        incoming_energy = incoming.weight.grad.float().square().sum(dim=1)
        if incoming.bias is not None:
            incoming_energy = incoming_energy + incoming.bias.grad.float().square()
        outgoing_energy = outgoing.weight.grad.float().square().sum(dim=0)
        if components == "incoming":
            energy = incoming_energy
        elif components == "outgoing":
            energy = outgoing_energy
        else:
            energy = incoming_energy + outgoing_energy
        energies.append(energy.detach().cpu())
    energy = torch.cat(energies).double()
    return (
        F.normalize(feature.detach().cpu().double(), dim=0),
        (energy / energy.sum().clamp_min(1e-30)).float(),
    )


def _scope_layers(experiment: ChronosSensitivity, scope: str) -> list[int]:
    if scope == "all":
        return list(range(len(experiment.ff_pairs)))
    if scope.startswith("layer"):
        layer = int(scope.removeprefix("layer"))
        if not 0 <= layer < len(experiment.ff_pairs):
            raise ValueError(f"invalid layer scope: {scope}")
        return [layer]
    raise ValueError(f"unsupported scope: {scope}")


def activation_attribution(
    experiment: ChronosSensitivity,
    x: torch.Tensor,
    scope: str,
    *,
    signed_degradation: bool = False,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Return functional, query representation, and action-matched scores.

    The exact action sets selected inputs of an FF output projection to zero.
    Its first-order functional change is ``gradient * activation`` summed over
    every token.  The forecasting functional has no preferred sign, so ranking
    uses the absolute predicted change.
    """

    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = _scope_layers(experiment, scope)
    for layer in layers:
        outgoing = experiment.ff_pairs[layer][2]

        def capture(_module, inputs, layer=layer):
            captured[layer] = inputs[0]

        handles.append(outgoing.register_forward_pre_hook(capture))
    try:
        experiment.model.zero_grad(set_to_none=True)
        functional, representation = experiment._forward_components(x)
        activations = tuple(captured[layer] for layer in layers)
        gradients = torch.autograd.grad(functional, activations)
        scores = torch.zeros(experiment.groups, dtype=torch.float64)
        for layer, activation, gradient in zip(layers, activations, gradients):
            dims = tuple(range(activation.ndim - 1))
            local = (gradient.detach().float() * activation.detach().float()).sum(
                dim=dims
            )
            start = layer * experiment.units_per_layer
            local = local if signed_degradation else local.abs()
            scores[start : start + experiment.units_per_layer] = local.double().cpu()
        return (
            float(functional.detach()),
            F.normalize(representation.detach().cpu().double(), dim=0),
            scores,
        )
    finally:
        for handle in handles:
            handle.remove()
        experiment.model.zero_grad(set_to_none=True)


@contextmanager
def zero_activation_groups(
    experiment: ChronosSensitivity,
    selected: torch.Tensor,
) -> Iterator[None]:
    """Zero selected FF hidden features while preserving model parameters."""

    selected = selected.detach().long().cpu()
    handles = []
    for layer, (_, _incoming, outgoing) in enumerate(experiment.ff_pairs):
        start = layer * experiment.units_per_layer
        local = selected[(selected >= start) & (selected < start + experiment.units_per_layer)]
        local = local - start
        if not len(local):
            continue

        def intervene(_module, inputs, local=local):
            activation = inputs[0].clone()
            activation[..., local.to(activation.device)] = 0
            return (activation, *inputs[1:])

        handles.append(outgoing.register_forward_pre_hook(intervene))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def query_representation(
    experiment: ChronosSensitivity,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute only the frozen encoder coordinate needed for a KPSA query."""

    context = (
        experiment.context(x) if hasattr(experiment, "context") else x
    )[None].to(experiment.device)
    hidden, _loc_scale, _input_embeds, _attention_mask = experiment.model.encode(
        context
    )
    return F.normalize(hidden[0, -1].float(), dim=0)


def _method_scores(
    atlas: dict,
    query_features: torch.Tensor,
    query_categories: torch.Tensor,
    *,
    scale: float,
) -> dict[str, torch.Tensor]:
    source_features = atlas["source_features"].double()
    source_profiles = atlas["source_profiles"].double()
    source_categories = atlas["source_categories"].double()
    query_features = query_features.double()
    query_categories = query_categories.double()
    permutation = atlas["permutation"].long()
    denominator = 2 * float(atlas["median_squared_distance"]) * scale**2
    kernel = torch.exp(
        -torch.cdist(source_features, query_features).square() / denominator
    )
    shuffled = torch.exp(
        -torch.cdist(source_features[permutation], query_features).square()
        / denominator
    )
    categorical_kernel = source_categories @ query_categories.T
    similarity = source_features @ query_features.T
    return {
        "exact_rbf": source_profiles @ kernel / len(source_features),
        "matched_shuffle": source_profiles @ shuffled / len(source_features),
        "nearest": source_profiles[:, similarity.argmax(dim=0)],
        "scalar": source_profiles.mean(dim=1, keepdim=True).expand(
            -1, len(query_features)
        ),
        "temporal_categorical": (
            source_profiles @ categorical_kernel / len(source_features)
        ),
    }


def aggregate_feature_bundles(
    scores: torch.Tensor,
    *,
    units_per_layer: int,
    bundle_size: int,
) -> torch.Tensor:
    """Aggregate contiguous, within-layer feature scores into fixed bundles."""

    if bundle_size < 1 or units_per_layer % bundle_size:
        raise ValueError("bundle size must be a positive divisor of layer width")
    if scores.shape[0] % units_per_layer:
        raise ValueError("score rows must contain complete FF layers")
    layers = scores.shape[0] // units_per_layer
    trailing = scores.shape[1:]
    return scores.reshape(
        layers, units_per_layer // bundle_size, bundle_size, *trailing
    ).sum(dim=2).reshape(layers * (units_per_layer // bundle_size), *trailing)


def expand_bundle_selection(
    selected: torch.Tensor,
    *,
    units_per_layer: int,
    bundle_size: int,
) -> torch.Tensor:
    """Expand global bundle indices into global fine-feature indices."""

    bundles_per_layer = units_per_layer // bundle_size
    fine = []
    for value in selected.detach().long().cpu().tolist():
        layer, bundle = divmod(value, bundles_per_layer)
        start = layer * units_per_layer + bundle * bundle_size
        fine.extend(range(start, start + bundle_size))
    return torch.tensor(fine, dtype=torch.long)


def bundle_scope(
    experiment: ChronosSensitivity,
    scope: str,
    bundle_size: int,
) -> torch.Tensor:
    bundles_per_layer = experiment.units_per_layer // bundle_size
    if scope == "all":
        return torch.arange(len(experiment.ff_pairs) * bundles_per_layer)
    layer = _scope_layers(experiment, scope)[0]
    start = layer * bundles_per_layer
    return torch.arange(start, start + bundles_per_layer)


def _summarize(
    records: list[dict],
    methods: tuple[str, ...],
    fractions: tuple[float, ...],
) -> tuple[dict, dict]:
    summaries, comparisons = {}, {}
    for fraction in fractions:
        key = f"{fraction:g}"
        values = {
            method: [
                float(row["causal_effect"])
                for row in records
                if row["fraction"] == fraction and row["method"] == method
            ]
            for method in methods
        }
        summaries[key] = {
            method: _paired_summary(method_values)
            for method, method_values in values.items()
        }
        comparisons[key] = {"exact_rbf": {}}
        for baseline in methods:
            if baseline == "exact_rbf":
                continue
            differences = [
                left - right
                for left, right in zip(values["exact_rbf"], values[baseline])
            ]
            comparisons[key]["exact_rbf"][f"vs_{baseline}"] = _paired_summary(
                differences
            )
        scalar = summaries[key]["scalar"]["mean"]
        oracle = summaries[key]["activation_attribution"]["mean"]
        current = summaries[key]["exact_rbf"]["mean"]
        comparisons[key]["exact_rbf"]["activation_gap_recovered"] = (
            (current - scalar) / (oracle - scalar) if oracle > scalar else None
        )
    return summaries, comparisons


def run_profile(args) -> dict:
    frame, values = _load_values(args)
    experiment = _make_experiment(args, values)
    source_ends = (
        np.linspace(args.source_start, args.source_stop, args.source_count)
        .round()
        .astype(int)
    )
    features, profiles = [], []
    started = time.time()
    for position, end in enumerate(source_ends):
        feature, profile = parameter_profile(
            experiment,
            experiment.window(int(end)),
            args.group_components,
        )
        features.append(feature)
        profiles.append(profile)
        if (position + 1) % 100 == 0:
            print(
                "source profile",
                position + 1,
                "elapsed",
                round(time.time() - started, 1),
                flush=True,
            )
    source_features = torch.stack(features)
    source_profiles = torch.stack(profiles).T
    source_categories = _temporal_coordinates(frame, source_ends).float()
    median_distance = median_squared_distance(source_features)
    permutation = torch.randperm(
        len(source_features), generator=torch.Generator().manual_seed(args.seed + 17)
    )
    payload = {
        "setting": "chronos_bolt_feature_localization_atlas",
        "model": args.model,
        "dataset": args.dataset_name,
        "target_column": args.target_column,
        "source_examples": len(source_ends),
        "source_end_range": [args.source_start, args.source_stop],
        "functional": args.functional,
        "group_components": args.group_components,
        "groups": experiment.groups,
        "representation": "final encoder regression-token state",
        "median_squared_distance": median_distance,
        "source_profile_seconds": time.time() - started,
    }
    args.atlas_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_ends": torch.from_numpy(source_ends),
            "source_features": source_features.float(),
            "source_profiles": source_profiles.float(),
            "source_categories": source_categories,
            "median_squared_distance": median_distance,
            "permutation": permutation,
            "functional": args.functional,
            "group_components": args.group_components,
        },
        args.atlas_cache,
    )
    return payload


def run_localization(args) -> dict:
    frame, values = _load_values(args)
    experiment = _make_experiment(args, values)
    atlas = torch.load(args.atlas_cache, map_location="cpu", weights_only=True)
    if atlas.get("functional", "trend") != args.functional:
        raise ValueError("atlas and query functional do not match")
    query_ends = (
        np.linspace(args.query_start, args.query_stop, args.query_count)
        .round()
        .astype(int)
    )
    query_features, attributions, baselines = [], [], []
    started = time.time()
    for query, end in enumerate(query_ends):
        baseline, feature, scores = activation_attribution(
            experiment,
            experiment.window(int(end)),
            args.scope,
            signed_degradation=args.functional == "neg_mse",
        )
        baselines.append(baseline)
        query_features.append(feature)
        attributions.append(scores)
        if (query + 1) % 25 == 0:
            print(
                "activation attribution",
                query + 1,
                "elapsed",
                round(time.time() - started, 1),
                flush=True,
            )
    query_features = torch.stack(query_features)
    attributions = torch.stack(attributions).T
    query_categories = _temporal_coordinates(frame, query_ends)
    predictions = _method_scores(
        atlas,
        query_features,
        query_categories,
        scale=args.scale,
    )
    predictions["activation_attribution"] = attributions
    methods = tuple(predictions)
    predictions = {
        method: aggregate_feature_bundles(
            scores,
            units_per_layer=experiment.units_per_layer,
            bundle_size=args.bundle_size,
        )
        for method, scores in predictions.items()
    }
    scope = bundle_scope(experiment, args.scope, args.bundle_size)
    records = []
    started = time.time()
    for query, end in enumerate(query_ends):
        x = experiment.window(int(end))
        baseline = baselines[query]
        for fraction in args.fractions:
            count = max(1, math.ceil(fraction * len(scope)))
            for method, scores in predictions.items():
                local = torch.topk(scores[scope, query], count).indices
                selected_bundles = scope[local].sort().values
                selected = expand_bundle_selection(
                    selected_bundles,
                    units_per_layer=experiment.units_per_layer,
                    bundle_size=args.bundle_size,
                )
                with zero_activation_groups(experiment, selected):
                    intervened = experiment.functional(x)
                change = baseline - intervened
                causal_effect = abs(change) if args.functional == "trend" else change
                records.append(
                    {
                        "query": query,
                        "end": int(end),
                        "method": method,
                        "fraction": fraction,
                        "selected_groups": count,
                        "selected_features": len(selected),
                        "functional_before": baseline,
                        "functional_after": intervened,
                        "signed_functional_change": change,
                        "absolute_functional_change": abs(change),
                        "squared_functional_change": change**2,
                        "causal_effect": causal_effect,
                    }
                )
        print(
            "exact deactivation",
            query + 1,
            "/",
            args.query_count,
            "elapsed",
            round(time.time() - started, 1),
            flush=True,
        )
    summaries, comparisons = _summarize(records, methods, args.fractions)
    return {
        "setting": "chronos_bolt_amortized_feature_localization",
        "model": args.model,
        "parameters": sum(parameter.numel() for parameter in experiment.model.parameters()),
        "dataset": args.dataset_name,
        "data": str(args.data),
        "target_column": args.target_column,
        "groups": experiment.groups,
        "scope": args.scope,
        "scoped_groups": len(scope),
        "protocol": {
            "source_examples": len(atlas["source_features"]),
            "query_examples": args.query_count,
            "query_end_range": [args.query_start, args.query_stop],
            "context_length": args.context_length,
            "forecast_horizon": args.horizon,
            "functional": (
                "median mean forecast minus last observed value"
                if args.functional == "trend"
                else "negative median-forecast mean squared error"
            ),
            "representation": "final encoder regression-token state",
            "sensitivity": "normalized squared parameter-gradient group energy",
            "grouping": "coupled incoming row and outgoing column for every encoder and decoder FF feature",
            "rbf_scale": args.scale,
            "fractions": args.fractions,
            "action": "zero selected FF hidden features at the input of their output projection",
            "functional_variant": args.functional,
            "group_components": atlas.get("group_components", "coupled"),
            "bundle_size": args.bundle_size,
            "primary_metric": (
                "absolute change in the forecast functional"
                if args.functional == "trend"
                else "degradation in negative median-forecast MSE"
            ),
            "action_matched_reference": (
                "absolute gradient-times-activation summed over tokens"
                if args.functional == "trend"
                else "signed gradient-times-activation degradation summed over tokens"
            ),
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "records": records,
    }


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure(callable_, repeats: int) -> tuple[dict, int]:
    times, peaks = [], []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _synchronize()
        started = time.perf_counter()
        callable_()
        _synchronize()
        times.append((time.perf_counter() - started) * 1000)
        peaks.append(torch.cuda.max_memory_allocated())
    return _paired_summary(times), int(max(peaks))


def run_benchmark(args) -> dict:
    _frame, values = _load_values(args)
    experiment = _make_experiment(args, values)
    atlas = torch.load(args.atlas_cache, map_location="cpu", weights_only=True)
    if atlas.get("functional", "trend") != args.functional:
        raise ValueError("atlas and query functional do not match")
    scope = bundle_scope(experiment, args.scope, args.bundle_size)
    source_features = atlas["source_features"].float().to(experiment.device)
    bundled_profiles = aggregate_feature_bundles(
        atlas["source_profiles"],
        units_per_layer=experiment.units_per_layer,
        bundle_size=args.bundle_size,
    )
    source_profiles = bundled_profiles[scope].float().to(experiment.device)
    denominator = 2 * float(atlas["median_squared_distance"]) * args.scale**2
    ends = (
        np.linspace(args.query_start, args.query_stop, args.query_count)
        .round()
        .astype(int)
    )
    windows = [experiment.window(int(end)) for end in ends]
    cursor = 0

    def atlas_query():
        nonlocal cursor
        representation = query_representation(
            experiment, windows[cursor % len(windows)]
        )
        kernel = torch.exp(
            -torch.cdist(source_features, representation[None]).square()
            / denominator
        )
        scores = source_profiles @ kernel / len(source_features)
        torch.topk(scores[:, 0], max(1, math.ceil(args.fractions[-1] * len(scope))))
        cursor += 1

    def attribution_query():
        nonlocal cursor
        _baseline, _feature, scores = activation_attribution(
            experiment,
            windows[cursor % len(windows)],
            args.scope,
            signed_degradation=args.functional == "neg_mse",
        )
        bundled = aggregate_feature_bundles(
            scores,
            units_per_layer=experiment.units_per_layer,
            bundle_size=args.bundle_size,
        )
        torch.topk(
            bundled[scope], max(1, math.ceil(args.fractions[-1] * len(scope)))
        )
        cursor += 1

    for _ in range(args.warmup):
        atlas_query()
        attribution_query()
    cursor = 0
    atlas_timing, atlas_peak = _measure(atlas_query, args.query_count)
    cursor = 0
    attribution_timing, attribution_peak = _measure(
        attribution_query, args.query_count
    )
    savings = attribution_timing["mean"] - atlas_timing["mean"]
    break_even = (
        args.source_profile_seconds * 1000 / savings if savings > 0 else None
    )
    return {
        "setting": "chronos_bolt_feature_localization_cost",
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
        "protocol": {
            "queries": args.query_count,
            "warmup_repeats": args.warmup,
            "scope": args.scope,
            "scoped_groups": len(scope),
            "bundle_size": args.bundle_size,
            "source_examples": len(source_features),
            "atlas_query_includes": [
                "Chronos encoder forward",
                "exact RBF responses to all source coordinates",
                "stored scoped-atlas matrix product",
                "top-k selection",
            ],
            "activation_attribution": "optimized action-matched gradient-times-activation over the same FF scope",
        },
        "online": {
            "exact_kpsa": {
                "milliseconds_per_query": atlas_timing,
                "peak_allocated_bytes": atlas_peak,
                "persistent_atlas_bytes": int(
                    source_features.numel() * source_features.element_size()
                    + source_profiles.numel() * source_profiles.element_size()
                ),
            },
            "activation_attribution": {
                "milliseconds_per_query": attribution_timing,
                "peak_allocated_bytes": attribution_peak,
            },
        },
        "offline": {
            "source_profile_seconds": args.source_profile_seconds,
            "break_even_queries_from_profile_cost_only": break_even,
            "note": "Source profiling time is the measured 1,600-window Chronos pass; data loading is excluded.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("profile", "localize", "benchmark"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--normalization-end", type=int, required=True)
    parser.add_argument("--source-start", type=int, default=600)
    parser.add_argument("--source-stop", type=int, default=8400)
    parser.add_argument("--source-count", type=int, default=1600)
    parser.add_argument("--query-start", type=int, required=True)
    parser.add_argument("--query-stop", type=int, required=True)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=260926)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atlas-cache", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--scope", default="layer11")
    parser.add_argument("--bundle-size", type=int, default=1)
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--source-profile-seconds", type=float, default=22.5)
    parser.add_argument(
        "--functional", choices=("trend", "neg_mse"), default="trend"
    )
    parser.add_argument(
        "--group-components",
        choices=("coupled", "incoming", "outgoing"),
        default="coupled",
    )
    args = parser.parse_args()
    args.fractions = tuple(args.fractions)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.stage == "profile":
        result = run_profile(args)
    elif args.stage == "localize":
        result = run_localization(args)
    else:
        result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
