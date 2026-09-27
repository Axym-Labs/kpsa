"""Cross-modal parameter-sensitivity atlases for pretrained forecasters.

This module applies the same kernel-atlas and exact antithetic parameter-
influence protocol used by the vision experiments to Chronos-Bolt.  The
forecasting representation, RBF kernel, scalar/matched-shuffle/nearest
controls, coupled FF-feature partition, and causal estimator are explicit so
the time-series result can be reproduced without experiment-specific glue.

Install the optional dependency with ``pip install .[timeseries]``.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, t

from .common import save_json, seed_everything

DEFAULT_MODEL = "amazon/chronos-bolt-small"
DEFAULT_SCALES = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
DEFAULT_FRACTIONS = (0.0002, 0.001)


def _load_pipeline(model_name: str, device: str):
    try:
        from chronos import BaseChronosPipeline
    except ImportError as error:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "Chronos is required; install this project with the timeseries extra"
        ) from error
    return BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.float32,
    )


def _paired_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": 1}
    half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    return {
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n": len(array),
    }


def _group_seed(seed: int, group: int, part: int) -> int:
    modulus = 2**63 - 25
    return int(
        (
            seed * 6_364_136_223_846_793_005
            + (group + 1) * 2_862_933_555_777_941_757
            + (part + 1) * 1_442_695_040_888_963_407
        )
        % modulus
    )


def _temporal_coordinates(frame: pd.DataFrame, ends: np.ndarray) -> torch.Tensor:
    coordinates = []
    for timestamp in frame.iloc[ends - 1]["date"]:
        coordinate = torch.zeros(24 + 7 + 12, dtype=torch.float64)
        coordinate[timestamp.hour] = 1
        coordinate[24 + timestamp.dayofweek] = 1
        coordinate[31 + timestamp.month - 1] = 1
        coordinates.append(coordinate / math.sqrt(3))
    return torch.stack(coordinates)


class ChronosSensitivity:
    """Chronos functional, representation, and coupled FF-feature partition."""

    def __init__(
        self,
        model_name: str,
        values: torch.Tensor,
        *,
        device: str = "cuda",
        context_length: int = 512,
        horizon: int = 24,
    ):
        self.pipeline = _load_pipeline(model_name, device)
        self.model = self.pipeline.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.values = values
        self.device = torch.device(device)
        self.context_length = context_length
        self.horizon = horizon
        self.ff_pairs = []
        for stack_name in ("encoder", "decoder"):
            stack = getattr(self.model, stack_name)
            for layer_index, block in enumerate(stack.block):
                dense = block.layer[-1].DenseReluDense
                self.ff_pairs.append(
                    (f"{stack_name}.{layer_index}", dense.wi, dense.wo)
                )
        self.units_per_layer = int(self.ff_pairs[0][1].weight.shape[0])
        self.groups = len(self.ff_pairs) * self.units_per_layer

    def window(self, end: int) -> torch.Tensor:
        return self.values[end - self.context_length : end]

    def _forward_components(self, x: torch.Tensor):
        context = x[None].to(self.device)
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
        functional = prediction[0, median_index, : self.horizon].mean() - context[0, -1]
        return functional, hidden[0, -1]

    def profile(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        functional, feature = self._forward_components(x)
        functional.backward()
        energies = []
        for _, incoming, outgoing in self.ff_pairs:
            energy = incoming.weight.grad.float().square().sum(dim=1)
            if incoming.bias is not None:
                energy = energy + incoming.bias.grad.float().square()
            energy = energy + outgoing.weight.grad.float().square().sum(dim=0)
            energies.append(energy.detach().cpu())
        energy = torch.cat(energies).double()
        representation = F.normalize(feature.detach().cpu().double(), dim=0)
        return representation, (energy / energy.sum()).float()

    @torch.no_grad()
    def functional(self, x: torch.Tensor) -> float:
        return float(self.functionals(x[None])[0])

    @torch.no_grad()
    def functionals(self, contexts: torch.Tensor) -> torch.Tensor:
        """Evaluate the scalar forecasting functional for a context batch."""
        contexts = contexts.to(self.device)
        hidden, loc_scale, input_embeds, attention_mask = self.model.encode(contexts)
        decoded = self.model.decode(input_embeds, attention_mask, hidden)
        batch = len(contexts)
        normalized = self.model.output_patch_embedding(decoded).view(
            batch,
            self.model.num_quantiles,
            self.model.chronos_config.prediction_length,
        )
        prediction = self.model.instance_norm.inverse(
            normalized.view(batch, -1), loc_scale
        ).view_as(normalized)
        median_index = int((self.model.quantiles.float() - 0.5).abs().argmin())
        return (
            prediction[:, median_index, : self.horizon].mean(dim=1)
            - contexts[:, -1]
        ).float().cpu()

    @torch.no_grad()
    def parameter_rms(self) -> float:
        squared = torch.zeros((), device=self.device, dtype=torch.float64)
        count = 0
        for _, incoming, outgoing in self.ff_pairs:
            for parameter in (incoming.weight, outgoing.weight):
                squared += parameter.double().square().sum()
                count += parameter.numel()
        return float((squared / count).sqrt())

    @contextmanager
    def additive_noise(
        self,
        selected: torch.Tensor,
        standard_deviation: float,
        *,
        seed: int,
        sign: int,
    ):
        changes = []
        try:
            with torch.no_grad():
                for group in selected.tolist():
                    layer_index, unit = divmod(int(group), self.units_per_layer)
                    _, incoming, outgoing = self.ff_pairs[layer_index]
                    views = (incoming.weight[unit], outgoing.weight[:, unit])
                    for part, view in enumerate(views):
                        original = view.detach().clone()
                        generator = torch.Generator(device=view.device).manual_seed(
                            _group_seed(seed, group, part)
                        )
                        noise = torch.empty_like(view)
                        noise.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
                        view.add_(noise, alpha=standard_deviation * sign)
                        changes.append((view, original))
            yield
        finally:
            with torch.no_grad():
                for view, original in reversed(changes):
                    view.copy_(original)


def _load_values(args) -> tuple[pd.DataFrame, torch.Tensor]:
    frame = pd.read_csv(args.data, parse_dates=["date"])
    raw = torch.tensor(frame[args.target_column].values, dtype=torch.float32)
    mean = raw[: args.normalization_end].mean()
    std = raw[: args.normalization_end].std().clamp_min(1e-6)
    return frame, (raw - mean) / std


def _collect(
    experiment: ChronosSensitivity,
    ends: np.ndarray,
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, profiles = [], []
    started = time.time()
    for position, end in enumerate(ends):
        feature, profile = experiment.profile(experiment.window(int(end)))
        features.append(feature)
        profiles.append(profile)
        if (position + 1) % 100 == 0:
            print(label, position + 1, "elapsed", round(time.time() - started, 1))
    return F.normalize(torch.stack(features).double(), dim=1), torch.stack(profiles).T


def _profile_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    top_counts: tuple[int, ...],
) -> dict:
    correlations = []
    coverages = {count: [] for count in top_counts}
    for query in range(truth.shape[1]):
        correlations.append(
            float(
                spearmanr(
                    prediction[:, query].numpy(), truth[:, query].numpy()
                ).statistic
            )
        )
        for count in top_counts:
            selected = set(torch.topk(prediction[:, query], count).indices.tolist())
            exact = set(torch.topk(truth[:, query], count).indices.tolist())
            coverages[count].append(len(selected & exact) / count)
    result = {
        "spearman_mean": float(np.nanmean(correlations)),
        "spearman_by_query": correlations,
    }
    for count, values in coverages.items():
        result[f"coverage{count}_mean"] = float(np.mean(values))
        result[f"coverage{count}_by_query"] = values
    return result


def _scope_development(
    predictions: dict[str, torch.Tensor],
    truth: torch.Tensor,
    *,
    scale: float,
    units_per_layer: int,
    fractions: tuple[float, ...],
) -> dict:
    """Select a named FF scope from held-out direct-gradient coverage."""
    groups = truth.shape[0]
    layers = groups // units_per_layer
    scopes = {
        "all": torch.arange(groups),
        "encoder": torch.arange(groups // 2),
        "decoder": torch.arange(groups // 2, groups),
    }
    scopes.update(
        {
            f"layer{layer}": torch.arange(
                layer * units_per_layer, (layer + 1) * units_per_layer
            )
            for layer in range(layers)
        }
    )
    method_scores = {
        "exact_rbf": predictions[f"latent_rbf_{scale:g}"],
        "matched_shuffle": predictions[f"latent_rbf_{scale:g}_shuffled"],
        "nearest": predictions["nearest"],
        "scalar": predictions["scalar"],
        "temporal_categorical": predictions["temporal_categorical"],
        "direct_gradient": truth,
    }
    results = {}
    eligible = []
    for scope_name, scope in scopes.items():
        scope_result = {}
        gap_recoveries = []
        passes = []
        for fraction in fractions:
            count = max(1, math.ceil(fraction * len(scope)))
            coverages = {}
            for method, scores in method_scores.items():
                values = []
                for query in range(truth.shape[1]):
                    local = torch.topk(scores[scope, query], count).indices
                    selected = scope[local]
                    values.append(
                        float(truth[selected, query].sum() / truth[scope, query].sum())
                    )
                coverages[method] = values
            scalar = np.asarray(coverages["scalar"])
            direct = np.asarray(coverages["direct_gradient"])
            exact = np.asarray(coverages["exact_rbf"])
            denominator = float(direct.mean() - scalar.mean())
            gap_recovery = (
                float((exact.mean() - scalar.mean()) / denominator)
                if denominator > 0
                else None
            )
            comparisons = {}
            for baseline in (
                "scalar",
                "matched_shuffle",
                "nearest",
                "temporal_categorical",
            ):
                comparisons[f"vs_{baseline}"] = _paired_summary(
                    (exact - np.asarray(coverages[baseline])).tolist()
                )
            scope_result[f"{fraction:g}"] = {
                "selected_groups": count,
                "coverage": {
                    method: _paired_summary(values)
                    for method, values in coverages.items()
                },
                "comparisons": comparisons,
                "oracle_gap_recovered": gap_recovery,
            }
            gap_recoveries.append(
                gap_recovery if gap_recovery is not None else -math.inf
            )
            passes.append(
                comparisons["vs_scalar"]["ci95_low"] > 0
                and comparisons["vs_matched_shuffle"]["ci95_low"] > 0
            )
        score = float(np.mean(gap_recoveries))
        scope_result["selection_score_mean_gap_recovery"] = score
        scope_result["passes_scalar_and_shuffle_at_all_budgets"] = all(passes)
        results[scope_name] = scope_result
        if scope_name.startswith("layer") and all(passes):
            eligible.append((score, scope_name))
    selected_scope = max(eligible)[1] if eligible else None
    return {
        "selection_metric": (
            "highest mean oracle-gap recovery across frozen fractions among "
            "individual layers whose paired coverage gain over scalar and "
            "matched shuffle has positive 95% intervals at every fraction"
        ),
        "selected_scope": selected_scope,
        "scopes": results,
    }


def run_profile(args) -> None:
    frame, values = _load_values(args)
    experiment = ChronosSensitivity(
        args.model,
        values,
        device=args.device,
        context_length=args.context_length,
        horizon=args.horizon,
    )
    source_ends = (
        np.linspace(args.source_start, args.source_stop, args.source_count)
        .round()
        .astype(int)
    )
    query_ends = (
        np.linspace(args.query_start, args.query_stop, args.query_count)
        .round()
        .astype(int)
    )
    source_features, source_profiles = _collect(experiment, source_ends, label="source")
    query_features, query_profiles = _collect(experiment, query_ends, label="query")
    distances = torch.pdist(source_features).square()
    median_distance = float(distances[distances > 0].median())
    permutation = torch.randperm(
        args.source_count,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    similarity = source_features @ query_features.T
    source_categories = _temporal_coordinates(frame, source_ends)
    query_categories = _temporal_coordinates(frame, query_ends)
    categorical_kernel = source_categories @ query_categories.T
    predictions = {
        "scalar": source_profiles.mean(dim=1, keepdim=True).expand(
            -1, args.query_count
        ),
        "nearest": source_profiles[:, similarity.argmax(dim=0)],
        "temporal_categorical": (
            source_profiles.double() @ categorical_kernel / args.source_count
        ),
    }
    for scale in args.scales:
        denominator = 2 * median_distance * scale**2
        kernel = torch.exp(
            -torch.cdist(source_features, query_features).square() / denominator
        )
        shuffled = torch.exp(
            -torch.cdist(source_features[permutation], query_features).square()
            / denominator
        )
        predictions[f"latent_rbf_{scale:g}"] = (
            source_profiles.double() @ kernel / args.source_count
        )
        predictions[f"latent_rbf_{scale:g}_shuffled"] = (
            source_profiles.double() @ shuffled / args.source_count
        )
    metrics = {
        name: _profile_metrics(
            prediction.float(), query_profiles, tuple(args.top_counts)
        )
        for name, prediction in predictions.items()
    }
    scope_development = _scope_development(
        predictions,
        query_profiles,
        scale=args.scale,
        units_per_layer=experiment.units_per_layer,
        fractions=tuple(args.fractions),
    )
    payload = {
        "setting": "chronos_bolt_time_series_profile_retrieval",
        "model": args.model,
        "dataset": args.dataset_name,
        "data": str(args.data),
        "target_column": args.target_column,
        "source_examples": args.source_count,
        "query_examples": args.query_count,
        "source_end_range": [args.source_start, args.source_stop],
        "query_end_range": [args.query_start, args.query_stop],
        "context_length": args.context_length,
        "forecast_horizon": args.horizon,
        "groups": experiment.groups,
        "grouping": "coupled incoming row and outgoing column for every encoder and decoder FF feature",
        "representation": "final encoder regression-token state",
        "median_squared_distance": median_distance,
        "metrics": metrics,
        "scope_development": scope_development,
    }
    save_json(args.output, payload)
    if args.atlas_cache:
        args.atlas_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_ends": torch.from_numpy(source_ends),
                "source_features": source_features.float(),
                "source_profiles": source_profiles.float(),
                "source_categories": source_categories.float(),
                "median_squared_distance": median_distance,
                "permutation": permutation,
                "query_ends": torch.from_numpy(query_ends),
                "query_features": query_features.float(),
                "query_profiles": query_profiles.float(),
                "units_per_layer": experiment.units_per_layer,
            },
            args.atlas_cache,
        )


def _scope_indices(experiment: ChronosSensitivity, scope: str) -> torch.Tensor:
    if scope == "all":
        return torch.arange(experiment.groups)
    if scope.startswith("layer"):
        layer = int(scope.removeprefix("layer"))
        if not 0 <= layer < len(experiment.ff_pairs):
            raise ValueError(f"invalid layer scope: {scope}")
        start = layer * experiment.units_per_layer
        return torch.arange(start, start + experiment.units_per_layer)
    raise ValueError(f"unsupported scope: {scope}")


def run_causal(args) -> None:
    frame, values = _load_values(args)
    experiment = ChronosSensitivity(
        args.model,
        values,
        device=args.device,
        context_length=args.context_length,
        horizon=args.horizon,
    )
    atlas = torch.load(args.atlas_cache, map_location="cpu", weights_only=True)
    source_features = atlas["source_features"].double()
    source_profiles = atlas["source_profiles"].double()
    source_categories = atlas["source_categories"].double()
    permutation = atlas["permutation"]
    median_distance = float(atlas["median_squared_distance"])
    query_ends = (
        np.linspace(args.query_start, args.query_stop, args.query_count)
        .round()
        .astype(int)
    )
    query_features, query_profiles = _collect(
        experiment, query_ends, label="query gradients"
    )
    query_categories = _temporal_coordinates(frame, query_ends)
    denominator = 2 * median_distance * args.scale**2
    kernel = torch.exp(
        -torch.cdist(source_features, query_features).square() / denominator
    )
    shuffled = torch.exp(
        -torch.cdist(source_features[permutation], query_features).square()
        / denominator
    )
    categorical_kernel = source_categories @ query_categories.T
    similarity = source_features @ query_features.T
    methods = {
        "exact_rbf": source_profiles @ kernel / len(source_features),
        "matched_shuffle": source_profiles @ shuffled / len(source_features),
        "nearest": source_profiles[:, similarity.argmax(dim=0)],
        "scalar": source_profiles.mean(dim=1, keepdim=True).expand(
            -1, args.query_count
        ),
        "temporal_categorical": (
            source_profiles @ categorical_kernel / len(source_features)
        ),
        "direct_gradient": query_profiles.double(),
    }
    scope = _scope_indices(experiment, args.scope)
    parameter_rms = experiment.parameter_rms()
    noise_std = parameter_rms * args.noise_scale
    records = []
    started = time.time()
    for query, end in enumerate(query_ends):
        x = experiment.window(int(end))
        for fraction in args.fractions:
            count = max(1, math.ceil(fraction * len(scope)))
            for method, scores in methods.items():
                local = torch.topk(scores[scope, query], count).indices
                selected = scope[local].sort().values
                coverage = float(
                    query_profiles[selected, query].sum()
                    / query_profiles[scope, query].sum()
                )
                for direction in range(args.directions):
                    intervention_seed = args.seed + query * 100_003 + direction * 1_009
                    with experiment.additive_noise(
                        selected, noise_std, seed=intervention_seed, sign=1
                    ):
                        plus = experiment.functional(x)
                    with experiment.additive_noise(
                        selected, noise_std, seed=intervention_seed, sign=-1
                    ):
                        minus = experiment.functional(x)
                    derivative = (plus - minus) / (2 * noise_std)
                    records.append(
                        {
                            "query": query,
                            "end": int(end),
                            "method": method,
                            "fraction": fraction,
                            "selected_groups": count,
                            "direction": direction,
                            "direct_gradient_energy_coverage": coverage,
                            "squared_susceptibility": derivative**2,
                        }
                    )
        print(
            "exact query",
            query + 1,
            "/",
            args.query_count,
            "elapsed",
            round(time.time() - started, 1),
        )

    def query_means(method: str, fraction: float, metric: str):
        grouped = defaultdict(list)
        for record in records:
            if record["method"] == method and record["fraction"] == fraction:
                grouped[record["query"]].append(record[metric])
        return {query: float(np.mean(values)) for query, values in grouped.items()}

    summaries, comparisons = {}, {}
    for fraction in args.fractions:
        key = f"{fraction:g}"
        summaries[key], comparisons[key] = {}, {}
        values_by_method = {}
        for method in methods:
            values = query_means(method, fraction, "squared_susceptibility")
            values_by_method[method] = values
            summaries[key][method] = _paired_summary(list(values.values()))
            coverage = query_means(method, fraction, "direct_gradient_energy_coverage")
            summaries[key][method]["gradient_energy_coverage"] = _paired_summary(
                list(coverage.values())
            )
        comparisons[key]["exact_rbf"] = {}
        for baseline in (
            "scalar",
            "matched_shuffle",
            "nearest",
            "temporal_categorical",
            "direct_gradient",
        ):
            differences = [
                values_by_method["exact_rbf"][query] - values_by_method[baseline][query]
                for query in sorted(
                    values_by_method["exact_rbf"].keys()
                    & values_by_method[baseline].keys()
                )
            ]
            comparisons[key]["exact_rbf"][f"vs_{baseline}"] = _paired_summary(
                differences
            )
        scalar_mean = summaries[key]["scalar"]["mean"]
        oracle_mean = summaries[key]["direct_gradient"]["mean"]
        comparisons[key]["exact_rbf"]["oracle_gap_recovered"] = (
            (summaries[key]["exact_rbf"]["mean"] - scalar_mean)
            / (oracle_mean - scalar_mean)
            if oracle_mean > scalar_mean
            else None
        )
    save_json(
        args.output,
        {
            "setting": "chronos_bolt_time_series_parameter_influence_circuits",
            "model": args.model,
            "parameters": sum(
                parameter.numel() for parameter in experiment.model.parameters()
            ),
            "dataset": args.dataset_name,
            "data": str(args.data),
            "target_column": args.target_column,
            "groups": experiment.groups,
            "scope": args.scope,
            "scoped_groups": len(scope),
            "protocol": {
                "source_examples": len(source_features),
                "query_examples": args.query_count,
                "query_end_range": [args.query_start, args.query_stop],
                "context_length": args.context_length,
                "forecast_horizon": args.horizon,
                "functional": "median mean forecast minus last observed value",
                "representation": "final encoder regression-token state",
                "grouping": "coupled incoming row and outgoing column for every encoder and decoder FF feature",
                "rbf_scale": args.scale,
                "fractions": args.fractions,
                "directions": args.directions,
                "noise_scale_relative_to_ff_parameter_rms": args.noise_scale,
                "ff_parameter_rms": parameter_rms,
                "coordinate_noise_standard_deviation": noise_std,
            },
            "summaries": summaries,
            "comparisons": comparisons,
            "records": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("profile", "causal"))
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
    parser.add_argument("--atlas-cache", type=Path)
    parser.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES)
    parser.add_argument("--top-counts", nargs="+", type=int, default=(5, 25, 37))
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--scope", default="layer11")
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    args = parser.parse_args()
    if args.stage == "causal" and args.atlas_cache is None:
        parser.error("causal stage requires --atlas-cache")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.stage == "profile":
        run_profile(args)
    else:
        run_causal(args)


if __name__ == "__main__":
    main()
