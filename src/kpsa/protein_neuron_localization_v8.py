"""KPSA neuron localization for protein masked modeling.

The assay mirrors the vision Application-1 contract: normalized squared
native-parameter-gradient profiles are indexed in a frozen pretrained
representation, queried on held-out examples, and evaluated by exact FF-feature
zero-deactivation against action-matched gradient-times-activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .common import save_json, seed_everything
from .kernel_sensitivity import median_squared_distance
from .timeseries_parameter_influence_v8 import _group_seed, _paired_summary

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
CANONICAL_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
REPRESENTATIONS = (
    "mask_last",
    "mask_last4",
    "sequence_mean",
    "logit_distribution",
    "mask_plus_logits",
)
COMPONENTS = ("coupled", "incoming", "outgoing")


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector.detach().float().flatten().cpu(), dim=0)


def _load_examples(path: Path, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    generator = np.random.default_rng(seed)
    generator.shuffle(rows)
    return rows


def _mask_position(example: dict) -> int:
    sequence = example["sequence"]
    candidates = [
        index
        for index, residue in enumerate(sequence)
        if residue in CANONICAL_AA and 8 <= index < len(sequence) - 8
    ]
    if not candidates:
        candidates = [
            index for index, residue in enumerate(sequence) if residue in CANONICAL_AA
        ]
    digest = hashlib.sha256(example["accession"].encode()).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


class EsmSensitivity:
    def __init__(self, model_name: str, device: str):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.ff_pairs = [
            (
                f"encoder.{index}",
                layer.intermediate.dense,
                layer.output.dense,
            )
            for index, layer in enumerate(self.model.esm.encoder.layer)
        ]
        self.units_per_layer = int(self.ff_pairs[0][1].weight.shape[0])
        self.groups = len(self.ff_pairs) * self.units_per_layer
        self.aa_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in CANONICAL_AA],
            device=self.device,
        )

    def prepare(self, example: dict) -> dict[str, torch.Tensor | int | str]:
        sequence = example["sequence"]
        residue_index = _mask_position(example)
        true_residue = sequence[residue_index]
        encoded = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        token_position = residue_index + 1
        true_id = int(input_ids[0, token_position])
        input_ids = input_ids.clone()
        input_ids[0, token_position] = self.tokenizer.mask_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_position": token_position,
            "true_id": true_id,
            "true_residue": true_residue,
        }

    def forward(
        self,
        prepared: dict[str, torch.Tensor | int | str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], bool]:
        outputs = self.model(
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            output_hidden_states=True,
        )
        position = int(prepared["token_position"])
        true_id = int(prepared["true_id"])
        logits = outputs.logits[0, position]
        competitors = logits[self.aa_ids]
        true_offset = CANONICAL_AA.index(str(prepared["true_residue"]))
        masked_competitors = competitors.clone()
        masked_competitors[true_offset] = -torch.inf
        functional = logits[true_id] - masked_competitors.max()
        hidden_states = outputs.hidden_states
        mask_last = hidden_states[-1][0, position]
        last4 = torch.cat(
            [
                _normalize(state[0, position]).to(self.device)
                for state in hidden_states[-4:]
            ]
        )
        valid = prepared["attention_mask"][0].bool()
        valid[0] = False
        valid[-1] = False
        sequence_mean = hidden_states[-1][0, valid].mean(dim=0)
        logit_distribution = competitors - competitors.mean()
        representations = {
            "mask_last": _normalize(mask_last),
            "mask_last4": _normalize(last4),
            "sequence_mean": _normalize(sequence_mean),
            "logit_distribution": _normalize(logit_distribution),
            "mask_plus_logits": _normalize(
                torch.cat(
                    (
                        _normalize(mask_last).to(self.device),
                        _normalize(logit_distribution).to(self.device),
                    )
                )
            ),
        }
        predicted_id = int(self.aa_ids[competitors.argmax()])
        return functional, representations, predicted_id == true_id

    @torch.no_grad()
    def functional(self, prepared: dict[str, torch.Tensor | int | str]) -> float:
        functional, _representations, _correct = self.forward(prepared)
        return float(functional)

    @torch.no_grad()
    def parameter_rms(self, scope: torch.Tensor, component: str) -> float:
        squared = torch.zeros((), device=self.device, dtype=torch.float64)
        count = 0
        for group in scope.tolist():
            layer_index, unit = divmod(int(group), self.units_per_layer)
            _name, incoming, outgoing = self.ff_pairs[layer_index]
            parts = []
            if component in ("incoming", "coupled"):
                parts.append(incoming.weight[unit])
                if incoming.bias is not None:
                    parts.append(incoming.bias[unit : unit + 1])
            if component in ("outgoing", "coupled"):
                parts.append(outgoing.weight[:, unit])
            for part in parts:
                squared += part.double().square().sum()
                count += part.numel()
        return float((squared / count).sqrt())

    @contextmanager
    def additive_parameter_noise(
        self,
        selected: torch.Tensor,
        standard_deviation: float,
        *,
        component: str,
        seed: int,
        sign: int,
    ):
        changes = []
        try:
            with torch.no_grad():
                for group in selected.tolist():
                    layer_index, unit = divmod(int(group), self.units_per_layer)
                    _name, incoming, outgoing = self.ff_pairs[layer_index]
                    views = []
                    if component in ("incoming", "coupled"):
                        views.append(incoming.weight[unit])
                        if incoming.bias is not None:
                            views.append(incoming.bias[unit : unit + 1])
                    if component in ("outgoing", "coupled"):
                        views.append(outgoing.weight[:, unit])
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


@contextmanager
def _capture_ff(experiment: EsmSensitivity):
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer, (_name, _incoming, outgoing) in enumerate(experiment.ff_pairs):

        def capture(_module, inputs, layer=layer):
            captured[layer] = inputs[0]

        handles.append(outgoing.register_forward_pre_hook(capture))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def zero_activation_groups(experiment: EsmSensitivity, selected: torch.Tensor):
    selected = selected.detach().long().cpu()
    handles = []
    for layer, (_name, _incoming, outgoing) in enumerate(experiment.ff_pairs):
        start = layer * experiment.units_per_layer
        local = (
            selected[
                (selected >= start) & (selected < start + experiment.units_per_layer)
            ]
            - start
        )
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


def _parameter_energies(experiment: EsmSensitivity) -> dict[str, torch.Tensor]:
    incoming_parts, outgoing_parts = [], []
    for _name, incoming, outgoing in experiment.ff_pairs:
        incoming_energy = incoming.weight.grad.float().square().sum(dim=1)
        if incoming.bias is not None:
            incoming_energy = incoming_energy + incoming.bias.grad.float().square()
        outgoing_energy = outgoing.weight.grad.float().square().sum(dim=0)
        incoming_parts.append(incoming_energy.detach().cpu())
        outgoing_parts.append(outgoing_energy.detach().cpu())
    incoming = torch.cat(incoming_parts).double()
    outgoing = torch.cat(outgoing_parts).double()
    energies = {
        "incoming": incoming,
        "outgoing": outgoing,
        "coupled": incoming + outgoing,
    }
    return {
        name: (energy / energy.sum().clamp_min(1e-30)).float()
        for name, energy in energies.items()
    }


def _source_example(experiment: EsmSensitivity, example: dict):
    prepared = experiment.prepare(example)
    experiment.model.zero_grad(set_to_none=True)
    functional, representations, correct = experiment.forward(prepared)
    functional.backward()
    profiles = _parameter_energies(experiment)
    experiment.model.zero_grad(set_to_none=True)
    return prepared, representations, profiles, correct


def _query_example(experiment: EsmSensitivity, example: dict):
    prepared = experiment.prepare(example)
    experiment.model.zero_grad(set_to_none=True)
    with _capture_ff(experiment) as captured:
        functional, representations, correct = experiment.forward(prepared)
        activations = tuple(
            captured[layer] for layer in range(len(experiment.ff_pairs))
        )
        gradients = torch.autograd.grad(functional, activations, retain_graph=True)
        attribution = torch.cat(
            [
                (gradient.detach().float() * activation.detach().float())
                .sum(dim=tuple(range(activation.ndim - 1)))
                .abs()
                .cpu()
                for activation, gradient in zip(activations, gradients)
            ]
        )
        functional.backward()
        profiles = _parameter_energies(experiment)
    experiment.model.zero_grad(set_to_none=True)
    return (
        prepared,
        float(functional.detach()),
        representations,
        attribution,
        profiles,
        correct,
    )


def _stack_representations(rows: list[dict[str, torch.Tensor]]):
    return {
        name: F.normalize(torch.stack([row[name] for row in rows]).double(), dim=1)
        for name in REPRESENTATIONS
    }


def _scope_indices(experiment: EsmSensitivity) -> dict[str, torch.Tensor]:
    width = experiment.units_per_layer
    scopes = {"all": torch.arange(experiment.groups)}
    for layer in range(len(experiment.ff_pairs)):
        scopes[f"layer{layer}"] = torch.arange(layer * width, (layer + 1) * width)
    return scopes


def _coverage(prediction, target, scope, count):
    prediction = prediction[scope]
    target = target[scope]
    selected = torch.topk(prediction, count, dim=0).indices
    return (
        (target.gather(0, selected).sum(dim=0) / target.sum(dim=0).clamp_min(1e-30))
        .detach()
        .cpu()
        .double()
        .tolist()
    )


def _screen_cell(methods, target, scope, fractions):
    cells = {}
    for fraction in fractions:
        count = max(1, math.ceil(fraction * len(scope)))
        values = {
            name: _coverage(prediction, target, scope, count)
            for name, prediction in methods.items()
        }
        values["activation_attribution"] = _coverage(target, target, scope, count)
        summaries = {name: _paired_summary(value) for name, value in values.items()}
        exact = values["exact_rbf"]
        comparisons = {
            f"vs_{baseline}": _paired_summary(
                [left - right for left, right in zip(exact, values[baseline])]
            )
            for baseline in (
                "scalar",
                "matched_shuffle",
                "nearest",
                "categorical",
                "direct_parameter",
            )
        }
        scalar = summaries["scalar"]["mean"]
        oracle = summaries["activation_attribution"]["mean"]
        comparisons["activation_gap_recovered"] = (
            (summaries["exact_rbf"]["mean"] - scalar) / (oracle - scalar)
            if oracle > scalar
            else None
        )
        comparisons["direct_parameter_activation_gap_recovered"] = (
            (summaries["direct_parameter"]["mean"] - scalar) / (oracle - scalar)
            if oracle > scalar
            else None
        )
        cells[f"{fraction:g}"] = {
            "selected_groups": count,
            "summaries": summaries,
            "comparisons": comparisons,
        }
    return cells


def run_screen(args):
    examples = _load_examples(args.data, args.seed)
    experiment = EsmSensitivity(args.model, args.device)
    source_rows = examples[: args.source_count]
    candidate_rows = examples[args.source_count :]
    source_representations = []
    source_profiles = {component: [] for component in COMPONENTS}
    source_categories = []
    started = time.time()
    for index, example in enumerate(source_rows):
        prepared, representations, profiles, _correct = _source_example(
            experiment, example
        )
        source_representations.append(representations)
        for component in COMPONENTS:
            source_profiles[component].append(profiles[component])
        category = torch.zeros(len(CANONICAL_AA))
        category[CANONICAL_AA.index(str(prepared["true_residue"]))] = 1
        source_categories.append(category)
        if (index + 1) % 100 == 0:
            print(
                "source",
                index + 1,
                "elapsed",
                round(time.time() - started, 1),
                flush=True,
            )
    query_representations, attributions, query_profiles, query_categories = (
        [],
        [],
        {c: [] for c in COMPONENTS},
        [],
    )
    query_examples = []
    for example in candidate_rows:
        prepared, _baseline, representations, attribution, profiles, correct = (
            _query_example(experiment, example)
        )
        if not correct:
            continue
        query_examples.append(example)
        query_representations.append(representations)
        attributions.append(attribution)
        for component in COMPONENTS:
            query_profiles[component].append(profiles[component])
        category = torch.zeros(len(CANONICAL_AA))
        category[CANONICAL_AA.index(str(prepared["true_residue"]))] = 1
        query_categories.append(category)
        if len(query_examples) >= args.query_count:
            break
    if len(query_examples) < args.query_count:
        raise RuntimeError(f"only {len(query_examples)} correct queries found")
    source_representations = _stack_representations(source_representations)
    query_representations = _stack_representations(query_representations)
    source_profiles = {
        name: torch.stack(rows).T for name, rows in source_profiles.items()
    }
    query_profiles = {
        name: torch.stack(rows).T.float().to(args.metric_device)
        for name, rows in query_profiles.items()
    }
    attribution = torch.stack(attributions).T.float().to(args.metric_device)
    source_categories = torch.stack(source_categories).float().to(args.metric_device)
    query_categories = torch.stack(query_categories).float().to(args.metric_device)
    permutation = torch.randperm(
        args.source_count, generator=torch.Generator().manual_seed(args.seed + 17)
    )
    scopes = _scope_indices(experiment)
    medians, ranking = {}, []
    for representation in REPRESENTATIONS:
        source = source_representations[representation]
        query = query_representations[representation]
        median = median_squared_distance(source)
        medians[representation] = median
        distances = torch.cdist(source, query).square().float().to(args.metric_device)
        shuffled_distances = (
            torch.cdist(source[permutation], query)
            .square()
            .float()
            .to(args.metric_device)
        )
        similarity = (source @ query.T).float().to(args.metric_device)
        categorical_kernel = source_categories @ query_categories.T
        for scale in args.scales:
            kernel = torch.exp(-distances / (2 * median * scale**2))
            shuffled = torch.exp(-shuffled_distances / (2 * median * scale**2))
            for component in COMPONENTS:
                profiles = source_profiles[component].to(args.metric_device)
                methods = {
                    "exact_rbf": profiles @ kernel / args.source_count,
                    "matched_shuffle": profiles @ shuffled / args.source_count,
                    "nearest": profiles[:, similarity.argmax(dim=0)],
                    "scalar": profiles.mean(dim=1, keepdim=True).expand(
                        -1, args.query_count
                    ),
                    "categorical": profiles @ categorical_kernel / args.source_count,
                    "direct_parameter": query_profiles[component],
                }
                for scope_name, scope in scopes.items():
                    cells = _screen_cell(
                        methods,
                        attribution,
                        scope.to(args.metric_device),
                        args.fractions,
                    )
                    recoveries = [
                        cell["comparisons"]["activation_gap_recovered"]
                        for cell in cells.values()
                    ]
                    passes = all(
                        cell["comparisons"][baseline]["ci95_low"] > 0
                        for cell in cells.values()
                        for baseline in ("vs_scalar", "vs_matched_shuffle")
                    )
                    ranking.append(
                        {
                            "representation": representation,
                            "component": component,
                            "scale": scale,
                            "scope": scope_name,
                            "mean_activation_gap_recovered": float(np.mean(recoveries)),
                            "passes_scalar_and_shuffle": passes,
                            "cells": cells,
                        }
                    )
    ranking.sort(key=lambda row: row["mean_activation_gap_recovered"], reverse=True)
    args.atlas_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_accessions": [row["accession"] for row in source_rows],
            "source_representations": {
                name: value.float() for name, value in source_representations.items()
            },
            "source_profiles": {
                name: value.float() for name, value in source_profiles.items()
            },
            "source_categories": source_categories.cpu(),
            "median_squared_distances": medians,
            "permutation": permutation,
        },
        args.atlas_cache,
    )
    return {
        "setting": "esm2_protein_feature_localization_screen",
        "model": args.model,
        "parameters": sum(
            parameter.numel() for parameter in experiment.model.parameters()
        ),
        "dataset": str(args.data),
        "source_examples": args.source_count,
        "correct_query_examples": args.query_count,
        "representations": REPRESENTATIONS,
        "components": COMPONENTS,
        "scales": args.scales,
        "scopes": list(scopes),
        "fractions": args.fractions,
        "query_accessions": [row["accession"] for row in query_examples],
        "ranking": ranking,
        "top_20": ranking[:20],
        "elapsed_seconds": time.time() - started,
    }


def _selected_methods(atlas, query, categories, representation, component, scale):
    source = atlas["source_representations"][representation].double()
    profiles = atlas["source_profiles"][component].double()
    permutation = atlas["permutation"].long()
    median = float(atlas["median_squared_distances"][representation])
    kernel = torch.exp(-torch.cdist(source, query).square() / (2 * median * scale**2))
    shuffled = torch.exp(
        -torch.cdist(source[permutation], query).square() / (2 * median * scale**2)
    )
    similarity = source @ query.T
    categorical = atlas["source_categories"].double() @ categories.double().T
    return {
        "exact_rbf": profiles @ kernel / len(source),
        "matched_shuffle": profiles @ shuffled / len(source),
        "nearest": profiles[:, similarity.argmax(dim=0)],
        "scalar": profiles.mean(dim=1, keepdim=True).expand(-1, len(query)),
        "categorical": profiles @ categorical / len(source),
    }


def run_causal(args):
    examples = _load_examples(args.data, args.seed)
    experiment = EsmSensitivity(args.model, args.device)
    atlas = torch.load(args.atlas_cache, map_location="cpu", weights_only=True)
    source_accessions = set(atlas["source_accessions"])
    excluded = (
        set(args.exclude_accessions.read_text().splitlines())
        if args.exclude_accessions
        else set()
    )
    (
        rows,
        prepared_rows,
        baselines,
        representations,
        attributions,
        profiles,
        categories,
    ) = [], [], [], [], [], [], []
    for example in examples:
        if (
            example["accession"] in source_accessions
            or example["accession"] in excluded
        ):
            continue
        prepared, baseline, representation_dict, attribution, profile_dict, correct = (
            _query_example(experiment, example)
        )
        if not correct:
            continue
        rows.append(example)
        prepared_rows.append(prepared)
        baselines.append(baseline)
        representations.append(representation_dict[args.representation])
        attributions.append(attribution)
        profiles.append(profile_dict[args.component])
        category = torch.zeros(len(CANONICAL_AA))
        category[CANONICAL_AA.index(str(prepared["true_residue"]))] = 1
        categories.append(category)
        if len(rows) >= args.query_count:
            break
    if len(rows) < args.query_count:
        raise RuntimeError(f"only {len(rows)} correct queries found")
    query = F.normalize(torch.stack(representations).double(), dim=1)
    methods = _selected_methods(
        atlas,
        query,
        torch.stack(categories),
        args.representation,
        args.component,
        args.scale,
    )
    methods["direct_parameter"] = torch.stack(profiles).T.double()
    methods["activation_attribution"] = torch.stack(attributions).T.double()
    scope = _scope_indices(experiment)[args.scope]
    records = []
    started = time.time()
    for query_index, (prepared, baseline) in enumerate(zip(prepared_rows, baselines)):
        for fraction in args.fractions:
            count = max(1, math.ceil(fraction * len(scope)))
            for method, scores in methods.items():
                selected = (
                    scope[torch.topk(scores[scope, query_index], count).indices]
                    .sort()
                    .values
                )
                with zero_activation_groups(experiment, selected):
                    intervened = experiment.functional(prepared)
                records.append(
                    {
                        "query": query_index,
                        "accession": rows[query_index]["accession"],
                        "method": method,
                        "fraction": fraction,
                        "selected_groups": count,
                        "causal_effect": abs(baseline - intervened),
                        "signed_functional_change": baseline - intervened,
                    }
                )
        if (query_index + 1) % 16 == 0:
            print(
                "exact",
                query_index + 1,
                "elapsed",
                round(time.time() - started, 1),
                flush=True,
            )
    summaries, comparisons = {}, {}
    for fraction in args.fractions:
        key = f"{fraction:g}"
        values = {
            method: [
                row["causal_effect"]
                for row in records
                if row["fraction"] == fraction and row["method"] == method
            ]
            for method in methods
        }
        summaries[key] = {
            method: _paired_summary(item) for method, item in values.items()
        }
        exact = values["exact_rbf"]
        comparisons[key] = {
            "exact_rbf": {
                f"vs_{baseline}": _paired_summary(
                    [left - right for left, right in zip(exact, values[baseline])]
                )
                for baseline in values
                if baseline != "exact_rbf"
            }
        }
        scalar = summaries[key]["scalar"]["mean"]
        oracle = summaries[key]["activation_attribution"]["mean"]
        comparisons[key]["exact_rbf"]["activation_gap_recovered"] = (
            (summaries[key]["exact_rbf"]["mean"] - scalar) / (oracle - scalar)
            if oracle > scalar
            else None
        )
    return {
        "setting": "esm2_protein_feature_localization_causal",
        "model": args.model,
        "parameters": sum(
            parameter.numel() for parameter in experiment.model.parameters()
        ),
        "dataset": str(args.data),
        "protocol": {
            "source_examples": len(atlas["source_accessions"]),
            "query_examples": args.query_count,
            "representation": args.representation,
            "component": args.component,
            "scope": args.scope,
            "scale": args.scale,
            "fractions": args.fractions,
            "functional": "true masked-amino-acid logit minus strongest alternate",
            "sensitivity": "normalized squared native-parameter-gradient group energy",
            "action": "zero selected ESM2 FF hidden features",
            "action_matched_reference": "absolute gradient-times-activation summed over tokens",
        },
        "query_accessions": [row["accession"] for row in rows],
        "summaries": summaries,
        "comparisons": comparisons,
        "records": records,
    }


def run_parameter_causal(args):
    """Evaluate retrieved groups by exact antithetic parameter perturbations."""
    examples = _load_examples(args.data, args.seed)
    experiment = EsmSensitivity(args.model, args.device)
    atlas = torch.load(args.atlas_cache, map_location="cpu", weights_only=True)
    source_accessions = set(atlas["source_accessions"])
    excluded = (
        set(args.exclude_accessions.read_text().splitlines())
        if args.exclude_accessions
        else set()
    )
    rows, prepared_rows, representations, profiles, categories = [], [], [], [], []
    for example in examples:
        if example["accession"] in source_accessions or example["accession"] in excluded:
            continue
        prepared, _baseline, representation_dict, _attribution, profile_dict, correct = (
            _query_example(experiment, example)
        )
        if not correct:
            continue
        rows.append(example)
        prepared_rows.append(prepared)
        representations.append(representation_dict[args.representation])
        profiles.append(profile_dict[args.component])
        category = torch.zeros(len(CANONICAL_AA))
        category[CANONICAL_AA.index(str(prepared["true_residue"]))] = 1
        categories.append(category)
        if len(rows) >= args.query_count:
            break
    if len(rows) < args.query_count:
        raise RuntimeError(f"only {len(rows)} correct queries found")

    query = F.normalize(torch.stack(representations).double(), dim=1)
    methods = _selected_methods(
        atlas,
        query,
        torch.stack(categories),
        args.representation,
        args.component,
        args.scale,
    )
    query_profiles = torch.stack(profiles).T.double()
    methods["direct_parameter"] = query_profiles
    scope = _scope_indices(experiment)[args.scope]
    parameter_rms = experiment.parameter_rms(scope, args.component)
    noise_std = parameter_rms * args.noise_scale
    records = []
    started = time.time()
    for query_index, prepared in enumerate(prepared_rows):
        for fraction in args.fractions:
            count = max(1, math.ceil(fraction * len(scope)))
            for method, scores in methods.items():
                selected = (
                    scope[torch.topk(scores[scope, query_index], count).indices]
                    .sort()
                    .values
                )
                coverage = float(
                    query_profiles[selected, query_index].sum()
                    / query_profiles[scope, query_index].sum().clamp_min(1e-30)
                )
                for direction in range(args.directions):
                    intervention_seed = (
                        args.seed + query_index * 100_003 + direction * 1_009
                    )
                    with experiment.additive_parameter_noise(
                        selected,
                        noise_std,
                        component=args.component,
                        seed=intervention_seed,
                        sign=1,
                    ):
                        plus = experiment.functional(prepared)
                    with experiment.additive_parameter_noise(
                        selected,
                        noise_std,
                        component=args.component,
                        seed=intervention_seed,
                        sign=-1,
                    ):
                        minus = experiment.functional(prepared)
                    derivative = (plus - minus) / (2 * noise_std)
                    records.append(
                        {
                            "query": query_index,
                            "accession": rows[query_index]["accession"],
                            "method": method,
                            "fraction": fraction,
                            "selected_groups": count,
                            "direction": direction,
                            "direct_gradient_energy_coverage": coverage,
                            "squared_susceptibility": derivative**2,
                        }
                    )
        if (query_index + 1) % 8 == 0:
            print(
                "parameter causal",
                query_index + 1,
                "elapsed",
                round(time.time() - started, 1),
                flush=True,
            )

    def query_means(method: str, fraction: float, metric: str):
        grouped = defaultdict(list)
        for record in records:
            if record["method"] == method and record["fraction"] == fraction:
                grouped[record["query"]].append(record[metric])
        return {query_index: float(np.mean(values)) for query_index, values in grouped.items()}

    summaries, comparisons = {}, {}
    for fraction in args.fractions:
        key = f"{fraction:g}"
        values_by_method = {}
        summaries[key] = {}
        for method in methods:
            values = query_means(method, fraction, "squared_susceptibility")
            values_by_method[method] = values
            summaries[key][method] = _paired_summary(list(values.values()))
            coverage = query_means(method, fraction, "direct_gradient_energy_coverage")
            summaries[key][method]["gradient_energy_coverage"] = _paired_summary(
                list(coverage.values())
            )
        comparisons[key] = {"exact_rbf": {}}
        for baseline in methods:
            if baseline == "exact_rbf":
                continue
            differences = [
                values_by_method["exact_rbf"][index]
                - values_by_method[baseline][index]
                for index in sorted(
                    values_by_method["exact_rbf"].keys()
                    & values_by_method[baseline].keys()
                )
            ]
            comparisons[key]["exact_rbf"][f"vs_{baseline}"] = _paired_summary(
                differences
            )
        scalar = summaries[key]["scalar"]["mean"]
        oracle = summaries[key]["direct_parameter"]["mean"]
        comparisons[key]["exact_rbf"]["direct_parameter_gap_recovered"] = (
            (summaries[key]["exact_rbf"]["mean"] - scalar) / (oracle - scalar)
            if oracle > scalar
            else None
        )
    return {
        "setting": "esm2_protein_parameter_influence_circuits",
        "model": args.model,
        "parameters": sum(parameter.numel() for parameter in experiment.model.parameters()),
        "dataset": str(args.data),
        "protocol": {
            "source_examples": len(atlas["source_accessions"]),
            "query_examples": args.query_count,
            "representation": args.representation,
            "component": args.component,
            "scope": args.scope,
            "scale": args.scale,
            "fractions": args.fractions,
            "functional": "true masked-amino-acid logit minus strongest alternate",
            "sensitivity": "normalized squared native-parameter-gradient group energy",
            "action": "antithetic Rademacher perturbation of selected native parameter groups",
            "directions": args.directions,
            "noise_scale_relative_to_scoped_parameter_rms": args.noise_scale,
            "scoped_parameter_rms": parameter_rms,
            "coordinate_noise_standard_deviation": noise_std,
        },
        "query_accessions": [row["accession"] for row in rows],
        "summaries": summaries,
        "comparisons": comparisons,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("screen", "causal", "parameter_causal"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--seed", type=int, default=260926)
    parser.add_argument("--source-count", type=int, default=400)
    parser.add_argument("--query-count", type=int, default=48)
    parser.add_argument(
        "--scales", nargs="+", type=float, default=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=(0.0002, 0.001))
    parser.add_argument(
        "--representation", choices=REPRESENTATIONS, default="mask_last"
    )
    parser.add_argument("--component", choices=COMPONENTS, default="coupled")
    parser.add_argument("--scope", default="all")
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--atlas-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-accessions", type=Path)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    args = parser.parse_args()
    args.scales = tuple(args.scales)
    args.fractions = tuple(args.fractions)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.stage == "screen":
        result = run_screen(args)
    elif args.stage == "causal":
        result = run_causal(args)
    else:
        result = run_parameter_causal(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
