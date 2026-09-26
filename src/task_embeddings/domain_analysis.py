"""Explicit metric labels and model-clustered uncertainty for domain studies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis_v4 import mean_sd_ci
from .common import save_json


def normalize_profile_shape(values, sizes, blocks=None):
    """Remove task-wide (or per-tensor) energy, retaining only importance shape.

    Sizes count parameters, not groups. This is a diagnostic transformation,
    not residual-normalized OPG and not a change to the queried index.
    """
    import torch

    if values.ndim != 2 or sizes.shape != values.shape[:1]:
        raise ValueError("group-by-task profiles and matching group sizes required")
    if not bool(
        torch.isfinite(values).all()
        and torch.isfinite(sizes).all()
        and (sizes > 0).all()
    ):
        raise ValueError("finite profiles and positive finite sizes required")
    values, sizes = values.double(), sizes.double()
    blocks = [(0, len(values))] if blocks is None else blocks
    result = torch.empty_like(values)
    previous = 0
    for start, end in blocks:
        if start != previous or not start < end <= len(values):
            raise ValueError("ordered blocks must partition the profile rows")
        local, weights = values[start:end], sizes[start:end, None]
        energy = (local * weights).sum(0, keepdim=True) / weights.sum()
        result[start:end] = local / energy.clamp_min(1e-30)
        previous = end
    if previous != len(values):
        raise ValueError("blocks must cover every profile row")
    return result


def profile_shape_metrics(predicted, truth, source_mean, target_mean, sizes, blocks):
    """Column-wise rank diagnostics, with independent observed-task backgrounds."""
    from scipy.stats import spearmanr

    def correlate(a, b):
        result = []
        for x, y in zip(a.T.cpu().numpy(), b.T.cpu().numpy()):
            value = (
                float(spearmanr(x, y).statistic)
                if np.ptp(x) and np.ptp(y)
                else float("nan")
            )
            result.append(value if np.isfinite(value) else None)
        return result

    metrics = {
        "additive_residual_spearman": correlate(
            predicted - source_mean, truth - target_mean
        ),
    }
    for name, local in (
        ("shape_only_spearman", None),
        ("within_tensor_shape_spearman", blocks),
    ):
        if name.startswith("within") and local is None:
            metrics[name] = [None] * predicted.shape[1]
            continue
        a = normalize_profile_shape(predicted, sizes, local)
        a -= normalize_profile_shape(source_mean, sizes, local)
        b = normalize_profile_shape(truth, sizes, local)
        b -= normalize_profile_shape(target_mean, sizes, local)
        metrics[name] = correlate(a, b)
    return metrics


def shape_audit(paths, methods=("full", "tbe", "jl", "cluster", "gain_only")):
    """Reproduce the follow-up shape/cluster audit from saved review profiles."""
    import torch

    from .domain_applications import query_scores
    from .domain_optimizer import ParameterPartition
    from .domain_train import DomainConfig, make_model

    torch.set_num_threads(4)
    records = []
    for path in paths:
        result = json.loads(path.read_text())
        cache = torch.load(path.with_suffix(".pt"), weights_only=False, mmap=True)
        corpus_path = Path(cache["signature"]["corpus"])
        corpus = torch.load(corpus_path, weights_only=False, mmap=True)
        features = torch.load(
            corpus_path.parent / result["feature_file"], weights_only=False
        )["features"]
        with torch.device("meta"):
            model = make_model(
                DomainConfig(**result["model_configuration"]),
                corpus["vocab_size"],
                device="meta",
            )
        seed = result.get("control_seed", result["seed"])
        for kind, source in cache["source"].items():
            part = ParameterPartition(model, kind)
            spans = sorted(
                {(s.offset, s.count) for s in part.slices if ".layers." in s.name}
            )
            indices = torch.tensor(
                [i for start, count in spans for i in range(start, start + count)]
            )
            local, offset = [], 0
            for _, count in spans:
                local.append((offset, offset + count))
                offset += count
            blocks = local if kind != "tensor" else None
            sizes = part.sizes[indices]
            for estimator, atlas in source.items():
                truth = cache["target"][kind][estimator][indices]
                for fold in range(4):
                    queries = list(range(fold, len(features), 4))
                    observed = [t for t in range(len(features)) if t not in queries]
                    background = atlas[indices][:, observed].mean(1, keepdim=True)
                    answer_background = truth[:, observed].mean(1, keepdim=True)
                    for method in methods:
                        if method == "gain_only":
                            energy = (atlas[indices].double() * sizes[:, None]).sum(
                                0, keepdim=True
                            )
                            energy /= sizes.sum()
                            gain = query_scores(
                                energy,
                                features,
                                "tbe",
                                seed + 800,
                                observed=observed,
                                score_link="log",
                            )
                            scores = (
                                background.double()
                                * gain.double()
                                / energy[:, observed].mean()
                            )
                        else:
                            scores = query_scores(
                                atlas,
                                features,
                                method,
                                seed + 800,
                                observed=observed,
                                score_link=result["score_link"],
                                residual_scale=result.get("residual_scale", 1.0),
                            )[indices]
                        metrics = profile_shape_metrics(
                            scores[:, queries],
                            truth[:, queries],
                            background,
                            answer_background,
                            sizes,
                            blocks,
                        )
                        if method == "gain_only":
                            # Multiplying the pooled profile cannot change its shape.
                            for key in (
                                "shape_only_spearman",
                                "within_tensor_shape_spearman",
                            ):
                                metrics[key] = [None] * len(queries)
                        for i, task in enumerate(queries):
                            records.append(
                                {
                                    "model_source": result["model_source"],
                                    "seed": seed,
                                    "partition": kind,
                                    "estimator": estimator,
                                    "method": method,
                                    "task": task,
                                    "fold": fold,
                                    **{
                                        key: values[i]
                                        for key, values in metrics.items()
                                    },
                                }
                            )
    return {
        "complete": True,
        "sources": [str(p.resolve()) for p in paths],
        "definition": "Normalize core profiles and their independent observed-task backgrounds by parameter-weighted mean importance before computing residual Spearman.",
        "within_tensor_definition": "Normalize separately within each parent tensor or coupled SwiGLU family; undefined for whole-tensor groups.",
        "gain_only_definition": "Predict log task-wide mean importance from observed task features, then multiply the pooled profile by the predicted scalar; shape correlations are undefined.",
        "records": records,
    }


def translation_paired_interval(current, reference, repetitions=2000, seed=1307):
    """Paired document bootstrap of macro corpus chrF++, not sentence chrF."""
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=2)
    if not current or len(current) != len(reference):
        raise ValueError("paired nonempty language lists required")
    a = {r["language"]: r for r in current}
    b = {r["language"]: r for r in reference}
    if a.keys() != b.keys() or len(a) != len(current):
        raise ValueError("unique matched languages required")
    languages = sorted(a)
    documents = sorted({d for row in current for d in row["document_ids"]})
    positions = {doc: i for i, doc in enumerate(documents)}
    collected = []
    for lang in languages:
        x, y = a[lang], b[lang]
        if any(x[k] != y[k] for k in ("segment_ids", "document_ids", "references")):
            raise ValueError("unpaired translation examples")
        for row in (x, y):
            statistics = np.asarray(
                metric._extract_corpus_statistics(
                    row["predictions"], [row["references"]]
                ),
                dtype=np.int64,
            )
            by_document = np.zeros(
                (len(documents), statistics.shape[1]), dtype=np.int64
            )
            for doc, values in zip(row["document_ids"], statistics):
                by_document[positions[doc]] += values
            collected.append(by_document)
    statistics = np.stack(collected, 1)
    weights = np.random.default_rng(seed).multinomial(
        len(documents), np.full(len(documents), 1 / len(documents)), size=repetitions
    )
    weights = np.concatenate((np.ones((1, len(documents)), dtype=np.int64), weights))
    totals = np.einsum("bd,dcs->bcs", weights, statistics)
    values = np.array(
        [
            [metric._compute_score_from_stats(v.tolist()).score for v in sample]
            for sample in totals
        ]
    ).reshape(-1, len(languages), 2)
    differences = (values[:, :, 0] - values[:, :, 1]).mean(1)
    return {
        "mean": float(differences[0]),
        "ci95_low": float(np.quantile(differences[1:], 0.025)),
        "ci95_high": float(np.quantile(differences[1:], 0.975)),
        "documents": len(documents),
        "languages": len(languages),
        "bootstrap_repetitions": repetitions,
        "uncertainty_unit": "paired source documents; fixed model, index and language catalogue",
    }


def document_paired_interval(
    current, reference, document_ids, repetitions=5000, seed=1307
):
    """Resample source documents jointly across their translated languages.

    Conditional on this model, task catalogue, and index. Not model-seed or
    general-language uncertainty. Token-weighted NLL within each language,
    macro average across the fixed language catalogue.
    """
    if len(current) != len(reference) or not current:
        raise ValueError("paired nonempty language lists required")
    used = sorted({document_ids[i] for row in current for i in row["indices"]})
    lookup = {doc: i for i, doc in enumerate(used)}
    sums = np.zeros((len(used), len(current), 2))
    counts = np.zeros((len(used), len(current)))
    for task, (a, b) in enumerate(zip(current, reference)):
        if a["indices"] != b["indices"] or a["token_counts"] != b["token_counts"]:
            raise ValueError("unpaired evaluation examples")
        for i, index in enumerate(a["indices"]):
            doc = lookup[document_ids[index]]
            counts[doc, task] += a["token_counts"][i]
            sums[doc, task] += [a["loss_sums"][i], b["loss_sums"][i]]
    point = ((sums[:, :, 0].sum(0) - sums[:, :, 1].sum(0)) / counts.sum(0)).mean()
    weights = np.random.default_rng(seed).multinomial(
        len(used), np.full(len(used), 1 / len(used)), size=repetitions
    )
    denominators = weights @ counts
    valid = (denominators > 0).all(1)
    estimates = (
        (weights @ (sums[:, :, 0] - sums[:, :, 1]))[valid] / denominators[valid]
    ).mean(1)
    return {
        "mean": float(point),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "documents": len(used),
        "languages": len(current),
        "bootstrap_repetitions": repetitions,
        "uncertainty_unit": "paired source documents; model and language catalogue fixed",
    }


def summarize_endpoint(records, endpoint, metric):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        model = record.get("model_source", str(record["seed"]))
        for row in record[endpoint]:
            value = row.get(metric)
            if value is None:
                continue
            budget = row.get(
                "requested_removed_fraction",
                row.get("requested_high_precision_fraction"),
            )
            key = (row["partition"], row["estimator"], row["method"], budget)
            grouped[key][model].append(value)
    result = []
    for (partition, estimator, method, budget), models in grouped.items():
        means = {model: float(np.mean(values)) for model, values in models.items()}
        result.append(
            {
                "partition": partition,
                "estimator": estimator,
                "method": method,
                "budget": budget,
                "metric": metric,
                "model_means": means,
                **mean_sd_ci(list(means.values())),
            }
        )
    return result


def paired_differences(
    summary, method="tbe", baselines=("full", "jl", "mean", "wrong_task", "cluster")
):
    result = []
    for current in summary:
        if current["method"] != method:
            continue
        for reference in summary:
            if reference["method"] not in baselines:
                continue
            if any(
                current[k] != reference[k]
                for k in ("partition", "estimator", "budget", "metric")
            ):
                continue
            models = current["model_means"].keys() & reference["model_means"].keys()
            values = [
                current["model_means"][m] - reference["model_means"][m] for m in models
            ]
            if values:
                result.append(
                    {
                        k: current[k]
                        for k in ("partition", "estimator", "budget", "metric")
                    }
                    | {
                        "contrast": f"{method} minus {reference['method']}",
                        **mean_sd_ci(values),
                    }
                )
    return result


def summarize(paths):
    records = [json.loads(path.read_text()) for path in paths]
    fields = (
        "reference_samples_per_domain",
        "feature_file",
        "score_link",
        "protocol_version",
        "data_role",
        "causal_sigma",
        "model_parameters",
        "low_bits",
        "quant_group_size",
        "query_mode",
        "importance_coordinate",
        "intervention_scope",
        "residual_scale",
        "method_scales",
    )
    recipes = {
        json.dumps({k: record.get(k) for k in fields}, sort_keys=True)
        for record in records
    }
    if len(recipes) != 1:
        raise ValueError("different protocols must not be pooled")
    endpoints = {
        "ranking": [
            "residual_opg_spearman",
            "transformer_residual_opg_spearman",
            "raw_opg_spearman",
        ],
        "causal": ["residual_causal_spearman"],
        "pruning": ["nll_increase", "removed_parameter_fraction"],
        "quantization": ["nll_increase", "ideal_packed_bits_per_parameter"],
    }
    summaries = {}
    for endpoint, metrics in endpoints.items():
        summaries[endpoint] = []
        for metric in metrics:
            summaries[endpoint].extend(summarize_endpoint(records, endpoint, metric))
    return {
        "sources": [str(p) for p in paths],
        "configuration": json.loads(next(iter(recipes))),
        "uncertainty_unit": "independently trained model; domains and projection repeats averaged within model",
        "summaries": summaries,
        "paired": {key: paired_differences(value) for key, value in summaries.items()},
        "resources": records[0]["resources"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shape-audit", action="store_true")
    a = p.parse_args()
    save_json(a.output, shape_audit(a.files) if a.shape_audit else summarize(a.files))
