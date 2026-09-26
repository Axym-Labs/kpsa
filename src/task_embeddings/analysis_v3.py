from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .common import (
    arc_artifact_dir,
    embedding_fidelity,
    layer_balanced_indices,
    normalized_rows,
    save_json,
    task_aligned_importance,
)
from .controlled_v3 import task_mixtures


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(draws, len(array)), replace=True).mean(
        axis=1
    )
    low, high = np.quantile(samples, (0.025, 0.975))
    return mean, float(low), float(high)


def summarize_task_records(
    records: Sequence[dict],
    value_field: str,
    group_fields: Sequence[str],
    *,
    task_field: str = "task",
    draws: int = 10_000,
    seed: int = 0,
) -> list[dict]:
    within_task: dict[tuple, list[float]] = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in group_fields) + (record[task_field],)
        within_task[key].append(float(record[value_field]))
    by_group: dict[tuple, list[float]] = defaultdict(list)
    for key, values in within_task.items():
        by_group[key[:-1]].append(float(np.mean(values)))
    output = []
    for index, (key, values) in enumerate(sorted(by_group.items())):
        mean, low, high = bootstrap_mean_ci(values, draws=draws, seed=seed + index)
        row = {field: value for field, value in zip(group_fields, key)}
        row.update(
            {
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_tasks": len(values),
                "uncertainty_unit": "tasks within one trained checkpoint",
            }
        )
        output.append(row)
    return output


def global_circuit(
    selected: Sequence[torch.Tensor], layer_sizes: Sequence[int]
) -> set[int]:
    output: set[int] = set()
    offset = 0
    for indices, size in zip(selected, layer_sizes):
        output.update((indices.detach().cpu() + offset).tolist())
        offset += size
    return output


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _method_family(name: str) -> str:
    if "post_ief_semantic" in name or "post_ief_structured" in name:
        return "Post-hoc iEF task space"
    if "post_ief_onehot" in name:
        return "Full one-hot atlas"
    if "post_ief_jl" in name:
        return "Matched JL"
    if "online_late_raw" in name:
        return "Online raw EF"
    if "actgrad" in name:
        return "Activation x gradient"
    if "activation" in name:
        return "Activation"
    if "post_raw" in name or name == "raw_ef":
        return "Post-hoc raw EF"
    if "weight" in name:
        return "Weight magnitude"
    if name == "post_ief":
        return "Post-hoc iEF task space"
    if name == "online_raw_ef":
        return "Online raw EF"
    if name == "none":
        return "No protection"
    if name == "random":
        return "Random"
    return name


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for setting, result in results.items():
        for representation, metrics in result["source_fidelity"].items():
            rows.append(
                {
                    "setting": setting,
                    "representation": representation,
                    **metrics,
                }
            )
    return rows


def _interpretability_rows(results: dict[str, dict]) -> list[dict]:
    definitions = {
        "controlled": "selectivity",
        "vision": "target_selectivity",
        "language": "selective_loss_increase",
    }
    output = []
    for setting, field in definitions.items():
        selected = [
            {**record, "method_family": _method_family(record["method"])}
            for record in results[setting]["interpretability"]["records"]
            if record["intervention"] == "mean"
        ]
        summaries = summarize_task_records(
            selected,
            field,
            ("method_family", "fraction"),
            seed=100 + len(output),
        )
        for row in summaries:
            row.update(
                {
                    "setting": setting,
                    "metric": field,
                    "intervention": "mean",
                }
            )
            output.append(row)
    return output


def _pruning_recovery_rows(results: dict[str, dict]) -> list[dict]:
    output = []
    for setting, result in results.items():
        derived = []
        for record in result["pruning"]["records"]:
            recovered_field = (
                "recovered_accuracy" if setting == "vision" else "recovered_loss"
            )
            recovered = record.get(recovered_field)
            if recovered is None or not np.isfinite(float(recovered)):
                continue
            row = {**record, "method_family": _method_family(record["method"])}
            if setting == "vision":
                baseline = float(row["baseline_accuracy"])
                value = float(recovered / baseline) if baseline > 0 else float("nan")
            else:
                baseline = float(row["baseline_loss"])
                value = (
                    float(baseline / recovered)
                    if float(recovered) > 0
                    else float("nan")
                )
            row["relative_target_retention"] = value
            derived.append(row)
        summaries = summarize_task_records(
            derived,
            "relative_target_retention",
            ("method_family", "retained_fraction"),
            seed=250 + len(output),
        )
        for row in summaries:
            row.update(
                {
                    "setting": setting,
                    "metric": "relative_target_retention_after_recovery",
                }
            )
            output.append(row)
    return output


def _pruning_rows(results: dict[str, dict]) -> list[dict]:
    output = []
    for setting, result in results.items():
        derived = []
        for record in result["pruning"]["records"]:
            row = {**record, "method_family": _method_family(record["method"])}
            if setting == "vision":
                baseline = float(row["baseline_accuracy"])
                value = (
                    float(row["zero_shot_accuracy"] / baseline)
                    if baseline > 0
                    else float("nan")
                )
            else:
                baseline = float(row["baseline_loss"])
                value = (
                    float(baseline / row["zero_shot_loss"])
                    if row["zero_shot_loss"] > 0
                    else float("nan")
                )
            row["relative_target_retention"] = value
            derived.append(row)
        summaries = summarize_task_records(
            derived,
            "relative_target_retention",
            ("method_family", "retained_fraction"),
            seed=200 + len(output),
        )
        for row in summaries:
            row.update({"setting": setting, "metric": "relative_target_retention"})
            output.append(row)
    return output


def _continual_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for setting, result in results.items():
        for order, methods in result["continual_learning"].items():
            for method, payload in methods.items():
                rows.append(
                    {
                        "setting": setting,
                        "order": order,
                        "method": method,
                        "method_family": _method_family(method),
                        **payload["summary"],
                    }
                )
    return rows


def _online_rows(artifact_dir: Path) -> list[dict]:
    definitions = {
        "controlled": (
            artifact_dir / "controlled/controlled_v3_embeddings_seed1.npz",
            "structured4",
        ),
        "vision": (
            artifact_dir / "vision/vision_v3_embeddings_seed1.npz",
            "semantic32",
        ),
        "language": (
            artifact_dir / "language/language_v3_embeddings_seed1.npz",
            "semantic4",
        ),
    }
    rows = []
    for setting, (path, representation_name) in definitions.items():
        with np.load(path) as arrays:
            atlas = torch.from_numpy(arrays["atlas_ief"])
            if setting == "controlled":
                representation = normalized_rows(task_mixtures())
            else:
                representation = torch.from_numpy(arrays["semantic_representation"])
            reference = atlas @ representation
            candidate = torch.from_numpy(
                arrays[f"online_late_raw_{representation_name}"]
            )
            rows.append(
                {
                    "setting": setting,
                    "online_variant": "late_raw",
                    "representation": representation_name,
                    **embedding_fidelity(candidate, reference, representation),
                }
            )
    return rows


def _circuits(
    scores: torch.Tensor,
    layer_sizes: Sequence[int],
    fraction: float,
) -> list[set[int]]:
    return [
        global_circuit(
            layer_balanced_indices(scores[:, task], layer_sizes, fraction), layer_sizes
        )
        for task in range(scores.shape[1])
    ]


def _jaccard(first: set[int], second: set[int]) -> float:
    return len(first & second) / max(1, len(first | second))


def _composition_rows(artifact_dir: Path, results: dict[str, dict]) -> list[dict]:
    rows = []
    fraction = 0.05

    with np.load(
        artifact_dir / "controlled/controlled_v3_embeddings_seed1.npz"
    ) as arrays:
        atlas = torch.from_numpy(arrays["atlas_ief"])
        amplitude = torch.from_numpy(arrays["amplitude_ief"])
    mixtures = task_mixtures()
    structured = normalized_rows(mixtures)
    structured_scores = task_aligned_importance(
        atlas @ structured, structured, amplitude
    )
    full_scores = amplitude[:, None] * atlas
    layer_sizes = [results["controlled"]["configuration"]["d_ff"]] * results[
        "controlled"
    ]["configuration"]["n_layers"]
    direct = _circuits(structured_scores, layer_sizes, fraction)
    primitive = _circuits(full_scores[:, :4], layer_sizes, fraction)
    for task in range(4, 16):
        active = torch.where(mixtures[task] > 0)[0].tolist()
        union = set().union(*(primitive[index] for index in active))
        rows.extend(
            [
                {
                    "setting": "controlled",
                    "group": f"mixture_{task}",
                    "metric": "direct_vs_primitive_union_jaccard",
                    "value": _jaccard(direct[task], union),
                },
                {
                    "setting": "controlled",
                    "group": f"mixture_{task}",
                    "metric": "direct_circuit_recall_in_primitive_union",
                    "value": len(direct[task] & union) / max(1, len(direct[task])),
                },
            ]
        )

    from .vision import fine_to_coarse_mapping
    from .vision_v3 import continual_class_groups

    with np.load(artifact_dir / "vision/vision_v3_embeddings_seed1.npz") as arrays:
        atlas = torch.from_numpy(arrays["atlas_ief"])
        amplitude = torch.from_numpy(arrays["amplitude_ief"])
        semantic = torch.from_numpy(arrays["semantic_representation"])
    vision_scores = task_aligned_importance(atlas @ semantic, semantic, amplitude)
    vision_layers = [vision_scores.shape[0] // 12] * 12
    vision_circuits = _circuits(vision_scores, vision_layers, fraction)
    mapping = fine_to_coarse_mapping(
        Path(results["vision"]["configuration"]["data_root"])
    )
    groups = continual_class_groups(mapping)
    for group_kind, class_groups in groups.items():
        for group_index, classes in enumerate(class_groups):
            union = set().union(*(vision_circuits[class_id] for class_id in classes))
            rows.append(
                {
                    "setting": "vision",
                    "group": f"{group_kind}_{group_index}",
                    "group_kind": group_kind,
                    "metric": "five_class_union_fraction",
                    "value": len(union) / vision_scores.shape[0],
                }
            )

    with np.load(artifact_dir / "language/language_v3_embeddings_seed1.npz") as arrays:
        atlas = torch.from_numpy(arrays["atlas_ief"])
        amplitude = torch.from_numpy(arrays["amplitude_ief"])
        semantic = torch.from_numpy(arrays["semantic_representation"])
    language_scores = task_aligned_importance(atlas @ semantic, semantic, amplitude)
    language_layers = [language_scores.shape[0] // 28] * 28
    language_circuits = _circuits(language_scores, language_layers, fraction)
    for first in range(6):
        for second in range(first + 1, 6):
            related = first < 4 and second < 4
            if first >= 4 and second >= 4:
                continue
            rows.append(
                {
                    "setting": "language",
                    "group": f"tasks_{first}_{second}",
                    "group_kind": "classification_family"
                    if related
                    else "cross_family",
                    "metric": "pair_circuit_jaccard",
                    "value": _jaccard(
                        language_circuits[first], language_circuits[second]
                    ),
                }
            )
    return rows


def _efficiency_rows(artifact_dir: Path, results: dict[str, dict]) -> list[dict]:
    representation_names = {
        "controlled": "structured4",
        "vision": "semantic32",
        "language": "semantic4",
    }
    metric_names = {
        "controlled": "controlled_v3_metrics_seed1.json",
        "vision": "vision_v3_metrics_seed1.json",
        "language": "language_v3_metrics_seed1.json",
    }
    checkpoint_names = {
        "controlled": "controlled_v3_seed1.pt",
        "vision": "vision_v3_seed1.pt",
        "language": "language_v3_seed1.pt",
    }
    embedding_names = {
        "controlled": "controlled_v3_embeddings_seed1.npz",
        "vision": "vision_v3_embeddings_seed1.npz",
        "language": "language_v3_embeddings_seed1.npz",
    }
    rows = []
    for setting, result in results.items():
        directory = artifact_dir / setting
        embedding_path = directory / embedding_names[setting]
        with np.load(embedding_path) as arrays:
            atlas = torch.from_numpy(arrays["atlas_ief"])
            amplitude = torch.from_numpy(arrays["amplitude_ief"])
            if setting == "controlled":
                representation = normalized_rows(task_mixtures())
            else:
                representation = torch.from_numpy(arrays["semantic_representation"])
            embedding = atlas @ representation
            queries = representation[torch.arange(100) % representation.shape[0]]
            for _ in range(2):
                _ = (
                    (amplitude[:, None] * (embedding @ queries.T).clamp_min(0))
                    .sum()
                    .item()
                )
            timings = []
            for _ in range(7):
                started = time.perf_counter()
                _ = (
                    (amplitude[:, None] * (embedding @ queries.T).clamp_min(0))
                    .sum()
                    .item()
                )
                timings.append(time.perf_counter() - started)
            full_atlas_bytes = int(
                atlas.numel() * atlas.element_size()
                + amplitude.numel() * amplitude.element_size()
            )
            compressed_bytes = int(
                embedding.numel() * embedding.element_size()
                + amplitude.numel() * amplitude.element_size()
            )
            online_bytes = int(
                sum(
                    arrays[name].nbytes
                    for name in arrays.files
                    if name.startswith("online_")
                )
            )
        metric_path = directory / metric_names[setting]
        checkpoint_path = directory / checkpoint_names[setting]
        run_efficiency = result.get("efficiency", {})
        rows.append(
            {
                "setting": setting,
                "representation": representation_names[setting],
                "train_seconds": result["train"]["wall_seconds"],
                "total_wall_seconds": run_efficiency.get(
                    "total_wall_seconds", float("nan")
                ),
                "online_overhead_upper_bound": result["train"][
                    "online_accumulator_fraction_upper_bound"
                ],
                "posthoc_seconds": result["posthoc"]["wall_seconds"],
                "posthoc_samples": result["posthoc"]["samples"],
                "n_modules": result["model"]["n_modules"],
                "peak_cuda_allocated_bytes": run_efficiency.get(
                    "peak_cuda_allocated_bytes", 0
                ),
                "peak_cuda_reserved_bytes": run_efficiency.get(
                    "peak_cuda_reserved_bytes", 0
                ),
                "full_atlas_uncompressed_bytes": full_atlas_bytes,
                "compressed_embedding_uncompressed_bytes": compressed_bytes,
                "compression_ratio": compressed_bytes / full_atlas_bytes,
                "online_arrays_uncompressed_bytes": online_bytes,
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "embedding_artifact_bytes": embedding_path.stat().st_size,
                "metrics_artifact_bytes": metric_path.stat().st_size,
                "query_count": 100,
                "seconds_per_amortized_query": float(np.median(timings) / 100),
                "query_benchmark_device": "CPU",
            }
        )
    return rows


def aggregate(artifact_dir: Path) -> dict:
    results = {
        "controlled": _load_json(
            artifact_dir / "controlled/controlled_v3_metrics_seed1.json"
        ),
        "vision": _load_json(artifact_dir / "vision/vision_v3_metrics_seed1.json"),
        "language": _load_json(
            artifact_dir / "language/language_v3_metrics_seed1.json"
        ),
    }
    efficiency = _efficiency_rows(artifact_dir, results)
    return {
        "source_fidelity": _source_rows(results),
        "interpretability": _interpretability_rows(results),
        "pruning": _pruning_rows(results),
        "pruning_recovery": _pruning_recovery_rows(results),
        "continual_learning": _continual_rows(results),
        "online_fidelity": _online_rows(artifact_dir),
        "composition": _composition_rows(artifact_dir, results),
        "efficiency": efficiency,
        "sanity": {
            "controlled": {
                "model_loss": results["controlled"]["baseline_loss_mean"],
                **results["controlled"].get("sanity_baselines", {}),
            },
            "vision": {
                "accuracy": results["vision"]["baseline_accuracy_mean"],
                **results["vision"].get("sanity", {}),
            },
            "language": {
                "exact_match": results["language"]["baseline_exact_match_mean"],
                **results["language"].get("sanity", {}),
            },
        },
    }


PRIMARY = "#3F21B6"
SECONDARY = "#8C7AD3"
NEUTRAL = "#BDBDBD"
DARK = "#444444"
SETTING_LABELS = {
    "controlled": "Controlled Transformer",
    "vision": "DINOv2 / CIFAR-100",
    "language": "Qwen3-1.7B",
}


def _save_figure(figure, stem: Path) -> None:
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(stem.with_suffix(f".{suffix}"), transparent=True, dpi=220)


def plot_source_fidelity(payload: dict, output_dir: Path, style: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey="row")
    metrics = ("centered_kernel_alignment", "ranking_spearman_mean")
    labels = ("Centered kernel alignment", "Task-ranking Spearman")
    for column, setting in enumerate(("controlled", "vision", "language")):
        rows = [row for row in payload["source_fidelity"] if row["setting"] == setting]
        jl = sorted(
            [row for row in rows if row["representation"].startswith("jl")],
            key=lambda row: row["dimension"],
        )
        semantic = [
            row
            for row in rows
            if row["representation"].startswith("semantic")
            or row["representation"].startswith("structured")
        ]
        onehot = next(row for row in rows if row["representation"] == "onehot")
        for row_index, (metric, ylabel) in enumerate(zip(metrics, labels)):
            axis = axes[row_index, column]
            axis.plot(
                [row["dimension"] for row in jl],
                [row[metric] for row in jl],
                color=NEUTRAL,
                marker="o",
                label="JL",
            )
            axis.scatter(
                [row["dimension"] for row in semantic],
                [row[metric] for row in semantic],
                color=PRIMARY,
                marker="D",
                s=24,
                label="Semantic / known",
                zorder=3,
            )
            axis.scatter(
                [onehot["dimension"]],
                [onehot[metric]],
                color=DARK,
                marker="x",
                s=26,
                label="Full atlas",
                zorder=3,
            )
            axis.set_ylim(-0.05, 1.05)
            axis.set_xlabel("Task-space dimension")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(SETTING_LABELS[setting])
    axes[0, 0].legend(frameon=False, loc="lower right")
    figure.tight_layout()
    _save_figure(figure, output_dir / "source_fidelity")
    plt.close(figure)


def _method_color(method: str) -> str:
    if method == "Post-hoc iEF task space":
        return PRIMARY
    if method == "Online raw EF":
        return SECONDARY
    return NEUTRAL


def plot_interpretability(payload: dict, output_dir: Path, style: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    methods = (
        "Post-hoc iEF task space",
        "Full one-hot atlas",
        "Matched JL",
        "Online raw EF",
        "Post-hoc raw EF",
        "Activation",
        "Activation x gradient",
        "Weight magnitude",
        "Random",
    )
    short = (
        "iEF-task",
        "full",
        "JL",
        "online",
        "raw",
        "act",
        "act×grad",
        "weight",
        "random",
    )
    markers = ("D", "s", "^", "o", "P", "v", "X", "*", "x")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    for axis, setting in zip(axes, ("controlled", "vision", "language")):
        rows = [row for row in payload["interpretability"] if row["setting"] == setting]
        for method, label, marker in zip(methods, short, markers):
            values = sorted(
                [row for row in rows if row["method_family"] == method],
                key=lambda row: row["fraction"],
            )
            if not values:
                continue
            axis.plot(
                [row["fraction"] for row in values],
                [row["mean"] for row in values],
                color=_method_color(method),
                marker=marker,
                linestyle="-"
                if method in ("Post-hoc iEF task space", "Online raw EF")
                else "--",
                linewidth=1.0,
                markersize=3.2,
                label=label,
            )
        axis.axhline(0, color=DARK, linewidth=0.5)
        axis.set_title(SETTING_LABELS[setting])
        axis.set_xlabel("Intervened MLP features")
        axis.set_ylabel("Task selectivity")
    axes[-1].legend(
        frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )
    figure.tight_layout()
    _save_figure(figure, output_dir / "interpretability_selectivity")
    plt.close(figure)


def plot_pruning(payload: dict, output_dir: Path, style: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    methods = (
        "Post-hoc iEF task space",
        "Full one-hot atlas",
        "Matched JL",
        "Online raw EF",
        "Activation",
        "Random",
    )
    markers = ("D", "s", "^", "o", "v", "x")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for axis, setting in zip(axes, ("controlled", "vision", "language")):
        rows = [row for row in payload["pruning"] if row["setting"] == setting]
        for method, marker in zip(methods, markers):
            values = sorted(
                [row for row in rows if row["method_family"] == method],
                key=lambda row: row["retained_fraction"],
            )
            if not values:
                continue
            axis.plot(
                [row["retained_fraction"] for row in values],
                [row["mean"] for row in values],
                marker=marker,
                color=_method_color(method),
                linestyle="-"
                if method in ("Post-hoc iEF task space", "Online raw EF")
                else "--",
                linewidth=1.1,
                markersize=3.5,
                label=method,
            )
        recovered = sorted(
            [
                row
                for row in payload["pruning_recovery"]
                if row["setting"] == setting
                and row["method_family"] == "Post-hoc iEF task space"
            ],
            key=lambda row: row["retained_fraction"],
        )
        if recovered:
            axis.scatter(
                [row["retained_fraction"] for row in recovered],
                [row["mean"] for row in recovered],
                marker="*",
                s=30,
                facecolors="none",
                edgecolors=PRIMARY,
                linewidths=0.8,
                label="iEF-task + recovery",
                zorder=4,
            )
        axis.axhline(1.0, color=DARK, linewidth=0.5, linestyle=":")
        axis.set_title(SETTING_LABELS[setting])
        axis.set_xlabel("Retained MLP features")
        axis.set_xlim(0.05, 0.80)
    axes[0].set_ylabel("Relative target retention")
    axes[-1].legend(
        frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )
    figure.tight_layout()
    _save_figure(figure, output_dir / "pruning_retention")
    plt.close(figure)


def plot_continual(payload: dict, output_dir: Path, style: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    methods = (
        "Post-hoc iEF task space",
        "Online raw EF",
        "Post-hoc raw EF",
        "Activation",
        "Random",
        "No protection",
    )
    short = ("iEF-task", "online", "raw", "act", "random", "none")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for axis, setting in zip(axes, ("controlled", "vision", "language")):
        rows = [
            row for row in payload["continual_learning"] if row["setting"] == setting
        ]
        means = []
        for method in methods:
            values = [
                row["average_forgetting"]
                for row in rows
                if row["method_family"] == method
            ]
            means.append(float(np.mean(values)) if values else float("nan"))
        x = np.arange(len(methods))
        axis.bar(
            x,
            means,
            color=[_method_color(method) for method in methods],
            edgecolor=DARK,
            linewidth=0.35,
        )
        axis.set_xticks(x, short, rotation=50, ha="right")
        axis.set_title(SETTING_LABELS[setting])
        axis.set_ylabel("Average forgetting")
    figure.tight_layout()
    _save_figure(figure, output_dir / "continual_forgetting")
    plt.close(figure)


def plot_efficiency(payload: dict, output_dir: Path, style: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    rows = payload["efficiency"]
    settings = ("controlled", "vision", "language")
    short = ("Controlled", "Vision", "Language")
    lookup = {row["setting"]: row for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    panels = (
        ("total_wall_seconds", 60.0, "Full run (min)"),
        ("peak_cuda_allocated_bytes", 2**30, "Peak allocated (GiB)"),
        ("checkpoint_bytes", 2**30, "Checkpoint (GiB)"),
    )
    for axis, (field, divisor, ylabel) in zip(axes, panels):
        values = [lookup[setting][field] / divisor for setting in settings]
        axis.bar(
            np.arange(3), values, color=(PRIMARY, SECONDARY, NEUTRAL), edgecolor=DARK
        )
        axis.set_xticks(np.arange(3), short, rotation=35, ha="right")
        axis.set_ylabel(ylabel)
    figure.tight_layout()
    _save_figure(figure, output_dir / "efficiency")
    plt.close(figure)


def make_figures(payload: dict, output_dir: Path, style: Path) -> None:
    plot_source_fidelity(payload, output_dir, style)
    plot_interpretability(payload, output_dir, style)
    plot_pruning(payload, output_dir, style)
    plot_continual(payload, output_dir, style)
    plot_efficiency(payload, output_dir, style)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", type=Path, default=arc_artifact_dir("02_exploratory")
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=Path(
            "/home/davwis/.codex/plugins/cache/local-skills/local-scientific/"
            "0.1.0+codex.local/skills/sci-scientific-visualization/assets/paper.mplstyle"
        ),
    )
    args = parser.parse_args()
    payload = aggregate(args.artifact_dir)
    save_json(args.artifact_dir / "aggregate_metrics.json", payload)
    for name in (
        "source_fidelity",
        "interpretability",
        "pruning",
        "pruning_recovery",
        "continual_learning",
        "online_fidelity",
        "efficiency",
        "composition",
    ):
        _write_csv(args.artifact_dir / f"{name}.csv", payload[name])
    make_figures(payload, args.artifact_dir, args.style)
    print(f"V3_AGGREGATE={args.artifact_dir / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
