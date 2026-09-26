from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .common import arc_artifact_dir

PRIMARY = "#3F21B6"
SECONDARY = "#8C7AD3"
NEUTRAL = "#BDBDBD"
DARK_NEUTRAL = "#555555"


def selectivity_values(
    records: Sequence[dict], method: str, fraction: float, field: str
) -> list[float]:
    return [
        float(row[field])
        for row in records
        if row["method"] == method and math.isclose(float(row["fraction"]), fraction)
    ]


def mean_sem(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    sem = 0.0 if len(array) == 1 else float(array.std(ddof=1) / np.sqrt(len(array)))
    return float(array.mean()), sem


def finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def aggregate(root: Path) -> dict:
    controlled = [load_json(root / "controlled/controlled_metrics_seed1.json")]
    vision = [
        load_json(root / f"vision_corrected/vision_metrics_seed{seed}.json")
        for seed in (1, 2)
    ]
    language = [load_json(root / "language/language_metrics_seed1.json")]
    return {"controlled": controlled, "vision": vision, "language": language}


METHODS = {
    "controlled": [
        ("Full one-hot", "post_ief_onehot"),
        ("Compressed iEF", "post_ief_latent4"),
        ("Online raw EF", "train_late_raw_latent4"),
        ("Post-hoc raw EF", "post_raw_latent4"),
        ("Activation", "activation_latent4"),
        ("Activation × gradient", "actgrad_latent4"),
        ("Weight norm", "weight_norm"),
        ("Random", "random"),
    ],
    "vision": [
        ("Full one-hot", "post_ief_onehot"),
        ("Compressed iEF", "post_ief_jl32"),
        ("Online raw EF", "train_late_raw_jl32"),
        ("Post-hoc raw EF", "post_raw_jl32"),
        ("Activation", "activation_jl32"),
        ("Activation × gradient", "actgrad_jl32"),
        ("Weight norm", "weight_norm"),
        ("Random", "random"),
    ],
    "language": [
        ("Full one-hot", "post_ief_onehot"),
        ("Compressed iEF", "post_ief_semantic4"),
        ("Online raw EF", "train_late_raw_semantic4"),
        ("Post-hoc raw EF", "post_raw_semantic4"),
        ("Activation", "activation_semantic4"),
        ("Activation × gradient", "actgrad_semantic4"),
        ("Weight norm", "weight_norm"),
        ("Random", "random"),
    ],
}


SETTING_META = {
    "controlled": (
        0.05,
        "selective_drop",
        "Selective loss increase",
        "Controlled (5%)",
    ),
    "vision": (
        0.05,
        "selective_accuracy_drop",
        "Selective accuracy drop",
        "CIFAR-100 (5%)",
    ),
    "language": (
        0.03,
        "selective_accuracy_drop",
        "Selective accuracy drop",
        "T5 / GLUE (3%)",
    ),
}


def causal_summary(results: dict) -> list[dict]:
    rows = []
    for setting, runs in results.items():
        fraction, field, _axis, _label = SETTING_META[setting]
        records = [row for run in runs for row in run["causal"]["records"]]
        for label, method in METHODS[setting]:
            values = selectivity_values(records, method, fraction, field)
            mean, sem = mean_sem(values)
            rows.append(
                {
                    "setting": setting,
                    "fraction": fraction,
                    "method": label,
                    "method_key": method,
                    "mean": mean,
                    "sem_over_task_queries": sem,
                    "n_task_queries": len(values),
                    "n_seeds": len(runs),
                }
            )
    return rows


def save_table(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _method_style(label: str) -> tuple[str, str]:
    if label == "Compressed iEF":
        return PRIMARY, "o"
    if label in {"Full one-hot", "Online raw EF"}:
        return SECONDARY, "s" if label == "Full one-hot" else "D"
    return NEUTRAL, "o"


def causal_figure(
    results: dict, rows: Sequence[dict], output_dir: Path, style: Path
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65))
    for axis, setting in zip(axes, ("controlled", "vision", "language")):
        subset = [row for row in rows if row["setting"] == setting]
        positions = np.arange(len(subset))[::-1]
        for position, row in zip(positions, subset):
            color, marker = _method_style(row["method"])
            axis.errorbar(
                row["mean"],
                position,
                xerr=row["sem_over_task_queries"],
                fmt=marker,
                color=color,
                markeredgecolor=DARK_NEUTRAL if color == NEUTRAL else color,
                markersize=4,
                capsize=2,
                linewidth=1,
                zorder=3,
            )
        axis.axvline(0, color="#777777", linewidth=0.6, linestyle="--", zorder=1)
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [row["method"] for row in subset] if setting == "controlled" else []
        )
        axis.set_xlabel(SETTING_META[setting][2])
        axis.text(
            0.02,
            1.03,
            SETTING_META[setting][3],
            transform=axis.transAxes,
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    fig.subplots_adjust(wspace=0.18)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"causal_selectivity.{suffix}", transparent=True)
    plt.close(fig)


def _metric_mean(runs: Sequence[dict], variant: str, rep: str, metric: str) -> float:
    return float(np.mean([run["proxy_fidelity"][variant][rep][metric] for run in runs]))


def proxy_figure(results: dict, output_dir: Path, style: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    settings = ("controlled", "vision", "language")
    reps = {"controlled": "latent4", "vision": "jl32", "language": "semantic4"}
    metrics = (
        ("module_cosine_mean", "Module cosine"),
        ("task_ranking_spearman_mean", "Ranking Spearman"),
        ("topk_overlap_mean", "Top-5% overlap"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.25), sharey=True)
    x = np.arange(len(settings))
    for axis, (metric, label) in zip(axes, metrics):
        raw = [_metric_mean(results[s], "late_raw", reps[s], metric) for s in settings]
        corrected = [
            _metric_mean(results[s], "late_ief", reps[s], metric) for s in settings
        ]
        axis.plot(x, raw, color=PRIMARY, marker="o", label="Late raw EF")
        axis.plot(
            x,
            corrected,
            color=SECONDARY,
            marker="s",
            linestyle="--",
            label="Late online iEF",
        )
        axis.set_xticks(
            x, ["Controlled", "Vision", "Language"], rotation=25, ha="right"
        )
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].legend(frameon=False, loc="lower left")
    fig.subplots_adjust(wspace=0.28)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"online_proxy_fidelity.{suffix}", transparent=True)
    plt.close(fig)


def kernel_figure(results: dict, output_dir: Path, style: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    settings = ("controlled", "vision", "language")
    labels = ("Controlled", "Vision", "Language")
    primary_rep = {"controlled": "latent4", "vision": "jl32", "language": "semantic4"}
    random_rep = {"controlled": "random4", "vision": "random32", "language": "random4"}
    x = np.arange(3)
    primary = [
        float(
            np.mean(
                [
                    run["kernel_fidelity"][primary_rep[s]]["rho_spearman"]
                    for run in results[s]
                ]
            )
        )
        for s in settings
    ]
    random = [
        float(
            np.mean(
                [
                    run["kernel_fidelity"][random_rep[s]]["rho_spearman"]
                    for run in results[s]
                ]
            )
        )
        for s in settings
    ]
    onehot = [
        finite_mean(
            [run["kernel_fidelity"]["onehot"]["rho_spearman"] for run in results[s]]
        )
        for s in settings
    ]
    fig, axis = plt.subplots(figsize=(3.45, 2.8))
    axis.axhline(0, color="#777777", linewidth=0.6, linestyle="--")
    axis.plot(
        x, primary, color=PRIMARY, marker="o", label="Specified compressed geometry"
    )
    axis.plot(
        x, random, color=NEUTRAL, marker="x", linestyle="--", label="Random geometry"
    )
    finite = np.isfinite(onehot)
    axis.scatter(
        x[finite],
        np.asarray(onehot)[finite],
        color=SECONDARY,
        marker="s",
        label="One-hot",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Tangent-cosine Spearman")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
    )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"kernel_fidelity.{suffix}", transparent=True)
    plt.close(fig)


def aggregate_json(results: dict, causal_rows: Sequence[dict]) -> dict:
    vision_accuracy = [
        run["causal"]["baseline_accuracy_mean"] for run in results["vision"]
    ]
    return {
        "seed_counts": {setting: len(runs) for setting, runs in results.items()},
        "vision_accuracy_mean": float(np.mean(vision_accuracy)),
        "vision_accuracy_range": [
            float(min(vision_accuracy)),
            float(max(vision_accuracy)),
        ],
        "causal_summary": list(causal_rows),
    }


def run(args: argparse.Namespace) -> None:
    artifact_dir = Path(args.artifact_dir)
    results = aggregate(artifact_dir)
    rows = causal_summary(results)
    save_table(artifact_dir / "causal_summary.csv", rows)
    (artifact_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_json(results, rows), indent=2) + "\n"
    )
    style = Path(args.style)
    causal_figure(results, rows, artifact_dir, style)
    proxy_figure(results, artifact_dir, style)
    kernel_figure(results, artifact_dir, style)
    print(f"FIGURES={artifact_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default=str(arc_artifact_dir("01_exploratory"))
    )
    parser.add_argument(
        "--style",
        default="/home/davwis/.codex/plugins/cache/local-skills/local-scientific/0.1.0+codex.local/skills/sci-scientific-visualization/assets/paper.mplstyle",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
