"""Generate paper-facing tables, figures, and a report for the refined arc."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import save_json
from .refined_analysis import paired_t_summary, summarize_precision_records

PRIMARY = "#3F21B6"
SECONDARY = "#8C7AD3"
GRAY = "#777777"
LIGHT = "#BDBDBD"


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def effect_summary(left, right) -> dict:
    return paired_t_summary(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))


def errorbar(ax, x, summary, *, color=PRIMARY, marker="o", label=None):
    ax.errorbar(
        x,
        summary["mean"],
        yerr=[
            [summary["mean"] - summary["lower_95"]],
            [summary["upper_95"] - summary["mean"]],
        ],
        fmt=marker,
        color=color,
        capsize=2,
        linewidth=1,
        markersize=4,
        label=label,
    )


def cold_effects(artifacts: Path) -> list[dict]:
    rows = []
    controlled = []
    for path in sorted((artifacts / "controlled").glob("seed*.json")):
        payload = read_json(path)["cold_query"]["fine"]
        controlled.append(
            {
                metric: payload["semantic6"][f"mean_{metric}"]
                - payload["scalar_mass"][f"mean_{metric}"]
                for metric in ("spearman", "topk_recall", "ndcg")
            }
        )
    for metric in ("spearman", "topk_recall", "ndcg"):
        rows.append(
            {
                "setting": "Controlled decoder (3 seeds)",
                "representation": "Known semantic mixture",
                "functional": "output",
                "metric": metric,
                **paired_t_summary([item[metric] for item in controlled]),
            }
        )

    language = (
        ("Qwen2.5-3B", "Token PCA", "qwen2_5_3b/test.json", "affine_semantic"),
        ("Qwen3-1.7B", "Token PCA", "qwen3_1_7b/test.json", "affine_semantic"),
        (
            "Qwen2.5-3B",
            "Qwen3 task descriptions",
            "qwen2_5_3b_description/validation.json",
            "rbf_semantic",
        ),
        (
            "Qwen3-1.7B",
            "Qwen3 task descriptions",
            "qwen3_1_7b_description/validation.json",
            "rbf_semantic",
        ),
    )
    for setting, representation, relative, method in language:
        payload = read_json(artifacts / relative)["cold_query"]
        for metric in ("spearman", "topk_recall", "ndcg"):
            rows.append(
                {
                    "setting": setting,
                    "representation": representation,
                    "functional": "next-token loss",
                    "metric": metric,
                    **effect_summary(
                        payload[method][metric], payload["scalar_mass"][metric]
                    ),
                }
            )

    vision = read_json(artifacts / "vision/imagenet200_vitb16_dinov2.json")
    for functional, result in vision["functionals"].items():
        methods = result["representations"]["frozen_input_encoder"]
        for metric in ("spearman", "topk_recall", "ndcg"):
            rows.append(
                {
                    "setting": "ViT-B/16 ImageNet",
                    "representation": "DINOv2-Small",
                    "functional": functional,
                    "metric": metric,
                    **effect_summary(
                        methods["affine_semantic"][metric],
                        methods["scalar_mass"][metric],
                    ),
                }
            )
    return rows


def precision_effects(artifacts: Path) -> list[dict]:
    rows = []
    experiments = (
        ("Qwen2.5-3B", "Token PCA", "validation", "qwen2_5_3b/validation.json"),
        ("Qwen2.5-3B", "Token PCA", "test", "qwen2_5_3b/test.json"),
        ("Qwen3-1.7B", "Token PCA", "validation", "qwen3_1_7b/validation.json"),
        ("Qwen3-1.7B", "Token PCA", "test", "qwen3_1_7b/test.json"),
        (
            "Qwen2.5-3B",
            "Qwen3 task descriptions",
            "validation",
            "qwen2_5_3b_description/validation.json",
        ),
        (
            "Qwen3-1.7B",
            "Qwen3 task descriptions",
            "validation",
            "qwen3_1_7b_description/validation.json",
        ),
    )
    for model, representation, split, relative in experiments:
        payload = read_json(artifacts / relative)
        summary = summarize_precision_records(payload["precision"]["records"])
        for method, budgets in summary["paired_vs_scalar"].items():
            for budget, values in budgets.items():
                rows.append(
                    {
                        "model": model,
                        "representation": representation,
                        "split": split,
                        "method": method,
                        "budget": float(budget),
                        **values,
                    }
                )
    return rows


def causal_effects(artifacts: Path) -> list[dict]:
    payload = read_json(
        artifacts / "vision/imagenet200_vitb16_dinov2_causal.json"
    )
    grouped = defaultdict(dict)
    for row in payload["records"]:
        key = (
            row["functional"],
            float(row["requested_parameter_fraction"]),
            int(row["query_column"]),
        )
        grouped[key][row["method"]] = float(row["degradation"])
    output = []
    for functional in ("class_logit", "margin", "loss"):
        for budget in (0.0002, 0.001):
            query_rows = [
                methods
                for (local_functional, local_budget, _), methods in grouped.items()
                if local_functional == functional and local_budget == budget
            ]
            for method in (
                "dinov2_affine",
                "scalar_mass",
                "direct_gradient_oracle",
                "weight_magnitude",
                "random",
            ):
                if method == "random":
                    values = [
                        np.mean([row[f"random_{i}"] for i in range(3)])
                        for row in query_rows
                    ]
                else:
                    values = [row[method] for row in query_rows]
                output.append(
                    {
                        "functional": functional,
                        "budget": budget,
                        "method": method,
                        **paired_t_summary(values),
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_cold(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), constrained_layout=True)
    controlled = next(
        row
        for row in rows
        if row["setting"].startswith("Controlled") and row["metric"] == "spearman"
    )
    errorbar(axes[0], 0, controlled)
    axes[0].axhline(0, color=LIGHT, linewidth=0.8)
    axes[0].set_xticks([0], ["Controlled\ndecoder"])
    axes[0].set_ylabel(r"Vector $-$ scalar $\Delta$ Spearman")
    axes[0].set_xlim(-0.7, 0.7)

    chosen = [
        row
        for row in rows
        if row["metric"] == "spearman" and not row["setting"].startswith("Controlled")
    ]
    labels = []
    for index, row in enumerate(chosen):
        errorbar(axes[1], index, row)
        label = row["setting"].replace(" ImageNet", "")
        if row["functional"] != "next-token loss":
            label += "\n" + row["functional"].replace("class_", "")
        else:
            label += "\n" + ("description" if "descriptions" in row["representation"] else "token PCA")
        labels.append(label)
    axes[1].axhline(0, color=LIGHT, linewidth=0.8)
    axes[1].set_xticks(range(len(chosen)), labels, rotation=42, ha="right")
    axes[1].set_ylabel(r"Vector $-$ scalar $\Delta$ Spearman")
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + suffix))
    plt.close(fig)


def plot_precision(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ("Qwen2.5-3B", "Qwen3-1.7B")):
        selected = [
            row
            for row in rows
            if row["model"] == model
            and (
                (row["representation"] == "Token PCA" and row["split"] == "test" and row["method"] in {"affine_semantic", "direct_gradient_oracle"})
                or ("descriptions" in row["representation"] and row["method"] == "rbf_semantic")
            )
        ]
        styles = {
            "affine_semantic": (PRIMARY, "o", "Token PCA"),
            "rbf_semantic": (SECONDARY, "s", "Description RBF"),
            "direct_gradient_oracle": (GRAY, "^", "Direct gradient"),
        }
        for method, (color, marker, label) in styles.items():
            local = sorted(
                [row for row in selected if row["method"] == method],
                key=lambda row: row["budget"],
            )
            if not local:
                continue
            for index, row in enumerate(local):
                errorbar(
                    ax,
                    index,
                    row,
                    color=color,
                    marker=marker,
                    label=label if index == 0 else None,
                )
            ax.plot(
                range(len(local)),
                [row["mean"] for row in local],
                color=color,
                linewidth=0.8,
            )
        ax.axhline(0, color=LIGHT, linewidth=0.8)
        ax.set_title(model, fontsize=8)
        ax.set_xlabel("High-precision scope (%)")
        ax.set_xticks([0, 1], ["10", "30"])
        ax.set_xlim(-0.18, 1.18)
    axes[0].set_ylabel(r"NLL increase difference vs. scalar ($\downarrow$)")
    axes[1].legend(frameon=False, loc="upper left")
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + suffix))
    plt.close(fig)


def plot_causal(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=False, constrained_layout=True)
    names = {
        "dinov2_affine": "DINOv2",
        "scalar_mass": "Scalar",
        "direct_gradient_oracle": "Direct grad.",
        "random": "Random",
    }
    colors = [PRIMARY, SECONDARY, GRAY, LIGHT]
    for ax, functional in zip(axes, ("class_logit", "margin", "loss")):
        for offset, (budget, marker) in enumerate(((0.0002, "o"), (0.001, "s"))):
            for index, method in enumerate(names):
                row = next(
                    item
                    for item in rows
                    if item["functional"] == functional
                    and item["budget"] == budget
                    and item["method"] == method
                )
                x = index + (offset - 0.5) * 0.22
                errorbar(ax, x, row, color=colors[index], marker=marker)
        ax.axhline(0, color=LIGHT, linewidth=0.8)
        ax.set_xticks(range(len(names)), names.values(), rotation=35, ha="right")
        ax.set_title(functional.replace("_", " "), fontsize=8)
    axes[0].set_ylabel("Target degradation after mean ablation")
    axes[-1].plot([], [], "o", color="#333333", label="0.02%")
    axes[-1].plot([], [], "s", color="#333333", label="0.1%")
    axes[-1].legend(frameon=False, loc="upper left", title="Parameter budget")
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + suffix))
    plt.close(fig)


def report_text(cold: list[dict], precision: list[dict], causal: list[dict]) -> str:
    def find_cold(setting, representation, functional, metric="spearman"):
        return next(
            row
            for row in cold
            if row["setting"] == setting
            and row["representation"] == representation
            and row["functional"] == functional
            and row["metric"] == metric
        )

    controlled = find_cold(
        "Controlled decoder (3 seeds)", "Known semantic mixture", "output"
    )
    vision = find_cold("ViT-B/16 ImageNet", "DINOv2-Small", "class_logit")
    q25 = find_cold("Qwen2.5-3B", "Qwen3 task descriptions", "next-token loss")
    q3 = find_cold("Qwen3-1.7B", "Qwen3 task descriptions", "next-token loss")
    return f"""# Refined empirical report: representation-conditioned sensitivity

## Decision

The canonical normalized sensitivity atlas is implemented faithfully and is numerically stable, but the current evidence does **not** support a paper claim that the representation-valued component improves modern-model applications over scalar sensitivity mass. The controlled decoder shows a small repeatable effect; both modern language models and the ImageNet ViT collapse almost entirely to the scalar baseline. Under the specification's decision gate, neither precision nor mechanistic localization is yet a positive representation-specific pillar.

## Evidence at a glance

| Setting | Scale | Representation | Vector minus scalar Spearman (95% CI) | Downstream result |
|---|---:|---|---:|---|
| Controlled compositional decoder | 0.87M parameters, 3 seeds | Known 6D task semantics | {controlled['mean']:+.5f} [{controlled['lower_95']:+.5f}, {controlled['upper_95']:+.5f}] | Small top-tail retrieval gain; no nDCG gain |
| Qwen2.5-3B | 3.09B parameters, 20 domains | Qwen3-Embedding descriptions, RBF | {q25['mean']:+.6f} [{q25['lower_95']:+.6f}, {q25['upper_95']:+.6f}] | No mixed-precision gain over scalar on validation |
| Qwen3-1.7B | 1.72B parameters, 20 domains | Qwen3-Embedding descriptions, RBF | {q3['mean']:+.6f} [{q3['lower_95']:+.6f}, {q3['upper_95']:+.6f}] | No mixed-precision gain over scalar on validation |
| ViT-B/16, 200 ImageNet classes | 86.6M parameters | independent DINOv2-Small | {vision['mean']:+.6f} [{vision['lower_95']:+.6f}, {vision['upper_95']:+.6f}] | No causal gain over scalar under block-only mean ablation |

## Canonical-object validation

The estimator uses per-example group-relative squared-gradient shares, so every profile sums to one before averaging. The controlled Transformer uses 11,629 complete fine groups; the Qwen2.5-3B and Qwen3-1.7B models use 955,776 and 619,904 complete coupled-MLP/row groups; the ViT uses 161,746 complete groups. Maximum full-gradient partition discrepancies are approximately 1e-6 or below. Controlled fine-to-coarse aggregation is exact up to floating-point precision, and reference stability rises monotonically with sample count.

## Precision allocation

Across Qwen2.5-3B and Qwen3-1.7B, both token-distribution PCA and genuine Qwen3-Embedding task descriptions yield cold-query profiles close to scalar mass. The original token-PCA protocol was frozen on validation and confirmed on untouched test data. At both 10% and 30% high-precision budgets, vector-versus-scalar NLL intervals include zero or favor scalar. The direct held-out-gradient ranking also fails to beat scalar and is significantly worse in parts of the Qwen2.5 test, showing that this fake-quantization intervention is not aligned with the local gradient-energy oracle. This rules out precision as a positive embedding pillar in the present design.

## Vision localization and causal intervention

The primary vision representation is a full 384D frozen DINOv2-Small embedding trained by a self-supervised representation objective. Target-ViT penultimate CLS features, target class-logit vectors, and DINOv2 PCA-32 are ablations. All continuous spaces produce only tiny improvements over scalar mass. The causal check mean-ablates ranked groups inside transformer blocks, excluding the patch embedder, final norm, and classifier head. DINOv2 and scalar rankings have indistinguishable causal effects at 0.02% and 0.1% parameter budgets. Direct gradients beat scalar for margin and loss at the smaller budget, so the intervention has power to detect a better query-specific ranking. Weight magnitude is highly destructive under this intervention and must remain a strong baseline.

## Claim boundaries

- **Supported:** the normalized group-share object is well defined, additive across nested partitions, stable with reference size, and identifies causally important groups better than random in several settings.
- **Suggestive:** known semantic task coordinates yield a small representation-specific retrieval effect in the controlled decoder.
- **Unsupported:** continuous encoder geometry provides practically useful cold-query localization or task-conditioned precision beyond scalar mass on the tested modern models.
- **Unsupported:** the current two-pillar embedding paper described by the specification is empirically ready.

The real-model intervals treat tasks/classes as the sampling unit and condition on one checkpoint per architecture. They do not measure variation across pretrained checkpoints. The precision experiment uses simulated quantization rather than hardware kernels; the causal result uses 24 query images and parameter mean ablation. A 10B–15B replication was not run because the smaller workhorse and architecture replication both fail the prerequisite direct-oracle and vector-over-scalar checks.

## Artifacts

- `cold_query_effects.csv`: paired vector-minus-scalar retrieval effects.
- `precision_effects.csv`: paired NLL differences from scalar allocation.
- `causal_effects.csv`: absolute causal degradation summaries.
- `figure_cold_query.*`, `figure_precision.*`, and `figure_causal.*`: PDF and PNG figures.
- `summary.json`: machine-readable copies of every plotted interval.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style = Path(
        "/home/davwis/.codex/plugins/cache/local-skills/local-scientific/0.1.0+codex.local/skills/sci-scientific-visualization/assets/paper.mplstyle"
    )
    if style.exists():
        plt.style.use(style)
    cold = cold_effects(args.artifacts)
    precision = precision_effects(args.artifacts)
    causal = causal_effects(args.artifacts)
    write_csv(args.output / "cold_query_effects.csv", cold)
    write_csv(args.output / "precision_effects.csv", precision)
    write_csv(args.output / "causal_effects.csv", causal)
    save_json(
        args.output / "summary.json",
        {"cold_query": cold, "precision": precision, "causal": causal},
    )
    plot_cold(cold, args.output / "figure_cold_query")
    plot_precision(precision, args.output / "figure_precision")
    plot_causal(causal, args.output / "figure_causal")


if __name__ == "__main__":
    main()
