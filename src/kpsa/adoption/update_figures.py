"""Figures for the research update (one per question), SVG for the web and PNG for LaTeX."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ART = Path.home() / "main/workspace/kpsa-paper-internal/05_adoption_demo/artifacts/update"
STYLE = Path.home() / ".claude/skills/sci-scientific-visualization/assets/paper.mplstyle"
C = {"supervised": "#3F21B6", "dino": "#8C7AD3"}
LABEL = {"supervised": "Supervised ViT-B/16", "dino": "DINO ViT-B/16"}
BLOCKS = np.arange(1, 13)
W = 6.8


def setup():
    if STYLE.exists():
        plt.style.use(str(STYLE))
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(ART / f"{name}.{ext}", dpi=220, bbox_inches="tight", transparent=ext == "svg")
    plt.close(fig)


def block_axis(ax):
    ax.set_xticks(BLOCKS)
    ax.set_xlim(0.6, 12.4)
    ax.set_xlabel("Transformer block")


def fig_stability(S):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for ax, m in zip(axes, ("supervised", "dino")):
        q = S[m]["q1"]
        spread = np.array(q["class_spread"][:12])
        ax.fill_between(BLOCKS, spread[:, 1], spread[:, 3], color=C[m], alpha=0.15, lw=0)
        ax.plot(BLOCKS, q["same_class"][:12], color=C[m], lw=1.6, marker="o", ms=3, label="Same class")
        ax.plot(BLOCKS, q["same_group"][:12], color=C[m], lw=1.2, ls="--", label="Same supergroup, other class")
        ax.plot(BLOCKS, q["other_group"][:12], color="#8a8a8a", lw=1.2, ls=":", label="Different supergroups")
        ax.set_title(LABEL[m])
        block_axis(ax)
    axes[0].set_ylabel("Profile similarity\n(Bhattacharyya coefficient)")
    axes[1].legend(frameon=False, loc="lower left")
    save(fig, "fig1-stability")


def sel(S, m, key):
    obs = np.array([d[key] for d in S[m]["q2"][:12]])
    null = np.array([d[key] for d in S[m]["q2_null"][:12]])
    return obs - null


def fig_selectivity(S, levels):
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.42})
    ax = axes[0]
    for m in ("supervised", "dino"):
        ex = sel(S, m, "class")
        ax.plot(BLOCKS, ex, color=C[m], lw=1.6, marker="o", ms=3, label=LABEL[m])
    ax.set_ylabel("Class information (bits)")
    ax.set_title("Selectivity for classes")
    ax.legend(frameon=False)
    block_axis(ax)
    styles = {"250": "-", "50": "--"}
    for ax, m in zip(axes[1:], ("supervised", "dino")):
        for k in ("250", "50"):
            cls = sel(S, m, k + "_classes")
            g = sel(S, m, k)
            ax.plot(BLOCKS, g / cls, color=C[m], ls=styles[k], lw=1.4,
                    label=f"{levels[k]} supergroups (≤{k} classes)")
        ax.set_ylim(0, 1.05)
        ax.set_title(LABEL[m])
        block_axis(ax)
        ax.set_xticks([1, 4, 8, 12])
    axes[0].set_xticks([1, 4, 8, 12])
    axes[1].set_ylabel("Share carried by supergroups")
    axes[2].legend(frameon=False, loc="lower left")
    save(fig, "fig2-selectivity")


def fig_depth_space(S):
    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.35)
    for i, m in enumerate(("supervised", "dino")):
        ax = fig.add_subplot(gs[0, i])
        cm = np.array(S[m]["q3"]["class_layer_mass"])
        depth = np.array(S[m]["q3"]["depth"]["class_means"])
        img = cm[np.argsort(depth)]
        ax.imshow(img, aspect="auto", cmap="Purples", interpolation="nearest",
                  extent=(0.5, 12.5, 1000, 0), vmin=0, vmax=np.percentile(img, 99))
        ax.set_xticks([1, 4, 8, 12])
        ax.set_xlabel("Transformer block")
        ax.set_title(f"{LABEL[m]}\nη² of mean depth: {S[m]['q3']['depth']['eta2']:.2f}")
        if i == 0:
            ax.set_ylabel("Classes, sorted by mean depth")
    ax = fig.add_subplot(gs[0, 2])
    for m in ("supervised", "dino"):
        col = np.array(S[m]["q3"]["col"]["class_means"])
        row = np.array(S[m]["q3"]["row"]["class_means"])
        ax.scatter(col + 1, row + 1, s=3, color=C[m], alpha=0.5, lw=0,
                   label=f"{LABEL[m]} (η² row {S[m]['q3']['row']['eta2']:.2f}, col {S[m]['q3']['col']['eta2']:.2f})")
    allc = np.concatenate([np.array(S[m]["q3"]["col"]["class_means"]) for m in ("supervised", "dino")]) + 1
    allr = np.concatenate([np.array(S[m]["q3"]["row"]["class_means"]) for m in ("supervised", "dino")]) + 1
    ax.set_xlim(allc.min() - 0.3, allc.max() + 0.3)
    ax.set_ylim(allr.max() + 0.3, allr.min() - 0.3)
    ax.set_aspect("equal")
    names = S["names"]
    row = np.array(S["supervised"]["q3"]["row"]["class_means"]) + 1
    col = np.array(S["supervised"]["q3"]["col"]["class_means"]) + 1
    for i in list(np.argsort(row)[:2]) + list(np.argsort(row)[-2:]):
        ax.annotate(names[i], (col[i], row[i]), xytext=(4, 0), textcoords="offset points", fontsize=6, color="#333333", va="center")
    ax.set_xlabel("Horizontal patch index")
    ax.set_ylabel("Vertical patch index")
    ax.set_title("Class-mean centre of sensitivity")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), markerscale=3, fontsize=6.3)
    save(fig, "fig3-depth-space")


def fig_hierarchy(S):
    fig, ax = plt.subplots(figsize=(W * 0.55, 2.3))
    for m in ("supervised", "dino"):
        ax.plot(BLOCKS, S[m]["q4_rho"][:12], color=C[m], lw=1.6, marker="o", ms=3, label=LABEL[m])
        ax.axhline(S[m]["q4_rho"][12], color=C[m], lw=0.8, ls=":")
    ax.set_ylabel("Spearman ρ with WordNet\ncommon-ancestor depth")
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.legend(frameon=False)
    block_axis(ax)
    save(fig, "fig4-hierarchy")


def fig_sparsity(S):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.35})
    bins = np.logspace(2.5, np.log10(36864), 40)
    ax = axes[0]
    for m in ("supervised", "dino"):
        q = S[m]["q5"]
        img = np.array(q["eff_groups_image"])
        cls = np.array(q["eff_groups_class"])
        ax.hist(img, bins=bins, color=C[m], alpha=0.35, weights=np.ones_like(img) / len(img), label=f"{LABEL[m].split()[0]}: single images")
        ax.hist(cls, bins=bins, color=C[m], histtype="step", lw=1.4, weights=np.ones_like(cls) / len(cls), label=f"{LABEL[m].split()[0]}: class means")
    ax.set_xscale("log")
    ax.set_xlabel("Effective number of parameter groups (of 36,864)")
    ax.set_ylabel("Fraction of images or classes")
    ax.set_title("How many groups an image or class depends on")
    ax.set_ylim(0, 0.68)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)
    ax = axes[1]
    for m in ("supervised", "dino"):
        q = S[m]["q5"]
        eff = np.array(q["eff_classes_feature"])
        mass = np.array(q["feature_mass"])
        layer = np.arange(len(eff)) // 3072
        ok = np.isfinite(eff) & (eff > 0) & np.isfinite(mass)
        gm = [np.exp((mass[ok & (layer == b)] * np.log(eff[ok & (layer == b)])).sum() / mass[ok & (layer == b)].sum()) for b in range(12)]
        ax.plot(BLOCKS, gm, color=C[m], lw=1.6, marker="o", ms=3, label=LABEL[m])
    ax.set_yscale("log")
    ax.set_ylim(1, 1000)
    ax.set_ylabel("Effective number of classes\nper group (mass-weighted)")
    ax.set_title("How many classes a group serves")
    ax.legend(frameon=False, loc="lower left")
    block_axis(ax)
    save(fig, "fig5-sparsity")


def main():
    setup()
    S = json.loads((ART / "stats.json").read_text())
    levels = {k: len(set(v)) for k, v in S["levels"].items()}
    fig_stability(S)
    fig_selectivity(S, levels)
    fig_depth_space(S)
    fig_hierarchy(S)
    fig_sparsity(S)


if __name__ == "__main__":
    main()
