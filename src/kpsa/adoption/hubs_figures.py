"""Figures for research update 2 (always-sensitive Qwen3 features)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from kpsa.adoption.update_figures import STYLE, W  # noqa: E402

from kpsa.adoption.qwen_hubs import ART  # noqa: E402

BRAND, LIGHT, GREY = "#3F21B6", "#8C7AD3", "#9a9a9a"


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


def fig_shares(S):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3})
    ax = axes[0]
    n = 40
    mean = np.array(S["mean_sorted"][:n]) * 100
    est = np.array(S["est_mean_sorted"][:n]) * 100
    hub = np.array(S["mean_sorted_is_hub"][:n])
    x = np.arange(1, n + 1)
    ax.bar(x[hub], mean[hub], color=BRAND, width=0.8, label="15 always-sensitive features")
    ax.bar(x[~hub], mean[~hub], color=GREY, width=0.8, label="other features")
    ax.scatter(x, est, s=7, color="#111111", zorder=3, lw=0, label="KPSA estimate")
    ax.set_xlabel(f"Feature rank by mean sensitivity (of {S['n_features']:,})")
    ax.set_ylabel("Mean share of sensitivity (%)")
    ax.set_xlim(0, n + 1)
    ax.legend(frameon=False)
    ax.set_title("Mean sensitivity profile over 300,000 contexts")
    ax = axes[1]
    bins = np.linspace(0.2, 0.8, 37)
    e = np.array(S["exact_share"])
    k = np.array(S["est_share"])
    ax.hist(e, bins=bins, color=BRAND, alpha=0.35, label="exact")
    ax.hist(k, bins=bins, color=BRAND, histtype="step", lw=1.4, label="KPSA estimate")
    ax.set_xlabel("Share of a token's sensitivity in the 15 features")
    ax.set_ylabel("Held-out tokens")
    ax.set_title("Per token (3,000 held-out tokens)")
    ax.set_ylim(0, 900)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig1-shares")


def fig_ablate(R):
    rand = np.array([r[0] for r in R["random_same_blocks"]])
    rand_acc = np.array([r[1] for r in R["random_same_blocks"]])
    rows = [
        ("Intact", R["intact"], None, GREY),
        ("15 random features,\nsame blocks", (rand.mean(), rand_acc.mean()), (rand.min(), rand.max()), GREY),
        ("Next 15 by\nmean sensitivity", R["next15"], None, GREY),
        ("9 features of block 28", R["by_block"]["28"], None, LIGHT),
        ("6 features of block 3", R["by_block"]["3"], None, BRAND),
        ("All 15", R["all"], None, BRAND),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(rows))[::-1]
    for ax, j, label in ((axes[0], 0, "Next-token loss (nats)"), (axes[1], 1, "Top-1 accuracy")):
        vals = [r[1][j] for r in rows]
        ax.barh(y, vals, color=[r[3] for r in rows], height=0.65)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.2f}" if j == 0 else f" {v:.0%}", va="center", fontsize=6.5)
        ax.set_xlabel(label)
        ax.axvline(rows[0][1][j], color="#555555", lw=0.6, ls=":")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].set_xlim(0, 8)
    axes[1].set_xlim(0, 0.52)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    save(fig, "fig2-ablation")


def main():
    setup()
    fig_shares(json.loads((ART / "shares.json").read_text()))
    fig_ablate(json.loads((ART / "ablate.json").read_text()))


if __name__ == "__main__":
    main()
