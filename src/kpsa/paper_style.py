"""Shared visual language for KPSA paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# KPSA variants carry the strongest chroma. Controls use quieter blue/teal,
# orange, and red hues and never rely on color alone.
SEMANTIC = "#4B2E83"
CATEGORICAL = "#7656B5"
SEMANTIC_LIGHT = "#A89AD1"
CONSTANT = "#39858C"
SHUFFLE = "#78B7C5"
NEAREST = "#D58A62"
DIRECT = "#C96868"
TEXT = "#20232A"
GRID = "#DCE9EC"
WHITE = "#FFFFFF"

METHOD_STYLES = {
    "Semantic KPSA": {"color": SEMANTIC, "marker": "o", "linestyle": "-"},
    "Categorical KPSA": {"color": CATEGORICAL, "marker": "s", "linestyle": "--"},
    "KPSA-1NN": {"color": NEAREST, "marker": "^", "linestyle": ":"},
    "Constant sensitivity": {"color": CONSTANT, "marker": "x", "linestyle": "-."},
    "Matched shuffle": {"color": SHUFFLE, "marker": "D", "linestyle": (0, (2, 2))},
    "Direct gradient": {"color": DIRECT, "marker": "P", "linestyle": "-"},
    "Activation attribution": {"color": DIRECT, "marker": "P", "linestyle": "-"},
}


def apply_paper_style() -> None:
    """Apply the single rcParams contract used by every paper plot."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 6.8,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.2,
            "errorbar.capsize": 2.0,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_style(label: str, **overrides):
    style = dict(METHOD_STYLES[label])
    style.update(overrides)
    return style


def panel_title(axis, letter: str, title: str) -> None:
    axis.set_title(f"{letter}  {title}", loc="left", pad=4)


def finish_axis(axis, *, zero_line: bool = False, grid: str = "y") -> None:
    if zero_line:
        axis.axhline(0, color=GRID, linewidth=0.75, zorder=0)
    if grid:
        axis.grid(axis=grid, color=GRID, linewidth=0.55, alpha=0.9, zorder=0)
    axis.set_axisbelow(True)


def outside_legend(fig, axes, *, ncol=None, fontsize=6.8, y=1.045):
    """Place one deduplicated legend above a multipanel figure."""
    axes = np.asarray(axes, dtype=object).reshape(-1)
    handles_by_label = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                handles_by_label.setdefault(label, handle)
    labels = list(handles_by_label)
    if labels:
        fig.legend(
            [handles_by_label[label] for label in labels],
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=ncol or len(labels),
            frameon=False,
            fontsize=fontsize,
            handlelength=2.2,
            columnspacing=1.05,
            handletextpad=0.45,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    else:
        fig.tight_layout()


def save_plot(fig, output: Path | str) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            output.with_suffix(f".{extension}"),
            dpi=350,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.025,
        )
    plt.close(fig)


apply_paper_style()
