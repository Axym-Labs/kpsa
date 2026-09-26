from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .common import arc_artifact_dir

SETTINGS = ("controlled", "vision", "language")
LABELS = {
    "controlled": "Controlled Transformer",
    "vision": "DINOv2 / CIFAR-100",
    "language": "Qwen3-1.7B",
}
PRIMARY_FRACTION = {"controlled": 0.05, "vision": 0.05, "language": 0.01}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _one(rows: list[dict], **conditions) -> dict:
    matches = [
        row
        for row in rows
        if all(
            abs(float(row[key]) - float(value)) < 1e-9
            if isinstance(value, float)
            else row[key] == value
            for key, value in conditions.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def _f(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def _tex_pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}\\%"


def _source_table(payload: dict) -> tuple[str, str]:
    rows = []
    tex = []
    semantic_names = {
        "controlled": "structured4",
        "vision": "semantic32",
        "language": "semantic4",
    }
    jl_names = {"controlled": "jl4", "vision": "jl32", "language": "jl4"}
    for setting in SETTINGS:
        semantic = _one(
            payload["source_fidelity"],
            setting=setting,
            representation=semantic_names[setting],
        )
        jl = _one(
            payload["source_fidelity"],
            setting=setting,
            representation=jl_names[setting],
        )
        rows.append(
            f"| {LABELS[setting]} | {_f(semantic['centered_kernel_alignment'])} / "
            f"{_f(jl['centered_kernel_alignment'])} | {_f(semantic['ranking_spearman_mean'])} / "
            f"{_f(jl['ranking_spearman_mean'])} | {_f(semantic['topk_overlap_mean'])} / "
            f"{_f(jl['topk_overlap_mean'])} | {_f(semantic['knn_preservation'])} / "
            f"{_f(jl['knn_preservation'])} |"
        )
        tex.append(
            f"{LABELS[setting]} & {_f(semantic['centered_kernel_alignment'])}/{_f(jl['centered_kernel_alignment'])} & "
            f"{_f(semantic['ranking_spearman_mean'])}/{_f(jl['ranking_spearman_mean'])} & "
            f"{_f(semantic['topk_overlap_mean'])}/{_f(jl['topk_overlap_mean'])} & "
            f"{_f(semantic['knn_preservation'])}/{_f(jl['knn_preservation'])} \\\\"
        )
    markdown = "\n".join(
        [
            "| Setting | CKA task/JL | Ranking $\\rho$ task/JL | Top-5% overlap task/JL | k-NN task/JL |",
            "|---|---:|---:|---:|---:|",
            *rows,
        ]
    )
    return markdown, "\n".join(tex)


def _interpretability_table(payload: dict) -> tuple[str, str]:
    methods = (
        "Post-hoc iEF task space",
        "Full one-hot atlas",
        "Matched JL",
        "Activation",
        "Online raw EF",
        "Random",
    )
    rows, tex = [], []
    for setting in SETTINGS:
        values = [
            _one(
                payload["interpretability"],
                setting=setting,
                method_family=method,
                fraction=PRIMARY_FRACTION[setting],
            )["mean"]
            for method in methods
        ]
        rows.append(
            f"| {LABELS[setting]} | " + " | ".join(_f(value) for value in values) + " |"
        )
        tex.append(
            f"{LABELS[setting]} & "
            + " & ".join(_f(value) for value in values)
            + " \\\\"
        )
    header = (
        "| Setting | iEF-task | Full | JL | Activation | Online | Random |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows), "\n".join(tex)


def _pruning_table(payload: dict) -> tuple[str, str]:
    methods = (
        "Post-hoc iEF task space",
        "Full one-hot atlas",
        "Matched JL",
        "Activation",
        "Random",
    )
    rows, tex = [], []
    for setting in SETTINGS:
        values = [
            _one(
                payload["pruning"],
                setting=setting,
                method_family=method,
                retained_fraction=0.75,
            )["mean"]
            for method in methods
        ]
        rows.append(
            f"| {LABELS[setting]} | " + " | ".join(_f(value) for value in values) + " |"
        )
        tex.append(
            f"{LABELS[setting]} & "
            + " & ".join(_f(value) for value in values)
            + " \\\\"
        )
    header = (
        "| Setting | iEF-task | Full | JL | Activation | Random |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows), "\n".join(tex)


def _cl_table(payload: dict) -> tuple[str, str]:
    methods = (
        "Post-hoc iEF task space",
        "Post-hoc raw EF",
        "Online raw EF",
        "Random",
        "No protection",
    )
    rows, tex = [], []
    for setting in SETTINGS:
        for order in ("related", "dissimilar"):
            values = [
                _one(
                    payload["continual_learning"],
                    setting=setting,
                    order=order,
                    method_family=method,
                )["average_forgetting"]
                for method in methods
            ]
            rows.append(
                f"| {LABELS[setting]} | {order} | "
                + " | ".join(_f(value, 4) for value in values)
                + " |"
            )
            tex.append(
                f"{LABELS[setting]} & {order} & "
                + " & ".join(_f(value, 4) for value in values)
                + " \\\\"
            )
    header = (
        "| Setting | Order | iEF-task | Raw EF | Online | Random | None |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows), "\n".join(tex)


def _efficiency_table(payload: dict) -> tuple[str, str]:
    rows, tex = [], []
    for setting in SETTINGS:
        row = _one(payload["efficiency"], setting=setting)
        values = (
            row["total_wall_seconds"] / 60,
            row["train_seconds"],
            row["posthoc_seconds"],
            row["online_overhead_upper_bound"] * 100,
            row["peak_cuda_allocated_bytes"] / 2**30,
            row["seconds_per_amortized_query"] * 1e6,
        )
        rows.append(
            f"| {LABELS[setting]} | {_f(values[0], 2)} | {_f(values[1], 1)} | "
            f"{_f(values[2], 2)} | {_f(values[3], 1)}% | {_f(values[4], 2)} | "
            f"{_f(values[5], 1)} |"
        )
        tex.append(
            f"{LABELS[setting]} & {_f(values[0], 2)} & {_f(values[1], 1)} & "
            f"{_f(values[2], 2)} & {_f(values[3], 1)}\\% & {_f(values[4], 2)} & "
            f"{_f(values[5], 1)} \\\\"
        )
    header = (
        "| Setting | Full run min | Train s | Post-hoc s | Online upper bound | Peak GiB | Query µs |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows), "\n".join(tex)


def _composition(
    payload: dict, setting: str, metric: str, group_kind: str | None = None
) -> float:
    values = [
        row["value"]
        for row in payload["composition"]
        if row["setting"] == setting
        and row["metric"] == metric
        and (group_kind is None or row.get("group_kind") == group_kind)
    ]
    return mean(values)


def _build_markdown(payload: dict, raw: dict[str, dict]) -> str:
    source_table, _ = _source_table(payload)
    interpretability_table, _ = _interpretability_table(payload)
    pruning_table, _ = _pruning_table(payload)
    cl_table, _ = _cl_table(payload)
    efficiency_table, _ = _efficiency_table(payload)
    c = payload["sanity"]["controlled"]
    vision_accuracy = payload["sanity"]["vision"]["accuracy"]
    language_exact = payload["sanity"]["language"]["exact_match"]
    language_tasks = raw["language"]["baseline_task_exact_match"]
    online = {row["setting"]: row for row in payload["online_fidelity"]}
    recovery = {
        row["setting"]: row["mean"]
        for row in payload["pruning_recovery"]
        if row["method_family"] == "Post-hoc iEF task space"
    }
    return f"""# Task-space neuronal embeddings: v3 exploratory report

## Decision

**NO-GO for paper-scale expansion of the current method.** The low-dimensional
task spaces preserve task-specific module rankings better than matched JL
sketches in all three settings, and the structural circuit-overlap diagnostics
follow the expected relatedness ordering. The predeclared claim is stronger,
however: one
compressed atlas must support interpretability, pruning, and continual-learning
protection across all settings. It does not. The proposal fails gates 2--6 and
only partially satisfies source-fidelity gate 1.

| Gate | Result | Evidence |
|---|---|---|
| 1. Source-object fidelity | Mixed | Semantic/known spaces beat matched JL in ranking and top-5% overlap in all settings, but lose CKA and k-NN in controlled and vision. |
| 2. Interpretability | Fail | iEF-task is not stronger than cheap baselines in controlled or language and is only comparable in vision. |
| 3. Pruning | Fail | At 75% retained features, iEF-task retention is {_f(_one(payload["pruning"], setting="vision", method_family="Post-hoc iEF task space", retained_fraction=0.75)["mean"])} in vision and {_f(_one(payload["pruning"], setting="language", method_family="Post-hoc iEF task space", retained_fraction=0.75)["mean"])} in language, below several controls. |
| 4. Continual learning | Fail | Protection helps some cells but is not consistently better than random/no protection or competitive with raw/online EF. |
| 5. Compression advantage | Fail | Compressed rankings retain signal, but downstream utility does not retain the advantage of the full atlas. |
| 6. Online approximation | Fail | Post-hoc/online top-5% overlap is {_f(online["controlled"]["topk_overlap_mean"])}, {_f(online["vision"]["topk_overlap_mean"])}, and {_f(online["language"]["topk_overlap_mean"])}. |
| 7. Optional NTK-angle signal | Not run | This optional diagnostic cannot rescue or overturn the failed core gates. |

## Experimental scope

The study uses one seed and the same disjoint MLP-feature modules across all
applications. The controlled model is a six-layer, width-384 decoder-only
Transformer with 14.2M parameters and 16 tasks constructed from four known
sequence primitives. The vision model is a 22.1M-parameter, 12-block DINOv2
small ViT fine-tuned for 10,000 class-homogeneous CIFAR-100 batches; the
requested official DINOv3 model was access-gated. Strict deterministic CUDA
kernels required freezing the 526,080-element positional-embedding tensor. The
language model is Qwen3-1.7B, fully fine-tuned with Adafactor for 600 steps on
six instruction-formatted tasks. Frozen MiniLM description embeddings define
the external vision and language task spaces independently of module
sensitivities.

Model sanity checks passed after debugging. Controlled held-out half-MSE is
{_f(c["model_loss"])}, versus {_f(c["zero_predictor_half_mse"])} for the zero
predictor and {_f(c["task_mean_predictor_half_mse"])} for task means. Balanced
vision accuracy is {_pct(vision_accuracy)}, versus 1% chance. Language exact
match is {_pct(language_exact)} overall, with per-task values
`{[round(float(value), 3) for value in language_tasks]}`; GSM8K remains at zero,
so language conclusions are especially low-powered.

All confidence intervals below bootstrap tasks within a single trained
checkpoint. They measure cross-task variation, not training-seed uncertainty.

## Source-object fidelity

Each module's task-balanced sensitivity row forms the full atlas $P$; the
compressed object is $E=P\\Phi$. Large-module metrics use 200,000 deterministic
module pairs and at most 512 k-NN queries, while linear CKA remains exact.
Entries show semantic/known task space followed by matched-dimensional JL.

{source_table}

The result is consistent across settings: task semantics help most for
task-specific ranking and top-circuit recovery, whereas JL often better
preserves global Euclidean geometry. Qwen is the clearest compression result
(semantic4 CKA {_f(_one(payload["source_fidelity"], setting="language", representation="semantic4")["centered_kernel_alignment"])}, ranking $\\rho$
{_f(_one(payload["source_fidelity"], setting="language", representation="semantic4")["ranking_spearman_mean"])}), but four dimensions for six tasks is
only modest compression. Source fidelity does not fully pass gate 1.

![Source-object fidelity](source_fidelity.png)

## Interpretability

The table reports mean-ablation task selectivity at 5% of MLP features for
controlled/vision and 1% for language; the figure and CSV include every tested
fraction. Higher is better.

{interpretability_table}

Controlled selectivity is effectively zero for iEF-task ({_f(_one(payload["interpretability"], setting="controlled", method_family="Post-hoc iEF task space", fraction=0.05)["mean"], 4)}),
whereas the full atlas and JL are numerically larger but uncertain. Vision is
causally nontrivial: iEF-task reaches {_f(_one(payload["interpretability"], setting="vision", method_family="Post-hoc iEF task space", fraction=0.05)["mean"])},
but activation ({_f(_one(payload["interpretability"], setting="vision", method_family="Activation", fraction=0.05)["mean"])}) and JL
({_f(_one(payload["interpretability"], setting="vision", method_family="Matched JL", fraction=0.05)["mean"])}) are at least as strong. Language iEF-task is
{_f(_one(payload["interpretability"], setting="language", method_family="Post-hoc iEF task space", fraction=0.01)["mean"])}, with a wide task-bootstrap interval,
versus {_f(_one(payload["interpretability"], setting="language", method_family="Activation", fraction=0.01)["mean"])} for activation. Gate 2 fails.

![Mean-ablation selectivity across circuit sizes](interpretability_selectivity.png)

## Structured pruning and composition

Relative target retention is baseline loss divided by pruned loss for
controlled/language and pruned accuracy divided by baseline accuracy for
vision. The table reports 75% retained MLP features; 1.0 denotes baseline
performance.

{pruning_table}

Controlled pruning is saturated: iEF-task preserves baseline loss, but so do
activation, raw EF, weight magnitude, and the full atlas. Vision collapses to
zero target accuracy through 50% retained features for every method; at 75%,
iEF-task ({_f(_one(payload["pruning"], setting="vision", method_family="Post-hoc iEF task space", retained_fraction=0.75)["mean"])}) trails JL
({_f(_one(payload["pruning"], setting="vision", method_family="Matched JL", retained_fraction=0.75)["mean"])}) and random
({_f(_one(payload["pruning"], setting="vision", method_family="Random", retained_fraction=0.75)["mean"])}). Language iEF-task retains
{_f(_one(payload["pruning"], setting="language", method_family="Post-hoc iEF task space", retained_fraction=0.75)["mean"])}, versus
{_f(_one(payload["pruning"], setting="language", method_family="Matched JL", retained_fraction=0.75)["mean"])} for JL and
{_f(_one(payload["pruning"], setting="language", method_family="Random", retained_fraction=0.75)["mean"])} for random.

Recovery at 25% retained features gives relative retention
{_f(recovery["controlled"])}, {_f(recovery["vision"])}, and
{_f(recovery["language"])} for controlled, vision, and language. The vision
recovery score is identical across iEF-task, activation, and random on only
five evaluation examples per class, so it is not discriminating evidence.

The structural composition probes match the expected ordering but do not
repair causal pruning. Controlled mixture circuits recover
{_pct(_composition(payload, "controlled", "direct_circuit_recall_in_primitive_union"))}
of their selected features from primitive-circuit unions (mean Jaccard
{_f(_composition(payload, "controlled", "direct_vs_primitive_union_jaccard"))}).
Vision related-class unions occupy {_pct(_composition(payload, "vision", "five_class_union_fraction", "related"))}
of features versus {_pct(_composition(payload, "vision", "five_class_union_fraction", "dissimilar"))}
for dissimilar unions. Language classification-task circuit Jaccard is
{_f(_composition(payload, "language", "pair_circuit_jaccard", "classification_family"))}
versus {_f(_composition(payload, "language", "pair_circuit_jaccard", "cross_family"))}
cross-family.

![Zero-shot pruning and iEF-task recovery markers](pruning_retention.png)

## Continual learning

The same soft-protection rule is used for every estimator. Lower forgetting is
better; controlled/language values are loss increases, whereas vision values
are accuracy drops.

{cl_table}

iEF-task improves over no protection in controlled-dissimilar, both vision
orders, and language-dissimilar, but loses or ties the relevant controls in the
remaining comparisons. In vision, online EF has lower forgetting than iEF-task
in both orders. In language, raw EF is lower in both orders. This is mixed
application evidence and fails the predeclared consistency gate.

![Average forgetting](continual_forgetting.png)

## Online approximation and efficiency

Online late raw-EF agrees weakly with post-hoc normalized sensitivity at the
top-circuit level. Mean module cosine is {_f(online["controlled"]["module_cosine_mean"])},
{_f(online["vision"]["module_cosine_mean"])}, and {_f(online["language"]["module_cosine_mean"])};
the corresponding top-5% overlaps are {_f(online["controlled"]["topk_overlap_mean"])},
{_f(online["vision"]["topk_overlap_mean"])}, and {_f(online["language"]["topk_overlap_mean"])}.
Language agreement is the clearest failure.

{efficiency_table}

The online-time number is an upper bound because it includes an explicit GPU
synchronization used for timing. Post-hoc atlas construction itself is cheap
on the deliberately small reference sets. The retained Qwen checkpoint is
3.78 GiB and dominates the 4.1 GiB artifact directory.

![Runtime, accelerator memory, and checkpoint size](efficiency.png)

## Debugging record and limitations

The study did not stop at first execution. The implementation was corrected
after five consequential failures: captured non-leaf activations broke model
replication; a CPU/GPU task-mixture mismatch broke controlled CUDA training;
quadratic fidelity materialization exceeded PyTorch's quantile limit; a
120-step SGD Qwen run produced zero exact match; and the first vision runs were
nondeterministic because DINOv2's trainable positional interpolation used CUDA
bicubic backward. The final paths clear captured state, keep task tensors on
device, use deterministic sampled fidelity at scale, use full-parameter
Adafactor for Qwen, and use strict deterministic vision kernels with frozen
positional embeddings. An aggregation return-placement bug was also caught
before report generation.

The remaining scientific limitations are material. There is one training seed;
reference sets are only 8 examples/task (controlled), 2/class (vision), and
2/task (language); language validation has 8 examples/task; CL updates are
intentionally short; DINOv2 is a documented fallback for gated DINOv3; and the
optional off-diagonal tangent-kernel test was not run. The pruning intervention
is a severe hard retain-mask, so catastrophic vision pruning may partly diagnose
the intervention rather than task ranking. These limitations lower confidence
in individual effect sizes, but they do not justify a paper-scale go: several
cheap or random controls beat the proposal by large margins in decisive cells.

## Recommended next decision

Do not scale the current 3x3 program. If the idea is revisited, the cheapest
discriminating sequence is: (1) repair the controlled task so known primitive
structure produces measurable causal selectivity; (2) test calibrated residual
or soft feature gating in vision rather than hard MLP retention; (3) diagnose
why one-hot/JL and activation beat semantic iEF in language; and (4) repeat only
those repaired cells over three seeds. A positive result there would justify
reopening the atlas-compression question; additional model scale would not.

## Reproduction and artifacts

Run from `/home/davwis/main/workspace/kpsa`:

```bash
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m unittest discover -s tests -v
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.controlled_v3
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.vision_v3
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.language_v3
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.analysis_v3
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.report_v3
```

The exact environment is in [`environment.json`](environment.json), aggregate
metrics in [`aggregate_metrics.json`](aggregate_metrics.json), and flat tables
in the adjacent CSV files. Per-setting JSON contains complete task-level
records and configuration; NPZ files contain the atlases and online embeddings;
checkpoints permit follow-up analysis without retraining.
"""


def _build_tex(payload: dict, raw: dict[str, dict]) -> str:
    _, source_rows = _source_table(payload)
    _, interp_rows = _interpretability_table(payload)
    _, pruning_rows = _pruning_table(payload)
    _, cl_rows = _cl_table(payload)
    _, efficiency_rows = _efficiency_table(payload)
    online = {row["setting"]: row for row in payload["online_fidelity"]}
    c = payload["sanity"]["controlled"]
    vision_accuracy = payload["sanity"]["vision"]["accuracy"]
    language_exact = payload["sanity"]["language"]["exact_match"]
    return rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=0.72in]{{geometry}}
\usepackage{{microtype}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\setlist{{nosep,leftmargin=*}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\parindent}}{{0pt}}
\title{{Task-Space Neuronal Embeddings\\\large V3 One-Day-Scope Exploratory Report}}
\author{{Internal research record}}
\date{{22 September 2026}}
\begin{{document}}
\maketitle

\section{{Decision}}
\textbf{{NO-GO for paper-scale expansion of the current method.}} Low-dimensional
task spaces preserve task-specific rankings better than matched JL sketches in
all three settings, but one compressed atlas does not reliably support causal
interpretation, structured pruning, and continual-learning protection. The
proposal fails predeclared gates 2--6 and only partially satisfies gate 1.

\begin{{center}}\small
\begin{{tabularx}}{{\linewidth}}{{@{{}}l l X@{{}}}}
\toprule Gate & Result & Evidence \\\midrule
Source fidelity & Mixed & Task spaces beat JL in ranking and top-5\% overlap, but lose CKA and k-NN in controlled and vision.\\
Interpretability & Fail & Not stronger than cheap baselines in controlled or language; comparable in vision.\\
Pruning & Fail & Below multiple matched controls in vision and language.\\
Continual learning & Fail & Benefits are order-dependent and not consistently competitive with raw/online EF.\\
Compression utility & Fail & Ranking signal does not translate into retained downstream utility.\\
Online approximation & Fail & Top-5\% circuit agreement is low, especially for Qwen.\\
Optional NTK angles & Not run & Optional and unable to overturn failed core gates.\\
\bottomrule
\end{{tabularx}}
\end{{center}}

\section{{Setup and model sanity}}
The controlled setting uses a six-layer, width-384 decoder Transformer (14.2M
parameters) and 16 mixtures of four sequence primitives. Vision uses a
22.1M-parameter DINOv2-small ViT on CIFAR-100; official DINOv3 access was gated.
Strict deterministic CUDA execution required freezing only its interpolated
positional-embedding tensor. Language uses full-parameter Qwen3-1.7B with
Adafactor over six heterogeneous instruction tasks. One seed is used throughout.

Controlled held-out half-MSE is {_f(c["model_loss"])}, versus
{_f(c["zero_predictor_half_mse"])} for a zero predictor. Vision balanced
accuracy is {_tex_pct(vision_accuracy)}, and language exact match is
{_tex_pct(language_exact)}. GSM8K remains at zero. Confidence intervals bootstrap
tasks within one checkpoint and therefore do not estimate seed uncertainty.

\section{{Source-object fidelity}}
The full balanced atlas is $P$ and the task-space sketch is $E=P\Phi$. Large
settings use 200,000 deterministic module pairs and at most 512 k-NN queries;
linear CKA is exact. Entries report task-space/JL at matched dimension.

\begin{{center}}\small
\begin{{tabular}}{{lrrrr}}\toprule
Setting & CKA & Rank $\rho$ & Top-5\% & k-NN\\\midrule
{source_rows}
\bottomrule\end{{tabular}}
\end{{center}}

\begin{{figure}}[ht]\centering
\includegraphics[width=\linewidth]{{source_fidelity.pdf}}
\caption{{Source fidelity versus task-space dimension. Semantic or known task
spaces improve ranking and circuit overlap more consistently than global
geometry.}}
\end{{figure}}

\section{{Causal interpretability}}
Mean-ablation selectivity is reported at 5\% of MLP features for controlled and
vision and 1\% for language; higher is better.

\begin{{center}}\scriptsize
\begin{{tabular}}{{lrrrrrr}}\toprule
Setting & iEF-task & Full & JL & Activation & Online & Random\\\midrule
{interp_rows}
\bottomrule\end{{tabular}}
\end{{center}}

Controlled iEF-task selectivity is effectively zero. Vision iEF-task is causal
but does not exceed activation or JL. In language, activation exceeds iEF-task,
with wide task-bootstrap uncertainty. Thus task-space ranking is not the
strongest shared locator.

\begin{{figure}}[ht]\centering
\includegraphics[width=\linewidth]{{interpretability_selectivity.pdf}}
\caption{{Mean-ablation selectivity across tested circuit sizes.}}
\end{{figure}}

\section{{Structured pruning and composition}}
Relative retention is baseline loss divided by pruned loss for controlled and
language and pruned accuracy divided by baseline accuracy for vision. Values
below report 75\% retained MLP features.

\begin{{center}}\small
\begin{{tabular}}{{lrrrrr}}\toprule
Setting & iEF-task & Full & JL & Activation & Random\\\midrule
{pruning_rows}
\bottomrule\end{{tabular}}
\end{{center}}

Controlled pruning saturates for most informed methods. Vision has zero target
accuracy through 50\% retention for every method, and at 75\% iEF-task trails
JL and random. Language iEF-task also trails JL and random. Composition follows
the expected ordering: controlled primitive unions recall
{_tex_pct(_composition(payload, "controlled", "direct_circuit_recall_in_primitive_union"))}
of mixture selections; related vision unions are smaller than dissimilar unions;
and within-classification Qwen circuit Jaccard is
{_f(_composition(payload, "language", "pair_circuit_jaccard", "classification_family"))}
versus {_f(_composition(payload, "language", "pair_circuit_jaccard", "cross_family"))}
cross-family. These structural results do not recover pruning utility.

\begin{{figure}}[ht]\centering
\includegraphics[width=\linewidth]{{pruning_retention.pdf}}
\caption{{Zero-shot structured pruning; open stars mark iEF-task recovery.}}
\end{{figure}}

\section{{Continual learning}}
Lower forgetting is better. Controlled/language use loss increases, whereas
vision uses accuracy drops.

\begin{{center}}\scriptsize
\begin{{tabular}}{{llrrrrr}}\toprule
Setting & Order & iEF-task & Raw EF & Online & Random & None\\\midrule
{cl_rows}
\bottomrule\end{{tabular}}
\end{{center}}

iEF-task helps some cells but does not dominate relevant controls. Online EF is
better in both vision orders; raw EF is better in both language orders. The
protection gate therefore fails.

\begin{{figure}}[ht]\centering
\includegraphics[width=\linewidth]{{continual_forgetting.pdf}}
\caption{{Average forgetting, averaged over related and dissimilar orders.}}
\end{{figure}}

\section{{Online approximation and efficiency}}
Post-hoc/online top-5\% overlap is
{_f(online["controlled"]["topk_overlap_mean"])},
{_f(online["vision"]["topk_overlap_mean"])}, and
{_f(online["language"]["topk_overlap_mean"])} for controlled, vision, and
language. The online accumulator timing is an upper bound because measurement
includes explicit synchronization.

\begin{{center}}\scriptsize
\begin{{tabular}}{{lrrrrrr}}\toprule
Setting & Full min & Train s & Post-hoc s & Online & Peak GiB & Query $\mu$s\\\midrule
{efficiency_rows}
\bottomrule\end{{tabular}}
\end{{center}}

\begin{{figure}}[ht]\centering
\includegraphics[width=\linewidth]{{efficiency.pdf}}
\caption{{Full-run time, peak allocated CUDA memory, and checkpoint storage.}}
\end{{figure}}

\section{{Debugging, limitations, and next decision}}
Five consequential failures were diagnosed rather than hidden: captured
activation graphs broke cloning; task mixtures crossed CPU/GPU devices;
quadratic fidelity exceeded a quantile limit; short SGD Qwen training produced
zero exact match; and trainable DINOv2 positional interpolation caused CUDA
nondeterminism, producing discarded 76.4\% and 38.0\% vision runs. The final
vision run is deterministic and reaches {_tex_pct(vision_accuracy)}. A misplaced
aggregation return was also caught before reporting.

Limits remain: one seed, tiny balanced reference/validation sets, short CL
updates, DINOv2 rather than gated DINOv3, and no optional off-diagonal NTK test.
Hard retain-masking is also a severe vision intervention. These limitations
reduce precision but do not reverse large losses to cheap controls.

Do not scale the current matrix. A revisit should first obtain measurable
controlled causal selectivity, calibrate soft/residual vision pruning, and
diagnose the language gap to activation and JL, then repeat only repaired cells
over three seeds. More model scale is not the next discriminating experiment.

\section{{Artifacts}}
Exact configuration is embedded in each per-setting metrics JSON. The artifact
directory also contains the environment record, aggregate JSON/CSV tables,
atlases, checkpoints, and SVG/PDF/PNG figures. Reproduction commands are in the
Markdown report and repository README.

\end{{document}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", type=Path, default=arc_artifact_dir("02_exploratory")
    )
    args = parser.parse_args()
    payload = _load(args.artifact_dir / "aggregate_metrics.json")
    raw = {
        "controlled": _load(
            args.artifact_dir / "controlled/controlled_v3_metrics_seed1.json"
        ),
        "vision": _load(args.artifact_dir / "vision/vision_v3_metrics_seed1.json"),
        "language": _load(
            args.artifact_dir / "language/language_v3_metrics_seed1.json"
        ),
    }
    markdown_path = args.artifact_dir / "exploratory_report.md"
    tex_path = args.artifact_dir / "exploratory_report.tex"
    markdown_path.write_text(_build_markdown(payload, raw))
    tex_path.write_text(_build_tex(payload, raw))
    print(f"V3_REPORT_MARKDOWN={markdown_path}")
    print(f"V3_REPORT_TEX={tex_path}")


if __name__ == "__main__":
    main()
