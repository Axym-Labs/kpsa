"""Input-level (cluster) bootstrap for the circuit-overlap analysis (R4 W2/Q2).

The paper bins input pairs by representation similarity and applies a t
interval to the pair Jaccards, although pairs share inputs. Here we rerun the
paper's structure stage (semantic_circuit_evidence_v8, --structure-only),
capture the per-input selections, and resample *inputs* with replacement,
recomputing all pairs, bins, and the top-minus-bottom Jaccard increase in
every draw.

  stage dump-vision / dump-forecast: run the paper code and save the inputs
  stage bootstrap: compute the cluster-bootstrap intervals from a dump
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from .. import semantic_circuit_evidence_v8 as sce

EDGES = (0.0, 0.5, 0.8, 0.95, 0.99, 1.0)


def dump(stage_args, dump_path):
    original = sce._overlap_by_similarity

    def wrapped(features, semantic_scores, shuffled_scores, *, counts, seed, max_pairs=20_000):
        torch.save({"features": features.float().cpu(), "semantic": semantic_scores.float().cpu(),
                    "shuffled": shuffled_scores.float().cpu(), "counts": counts, "seed": seed}, dump_path)
        return original(features, semantic_scores, shuffled_scores, counts=counts, seed=seed, max_pairs=max_pairs)

    sce._overlap_by_similarity = wrapped
    sys.argv = ["semantic_circuit_evidence_v8", *stage_args]
    sce.main()


def selections(scores, count):
    return torch.topk(scores, count, dim=0).indices.T.numpy()  # (inputs, count)


def jaccard_matrix(sel, n_groups):
    """Pairwise Jaccard between all input selections (dense incidence)."""
    inc = np.zeros((len(sel), n_groups), dtype=np.float32)
    np.put_along_axis(inc, sel, 1.0, axis=1)
    inter = inc @ inc.T
    size = inc.sum(1)
    return inter / (size[:, None] + size[None, :] - inter)


def binned_increase(sim, jac, idx, rng, max_pairs):
    """Bins of pairs among resampled inputs `idx` (positions differ)."""
    n = len(idx)
    a = rng.integers(0, n, size=2 * max_pairs)
    b = rng.integers(0, n, size=2 * max_pairs)
    a, b = idx[a], idx[b]
    keep = a != b  # pairs of distinct original inputs only (duplicates from resampling dropped)
    a, b = a[keep][:max_pairs], b[keep][:max_pairs]
    s = sim[a, b]
    order = np.argsort(s)
    means = []
    for lo, hi in pairwise(EDGES):
        part = order[round(lo * len(order)):round(hi * len(order))]
        means.append(jac[a[part], b[part]].mean())
    return np.asarray(means)


def bootstrap(dump_path, draws=1000, seed=0, max_pairs=20_000, block=1):
    d = torch.load(dump_path, weights_only=False)
    f = d["features"].double()
    f = f / f.norm(dim=1, keepdim=True)
    sim = (f @ f.T).numpy()
    n, groups = len(f), d["semantic"].shape[0]
    rng = np.random.default_rng(seed)
    out = {"inputs": n, "draws": draws, "block": block, "quantile_edges": EDGES, "budgets": {}}
    for budget, count in d["counts"].items():
        jac = {m: jaccard_matrix(selections(d[m], count), groups) for m in ("semantic", "shuffled")}
        point = {m: binned_increase(sim, jac[m], np.arange(n), np.random.default_rng(d["seed"]), max_pairs)
                 for m in jac}
        stats = {"semantic": [], "shuffled": [], "difference": [], "semantic_top": []}
        for _ in range(draws):
            starts = rng.integers(0, n, size=-(-n // block))
            idx = ((starts[:, None] + np.arange(block)[None]).reshape(-1)[:n]) % n  # circular moving blocks
            bins = {m: binned_increase(sim, jac[m], idx, rng, max_pairs) for m in jac}
            inc = {m: bins[m][-1] - bins[m][0] for m in bins}
            stats["semantic"].append(inc["semantic"])
            stats["shuffled"].append(inc["shuffled"])
            stats["difference"].append(inc["semantic"] - inc["shuffled"])
            stats["semantic_top"].append(bins["semantic"][-1])
        ci = {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in stats.items()}
        out["budgets"][f"{budget:g}"] = {
            "selected_groups": count,
            "point_bin_means": {m: point[m].tolist() for m in point},
            "point_increase": {m: float(point[m][-1] - point[m][0]) for m in point},
            "cluster_bootstrap_ci95": ci,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dump", "bootstrap"))
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--block", type=int, default=1, help="moving-block length over time-ordered inputs")
    args, rest = parser.parse_known_args()
    if args.stage == "dump":
        dump(rest, args.dump)
    else:
        result = bootstrap(args.dump, args.draws, block=args.block)
        args.result.write_text(json.dumps(result, indent=1))
        print(json.dumps({b: {"point": v["point_increase"], "ci": v["cluster_bootstrap_ci95"]}
                          for b, v in result["budgets"].items()}, indent=1))


if __name__ == "__main__":
    main()
