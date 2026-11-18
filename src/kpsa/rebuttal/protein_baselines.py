"""Protein (ESM2-35M, masked residue) neuron localization: retrieval baselines.

Re-runs the paper's protein localization protocol (protein_neuron_localization_v8:
seeded shuffle of 3,000 Swiss-Prot sequences, first 1,600 as sources, normalized
mean final hidden state, exact RBF at scale .10, outgoing FF component, layer-3
scope, exact zero-deactivation, absolute margin change) with kNN and ridge
predictors. Development queries are the first 128 correctly answered candidates
after the sources; test queries are the next 128. Baseline hyperparameters and
the KPSA scale are chosen on development queries by causal gap recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..protein_neuron_localization_v8 import (
    CANONICAL_AA,
    EsmSensitivity,
    _load_examples,
    _query_example,
    _source_example,
    zero_activation_groups,
)
from .forecast_baselines import ridge_fit
from .vision_retrieval_eval import bootstrap_gap

SCALES = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
KS = (1, 2, 4, 8, 16, 32, 64)
RIDGE = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FRACTIONS = (0.0002, 0.0005, 0.001)
REPR, COMPONENT, LAYER = "sequence_mean", "outgoing", 3


def build_cache(args):
    examples = _load_examples(args.data, args.seed)
    exp = EsmSensitivity("facebook/esm2_t12_35M_UR50D", "cuda")
    reps, profs, cats = [], [], []
    t0 = time.time()
    for ex in examples[:1600]:
        prepared, r, p, _ = _source_example(exp, ex)
        reps.append(r[REPR])
        profs.append(p[COMPONENT])
        c = torch.zeros(len(CANONICAL_AA)); c[CANONICAL_AA.index(str(prepared["true_residue"]))] = 1
        cats.append(c)
    source_seconds = time.time() - t0
    q = {k: [] for k in ("acc", "rep", "prof", "attr", "cat", "base")}
    for ex in examples[1600:]:
        prepared, base, r, attr, p, correct = _query_example(exp, ex)
        if not correct:
            continue
        q["acc"].append(ex["accession"]); q["rep"].append(r[REPR]); q["prof"].append(p[COMPONENT])
        q["attr"].append(attr); q["base"].append(base)
        c = torch.zeros(len(CANONICAL_AA)); c[CANONICAL_AA.index(str(prepared["true_residue"]))] = 1
        q["cat"].append(c)
        if len(q["acc"]) >= 256:
            break
    zs = F.normalize(torch.stack(reps).double(), dim=1)
    d2 = torch.pdist(zs).square()
    return {
        "source_features": zs.float(), "source_profiles": torch.stack(profs).float(),
        "source_categories": torch.stack(cats), "source_seconds": source_seconds,
        "median_squared_distance": float(d2[d2 > 0].median()),
        "query_accessions": q["acc"], "query_features": F.normalize(torch.stack(q["rep"]).double(), dim=1).float(),
        "query_profiles": torch.stack(q["prof"]).float(), "query_attribution": torch.stack(q["attr"]).float(),
        "query_categories": torch.stack(q["cat"]), "query_baseline": torch.tensor(q["base"]),
        "units_per_layer": exp.units_per_layer,
    }


def predict(cache, rows):
    zs, S = cache["source_features"].double(), cache["source_profiles"].double()
    zq = cache["query_features"][rows].double()
    med = cache["median_squared_distance"]
    out = {"scalar": S.mean(0, keepdim=True).expand(len(zq), -1)}
    d2 = torch.cdist(zq, zs).square()
    for s in SCALES:
        out[f"kpsa@{s:g}"] = torch.exp(-d2 / (2 * med * s * s)) @ S / len(zs)
    order = (zq @ zs.T).argsort(1, descending=True)
    for k in KS:
        out[f"knn@{k}"] = S[order[:, :k]].mean(1)
    for rel in RIDGE:
        out[f"ridge@{rel:g}"] = ridge_fit(zs.float(), S.float(), rel)(zq.float()).double()
    out["categorical"] = (cache["query_categories"][rows].double() @ cache["source_categories"].double().T) @ S / len(zs)
    out["direct_parameter"] = cache["query_profiles"][rows].double()
    out["activation_attribution"] = cache["query_attribution"][rows].double()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path.home() / "main/data/protein/swissprot_reviewed_len80_400_3000.jsonl")
    parser.add_argument("--seed", type=int, default=260926)
    parser.add_argument("--cache", type=Path, default=Path.home() / "main/data/kpsa-rebuttal/protein_cache.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.cache.exists():
        cache = torch.load(args.cache, weights_only=False)
    else:
        cache = build_cache(args)
        torch.save(cache, args.cache)
    upl = cache["units_per_layer"]
    scope = torch.arange(LAYER * upl, (LAYER + 1) * upl)
    counts = [max(1, math.ceil(f * len(scope))) for f in FRACTIONS]
    examples = {ex["accession"]: ex for ex in _load_examples(args.data, args.seed)}
    exp = EsmSensitivity("facebook/esm2_t12_35M_UR50D", "cuda")
    nq = len(cache["query_accessions"])
    splits = {"dev": list(range(0, min(128, nq))), "test": list(range(128, nq))}
    result = {"feature_counts": counts, "sources": 1600, "splits": {k: len(v) for k, v in splits.items()},
              "protocol": "paper protein localization; dev = first 128 correct candidates, test = next ones"}
    effects = {}
    for split, rows in splits.items():
        preds = predict(cache, rows)
        rec = {m: {f: [] for f in FRACTIONS} for m in preds}
        memo = {}
        for i, row in enumerate(rows):
            acc = cache["query_accessions"][row]
            prepared = exp.prepare(examples[acc])
            base = float(cache["query_baseline"][row])
            for m, p in preds.items():
                for f, c in zip(FRACTIONS, counts):
                    sel = scope[torch.topk(p[i, scope], c).indices].sort().values
                    key = (row, tuple(sel.tolist()))
                    if key not in memo:
                        with zero_activation_groups(exp, sel):
                            memo[key] = abs(base - exp.functional(prepared))
                    rec[m][f].append(memo[key])
        effects[split] = rec
        print(split, "assayed", len(rows), flush=True)
    summary = {}
    for split, rec in effects.items():
        summary[split] = {}
        for m in rec:
            summary[split][m] = {}
            for f in FRACTIONS:
                a, c, o = (np.asarray(rec[x][f]) for x in (m, "scalar", "activation_attribution"))
                summary[split][m][f"{f:g}"] = bootstrap_gap(a, c, o)
    chosen = {}
    for m in summary["dev"]:
        if "@" not in m:
            continue
        fam = m.split("@")[0]
        score = np.mean([summary["dev"][m][f"{f:g}"][0] for f in FRACTIONS])
        if fam not in chosen or score > chosen[fam][1]:
            chosen[fam] = (m, float(score))
    result["dev_chosen"] = {k: v[0] for k, v in chosen.items()}
    result["summary"] = summary
    result["records_test"] = {m: {f"{f:g}": v for f, v in r.items()} for m, r in effects["test"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    for m in ["kpsa@0.1", result["dev_chosen"]["kpsa"], "knn@1", result["dev_chosen"]["knn"],
              result["dev_chosen"]["ridge"], "categorical", "direct_parameter"]:
        s = summary["test"][m]
        print(f"{m:14s} " + " ".join(f"{f}:{s[f][0]:.3f}[{s[f][1]:.2f},{s[f][2]:.2f}]" for f in s), flush=True)


if __name__ == "__main__":
    main()
