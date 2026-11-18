"""Protein (ESM2-35M) baselines on the paper's exact query sets.

Development queries are the paper's 48 screen accessions; test queries are the
128 accessions of the frozen neuron-localization confirmation and the 48 of the
frozen parameter-perturbation confirmation. Sources are the paper's first 1,600
sequences of the seeded shuffle. Perturbation seeds follow the paper
(260,926 + position x 100,003 + direction x 1,009, 16 directions, sigma = 3% of
the scoped outgoing-parameter RMS). Shared methods are checked against the
frozen paper records.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..protein_neuron_localization_v8 import (
    CANONICAL_AA,
    EsmSensitivity,
    _load_examples,
    _query_example,
    zero_activation_groups,
)
from .protein_baselines import FRACTIONS, KS, RIDGE, SCALES
from .forecast_baselines import ridge_fit
from .vision_retrieval_eval import bootstrap_gap

HDD = Path("/run/media/davwis/Volume/main/workspace/kpsa-internal/06_strenghening/artifacts/protein")
SEED = 260926


def query_block(exp, examples, accessions):
    out = {k: [] for k in ("rep", "prof", "attr", "cat", "base", "prepared")}
    for acc in accessions:
        prepared, base, r, attr, p, correct = _query_example(exp, examples[acc])
        out["rep"].append(r["sequence_mean"]); out["prof"].append(p["outgoing"]); out["attr"].append(attr)
        out["base"].append(base); out["prepared"].append(prepared)
        c = torch.zeros(len(CANONICAL_AA)); c[CANONICAL_AA.index(str(prepared["true_residue"]))] = 1
        out["cat"].append(c)
    return {"z": F.normalize(torch.stack(out["rep"]).double(), dim=1), "prof": torch.stack(out["prof"]).double(),
            "attr": torch.stack(out["attr"]).double(), "cat": torch.stack(out["cat"]).double(),
            "base": out["base"], "prepared": out["prepared"]}


def predict(cache, q):
    zs, S = cache["source_features"].double(), cache["source_profiles"].double()
    med = cache["median_squared_distance"]
    out = {"scalar": S.mean(0, keepdim=True).expand(len(q["z"]), -1)}
    d2 = torch.cdist(q["z"], zs).square()
    for s in SCALES:
        out[f"kpsa@{s:g}"] = torch.exp(-(d2 - d2.min(1, keepdim=True).values) / (2 * med * s * s)) @ S / len(zs)
    order = (q["z"] @ zs.T).argsort(1, descending=True)
    for k in KS:
        out[f"knn@{k}"] = S[order[:, :k]].mean(1)
    for rel in RIDGE:
        out[f"ridge@{rel:g}"] = ridge_fit(zs.float(), S.float(), rel)(q["z"].float()).double()
    out["categorical"] = (q["cat"] @ cache["source_categories"].double().T) @ S / len(zs)
    out["direct_parameter"] = q["prof"]
    out["activation_attribution"] = q["attr"]
    return out


def choose(scores):
    chosen = {}
    for m, v in scores.items():
        if "@" in m:
            fam = m.split("@")[0]
            if fam not in chosen or v > scores[chosen[fam]]:
                chosen[fam] = m
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path.home() / "main/data/protein/swissprot_reviewed_len80_400_3000.jsonl")
    parser.add_argument("--cache", type=Path, default=Path.home() / "main/data/kpsa-rebuttal/protein_cache.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    cache = torch.load(args.cache, weights_only=False)  # sources from protein_baselines (paper's first 1,600)
    examples = {ex["accession"]: ex for ex in _load_examples(args.data, SEED)}
    exp = EsmSensitivity("facebook/esm2_t12_35M_UR50D", "cuda")
    upl = exp.units_per_layer
    scope = torch.arange(3 * upl, 4 * upl)
    counts = [max(1, math.ceil(f * len(scope))) for f in FRACTIONS]
    paper1 = json.loads((HDD / "esm2_t12_feature_localization_causal_n1600_frozen_confirmation_budgets.json").read_text())
    paper2 = json.loads((HDD / "esm2_t12_parameter_influence_causal_n1600_frozen_confirmation_budgets.json").read_text())
    dev_acc = (HDD / "esm2_t12_n1600_screen_query_accessions.txt").read_text().split()
    sets = {"dev": query_block(exp, examples, dev_acc),
            "app1": query_block(exp, examples, paper1["query_accessions"]),
            "app2": query_block(exp, examples, paper2["query_accessions"])}
    preds = {k: predict(cache, q) for k, q in sets.items()}

    def deactivation(split, methods):
        rec = {m: {f: [] for f in FRACTIONS} for m in methods}
        memo = {}
        for i, prepared in enumerate(sets[split]["prepared"]):
            base = sets[split]["base"][i]
            for m in methods:
                for f, c in zip(FRACTIONS, counts):
                    sel = scope[torch.topk(preds[split][m][i, scope], c).indices].sort().values
                    key = (i, tuple(sel.tolist()))
                    if key not in memo:
                        with zero_activation_groups(exp, sel):
                            memo[key] = abs(base - exp.functional(prepared))
                    rec[m][f].append(memo[key])
        return rec

    def gaps(rec, ref):
        return {m: {f"{f:g}": bootstrap_gap(np.asarray(rec[m][f]), np.asarray(rec["scalar"][f]), np.asarray(rec[ref][f]))
                    for f in FRACTIONS} for m in rec}

    def coverage(split, m):
        p, t = preds[split][m][:, scope], sets[split]["prof"][:, scope]
        return float(np.mean([(t.gather(1, torch.topk(p, c, dim=1).indices).sum(1) / t.sum(1)).mean() for c in counts]))

    result = {"feature_counts": counts, "n": {k: len(v["base"]) for k, v in sets.items()}}
    # Application 1: dev choice by causal gap on the screen queries.
    dev_rec = deactivation("dev", list(preds["dev"]))
    dev_gap = gaps(dev_rec, "activation_attribution")
    chosen1 = choose({m: float(np.mean([v[f"{f:g}"][0] for f in FRACTIONS])) for m, v in dev_gap.items()})
    methods1 = list(dict.fromkeys(["scalar", "kpsa@0.1", chosen1["kpsa"], "knn@1", chosen1["knn"], chosen1["ridge"],
                                   "categorical", "direct_parameter", "activation_attribution"]))
    rec1 = deactivation("app1", methods1)
    result["app1"] = {"dev_chosen": chosen1, "summary": gaps(rec1, "activation_attribution"),
                      "records": {m: {f"{f:g}": v for f, v in r.items()} for m, r in rec1.items()}}
    # Paper reproduction for application 1 (exact_rbf = kpsa@0.1, nearest = knn@1).
    pr = {}
    for x in paper1["records"]:
        pr.setdefault((x["method"], x["fraction"]), {})[x["accession"]] = x["causal_effect"]
    accs = paper1["query_accessions"]
    result["app1"]["paper_reproduction"] = {
        ours: float(max(abs(rec1[ours][f][i] - pr[(theirs, f)][a]) / max(abs(pr[(theirs, f)][a]), 1e-9)
                        for f in FRACTIONS for i, a in enumerate(accs)))
        for ours, theirs in (("kpsa@0.1", "exact_rbf"), ("knn@1", "nearest"), ("scalar", "scalar"),
                             ("categorical", "categorical"), ("activation_attribution", "activation_attribution"))}
    # Application 2: dev choice by direct-gradient energy coverage on the screen queries.
    chosen2 = choose({m: coverage("dev", m) for m in preds["dev"] if m not in ("activation_attribution",)})
    methods2 = list(dict.fromkeys(["scalar", "kpsa@0.1", chosen2["kpsa"], "knn@1", chosen2["knn"], chosen2["ridge"],
                                   "categorical", "direct_parameter"]))
    noise = exp.parameter_rms(scope, "outgoing") * 0.03
    rec2 = {m: {f: [] for f in FRACTIONS} for m in methods2}
    memo = {}
    for i, prepared in enumerate(sets["app2"]["prepared"]):
        for m in methods2:
            for f, c in zip(FRACTIONS, counts):
                sel = scope[torch.topk(preds["app2"][m][i, scope], c).indices].sort().values
                key = (i, tuple(sel.tolist()))
                if key not in memo:
                    vals = []
                    for d in range(16):
                        s = SEED + i * 100_003 + d * 1_009
                        with exp.additive_parameter_noise(sel, noise, component="outgoing", seed=s, sign=1):
                            plus = exp.functional(prepared)
                        with exp.additive_parameter_noise(sel, noise, component="outgoing", seed=s, sign=-1):
                            minus = exp.functional(prepared)
                        vals.append(((plus - minus) / (2 * noise)) ** 2)
                    memo[key] = float(np.mean(vals))
                rec2[m][f].append(memo[key])
    result["app2"] = {"dev_chosen": chosen2, "summary": gaps(rec2, "direct_parameter"),
                      "coverage_test": {m: coverage("app2", m) for m in methods2},
                      "records": {m: {f"{f:g}": v for f, v in r.items()} for m, r in rec2.items()}}
    pq = {}
    for x in paper2["records"]:
        pq.setdefault((x["method"], x["fraction"], x["accession"]), []).append(x["squared_susceptibility"])
    accs2 = paper2["query_accessions"]
    result["app2"]["paper_reproduction"] = {
        ours: float(max(abs(rec2[ours][f][i] - np.mean(pq[(theirs, f, a)])) / max(abs(np.mean(pq[(theirs, f, a)])), 1e-9)
                        for f in FRACTIONS for i, a in enumerate(accs2)))
        for ours, theirs in (("kpsa@0.1", "exact_rbf"), ("knn@1", "nearest"), ("scalar", "scalar"),
                             ("direct_parameter", "direct_parameter"))}
    args.output.write_text(json.dumps(result, indent=1))
    for app in ("app1", "app2"):
        print(app, "chosen", result[app]["dev_chosen"], "repro", result[app]["paper_reproduction"])
        for m, s in result[app]["summary"].items():
            print(f"  {m:24s}", {f: f"{100*v[0]:.1f} [{100*v[1]:.0f},{100*v[2]:.0f}]" for f, v in s.items()})


if __name__ == "__main__":
    main()
