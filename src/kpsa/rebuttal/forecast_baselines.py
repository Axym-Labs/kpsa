"""Forecasting (Chronos-Bolt, ETTh1/Weather): retrieval baselines, dependence-aware
intervals, and neighbour diagnostics for the rebuttal.

Re-runs the paper's protocol (timeseries_parameter_influence_v8: 1,600 evenly
spaced source windows, 128 query windows, final-encoder-state RBF at scale .25,
final-decoder FF scope, exact antithetic perturbation assay with 16 directions,
noise 3% of FF RMS, the same per-query seeds) and adds kNN and ridge
predictors. Hyperparameters of the added baselines (k, ridge strength) are
chosen on ETTh1 development windows (endpoints 8,900-12,000, between the
source and query ranges) by direct-gradient energy coverage and frozen for
Weather, as the paper did for KPSA's scale.

Outputs per-query records so that intervals can respect temporal dependence
(moving-block bootstrap over time-ordered queries; non-overlapping subsets).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..timeseries_parameter_influence_v8 import (
    ChronosSensitivity,
    _collect,
    _load_values,
    _temporal_coordinates,
)

SCALES = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
KS = (1, 2, 4, 8, 16, 32, 64)
RIDGE = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FRACTIONS = (0.0002, 0.0005, 0.001)
DATASETS = {
    "ETTh1": dict(data="~/main/data/ett/ETTh1.csv", target="OT", norm_end=8400,
                  source=(600, 8400), query=(12300, 16800), dev=(8900, 12000)),
    "Weather": dict(data="~/main/data/weather_autoformer/weather.csv", target="T (degC)",
                    norm_end=24000, source=(600, 24000), query=(33000, 50000), dev=None),
}


def ridge_fit(x, y, rel):
    xm, ym = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc = (x - xm).double()
    gram = xc.T @ xc
    lam = rel * float(torch.trace(gram)) / gram.shape[0]
    w = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device), xc.T @ (y - ym).double())
    return lambda q: ((q - xm).double() @ w + ym.double()).float()


def predict_all(cache, qkey, n=None):
    """{method: (Q, G)} predictions from the first n (evenly thinned) sources."""
    zs, S = cache["source_features"].double(), cache["source_profiles"].double()  # (N,d), (N,G)
    if n is not None and n < len(zs):
        keep = torch.linspace(0, len(zs) - 1, n).round().long()
        zs, S = zs[keep], S[keep]
        cats = cache["source_categories"][keep].double()
    else:
        cats = cache["source_categories"].double()
    zq = cache[f"{qkey}_features"].double()
    med = cache["median_squared_distance"]
    out = {"scalar": S.mean(0, keepdim=True).expand(len(zq), -1)}
    d2 = torch.cdist(zq, zs).square()
    for s in SCALES:
        w = torch.exp(-d2 / (2 * med * s * s))
        out[f"kpsa@{s:g}"] = w @ S / len(zs)
    order = (zq @ zs.T).argsort(1, descending=True)
    for k in KS:
        out[f"knn@{k}"] = S[order[:, :k]].mean(1)
    for rel in RIDGE:
        out[f"ridge@{rel:g}"] = ridge_fit(zs.float(), S.float(), rel)(zq.float()).double()
    out["temporal_categorical"] = (cache[f"{qkey}_categories"].double() @ cats.T) @ S / len(zs)
    out["direct_gradient"] = cache[f"{qkey}_profiles"].double()
    return {k: v.float() for k, v in out.items()}


def coverage(pred, truth, scope, count):
    """Fraction of in-scope direct-gradient energy in the top-`count` predicted groups."""
    p, t = pred[:, scope], truth[:, scope]
    idx = torch.topk(p, count, dim=1).indices
    return (t.gather(1, idx).sum(1) / t.sum(1)).numpy()


def neighbour_curve(cache, qkey, scope, count, ranks=(1, 2, 4, 8, 16, 32, 64, 128)):
    """Coverage achieved by the single j-th nearest source profile, and by the mean of the top-j."""
    zs, S = cache["source_features"].double(), cache["source_profiles"]
    zq, T = cache[f"{qkey}_features"].double(), cache[f"{qkey}_profiles"]
    order = (zq @ zs.T).argsort(1, descending=True)
    single = {r: float(coverage(S[order[:, r - 1]], T, scope, count).mean()) for r in ranks}
    sims = (zq @ zs.T).sort(1, descending=True).values
    return {"single_jth_neighbour_coverage": single,
            "cosine_of_jth_neighbour": {r: float(sims[:, r - 1].mean()) for r in ranks}}


def block_bootstrap_gap(method, const, oracle, block, draws=5000, seed=0):
    """Circular moving-block bootstrap over time-ordered queries."""
    rng = np.random.default_rng(seed)
    n = len(method)
    nb = math.ceil(n / block)
    stats = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + np.arange(block)[None]).reshape(-1)[:n] % n
        stats.append((method[idx].mean() - const[idx].mean()) / (oracle[idx].mean() - const[idx].mean()))
    point = (method.mean() - const.mean()) / (oracle.mean() - const.mean())
    return float(point), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def iid_bootstrap_gap(method, const, oracle, draws=5000, seed=0):
    return block_bootstrap_gap(method, const, oracle, 1, draws, seed)


def stage_profile(args, spec):
    frame, values = _load_values(argparse.Namespace(
        data=Path(spec["data"]).expanduser(), target_column=spec["target"], normalization_end=spec["norm_end"]))
    exp = ChronosSensitivity("amazon/chronos-bolt-small", values, device="cuda")
    out = {"units_per_layer": exp.units_per_layer}
    sets = {"source": (*spec["source"], 1600), "query": (*spec["query"], 128)}
    if spec["dev"]:
        sets["dev"] = (*spec["dev"], 128)
    t0 = time.time()
    for key, (a, b, n) in sets.items():
        ends = np.linspace(a, b, n).round().astype(int)
        feats, profs = _collect(exp, ends, label=key)
        out[f"{key}_ends"] = torch.from_numpy(ends)
        out[f"{key}_features"] = feats.float()
        out[f"{key}_profiles"] = profs.T.contiguous().float()  # (n, G)
        out[f"{key}_categories"] = _temporal_coordinates(frame, ends).float()
        out[f"{key}_seconds"] = time.time() - t0
    d = torch.pdist(out["source_features"].double()).square()
    out["median_squared_distance"] = float(d[d > 0].median())
    out["source_categories"] = out["source_categories"]
    return out


def stage_causal(args, spec, cache, methods_to_assay, scope):
    frame, values = _load_values(argparse.Namespace(
        data=Path(spec["data"]).expanduser(), target_column=spec["target"], normalization_end=spec["norm_end"]))
    exp = ChronosSensitivity("amazon/chronos-bolt-small", values, device="cuda")
    preds = predict_all(cache, "query")
    noise = exp.parameter_rms() * 0.03
    seed = 260926
    ends = cache["query_ends"].tolist()
    deriv_cache = {}
    records = {m: {f: [] for f in FRACTIONS} for m in methods_to_assay}
    t0 = time.time()
    for q, end in enumerate(ends):
        x = exp.window(int(end))
        for f in FRACTIONS:
            count = max(1, math.ceil(f * len(scope)))
            for m in methods_to_assay:
                local = torch.topk(preds[m][q, scope], count).indices
                selected = scope[local].sort().values
                key = (q, tuple(selected.tolist()))
                if key not in deriv_cache:
                    vals = []
                    for direction in range(16):
                        s = seed + q * 100_003 + direction * 1_009
                        with exp.additive_noise(selected, noise, seed=s, sign=1):
                            plus = exp.functional(x)
                        with exp.additive_noise(selected, noise, seed=s, sign=-1):
                            minus = exp.functional(x)
                        vals.append(((plus - minus) / (2 * noise)) ** 2)
                    deriv_cache[key] = float(np.mean(vals))
                records[m][f].append(deriv_cache[key])
        if (q + 1) % 16 == 0:
            print(f"  causal {q + 1}/{len(ends)} {time.time() - t0:.0f}s", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "main/data/kpsa-rebuttal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chosen", type=Path, help="JSON with ETTh1 dev-chosen hyperparameters (for Weather)")
    args = parser.parse_args()
    torch.manual_seed(260926)
    torch.set_float32_matmul_precision("high")  # as in the paper's forecasting runs
    spec = DATASETS[args.dataset]
    cache_path = args.cache_dir / f"forecast_{args.dataset}_cache.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False)
    else:
        cache = stage_profile(args, spec)
        torch.save(cache, cache_path)
    upl = cache["units_per_layer"]
    scope = torch.arange(11 * upl, 12 * upl)  # paper scope: layer11 = final decoder FF
    counts = [max(1, math.ceil(f * len(scope))) for f in FRACTIONS]
    result = {"dataset": args.dataset, "feature_counts": counts, "scope": "layer11 (final decoder FF)",
              "sources": int(len(cache["source_features"])), "queries": int(len(cache["query_features"]))}

    # Hyperparameter choice for added baselines (ETTh1 dev windows; frozen for Weather).
    if spec["dev"]:
        dev = predict_all(cache, "dev")
        truth = cache["dev_profiles"]
        dev_cov = {m: float(np.mean([coverage(p, truth, scope, c).mean() for c in counts])) for m, p in dev.items()}
        fam = lambda m: m.split("@")[0]
        chosen = {}
        for m, v in dev_cov.items():
            if "@" in m and (fam(m) not in chosen or v > dev_cov[chosen[fam(m)]]):
                chosen[fam(m)] = m
        result["dev_coverage"] = dev_cov
        result["dev_chosen"] = chosen
    else:
        chosen = json.loads(args.chosen.read_text())["dev_chosen"]
        result["dev_chosen_from_ETTh1"] = chosen
    test = predict_all(cache, "query")
    truth = cache["query_profiles"]
    result["test_coverage"] = {m: {f"{f:g}": float(coverage(p, truth, scope, c).mean()) for f, c in zip(FRACTIONS, counts)}
                               for m, p in test.items()}
    # Diagnostics: neighbour-rank curve, kernel effective sample size, source thinning.
    result["neighbour_curve"] = {f"{f:g}": neighbour_curve(cache, "query", scope, c) for f, c in zip(FRACTIONS, counts)}
    zs, zq = cache["source_features"].double(), cache["query_features"].double()
    d2 = torch.cdist(zq, zs).square()
    ess = {}
    for s in SCALES:
        w = torch.exp(-d2 / (2 * cache["median_squared_distance"] * s * s))
        ess[f"{s:g}"] = float(((w.sum(1) ** 2) / (w.square().sum(1))).mean())
    result["kernel_effective_sources"] = ess
    thin = {}
    for n in (25, 50, 100, 200, 400, 800, 1600):
        p = predict_all(cache, "query", n)
        thin[str(n)] = {m: float(np.mean([coverage(p[m], truth, scope, c).mean() for c in counts]))
                        for m in ("kpsa@0.25", "knn@1", chosen.get("knn", "knn@1"), chosen.get("ridge", "ridge@0.01"), "scalar")}
    result["source_thinning_mean_coverage"] = thin

    assay = ["scalar", "kpsa@0.25", "knn@1", chosen["knn"], chosen["ridge"], chosen["kpsa"],
             "temporal_categorical", "direct_gradient"]
    assay = list(dict.fromkeys(assay))
    records = stage_causal(args, spec, cache, assay, scope)
    result["assayed"] = assay
    result["records"] = {m: {f"{f:g}": v for f, v in r.items()} for m, r in records.items()}
    stride = float(np.diff(cache["query_ends"].numpy()).mean())
    block = math.ceil(512 / stride) + 1
    nonoverlap = max(1, math.ceil(512 / stride))
    result["query_stride"] = stride
    result["block_length_queries"] = block
    summary = {}
    for m in assay:
        summary[m] = {}
        for f in FRACTIONS:
            a = np.asarray(records[m][f]); c = np.asarray(records["scalar"][f]); o = np.asarray(records["direct_gradient"][f])
            sub = np.arange(0, len(a), nonoverlap)
            summary[m][f"{f:g}"] = {
                "mean_susceptibility": float(a.mean()),
                "gap_iid": iid_bootstrap_gap(a, c, o),
                "gap_block": block_bootstrap_gap(a, c, o, block),
                "gap_nonoverlapping_subset": iid_bootstrap_gap(a[sub], c[sub], o[sub]),
                "nonoverlapping_n": int(len(sub)),
            }
    result["summary"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    for m in assay:
        s = summary[m]["0.001"]
        print(f"{m:22s} gap@0.1% {s['gap_iid'][0]:.3f} iid[{s['gap_iid'][1]:.2f},{s['gap_iid'][2]:.2f}] "
              f"block[{s['gap_block'][1]:.2f},{s['gap_block'][2]:.2f}] sub(n={s['nonoverlapping_n']}) "
              f"[{s['gap_nonoverlapping_subset'][1]:.2f},{s['gap_nonoverlapping_subset'][2]:.2f}]", flush=True)


if __name__ == "__main__":
    main()
