"""KPSA vs. nearest-neighbour, kNN, class-average and learned linear predictors.

Setting: the Figure 5 neuron-localization task (ViT-B/16, DINOv2-S query
space, exact zero-deactivation of coupled MLP features, 200 ImageNet classes).
All predictors use the same source images, the same normalized source
profiles, and the same query representation; only the map from query
representation to a profile estimate differs.

Hyperparameters of every family (k, ridge strength, KPSA bandwidth) are chosen
on development queries (offset 48) by mean gap recovery across budgets; test
queries are offsets 49 and 47. The paper's frozen KPSA settings are also
reported unchanged.
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

from ..kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .vision_cache import MODEL

FRACTIONS = (0.0002, 0.0005, 0.001)
EXACT_SCALES = (0.025, 0.05, 0.1, 0.2, 0.4)
PROTO_SCALES = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)
KS = (1, 2, 4, 8, 16, 32, 64)
RIDGE = (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0)


def counts_for(sizes, fractions):
    if not torch.allclose(sizes, sizes[0].expand_as(sizes)):
        raise RuntimeError("expected equal-size coupled MLP groups")
    out = []
    for f in fractions:
        target = f * float(sizes.sum())
        cumulative = sizes.double().cumsum(0)
        out.append(int((cumulative < target).sum()) + 1)
    return out


def rbf(q, s, bw2):
    """RBF weights up to a per-query positive factor (which leaves every ranking
    unchanged): distances are shifted by each query's nearest-source distance so
    that narrow bandwidths approach 1NN instead of underflowing to zero."""
    d2 = (2 - 2 * q @ s.T).clamp_min(0)
    return torch.exp(-(d2 - d2.min(1, keepdim=True).values) / (2 * bw2))


def ridge_fit(x, y, rel):
    """Ridge with unpenalized intercept; rel scales lambda by tr(X^T X)/d."""
    xm = x.mean(0, keepdim=True)
    ym = y.mean(0, keepdim=True)
    xc = (x - xm).double()
    gram = xc.T @ xc
    lam = rel * float(torch.trace(gram)) / gram.shape[0]
    w = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], device=x.device, dtype=gram.dtype),
                           xc.T @ (y - ym).double())
    return xm, ym, w.float()


def ridge_predict(model, x):
    xm, ym, w = model
    return (x - xm) @ w + ym


def predictions(cache, n, qname, proto_maps, device):
    """Return {method: (Q, G) scores} for the first n sources."""
    S = cache["source_profiles"][:n].to(device)
    zs = cache["features"]["source"][:n].to(device)
    zq = cache["features"][qname].to(device)
    ys = cache["labels"]["source"][:n].to(device)
    yq = cache["labels"][qname].to(device)
    out = {}
    out["constant"] = S.mean(0, keepdim=True).expand(len(zq), -1)
    med = median_squared_distance(zs.cpu()) if n > 1 else 1.0
    for s in EXACT_SCALES:
        w = rbf(zq, zs, med * s * s)
        out[f"kpsa_exact@{s:g}"] = (w @ S) / n
    for s, pmap in proto_maps.items():
        ps = pmap.transform(zs.cpu()).float().to(device)
        pq = pmap.transform(zq.cpu()).float().to(device)
        atlas = ps.T @ S / n  # (D, G): the stored KPSA atlas
        out[f"kpsa_proto@{s:g}"] = pq @ atlas
        if s == 0.025:
            phi_s, phi_q = ps, pq
    sim = zq @ zs.T
    order = sim.argsort(dim=1, descending=True)
    for k in KS:
        if k > n:
            continue
        idx = order[:, :k]
        out[f"knn@{k}"] = S[idx].mean(1)
    # Class average = categorical KPSA (delta kernel on the true class).
    classes = torch.unique(ys)
    means = torch.stack([S[ys == c].mean(0) for c in classes])
    lookup = {int(c): i for i, c in enumerate(classes)}
    out["class_average"] = means[torch.tensor([lookup[int(c)] for c in yq], device=device)]
    for rel in RIDGE:
        out[f"ridge_z@{rel:g}"] = ridge_predict(ridge_fit(zs, S, rel), zq)
        out[f"ridge_phi@{rel:g}"] = ridge_predict(ridge_fit(phi_s, S, rel), phi_q)
    del S
    return out


class BatchedZeroAblation:
    """Per-batch-row masks on the input of every fc2 (coupled MLP features)."""

    def __init__(self, model):
        self.blocks = [b.mlp.fc2 for b in model.blocks]
        self.masks = None
        self.handles = [m.register_forward_pre_hook(self._hook(i)) for i, m in enumerate(self.blocks)]

    def _hook(self, layer):
        def hook(_module, inputs):
            if self.masks is None:
                return None
            a = inputs[0]
            return (a * self.masks[:, layer][:, None, :].to(a.dtype), *inputs[1:])
        return hook


def margins_of(logits, labels):
    alt = logits.clone()
    rows = torch.arange(len(labels), device=logits.device)
    alt[rows, labels] = -torch.inf
    return logits[rows, labels] - alt.max(1).values


@torch.no_grad()
def causal_assay(model, ablation, image, label, selections, layers=12, width=3072, chunk=128):
    """selections: list of index tensors (into the 36,864 scope). Returns degradations."""
    device = next(model.parameters()).device
    rows = len(selections) + 1  # row 0 = intact
    masks = torch.ones(rows, layers * width, device=device)
    for r, sel in enumerate(selections, start=1):
        masks[r, sel.to(device)] = 0
    masks = masks.view(rows, layers, width)
    values = []
    for start in range(0, rows, chunk):
        m = masks[start:start + chunk]
        ablation.masks = m
        x = image[None].to(device).expand(len(m), -1, -1, -1)
        logits = model(x).float()
        values.append(margins_of(logits, torch.full((len(m),), label, device=device)))
    ablation.masks = None
    values = torch.cat(values)
    return (values[0] - values[1:]).cpu(), float(values[0])


def bootstrap_gap(method, const, oracle, draws=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(method)
    idx = rng.integers(0, n, size=(draws, n))
    num = method[idx].mean(1) - const[idx].mean(1)
    den = oracle[idx].mean(1) - const[idx].mean(1)
    stats = num / den
    point = (method.mean() - const.mean()) / (oracle.mean() - const.mean())
    return float(point), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def bootstrap_diff_gap(a, b, const, oracle, draws=5000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(draws, n))
    den = oracle[idx].mean(1) - const[idx].mean(1)
    stats = (a[idx].mean(1) - b[idx].mean(1)) / den
    point = (a.mean() - b.mean()) / (oracle.mean() - const.mean())
    return float(point), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def storage_bytes(family, n, groups=36864, dim=384, protos=800, classes=200):
    f32 = 4
    if family in ("kpsa_proto", "ridge_phi"):
        return (protos * groups + protos * dim) * f32
    if family in ("kpsa_exact", "knn"):
        return (n * groups + n * dim) * f32
    if family == "class_average":
        return classes * groups * f32
    if family == "ridge_z":
        return (dim + 1) * groups * f32 + dim * f32
    if family == "constant":
        return groups * f32
    return 0


def paper_subset(records, ids, cache):
    """Offset 49, 100 evenly spaced class columns, correct only (paper protocol)."""
    columns = set(torch.linspace(0, 199, 100).round().long().unique().tolist())
    keep = [i for i, (q, row) in enumerate(ids) if q == "query49" and row in columns]
    out = {}
    for m in ("kpsa_proto@0.025", "kpsa_exact@0.1", "knn@1", "class_average"):
        out[m] = {}
        for f in FRACTIONS:
            a = np.asarray(records[m][f])[keep]
            c = np.asarray(records["constant"][f])[keep]
            o = np.asarray(records["activation_attribution"][f])[keep]
            out[m][f"{f:g}"] = bootstrap_gap(a, c, o)
    out["n"] = len(keep)
    return out


def family_of(name):
    return name.split("@")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=int, nargs="+", default=[200, 400, 800, 1600, 3200, 6400])
    parser.add_argument("--max-queries", type=int, default=0, help="debug: cap queries per split")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda")

    import timm
    from torchvision.datasets import ImageFolder

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    sizes = cache["scope_sizes"]
    counts = counts_for(sizes, FRACTIONS)
    print("feature counts per budget", counts, flush=True)

    construction = cache["features"]["prototype"]
    cmed = median_squared_distance(construction)
    proto_maps = {
        s: fit_prototype_response_map(construction, 800, bandwidth_squared=cmed * s * s, seed=91_027 + 800)
        for s in PROTO_SCALES
    }

    model = timm.create_model(MODEL, pretrained=True).to(device).eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    data = Path.home() / "main/data/imagenet-1k-256"
    dataset = ImageFolder(data / "validation", transform=transform)
    ablation = BatchedZeroAblation(model)
    from ..parameter_groups import ParameterPartition
    from ..vision_causal_refined import mlp_feature_layout
    layout = mlp_feature_layout(model, ParameterPartition(model, "swiglu"))
    offsets = [int(item["offset"]) for item in layout]
    if [item["name"] for item in layout] != [f"blocks.{i}.mlp" for i in range(12)] or offsets != sorted(offsets):
        raise RuntimeError("scope order must be blocks 0..11 with 3,072 contiguous features")

    # Fidelity check: batched masked ablation equals the paper's single-image hook.
    from ..vision_causal_refined import functional_value, zero_ablate_mlp_activations
    image, label = dataset[int(cache["roles"]["query49"][0])]
    scope_index = torch.arange(36864)
    attr = cache["query49_attribution"][0]
    sel = torch.topk(attr.clamp_min(0), 37).indices
    deg, intact = causal_assay(model, ablation, image, int(label), [sel, sel[:8]])
    full = torch.zeros(layout[-1]["offset"] + 3072, dtype=torch.bool)
    for i in sel.tolist():
        full[layout[i // 3072]["offset"] + i % 3072] = True
    base = functional_value(model, image, int(label), "margin")
    with zero_ablate_mlp_activations(layout, full):
        single = base - functional_value(model, image, int(label), "margin")
    print(f"fidelity: batched {float(deg[0]):.6f} vs single {single:.6f}; intact {intact:.5f} vs {base:.5f}",
          flush=True)
    if abs(float(deg[0]) - single) > 1e-3 * max(1.0, abs(single)):
        raise RuntimeError("batched ablation disagrees with the paper hook")
    del scope_index

    splits = {"dev": ["query48"], "test": ["query49", "query47"]}
    result = {"protocol": {
        "model": MODEL, "query_space": "DINOv2-S CLS (bf16 encode, L2-normalized)",
        "fractions": FRACTIONS, "feature_counts": counts,
        "exact_scales": EXACT_SCALES, "proto_scales": PROTO_SCALES, "ks": KS, "ridge_rel": RIDGE,
        "paper_kpsa": {"kpsa_proto": 0.025, "kpsa_exact": 0.1},
        "selection": "per family and N: hyperparameter maximizing mean dev gap recovery over budgets",
        "test_queries": "offsets 49 and 47, correctly classified only",
        "torch": torch.__version__,
    }, "by_n": {}}

    # Per-query intact data and correctness.
    per_split_queries = {}
    for split, names in splits.items():
        qs = []
        for qname in names:
            correct = cache[f"{qname}_correct"]
            for row in correct.nonzero().flatten().tolist():
                qs.append((qname, row))
        if args.max_queries:
            qs = qs[: args.max_queries]
        per_split_queries[split] = qs
        print(split, "queries", len(qs), flush=True)

    # Oracle (activation attribution) and direct-profile selections are N-independent.
    t_start = time.perf_counter()
    for n in args.sources:
        t0 = time.perf_counter()
        preds = {q: predictions(cache, n, q, proto_maps, device) for q in ("query47", "query48", "query49")}
        methods = list(preds["query48"].keys()) + ["activation_attribution", "direct_profile"]
        records = {split: {m: {f: [] for f in FRACTIONS} for m in methods} for split in splits}
        energy = {split: {m: {f: [] for f in FRACTIONS} for m in methods} for split in splits}
        overlap = {split: {m: {f: [] for f in FRACTIONS} for m in methods} for split in splits}
        kmax = max(counts)
        for split, qs in per_split_queries.items():
            for qname, row in qs:
                index = int(cache["roles"][qname][row])
                image, label = dataset[index]
                own = cache[f"{qname}_profiles"][row].to(device)
                attr = cache[f"{qname}_attribution"][row].to(device)
                tops = {}
                for m in methods:
                    if m == "activation_attribution":
                        s = attr
                    elif m == "direct_profile":
                        s = own
                    else:
                        s = preds[qname][m][row]
                    tops[m] = torch.topk(s.clamp_min(0), kmax).indices
                own_best = torch.topk(own, kmax).values.cumsum(0)
                attr_top = torch.topk(attr.clamp_min(0), kmax).indices
                selections, keys = [], []
                for m in methods:
                    for f, c in zip(FRACTIONS, counts):
                        sel = tops[m][:c]
                        selections.append(sel)
                        keys.append((m, f))
                        energy[split][m][f].append(float(own[sel].sum() / own_best[c - 1]))
                        overlap[split][m][f].append(
                            len(set(sel.tolist()) & set(attr_top[:c].tolist())) / c)
                deg, _ = causal_assay(model, ablation, image, int(label), selections)
                for (m, f), d in zip(keys, deg.tolist()):
                    records[split][m][f].append(d)
        # Summaries.
        summary = {}
        for split in splits:
            summary[split] = {}
            for m in methods:
                summary[split][m] = {}
                for f in FRACTIONS:
                    a = np.asarray(records[split][m][f])
                    c = np.asarray(records[split]["constant"][f])
                    o = np.asarray(records[split]["activation_attribution"][f])
                    point, lo, hi = bootstrap_gap(a, c, o)
                    summary[split][m][f"{f:g}"] = {
                        "mean_degradation": float(a.mean()),
                        "gap_recovered": point, "gap_ci95": [lo, hi],
                        "energy_captured": float(np.mean(energy[split][m][f])),
                        "attribution_overlap": float(np.mean(overlap[split][m][f])),
                        "n": len(a),
                    }
        # Dev selection per family.
        families = {}
        for m in methods:
            families.setdefault(family_of(m), []).append(m)
        chosen = {}
        for fam, members in families.items():
            best = max(members, key=lambda m: np.mean([summary["dev"][m][f"{f:g}"]["gap_recovered"] for f in FRACTIONS]))
            chosen[fam] = best
        # Paired test differences: KPSA (paper settings and dev-chosen) minus each baseline.
        diffs = {}
        anchors = {"kpsa_proto_paper": "kpsa_proto@0.025", "kpsa_proto_dev": chosen["kpsa_proto"],
                   "kpsa_exact_dev": chosen["kpsa_exact"]}
        for aname, am in anchors.items():
            diffs[aname] = {}
            for fam, bm in chosen.items():
                if fam in ("kpsa_proto", "kpsa_exact", "activation_attribution"):
                    continue
                diffs[aname][bm] = {}
                for f in FRACTIONS:
                    a = np.asarray(records["test"][am][f]); b = np.asarray(records["test"][bm][f])
                    c = np.asarray(records["test"]["constant"][f]); o = np.asarray(records["test"]["activation_attribution"][f])
                    diffs[aname][bm][f"{f:g}"] = bootstrap_diff_gap(a, b, c, o)
        result["by_n"][str(n)] = {
            "summary": summary, "dev_chosen": chosen, "paired_test_differences": diffs,
            "storage_bytes": {fam: storage_bytes(fam, n) for fam in families},
            "records_test": {m: {f"{f:g}": records["test"][m][f] for f in FRACTIONS} for m in methods},
            "test_query_ids": per_split_queries["test"],
            "paper_subset": paper_subset(records["test"], per_split_queries["test"], cache),
            "seconds": time.perf_counter() - t0,
        }
        line = " ".join(
            f"{fam}={summary['test'][chosen[fam]]['0.001']['gap_recovered']:.3f}" for fam in chosen)
        print(f"N={n} ({time.perf_counter()-t0:.0f}s) test gap@0.1% dev-chosen: {line} | paper-proto "
              f"{summary['test']['kpsa_proto@0.025']['0.001']['gap_recovered']:.3f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1))
        del preds
        torch.cuda.empty_cache()

    # Paper-subset check (offset 49, 100 evenly spaced classes, N=6,400, paper KPSA).
    result["seconds_total"] = time.perf_counter() - t_start
    args.output.write_text(json.dumps(result, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
