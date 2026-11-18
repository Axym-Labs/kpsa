"""Signed and magnitude extensions of the vision atlas (R3 W2, R4 suggestion).

1. Signed atlas: the same kernel machinery applied to the signed, action-aligned
   source target (zero-deactivation gradient-times-activation per MLP feature,
   L1-normalized per source, or raw). Selections are evaluated with the exact
   zero-deactivation assay of the paper, against the energy atlas.
2. Magnitude: the total gradient energy discarded by normalization is one extra
   scalar per source; we estimate its logarithm for held-out queries with the
   same kernel weights and report rank correlation with the measured value.

Setting and splits are those of vision_retrieval_eval (N = 6,400 sources,
paper KPSA prototype map, test offsets 49 and 47).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from ..kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .vision_cache import MODEL, attribution_scope
from .vision_retrieval_eval import BatchedZeroAblation, bootstrap_gap, causal_assay, counts_for, rbf

FRACTIONS = (0.0002, 0.0005, 0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import timm
    from torchvision.datasets import ImageFolder

    from ..parameter_groups import ParameterPartition
    from ..vision_causal_refined import mlp_feature_layout, mlp_feature_scope

    cache = torch.load(args.cache, weights_only=False)
    device = torch.device("cuda")
    model = timm.create_model(MODEL, pretrained=True).to(device).eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    dataset = ImageFolder(Path.home() / "main/data/imagenet-1k-256/validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    if args.source_attribution.exists():
        src_attr = torch.load(args.source_attribution)
    else:
        src_attr, secs = attribution_scope(model, dataset, cache["roles"]["source"], partition, layout, scope)
        torch.save(src_attr, args.source_attribution)
        print(f"source attributions in {secs:.0f}s", flush=True)

    counts = counts_for(cache["scope_sizes"], FRACTIONS)
    construction = cache["features"]["prototype"]
    pmap = fit_prototype_response_map(construction, 800, bandwidth_squared=median_squared_distance(construction) * 0.025**2,
                                      seed=91_027 + 800)
    zs = cache["features"]["source"]
    phi_s = pmap.transform(zs).float().to(device)
    S = cache["source_profiles"].to(device)
    A = src_attr.to(device)
    A_l1 = A / A.abs().sum(1, keepdim=True).clamp_min(1e-30)
    atlases = {"energy": phi_s.T @ S, "signed_l1": phi_s.T @ A_l1, "signed_raw": phi_s.T @ A,
               "positive_l1": phi_s.T @ A_l1.clamp_min(0)}
    ablation = BatchedZeroAblation(model)
    result = {"feature_counts": counts, "atlases": list(atlases), "records": {}, "summary": {}}
    rec = {m: {f: [] for f in FRACTIONS} for m in [*atlases, "constant", "activation_attribution",
                                                    "nn1_signed_l1", "constant_signed"]}
    mag = {"true": [], "kpsa_nw": [], "nn1": [], "constant": []}
    log_tot_s = cache["source_totals"].log().float().to(device)
    med = median_squared_distance(zs)
    nn_sim_src = zs.to(device)
    for qname in ("query49", "query47"):
        zq = cache["features"][qname]
        phi_q = pmap.transform(zq).float().to(device)
        scores = {name: phi_q @ atlas for name, atlas in atlases.items()}
        w = rbf(zq.to(device), nn_sim_src, med * 0.1**2)
        nn = (zq.to(device) @ nn_sim_src.T).argmax(1)
        for row in cache[f"{qname}_correct"].nonzero().flatten().tolist():
            mag["true"].append(float(cache[f"{qname}_totals"][row].log()))
            mag["kpsa_nw"].append(float((w[row] @ log_tot_s) / w[row].sum()))
            mag["nn1"].append(float(log_tot_s[nn[row]]))
            mag["constant"].append(float(log_tot_s.mean()))
            image, label = dataset[int(cache["roles"][qname][row])]
            per = {name: s[row] for name, s in scores.items()}
            per["constant"] = S.mean(0)
            per["constant_signed"] = A_l1.mean(0)
            per["activation_attribution"] = cache[f"{qname}_attribution"][row].to(device)
            per["nn1_signed_l1"] = A_l1[nn[row]]
            sels, keys = [], []
            for m, s in per.items():
                top = torch.topk(s, max(counts)).indices
                for f, c in zip(FRACTIONS, counts):
                    sels.append(top[:c]); keys.append((m, f))
            deg, _ = causal_assay(model, ablation, image, int(label), sels)
            for (m, f), d in zip(keys, deg.tolist()):
                rec[m][f].append(d)
    for m in rec:
        result["summary"][m] = {}
        for f in FRACTIONS:
            a, c, o = (np.asarray(rec[x][f]) for x in (m, "constant", "activation_attribution"))
            result["summary"][m][f"{f:g}"] = bootstrap_gap(a, c, o)
    result["records"] = {m: {f"{f:g}": v for f, v in r.items()} for m, r in rec.items()}
    t = np.asarray(mag["true"])
    result["magnitude_log_total_energy"] = {
        k: {"spearman": float(spearmanr(t, np.asarray(v)).statistic) if k != "constant" else 0.0,
            "pearson": float(pearsonr(t, np.asarray(v)).statistic) if k != "constant" else 0.0,
            "rmse": float(np.sqrt(np.mean((t - np.asarray(v)) ** 2)))}
        for k, v in mag.items() if k != "true"}
    result["magnitude_log_total_energy"]["true_std"] = float(t.std())
    args.output.write_text(json.dumps(result, indent=1))
    for m in result["summary"]:
        print(m, {f: round(v[0], 3) for f, v in result["summary"][m].items()})
    print(json.dumps(result["magnitude_log_total_energy"], indent=1))


if __name__ == "__main__":
    main()
