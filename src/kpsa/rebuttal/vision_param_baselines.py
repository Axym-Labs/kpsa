"""Figure 2 vision setting (exact parameter perturbation) with retrieval baselines.

Protocol from Table A1 (Sensitivity / ImageNet-1k row): ViT-B/16, 36,864
coupled MLP features, profiles normalized over the full partition, DINOv2-S
CLS with a 1,000-prototype RBF map (offset-43 construction images, scale .10),
N = 8,000 sources (offsets 0-7 of all 1,000 classes), queries at offset 45 for
200 evenly spaced classes (correct only), 32 antithetic Rademacher directions
with sigma = 3% of the scoped parameter RMS, 8/19/37 groups. Development
queries (offset 44, same classes) choose k, ridge strength and the exact-RBF
scale by direct-gradient energy coverage. The perturbation seeds are ours, so
numbers are protocol-matched rather than identical to the paper's run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from ..parameter_groups import ParameterPartition
from ..parameter_influence_circuits_v8 import additive_group_noise, margin_values, scoped_parameter_rms
from ..parameter_influence_interpretability_v8 import _indices_from_starts, _profile_scoped_normalized
from ..vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from ..vision_sample_refined import encode_indices
from .forecast_baselines import ridge_fit
from .vision_retrieval_eval import bootstrap_gap, rbf

COUNTS = (8, 19, 37)
EXACT_SCALES = (0.05, 0.1, 0.2, 0.4)
KS = (1, 2, 4, 8, 16, 32, 64)
RIDGE = (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0)


def predict(zs, S, zq, yq, ys, phi_s, phi_q, med):
    out = {"scalar": S.mean(0, keepdim=True).expand(len(zq), -1)}
    out["kpsa_proto@0.1"] = phi_q @ (phi_s.T @ S) / len(S)
    for s in EXACT_SCALES:
        out[f"kpsa_exact@{s:g}"] = rbf(zq, zs, med * s * s) @ S / len(S)
    order = (zq @ zs.T).argsort(1, descending=True)
    for k in KS:
        out[f"knn@{k}"] = S[order[:, :k]].mean(1)
    classes = torch.unique(ys)
    means = torch.stack([S[ys == c].mean(0) for c in classes])
    pos = {int(c): i for i, c in enumerate(classes)}
    out["class_average"] = means[torch.tensor([pos[int(c)] for c in yq], device=S.device)]
    for rel in RIDGE:
        out[f"ridge_z@{rel:g}"] = ridge_fit(zs, S, rel)(zq)
        out[f"ridge_phi@{rel:g}"] = ridge_fit(phi_s, S, rel)(phi_q)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path.home() / "main/data/imagenet-1k-256")
    parser.add_argument("--cache", type=Path, default=Path.home() / "main/data/kpsa-rebuttal/vision_param_cache.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directions", type=int, default=32)
    args = parser.parse_args()
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    raw = ImageFolder(args.data / "validation")
    starts = torch.arange(0, 50_000, 50)
    cols = torch.linspace(0, 999, 200).round().long()
    roles = {"source": _indices_from_starts(starts, range(8)), "construction": _indices_from_starts(starts, [43]),
             "dev": _indices_from_starts(starts, [44])[cols], "test": _indices_from_starts(starts, [45])[cols]}
    model = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True).cuda().eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    dataset = ImageFolder(args.data / "validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    scope_idx = scope.nonzero().flatten()
    if args.cache.exists():
        c = torch.load(args.cache, weights_only=False)
    else:
        enc = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).cuda().eval()
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
        c = {"features": {k: encode_indices(enc, proc, raw, v) for k, v in roles.items()}}
        del enc
        for k in ("source", "dev", "test"):
            prof, _, secs, correct = _profile_scoped_normalized(model, dataset, roles[k], partition, scope)
            c[f"{k}_profiles"] = prof.T.contiguous()
            c[f"{k}_correct"] = correct
            print(k, "profiled", secs, flush=True)
        torch.save(c, args.cache)
    labels = {k: torch.tensor([raw.targets[int(i)] for i in v]) for k, v in roles.items()}
    cons = c["features"]["construction"]
    pmap = fit_prototype_response_map(cons, 1000, bandwidth_squared=median_squared_distance(cons) * 0.1**2, seed=91_027 + 1000)
    dev_ = "cuda"
    S = c["source_profiles"].to(dev_)
    zs = c["features"]["source"].to(dev_)
    phi_s = pmap.transform(c["features"]["source"]).float().to(dev_)
    med = median_squared_distance(c["features"]["source"][torch.randperm(8000, generator=torch.Generator().manual_seed(0))[:4000]])
    preds = {}
    for k in ("dev", "test"):
        preds[k] = predict(zs, S, c["features"][k].to(dev_), labels[k].to(dev_), labels["source"].to(dev_),
                           phi_s, pmap.transform(c["features"][k]).float().to(dev_), med)

    def coverage(pred, truth, count):
        idx = torch.topk(pred, count, dim=1).indices
        return (truth.gather(1, idx).sum(1) / torch.topk(truth, count, dim=1).values.sum(1))

    result = {"counts": COUNTS, "coverage": {}}
    for k in ("dev", "test"):
        truth = c[f"{k}_profiles"].to(dev_)
        ok = c[f"{k}_correct"].to(dev_)
        result["coverage"][k] = {m: {str(n): float(coverage(p, truth, n)[ok].mean()) for n in COUNTS} for m, p in preds[k].items()}
    chosen = {}
    for m, v in result["coverage"]["dev"].items():
        if "@" in m:
            fam = m.split("@")[0]
            score = np.mean(list(v.values()))
            if fam not in chosen or score > chosen[fam][1]:
                chosen[fam] = (m, score)
    chosen = {f: m for f, (m, _) in chosen.items()}
    result["dev_chosen"] = chosen
    assay = list(dict.fromkeys(["scalar", "kpsa_proto@0.1", "kpsa_exact@0.1", chosen["kpsa_exact"], "knn@1",
                                chosen["knn"], "class_average", chosen["ridge_z"], chosen["ridge_phi"]]))
    noise = scoped_parameter_rms(partition, scope) * 0.03
    truth = c["test_profiles"]
    preds["test"]["direct"] = truth.to(dev_)
    assay.append("direct")
    rows = c["test_correct"].nonzero().flatten().tolist()
    rec = {m: {n: [] for n in COUNTS} for m in assay}
    memo = {}
    t0 = time.time()
    for pos, row in enumerate(rows):
        image, label = dataset[int(roles["test"][row])]
        images, lab = image[None], torch.tensor([int(label)])
        for m in assay:
            top = torch.topk(preds["test"][m][row], max(COUNTS)).indices.cpu()
            for n in COUNTS:
                key = (row, tuple(sorted(top[:n].tolist())))
                if key not in memo:
                    sel = torch.zeros(partition.n_groups, dtype=torch.bool)
                    sel[scope_idx[top[:n]]] = True
                    vals = []
                    for d in range(args.directions):
                        seed = 271_828 + row * 100_003 + d * 1_009
                        with torch.no_grad(), additive_group_noise(partition, sel, noise, seed=seed, sign=1):
                            plus = float(margin_values(model, images, lab)[0])
                        with torch.no_grad(), additive_group_noise(partition, sel, noise, seed=seed, sign=-1):
                            minus = float(margin_values(model, images, lab)[0])
                        vals.append(((plus - minus) / (2 * noise)) ** 2)
                    memo[key] = float(np.mean(vals))
                rec[m][n].append(memo[key])
        if (pos + 1) % 20 == 0:
            print(f"  {pos + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    result["assayed"] = assay
    result["n"] = len(rows)
    result["summary"] = {m: {str(n): bootstrap_gap(np.asarray(rec[m][n]), np.asarray(rec["scalar"][n]),
                                                    np.asarray(rec["direct"][n])) for n in COUNTS} for m in assay}
    result["records"] = {m: {str(n): v for n, v in r.items()} for m, r in rec.items()}
    args.output.write_text(json.dumps(result, indent=1))
    for m in assay:
        print(f"{m:18s}", {n: round(v[0] * 100, 1) for n, v in result["summary"][m].items()},
              "cov", {n: round(v * 100, 1) for n, v in result["coverage"]["test"][m].items()} if m in result["coverage"]["test"] else "")


if __name__ == "__main__":
    main()
