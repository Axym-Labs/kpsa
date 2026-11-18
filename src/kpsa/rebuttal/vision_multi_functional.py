"""Several functionals, one query space: accuracy of M atlases (Fig 5 setting).

Functional j is the logit margin of the class ranked j-th by the intact model
(j = 1 is the predicted-class margin): F_j(x) = z_{c_j(x)}(x) - max_{c != c_j(x)} z_c(x),
with c_j fixed from the intact forward pass. One atlas per functional is built
from the same 6,400 sources (exact batched profiles), the same DINOv2-S query
representation, and the same prototype map. Every atlas is evaluated with the
exact zero-deactivation assay of its own functional against its own
action-matched activation attribution. Because profiles are unsigned and, for
j > 1, the strongest features of F_j mostly support the top class (zeroing them
raises F_j), the assay scores the absolute change |dF_j| and the reference ranks
features by the absolute first-order effect |sum_t g_t a_t|, as in the paper's
protein localization. Hyperparameters (prototype bandwidth, k)
are chosen once for all functionals on the offset-48 queries; test queries are
offsets 49 and 47 (all queries; correctness is not defined for j > 1).
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
from ..vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from .batched_profiles import BatchedPartitionEnergy, freeze
from .vision_cache import MODEL
from .vision_retrieval_eval import BatchedZeroAblation, bootstrap_gap, causal_assay, counts_for

FRACTIONS = (0.0002, 0.0005, 0.001)
PROTO = (0.025, 0.05, 0.1, 0.2)
KS = (1, 2, 4, 8, 16, 32)


def load_images(dataset, indices):
    return torch.stack([dataset[int(i)][0] for i in indices])


def attribution(model, images, labels, batch=32):
    """Zero-deactivation gradient-times-activation per coupled MLP feature (fp32)."""
    captured = {}
    hooks = [b.mlp.fc2.register_forward_pre_hook(lambda _m, inp, i=i: captured.__setitem__(i, inp[0]))
             for i, b in enumerate(model.blocks)]
    out = []
    try:
        for s in range(0, len(images), batch):
            x = images[s:s + batch].float().cuda().requires_grad_(True)
            lab = labels[s:s + batch].cuda()
            logits = model(x).float()
            rows = torch.arange(len(lab), device=x.device)
            alt = logits.clone(); alt[rows, lab] = -torch.inf
            obj = (logits[rows, lab] - alt.max(1).values).sum()
            acts = [captured[i] for i in range(12)]
            grads = torch.autograd.grad(obj, acts)
            # Absolute first-order zero-deactivation effect: the unsigned profile cannot know
            # the sign of a feature's effect on F_j, so the assay measures |change| (as for protein).
            out.append(torch.cat([(g * a).sum(1) for g, a in zip(grads, acts)], 1).abs().detach().cpu())
            captured.clear()
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(out)


@torch.no_grad()
def ranked_classes(model, images, m, batch=64):
    return torch.cat([model(images[s:s + batch].float().cuda()).float().topk(m, dim=1).indices.cpu()
                      for s in range(0, len(images), batch)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / "main/data/kpsa-rebuttal/vision_cache.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functionals", type=int, default=16)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    import timm
    from torchvision.datasets import ImageFolder

    cache = torch.load(args.cache, weights_only=False)
    model = timm.create_model(MODEL, pretrained=True).cuda().eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    dataset = ImageFolder(Path.home() / "main/data/imagenet-1k-256/validation", transform=transform)
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope_idx = mlp_feature_scope(partition, layout).nonzero().flatten().cuda()
    counts = counts_for(cache["scope_sizes"], FRACTIONS)
    freeze(model)
    engine = BatchedPartitionEnergy(model, partition)
    ablation = BatchedZeroAblation(model)

    t0 = time.time()
    imgs = {k: load_images(dataset, cache["roles"][k]) for k in ("source", "query48", "query49", "query47")}
    imgs["test"] = torch.cat([imgs.pop("query49"), imgs.pop("query47")])
    imgs["dev"] = imgs.pop("query48")
    feats = {"source": cache["features"]["source"], "dev": cache["features"]["query48"],
             "test": torch.cat([cache["features"]["query49"], cache["features"]["query47"]])}
    ranks = {k: ranked_classes(model, v, args.functionals) for k, v in imgs.items()}
    print(f"images and ranks in {time.time() - t0:.0f}s", flush=True)

    cons = cache["features"]["prototype"]
    cmed = median_squared_distance(cons)
    pmaps = {s: fit_prototype_response_map(cons, 800, bandwidth_squared=cmed * s * s, seed=91_027 + 800) for s in PROTO}
    phi = {s: {k: pm.transform(v).float().cuda() for k, v in feats.items()} for s, pm in pmaps.items()}
    zs = feats["source"].cuda()
    order = {k: (feats[k].cuda() @ zs.T).argsort(1, descending=True)[:, :max(KS)] for k in ("dev", "test")}

    def run_split(split, S, attr, labels, methods):
        rec = {m: {f: [] for f in FRACTIONS} for m in methods}
        for i in range(len(labels)):
            sels, keys = [], []
            for m in methods:
                if m == "activation_attribution":
                    s = attr[i].cuda()
                elif m == "constant":
                    s = S.mean(0)
                elif m.startswith("kpsa"):
                    sc = float(m.split("@")[1])
                    s = phi[sc][split][i] @ atlases[sc]
                else:
                    k = int(m.split("@")[1])
                    s = S[order[split][i, :k]].mean(0)
                top = torch.topk(s, max(counts)).indices
                for f, c in zip(FRACTIONS, counts):
                    sels.append(top[:c]); keys.append((m, f))
            deg, _ = causal_assay(model, ablation, imgs[split][i].float(), int(labels[i]), sels)
            for (m, f), d in zip(keys, deg.abs().tolist()):
                rec[m][f].append(d)
        return rec

    candidates = ["constant", "activation_attribution"] + [f"kpsa@{s:g}" for s in PROTO] + [f"knn@{k}" for k in KS]
    per_j = {}
    construction_seconds = []
    for j in range(args.functionals):
        t1 = time.time()
        lab_s = ranks["source"][:, j]
        shares = []
        for s in range(0, len(lab_s), 32):
            e = engine(imgs["source"][s:s + 32].float(), lab_s[s:s + 32])
            shares.append((e / e.sum(1, keepdim=True))[:, scope_idx].float())
        S = torch.cat(shares)
        torch.cuda.synchronize()
        construction_seconds.append(time.time() - t1)
        atlases = {s: phi[s]["source"].T @ S / len(S) for s in PROTO}
        out = {}
        for split in ("dev", "test"):
            lab = ranks[split][:, j]
            attr = attribution(model, imgs[split], lab)
            out[split] = run_split(split, S, attr, lab, candidates)
        per_j[j + 1] = out
        print(f"functional {j + 1}: profiles {construction_seconds[-1]:.0f}s, total {time.time() - t1:.0f}s", flush=True)
        del S, atlases
        torch.cuda.empty_cache()

    def gap(rec, m, f):
        return bootstrap_gap(np.asarray(rec[m][f]), np.asarray(rec["constant"][f]), np.asarray(rec["activation_attribution"][f]))

    dev_score = {m: np.mean([gap(per_j[j]["dev"], m, f)[0] for j in per_j for f in FRACTIONS]) for m in candidates}
    chosen = {fam: max((m for m in candidates if m.startswith(fam + "@")), key=dev_score.get) for fam in ("kpsa", "knn")}
    report = ["constant", "kpsa@0.025", chosen["kpsa"], "knn@1", chosen["knn"], "activation_attribution"]
    report = list(dict.fromkeys(report))
    summary = {str(j): {m: {f"{f:g}": gap(per_j[j]["test"], m, f) for f in FRACTIONS} for m in report} for j in per_j}
    pooled = {}
    for m in report:
        pooled[m] = {}
        for f in FRACTIONS:
            vals = [summary[str(j)][m][f"{f:g}"][0] for j in per_j]
            pooled[m][f"{f:g}"] = {"mean_over_functionals": float(np.mean(vals)), "min": float(np.min(vals)),
                                   "max": float(np.max(vals))}
    result = {"functionals": args.functionals, "feature_counts": counts, "dev_chosen": chosen,
              "test_n": len(ranks["test"]), "construction_seconds_per_functional": construction_seconds,
              "summary_by_functional": summary, "mean_over_functionals": pooled,
              "records_test": {str(j): {m: {f"{f:g}": per_j[j]["test"][m][f] for f in FRACTIONS} for m in report} for j in per_j}}
    args.output.write_text(json.dumps(result, indent=1))
    for m in report:
        print(f"{m:24s}", {f: round(100 * v["mean_over_functionals"], 1) for f, v in pooled[m].items()},
              "j=1..4 @0.1%:", [round(100 * summary[str(j)][m]["0.001"][0], 1) for j in range(1, min(5, args.functionals + 1))])


if __name__ == "__main__":
    main()
