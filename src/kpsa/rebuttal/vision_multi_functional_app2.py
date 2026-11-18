"""Several functionals with the paper's sign-free assay (exact parameter perturbation).

Functional j is the margin of the class ranked j-th by the intact model (fixed
per input). Each functional gets its own atlas of the paper's normalized
parameter-energy profile (36,864 coupled MLP features, normalized over the full
partition) from the same 6,400 sources, the same DINOv2-S query representation,
and the same prototype map. Selections are evaluated by the exact antithetic
perturbation assay (Equation protocol-susceptibility; sigma = 3% of the scoped
parameter RMS) of the same functional, normalized between constant sensitivity
(0%) and the query's own direct-gradient profile (100%).
Hyperparameters are frozen from the single-functional comparison (prototype
scale .1 and k = 16 chosen there on offset-48 development queries); the paper's
.025 prototype scale is also reported.
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
from ..vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from .batched_profiles import BatchedPartitionEnergy, freeze
from .vision_cache import MODEL
from .vision_multi_functional import load_images, ranked_classes
from .vision_retrieval_eval import bootstrap_gap

COUNTS = (8, 37)
METHODS = ("constant", "kpsa@0.025", "kpsa@0.1", "knn@1", "knn@16", "direct")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / "main/data/kpsa-rebuttal/vision_cache.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functionals", type=int, default=16)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--directions", type=int, default=8)
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
    scope = mlp_feature_scope(partition, layout)
    scope_idx = scope.nonzero().flatten()
    noise = scoped_parameter_rms(partition, scope) * 0.03
    freeze(model)
    engine = BatchedPartitionEnergy(model, partition)

    cols = torch.linspace(0, 199, args.queries).round().long().unique()
    q_idx = cache["roles"]["query49"][cols]
    imgs_s = load_images(dataset, cache["roles"]["source"])
    imgs_q = load_images(dataset, q_idx)
    ranks_s = ranked_classes(model, imgs_s, args.functionals)
    ranks_q = ranked_classes(model, imgs_q, args.functionals)
    cons = cache["features"]["prototype"]
    cmed = median_squared_distance(cons)
    phi = {}
    for s in (0.025, 0.1):
        pm = fit_prototype_response_map(cons, 800, bandwidth_squared=cmed * s * s, seed=91_027 + 800)
        phi[s] = (pm.transform(cache["features"]["source"]).float().cuda(),
                  pm.transform(cache["features"]["query49"][cols]).float().cuda())
    zs = cache["features"]["source"].cuda()
    order = (cache["features"]["query49"][cols].cuda() @ zs.T).argsort(1, descending=True)[:, :16]

    def profiles(images, labels):
        out = []
        for s in range(0, len(images), 32):
            e = engine(images[s:s + 32], labels[s:s + 32])
            out.append((e / e.sum(1, keepdim=True))[:, scope_idx.cuda()].float())
        return torch.cat(out)

    rec = {j: {m: {c: [] for c in COUNTS} for m in METHODS} for j in range(1, args.functionals + 1)}
    t0 = time.time()
    for j in range(1, args.functionals + 1):
        S = profiles(imgs_s, ranks_s[:, j - 1])
        Q = profiles(imgs_q, ranks_q[:, j - 1])
        scores = {"constant": S.mean(0, keepdim=True).expand(len(Q), -1), "direct": Q,
                  "knn@1": S[order[:, 0]], "knn@16": S[order].mean(1)}
        for s, (ps, pq) in phi.items():
            scores[f"kpsa@{s:g}"] = pq @ (ps.T @ S / len(S))
        for i in range(len(Q)):
            image = imgs_q[i:i + 1]
            lab = ranks_q[i:i + 1, j - 1]
            memo = {}
            for m in METHODS:
                top = torch.topk(scores[m][i], max(COUNTS)).indices.cpu()
                for c in COUNTS:
                    key = tuple(sorted(top[:c].tolist()))
                    if key not in memo:
                        sel = torch.zeros(partition.n_groups, dtype=torch.bool)
                        sel[scope_idx[top[:c]]] = True
                        vals = []
                        for d in range(args.directions):
                            seed = 161_803 + j * 10_007 + int(cols[i]) * 100_003 + d * 1_009
                            with torch.no_grad(), additive_group_noise(partition, sel, noise, seed=seed, sign=1):
                                plus = float(margin_values(model, image, lab)[0])
                            with torch.no_grad(), additive_group_noise(partition, sel, noise, seed=seed, sign=-1):
                                minus = float(margin_values(model, image, lab)[0])
                            vals.append(((plus - minus) / (2 * noise)) ** 2)
                        memo[key] = float(np.mean(vals))
                    rec[j][m][c].append(memo[key])
        print(f"functional {j}: {time.time() - t0:.0f}s", flush=True)
        del S, Q, scores
        torch.cuda.empty_cache()

    def gap(j, m, c):
        return bootstrap_gap(*(np.asarray(rec[j][x][c]) for x in (m, "constant", "direct")))

    summary = {str(j): {m: {str(c): gap(j, m, c) for c in COUNTS} for m in METHODS} for j in rec}
    pooled = {m: {str(c): float(np.mean([summary[str(j)][m][str(c)][0] for j in rec])) for c in COUNTS} for m in METHODS}
    # Pooled interval: bootstrap over queries jointly for all functionals (same queries).
    rng = np.random.default_rng(0)
    pooled_ci = {}
    n = len(cols)
    for m in METHODS:
        pooled_ci[m] = {}
        for c in COUNTS:
            arr = {x: np.stack([np.asarray(rec[j][x][c]) for j in rec]) for x in (m, "constant", "direct")}
            draws = []
            for _ in range(2000):
                idx = rng.integers(0, n, n)
                num = arr[m][:, idx].mean(1) - arr["constant"][:, idx].mean(1)
                den = arr["direct"][:, idx].mean(1) - arr["constant"][:, idx].mean(1)
                draws.append(np.mean(num / den))
            pooled_ci[m][str(c)] = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    result = {"functionals": args.functionals, "queries": int(n), "directions": args.directions, "counts": COUNTS,
              "summary_by_functional": summary, "mean_over_functionals": pooled, "mean_over_functionals_ci95": pooled_ci,
              "records": {str(j): {m: {str(c): v for c, v in r.items()} for m, r in rec[j].items()} for j in rec}}
    args.output.write_text(json.dumps(result, indent=1))
    for m in METHODS:
        print(f"{m:12s} mean over functionals", {c: round(100 * v, 1) for c, v in pooled[m].items()},
              "CI", {c: [round(100 * x, 1) for x in v] for c, v in pooled_ci[m].items()},
              "| j=1,2,4,8,16 @37:", [round(100 * summary[str(j)][m]["37"][0], 1) for j in (1, 2, 4, 8, 16) if str(j) in summary])


if __name__ == "__main__":
    main()
