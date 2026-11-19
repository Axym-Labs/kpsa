"""Several functionals: category log-probabilities, exact parameter perturbation (App 2 assay).

Functional j is log p(C_j | x) = logsumexp_{c in C_j} z_c(x) - logsumexp_c z_c(x)
for the 16 entry-level categories of 16-class-ImageNet (Geirhos et al., ICLR
2019; 207 ImageNet classes; mapping in imagenet16_categories.json). The
categories are complementary coarse functionals defined for every input.
Each functional gets its own atlas of the paper's normalized parameter-energy
profile (36,864 coupled MLP features, normalized over the full partition)
from the same 6,400 sources, the same DINOv2-S query space, and the same
prototype map. Selections are scored by the exact antithetic perturbation
assay of the same functional (sigma = 3% of the scoped parameter RMS),
normalized between constant sensitivity (0%) and the query's own
direct-gradient profile (100%). KPSA hyperparameters are frozen from the
single-functional comparison (prototype scale .1); KPSA-1NN is the
reviewer-requested nearest-profile comparison.
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
from ..parameter_influence_circuits_v8 import additive_group_noise, scoped_parameter_rms
from ..vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from .batched_profiles import BatchedPartitionEnergy, freeze
from .vision_cache import MODEL
from .vision_multi_functional import load_images
from .vision_retrieval_eval import bootstrap_gap

COUNTS = (8, 37)
METHODS = ("constant", "kpsa@0.1", "knn@1", "direct")
CATEGORIES = Path.home() / "main/data/kpsa-rebuttal/imagenet16_categories.json"


def category_functional(indices):
    def f(logits):
        return torch.logsumexp(logits[:, indices], 1) - torch.logsumexp(logits, 1)
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / "main/data/kpsa-rebuttal/vision_cache.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--sources", type=int, default=6400,
                        help="first N sources (class-balanced prefixes of 200k)")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    import timm
    from torchvision.datasets import ImageFolder

    cats = json.loads(CATEGORIES.read_text())
    names = sorted(cats)
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
    src = cache["roles"]["source"][:args.sources]
    imgs_s = load_images(dataset, src)
    imgs_q = load_images(dataset, cache["roles"]["query49"][cols])
    dummy_s = torch.zeros(len(imgs_s), dtype=torch.long)
    dummy_q = torch.zeros(len(imgs_q), dtype=torch.long)
    cons = cache["features"]["prototype"]
    pm = fit_prototype_response_map(cons, 800, bandwidth_squared=median_squared_distance(cons) * 0.1 ** 2,
                                    seed=91_027 + 800)
    ps = pm.transform(cache["features"]["source"][:args.sources]).float().cuda()
    pq = pm.transform(cache["features"]["query49"][cols]).float().cuda()
    zs = cache["features"]["source"][:args.sources].cuda()
    nearest = (cache["features"]["query49"][cols].cuda() @ zs.T).argmax(1)

    def profiles(images, dummy, fn):
        out = []
        for s in range(0, len(images), 32):
            e = engine(images[s:s + 32], dummy[s:s + 32], functional=fn)
            out.append((e / e.sum(1, keepdim=True))[:, scope_idx.cuda()].float())
        return torch.cat(out)

    rec = {c: {m: {n: [] for n in COUNTS} for m in METHODS} for c in names}
    prob = {}
    t0 = time.time()
    for j, c in enumerate(names):
        idx = torch.tensor(cats[c], device="cuda")
        fn = category_functional(idx)
        S = profiles(imgs_s, dummy_s, fn)
        Q = profiles(imgs_q, dummy_q, fn)
        scores = {"constant": S.mean(0, keepdim=True).expand(len(Q), -1), "direct": Q,
                  "knn@1": S[nearest], "kpsa@0.1": pq @ (ps.T @ S / len(S))}
        with torch.no_grad():
            prob[c] = float(torch.cat([fn(model(imgs_q[s:s + 50].cuda()).float()).exp()
                                       for s in range(0, len(imgs_q), 50)]).mean())
        for i in range(len(Q)):
            image = imgs_q[i:i + 1].cuda()
            memo = {}
            for m in METHODS:
                top = torch.topk(scores[m][i], max(COUNTS)).indices.cpu()
                for n in COUNTS:
                    key = tuple(sorted(top[:n].tolist()))
                    if key not in memo:
                        sel = torch.zeros(partition.n_groups, dtype=torch.bool)
                        sel[scope_idx[top[:n]]] = True
                        vals = []
                        for d in range(args.directions):
                            seed = 271_828 + j * 10_007 + int(cols[i]) * 100_003 + d * 1_009
                            with torch.no_grad(), additive_group_noise(partition, sel, noise, seed=seed, sign=1):
                                plus = float(fn(model(image).float())[0])
                            with torch.no_grad(), additive_group_noise(partition, sel, noise, seed=seed, sign=-1):
                                minus = float(fn(model(image).float())[0])
                            vals.append(((plus - minus) / (2 * noise)) ** 2)
                        memo[key] = float(np.mean(vals))
                    rec[c][m][n].append(memo[key])
        print(f"{c}: {time.time() - t0:.0f}s", flush=True)
        del S, Q, scores
        torch.cuda.empty_cache()

    summary = {c: {m: {str(n): bootstrap_gap(*(np.asarray(rec[c][x][n]) for x in (m, "constant", "direct")))
                       for n in COUNTS} for m in METHODS} for c in names}
    rng = np.random.default_rng(0)
    pooled, pooled_ci = {}, {}
    q = len(cols)
    for m in METHODS:
        pooled[m], pooled_ci[m] = {}, {}
        for n in COUNTS:
            pooled[m][str(n)] = float(np.mean([summary[c][m][str(n)][0] for c in names]))
            arr = {x: np.stack([np.asarray(rec[c][x][n]) for c in names]) for x in (m, "constant", "direct")}
            draws = []
            for _ in range(2000):
                b = rng.integers(0, q, q)
                num = arr[m][:, b].mean(1) - arr["constant"][:, b].mean(1)
                den = arr["direct"][:, b].mean(1) - arr["constant"][:, b].mean(1)
                draws.append(np.mean(num / den))
            pooled_ci[m][str(n)] = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    result = {"functional": "log p(category | x), 16-class-ImageNet categories", "categories": names,
              "sources": int(args.sources),
              "queries": int(q), "directions": args.directions, "counts": COUNTS, "mean_category_probability": prob,
              "summary_by_functional": summary, "mean_over_functionals": pooled,
              "mean_over_functionals_ci95": pooled_ci,
              "records": {c: {m: {str(n): v for n, v in r.items()} for m, r in rec[c].items()} for c in names}}
    args.output.write_text(json.dumps(result, indent=1))
    for m in METHODS:
        print(f"{m:10s}", {n: round(100 * v, 1) for n, v in pooled[m].items()},
              "CI", {n: [round(100 * x, 1) for x in v] for n, v in pooled_ci[m].items()},
              "| per category @37:", {c: round(100 * summary[c][m]["37"][0], 1) for c in names})


if __name__ == "__main__":
    main()
