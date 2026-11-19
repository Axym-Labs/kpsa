"""Timed end-to-end run for several functionals (one process per side).

kpsa: build M atlases from N source images, then answer Q queries.
  construction = one forward and M backward passes per source batch through the
  batched exact group-energy engine (as direct_multi), encoding of the N sources
  and of the 800 prototype-construction images with DINOv2-S, k-means prototype
  fitting, and accumulation of the M atlases; queries = DINOv2-S pass + stacked
  atlas product + top-k for all M functionals.
attribution: answer the same Q queries with M sequential backward passes.
Inputs are one resident batch of 32 pre-transformed images, reused (preprocessing
is excluded for both sides, as in vision_cost_bench); everything is synchronized
and timed with a wall clock around the whole run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .vision_cost_bench import GROUPS, Encoder, build, load_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("kpsa", "attribution"), required=True)
    parser.add_argument("--functionals", type=int, required=True)
    parser.add_argument("--sources", type=int, default=800)
    parser.add_argument("--queries", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    k, batch = args.functionals, 32
    vit_px, vit_lab = (t.cuda() for t in load_images(batch, "vit"))
    dino_px, _ = load_images(batch, "dino")
    dino_px = dino_px.cuda()
    rec = vars(args).copy()
    rec["output"] = str(args.output)
    n_query_batches = math.ceil(args.queries / batch)
    if args.side == "kpsa":
        from ..kernel_sensitivity import fit_prototype_response_map, median_squared_distance

        profile_run, _ = build("direct_multi", "fp32", args.sources, k, 37)
        encoder = Encoder("fp32")
        query_run, _ = build("kpsa_multi_stacked", "fp32", args.sources, k, 37)
        # warm-up of every stage outside the timer
        profile_run(vit_px, vit_lab); encoder(dino_px); query_run(dino_px, vit_lab)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(math.ceil(args.sources / batch)):
            profile_run(vit_px, vit_lab)          # exact profiles for all k functionals
        feats = [encoder(dino_px).cpu() for _ in range(math.ceil((args.sources + 800) / batch))]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        z = F.normalize(torch.cat(feats)[: args.sources + 800].float(), dim=1)
        pmap = fit_prototype_response_map(z[args.sources:], 800,
                                          bandwidth_squared=median_squared_distance(z[args.sources:]) * 0.01, seed=0)
        phi = pmap.transform(z[: args.sources]).float().cuda()
        profiles = torch.rand(args.sources, GROUPS, device="cuda")
        atlases = [phi.T @ profiles / args.sources for _ in range(k)]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(n_query_batches):
            query_run(dino_px, vit_lab)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        rec.update(construction_profiles_encoding_s=t1 - t0, fit_accumulate_s=t2 - t1, queries_s=t3 - t2,
                   total_s=t3 - t0)
        del atlases
    else:
        run, _ = build("attribution_multi", "fp32", 6400, k, 37)
        run(vit_px, vit_lab)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_query_batches):
            run(vit_px, vit_lab)
        torch.cuda.synchronize()
        rec.update(total_s=time.perf_counter() - t0)
    rec["queries_answered"] = n_query_batches * batch
    rec["images_per_second"] = rec["queries_answered"] / rec["total_s"]
    with args.output.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
