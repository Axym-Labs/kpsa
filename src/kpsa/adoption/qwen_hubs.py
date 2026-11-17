"""Always-sensitive MLP features of Qwen3-0.6B-Base (research update 2).

The features whose mean exact sensitivity share over the 300k atlas sources
exceeds 0.4 % ("hub" features). Measures:
  shares    share of each held-out input's sensitivity in the hubs, exact and
            KPSA-estimated; mean profile ranking
  ablate    next-token NLL / top-1 accuracy on held-out paragraphs with the
            hubs zeroed, each block's hubs zeroed, each hub zeroed alone, and
            controls (random features of the same blocks, the next 15 by mean
            share); peak activation magnitudes
  rollouts  greedy continuations of a few prompts, intact vs zeroed
Outputs: kpsa-paper-internal/05_adoption_demo/artifacts/hubs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kpsa.adoption.language import OUT, Models, responses
from kpsa.adoption.language_demo import NF, Zero, choose

ART = Path.home() / "main/workspace/kpsa-paper-internal/05_adoption_demo/artifacts/hubs"
THRESHOLD = 0.004
LEN = 96
N_PARA = 512


def hubs():
    constant = torch.load(OUT / "constant_profile.pt")
    idx = (constant > THRESHOLD).nonzero().flatten()
    return constant, idx


@torch.no_grad()
def stage_shares(args):
    constant, idx = hubs()
    cand, scale, key, _ = choose()
    pmap = torch.load(OUT / "prototype_maps.pt")[cand]
    atlas = torch.load(OUT / "atlases.pt")[key].cuda()
    reps = torch.load(OUT / "heldout_reps.pt")[cand].cuda().float()
    exact = torch.from_numpy(np.load(OUT / "heldout_profiles.npy")).cuda().float()
    exact = exact / exact.sum(1, keepdim=True)
    est = responses(reps, pmap, scale) @ atlas.T
    est = est / est.sum(1, keepdim=True)
    order = constant.argsort(descending=True)
    out = {
        "hubs": [[int(i) // NF, int(i) % NF, float(constant[i])] for i in idx],
        "hub_total": float(constant[idx].sum()),
        "mean_sorted": constant[order[:200]].tolist(),
        "mean_sorted_is_hub": [bool((idx == i).any()) for i in order[:200].tolist()],
        "est_mean_sorted": est.mean(0).cpu()[order[:200]].tolist(),
        "exact_share": exact[:, idx.cuda()].sum(1).cpu().tolist(),
        "est_share": est[:, idx.cuda()].sum(1).cpu().tolist(),
        "next15": [int(i) for i in order[len(idx) : 2 * len(idx)]],
        "n_features": int(constant.numel()),
        # top-15 of each input, how many are hubs
        "hubs_in_top15": (exact.topk(15, 1).indices[..., None] == idx.cuda()).any(-1).sum(1).cpu().tolist(),
        "est_hubs_in_top15": (est.topk(15, 1).indices[..., None] == idx.cuda()).any(-1).sum(1).cpu().tolist(),
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "shares.json").write_text(json.dumps(out))
    print("hubs", out["hubs"], "total", out["hub_total"])
    print("exact share median", np.median(out["exact_share"]), "est", np.median(out["est_share"]))


def paragraphs(tok):
    items = json.loads((OUT / "heldout.json").read_text())
    seen, out = set(), []
    for it in items:
        key = tuple(it["ids"][:16])
        if len(it["ids"]) >= LEN and key not in seen:
            seen.add(key)
            out.append(it["ids"][:LEN])
    return torch.tensor(out[:N_PARA])


def evaluate(m, zero, data, features, batch=64):
    """Mean next-token NLL and top-1 accuracy with `features` (flat ids) zeroed."""
    masks = torch.ones(len(m.mlps), 1, NF, device="cuda", dtype=torch.bfloat16)
    for i in features:
        masks[i // NF, 0, i % NF] = 0
    zero.masks = masks if len(features) else None
    nll, acc, n = 0.0, 0.0, 0
    for s in range(0, len(data), batch):
        x = data[s : s + batch].cuda()
        logits = m.lm(input_ids=x).logits[:, :-1].float()
        y = x[:, 1:]
        nll += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        acc += (logits.argmax(-1) == y).sum().item()
        n += y.numel()
    zero.masks = None
    return nll / n, acc / n


@torch.no_grad()
def stage_ablate(args):
    constant, idx = hubs()
    shares = json.loads((ART / "shares.json").read_text())
    m = Models(embed=False)
    zero = Zero(m.mlps)
    data = paragraphs(m.tok)
    hub = idx.tolist()
    res = {"n_tokens": int(data[:, 1:].numel()), "n_paragraphs": len(data)}
    res["intact"] = evaluate(m, zero, data, [])
    res["all"] = evaluate(m, zero, data, hub)
    blocks = sorted({i // NF for i in hub})
    res["by_block"] = {str(b + 1): evaluate(m, zero, data, [i for i in hub if i // NF == b]) for b in blocks}
    res["single"] = {f"{i // NF + 1}_{i % NF}": evaluate(m, zero, data, [i]) for i in hub}
    res["next15"] = evaluate(m, zero, data, shares["next15"])
    g = torch.Generator().manual_seed(0)
    rand = []
    for _ in range(args.draws):
        pick = []
        for b in blocks:
            k = sum(1 for i in hub if i // NF == b)
            cands = torch.tensor([b * NF + f for f in range(NF) if b * NF + f not in hub])
            pick += cands[torch.randperm(len(cands), generator=g)[:k]].tolist()
        rand.append(evaluate(m, zero, data, pick))
    res["random_same_blocks"] = rand
    # peak |activation| (down_proj input) of the hubs vs all features of their blocks
    zero.acts = {}
    m.lm(input_ids=data[:64].cuda())
    peaks = {}
    for b in blocks:
        a = zero.acts[b].float().abs().amax((0, 1))
        peaks[str(b + 1)] = {"hub": {str(i % NF): float(a[i % NF]) for i in hub if i // NF == b},
                             "median_feature": float(a.median()), "p99_feature": float(a.quantile(0.99))}
    zero.acts = None
    res["peaks"] = peaks
    (ART / "ablate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("single", "random_same_blocks")}, indent=1))
    print("random", np.mean([r[0] for r in rand]), np.max([r[0] for r in rand]))
    print("single", res["single"])


PROMPTS = [
    "The river flows through the old town and",
    "In 1905, Albert Einstein published",
    "To make bread, you first",
    "The capital of France is",
]


@torch.no_grad()
def stage_rollouts(args):
    _, idx = hubs()
    m = Models(embed=False)
    zero = Zero(m.mlps)
    hub = idx.tolist()
    blocks = sorted({i // NF for i in hub})
    conditions = {"intact": [], "all": hub}
    for b in blocks:
        conditions[f"block{b + 1}"] = [i for i in hub if i // NF == b]
    out = {}
    for prompt in PROMPTS:
        out[prompt] = {}
        for name, feats in conditions.items():
            masks = torch.ones(len(m.mlps), 1, NF, device="cuda", dtype=torch.bfloat16)
            for i in feats:
                masks[i // NF, 0, i % NF] = 0
            zero.masks = masks if feats else None
            x = m.tok(prompt, return_tensors="pt").input_ids.cuda()
            y = m.lm.generate(x, max_new_tokens=24, do_sample=False, pad_token_id=m.tok.eos_token_id)
            zero.masks = None
            out[prompt][name] = m.tok.decode(y[0, x.shape[1] :])
            print(name, "|", prompt, "|", repr(out[prompt][name]), flush=True)
    (ART / "rollouts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["shares", "ablate", "rollouts"])
    parser.add_argument("--draws", type=int, default=10)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    {"shares": stage_shares, "ablate": stage_ablate, "rollouts": stage_rollouts}[args.stage](args)


if __name__ == "__main__":
    main()
