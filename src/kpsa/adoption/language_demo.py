"""Precompute the language half of the KPSA demo (Qwen3-0.6B-Base).

Stages:
  targets   per passage and token: model probability and alternatives, KPSA
            ranking (top 0.1 %), exact deactivation effects, layer mass and
            atlas neighbours
  features  feature cards: top-activating Wikipedia snippets per feature
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kpsa.adoption.language import OUT, Models, paragraphs, responses

SITE = Path.home() / "main/pinned/axym-website/public/demo/kpsa/language"
BUDGETS = [(0.02, 17), (0.05, 43), (0.1, 86)]
N_SINGLE = 5
NF = 3072


def choose():
    results = json.loads((OUT / "eval.json").read_text())
    key = max((k for k in results if k != "constant"), key=lambda k: results[k])
    c, s = key.split("|")
    return c, float(s), key, results


def trim(ids, tok, min_tokens=60):
    """Cut a truncated paragraph after its last full sentence."""
    for end in range(len(ids), min_tokens, -1):
        if tok.decode(ids[end - 1 : end]).rstrip().endswith((".", "!", "?")):
            return ids[:end]
    return ids


class Zero:
    def __init__(self, mlps):
        self.masks = None
        self.acts = None
        self.handles = [m.down_proj.register_forward_pre_hook(self._hook(i)) for i, m in enumerate(mlps)]

    def _hook(self, layer):
        def hook(module, args):
            x = args[0]
            if self.acts is not None:
                self.acts[layer] = x.detach()
            if self.masks is not None:
                x = x * self.masks[layer][:, None, :]
            return (x,)

        return hook


@torch.no_grad()
def stage_targets(args):
    cand, scale, key, results = choose()
    print("query space", cand, scale, results, flush=True)
    m = Models(embed=cand == "emb")
    tok = m.tok
    zero = Zero(m.mlps)
    pmaps = torch.load(OUT / "prototype_maps.pt")
    atlas = torch.load(OUT / "atlases.pt")[key].cuda()
    src_reps = torch.load(OUT / "source_reps.pt")[cand].cuda().float()
    sources = json.loads((OUT / "sources.json").read_text())
    bw2 = scale**2 * pmaps[cand]["median"]
    passages = json.loads((OUT / "demo_passages.json").read_text())[: args.limit or None]
    index = {"model": "Qwen3-0.6B-Base", "queryModel": {"own": "Qwen3-0.6B-Base hidden state", "own_tgt": "Qwen3-0.6B-Base hidden state and target embedding", "emb": "Qwen3-Embedding-0.6B"}[cand],
             "layers": len(m.mlps), "featuresPerLayer": NF, "atlasSources": len(sources),
             "budgets": [{"pct": p, "k": k} for p, k in BUDGETS], "passages": []}
    (SITE / "p").mkdir(parents=True, exist_ok=True)
    needed = set()
    for n, item in enumerate(passages):
        pid = f"p{n:04d}"
        ids = trim(item["ids"], tok)
        x = torch.tensor(ids, device="cuda")[None]
        zero.masks = None
        hid = m.lm.model(input_ids=x).last_hidden_state[0]
        probs = m.lm.lm_head(hid).float().softmax(-1)
        T = len(ids)
        targets = list(range(1, T))
        tgt_ids = torch.tensor(ids[1:], device="cuda")
        own = F.normalize(hid[:-1].float(), dim=-1)
        if cand == "own":
            z = own
        elif cand == "own_tgt":
            z = torch.cat((own, F.normalize(m.lm.model.embed_tokens.weight[tgt_ids].float(), dim=-1)), -1) / 2**0.5
        else:
            z = torch.cat([m.embed_texts([tok.decode(ids[: t + 1]) for t in targets[i : i + 64]]) for i in range(0, len(targets), 64)])
        scores = responses(z, pmaps[cand], scale) @ atlas.T
        scores = scores / scores.sum(1, keepdim=True)
        top = scores.topk(86, dim=1)
        # exact deactivation: rows = targets x (single features + budget sets)
        per = N_SINGLE + len(BUDGETS)
        rows = len(targets) * per
        p_after = torch.empty(rows, device="cuda")
        for start in range(0, rows, args.rows):
            stop = min(rows, start + args.rows)
            masks = torch.ones(len(m.mlps), stop - start, NF, device="cuda", dtype=torch.bfloat16)
            for r in range(start, stop):
                ti, kind = divmod(r, per)
                idx = top.indices[ti]
                chosen = idx[kind : kind + 1] if kind < N_SINGLE else idx[: BUDGETS[kind - N_SINGLE][1]]
                masks[chosen // NF, r - start, chosen % NF] = 0
            zero.masks = masks
            h = m.lm.model(input_ids=x.expand(stop - start, -1)).last_hidden_state
            zero.masks = None
            t_rows = torch.arange(start, stop, device="cuda") // per  # target ti is predicted at position ti
            logits = m.lm.lm_head(h[torch.arange(stop - start, device="cuda"), t_rows]).float()
            p_after[start:stop] = logits.softmax(-1).gather(1, tgt_ids[t_rows][:, None])[:, 0]
        p_after = p_after.reshape(len(targets), per).cpu().numpy()
        w = torch.exp(-torch.cdist(z, src_reps).square() / (2 * bw2))
        nb = w.topk(3, dim=1)
        out_targets = {}
        top5 = probs[:-1].topk(5, dim=1)
        p_true = probs[:-1].gather(1, tgt_ids[:, None])[:, 0]
        for ti, t in enumerate(targets):
            feats = [[int(i) // NF, int(i) % NF, round(float(v), 6)] for v, i in zip(top.values[ti], top.indices[ti])]
            needed.update((a, b) for a, b, _ in feats)
            mass = [0.0] * len(m.mlps)
            for a, _, v in feats:
                mass[a] += v
            tot = sum(mass)
            neigh = []
            for v, i in zip(nb.values[ti], nb.indices[ti]):
                s = sources[int(i)]
                ctx = s["ids"][max(0, s["t"] - 24) : s["t"]]
                neigh.append({"text": ("…" if s["t"] > 24 else "") + tok.decode(ctx), "target": tok.decode(s["ids"][s["t"] : s["t"] + 1]),
                              "w": round(float(v / w[ti].sum()), 4)})
            out_targets[str(t)] = {
                "p": round(float(p_true[ti]), 4),
                "top": [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(top5.values[ti], top5.indices[ti])],
                "features": feats,
                "deact": [round(float(v), 4) for v in p_after[ti, :N_SINGLE]],
                "setEffect": [round(float(v), 4) for v in p_after[ti, N_SINGLE:]],
                "layerMass": [round(v / tot, 4) for v in mass],
                "neighbors": neigh,
            }
        tokens = [tok.decode([i]) for i in ids]
        (SITE / f"p/{pid}.json").write_text(json.dumps({"id": pid, "tokens": tokens, "targets": out_targets}, separators=(",", ":")))
        index["passages"].append({"id": pid, "title": item["title"], "url": item["url"], "text": tok.decode(ids)})
        if n % 50 == 0:
            print(n, item["title"], len(needed), flush=True)
    (SITE / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    torch.save(sorted(needed), OUT / "needed.pt")
    print("passages", len(index["passages"]), "features", len(needed))


@torch.no_grad()
def stage_features(args):
    m = Models(embed=False)
    tok = m.tok
    zero = Zero(m.mlps)
    needed = torch.load(OUT / "needed.pt")
    by_layer = {}
    for l, f in needed:
        by_layer.setdefault(l, []).append(f)
    layer_idx = {l: torch.tensor(fs, device="cuda") for l, fs in by_layer.items()}
    windows = paragraphs(1, tok, 32, args.windows, seed=3)
    k = args.k
    best_v = {l: torch.full((k, len(fs)), -1e9, device="cuda") for l, fs in by_layer.items()}
    best_i = {l: torch.zeros((k, len(fs)), dtype=torch.long, device="cuda") for l, fs in by_layer.items()}
    best_p = {l: torch.zeros((k, len(fs)), dtype=torch.long, device="cuda") for l, fs in by_layer.items()}
    order = sorted(range(len(windows)), key=lambda i: len(windows[i]["ids"]))
    for start in range(0, len(order), args.batch):
        chunk = order[start : start + args.batch]
        ids, mask = m.batch_inputs([windows[i]["ids"] for i in chunk])
        zero.acts = {}
        m.lm.model(input_ids=ids, attention_mask=mask)
        valid = mask.bool()
        for l, fidx in layer_idx.items():
            a = zero.acts[l][:, :, fidx].float().masked_fill(~valid[:, :, None], -1e9)
            v, pos = a.max(1)
            wid = torch.tensor(chunk, device="cuda")[:, None].expand_as(v)
            cat_v = torch.cat((best_v[l], v))
            topi = cat_v.topk(k, 0).indices
            best_v[l] = cat_v.gather(0, topi)
            best_i[l] = torch.cat((best_i[l], wid)).gather(0, topi)
            best_p[l] = torch.cat((best_p[l], pos - (~valid).sum(1)[:, None])).gather(0, topi)
        zero.acts = None
        if start % (args.batch * 200) == 0:
            print(start, len(order), flush=True)
    (SITE / "f").mkdir(parents=True, exist_ok=True)
    wanted = {}
    for l, fs in by_layer.items():
        for j, f in enumerate(fs):
            for r in range(k):
                wanted.setdefault(int(best_i[l][r, j]), []).append((l, f, j, r))
    snippets = {}
    win_list = sorted(wanted)
    for start in range(0, len(win_list), args.batch):
        chunk = win_list[start : start + args.batch]
        ids, mask = m.batch_inputs([windows[i]["ids"] for i in chunk])
        zero.acts = {}
        m.lm.model(input_ids=ids, attention_mask=mask)
        pad = (~mask.bool()).sum(1)
        for b, wid in enumerate(chunk):
            for l, f, j, r in wanted[wid]:
                pos = int(best_p[l][r, j])
                lo, hi = max(0, pos - 16), min(len(windows[wid]["ids"]), pos + 5)
                acts = zero.acts[l][b, pad[b] + lo : pad[b] + hi, f].float().cpu().numpy()
                snippets[(l, f, r)] = (wid, lo, hi, acts, float(best_v[l][r, j]))
        zero.acts = None
    for l, fs in by_layer.items():
        for f in fs:
            entries = []
            peak = max(snippets[(l, f, r)][4] for r in range(k))
            for r in range(k):
                wid, lo, hi, acts, v = snippets[(l, f, r)]
                w = windows[wid]
                entries.append({"tokens": [tok.decode([i]) for i in w["ids"][lo:hi]],
                                "acts": [int(max(0.0, min(255.0, a / peak * 255))) for a in acts],
                                "max": round(v, 3), "title": w["title"], "url": w["url"]})
            (SITE / f"f/{l}_{f}.json").write_text(json.dumps({"id": f"{l}_{f}", "top": entries}, separators=(",", ":")))
    print("cards", len(needed), "windows", len(win_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["targets", "features"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--windows", type=int, default=120_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    {"targets": stage_targets, "features": stage_features}[args.stage](args)


if __name__ == "__main__":
    main()
