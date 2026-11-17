"""Add set interventions and atlas weights to the language demo passages.

For every target token and budget, the top-ranked features' activations
(down_proj inputs) are multiplied by each factor in FACTORS, and the target
token's probability is recorded. Atlas neighbours are the sources with the
largest readout weight phi(z*) . phi(z_i), the weight the atlas itself applies
to source i's profile.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from kpsa.adoption.language import OUT, Models, responses
from kpsa.adoption.language_demo import BUDGETS, NF, SITE, Zero, choose, trim

FACTORS = (0.5, 0.0, 3.0, 10.0)
NEIGHBORS = 5


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    cand, scale, _, _ = choose()
    assert cand == "own_tgt"
    m = Models(embed=False)
    tok = m.tok
    zero = Zero(m.mlps)
    pmap = torch.load(OUT / "prototype_maps.pt")[cand]
    src = torch.load(OUT / "source_reps.pt")[cand]
    phi_src = torch.cat([responses(src[i : i + 50_000].cuda().float(), pmap, scale) for i in range(0, len(src), 50_000)])
    del src
    sources = json.loads((OUT / "sources.json").read_text())
    passages = json.loads((OUT / "demo_passages.json").read_text())[: args.limit]
    gen = torch.Generator(device="cuda").manual_seed(0)
    per = len(BUDGETS) * len(FACTORS)
    for n, item in enumerate(passages):
        path = SITE / f"p/p{n:04d}.json"
        data = json.loads(path.read_text())
        ids = trim(item["ids"], tok)
        assert len(ids) == len(data["tokens"])
        x = torch.tensor(ids, device="cuda")[None]
        zero.masks = None
        hid = m.lm.model(input_ids=x).last_hidden_state[0]
        tgt_ids = torch.tensor(ids[1:], device="cuda")
        own = F.normalize(hid[:-1].float(), dim=-1)
        z = torch.cat((own, F.normalize(m.lm.model.embed_tokens.weight[tgt_ids].float(), dim=-1)), -1) / 2**0.5
        phi = responses(z, pmap, scale)
        n_t = len(ids) - 1
        # the stored ranking (language_demo targets), so effects match the listed features
        ranked = torch.tensor([[a * NF + b for a, b, _ in data["targets"][str(ti + 1)]["features"]] for ti in range(n_t)], device="cuda")
        rows = n_t * per
        p_after = torch.empty(rows, device="cuda")
        for start in range(0, rows, args.rows):
            stop = min(rows, start + args.rows)
            masks = torch.ones(len(m.mlps), stop - start, NF, device="cuda", dtype=torch.bfloat16)
            for r in range(start, stop):
                ti, kind = divmod(r, per)
                j, a = divmod(kind, len(FACTORS))
                chosen = ranked[ti, : BUDGETS[j][1]]
                masks[chosen // NF, r - start, chosen % NF] = FACTORS[a]
            zero.masks = masks
            h = m.lm.model(input_ids=x.expand(stop - start, -1)).last_hidden_state
            zero.masks = None
            t_rows = torch.arange(start, stop, device="cuda") // per
            logits = m.lm.lm_head(h[torch.arange(stop - start, device="cuda"), t_rows]).float()
            p_after[start:stop] = logits.softmax(-1).gather(1, tgt_ids[t_rows][:, None])[:, 0]
        p_after = p_after.reshape(n_t, len(BUDGETS), len(FACTORS)).cpu().tolist()
        w = phi @ phi_src.T
        w = w / w.sum(1, keepdim=True)
        half = ((w.sort(1, descending=True).values.cumsum(1) < 0.5).sum(1) + 1).tolist()
        order = torch.argsort(w + torch.rand(w.shape, generator=gen, device="cuda") * 1e-12, dim=1, descending=True)[:, :NEIGHBORS]
        for ti in range(n_t):
            target = data["targets"][str(ti + 1)]
            target["sets"] = [[float(f"{v:.3g}") for v in row] for row in p_after[ti]]
            target.pop("setEffect", None)
            neigh = []
            for i in order[ti].tolist():
                s = sources[i]
                ctx = s["ids"][max(0, s["t"] - 20) : s["t"]]
                neigh.append({"text": ("…" if s["t"] > 20 else "") + tok.decode(ctx),
                              "target": tok.decode(s["ids"][s["t"] : s["t"] + 1]), "w": float(f"{float(w[ti, i]):.3g}")})
            target["neighbors"] = neigh
            target["neighborsHalf"] = half[ti]
        path.write_text(json.dumps(data, separators=(",", ":")))
        if n % 50 == 0:
            print(n, item["title"], p_after[0], half[:5], flush=True)
    index = json.loads((SITE / "index.json").read_text())
    index["factors"] = list(FACTORS)
    (SITE / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print("done", flush=True)


if __name__ == "__main__":
    main()
