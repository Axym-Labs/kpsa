"""Recompute atlas-neighbour weights of the language demo in the log domain.

The first pass divided underflowed kernel values (0/0) for queries far from
all sources; here the weights are a softmax of -d^2 / (2 h^2).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from kpsa.adoption.language import OUT, Models
from kpsa.adoption.language_demo import SITE, choose, trim


@torch.no_grad()
def main():
    cand, scale, _, _ = choose()
    assert cand == "own_tgt"
    m = Models(embed=False)
    tok = m.tok
    pmaps = torch.load(OUT / "prototype_maps.pt")
    bw2 = scale**2 * pmaps[cand]["median"]
    src = torch.load(OUT / "source_reps.pt")[cand].cuda().float()
    sources = json.loads((OUT / "sources.json").read_text())
    passages = json.loads((OUT / "demo_passages.json").read_text())
    for n, item in enumerate(passages[:600]):
        path = SITE / f"p/p{n:04d}.json"
        data = json.loads(path.read_text())
        ids = trim(item["ids"], tok)
        assert len(ids) == len(data["tokens"])
        x = torch.tensor(ids, device="cuda")[None]
        hid = m.lm.model(input_ids=x).last_hidden_state[0]
        tgt = torch.tensor(ids[1:], device="cuda")
        own = F.normalize(hid[:-1].float(), dim=-1)
        z = torch.cat((own, F.normalize(m.lm.model.embed_tokens.weight[tgt].float(), dim=-1)), -1) / 2**0.5
        logw = -torch.cdist(z, src).square() / (2 * bw2)
        w = logw.softmax(1)
        nb = w.topk(3, dim=1)
        for ti in range(len(ids) - 1):
            neigh = []
            for v, i in zip(nb.values[ti], nb.indices[ti]):
                s = sources[int(i)]
                ctx = s["ids"][max(0, s["t"] - 20) : s["t"]]
                neigh.append({"text": ("…" if s["t"] > 20 else "") + tok.decode(ctx),
                              "target": tok.decode(s["ids"][s["t"] : s["t"] + 1]), "w": round(float(v), 4)})
            data["targets"][str(ti + 1)]["neighbors"] = neigh
        path.write_text(json.dumps(data, separators=(",", ":")))
    print("done")


if __name__ == "__main__":
    main()
