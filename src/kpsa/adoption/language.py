"""KPSA atlas for Qwen3-0.6B-Base under its next-token NLL.

Stages (run in order):
  corpus   Wikipedia paragraphs -> source, construction, held-out and demo sets
  check    batched Gram energies (bf16 model) vs fp32 single-sample .grad
  atlas    prototypes per query-space candidate, then one pass over the
           sources that accumulates the atlas for every candidate x bandwidth
  eval     held-out energy capture at the 0.1 % budget for each atlas

A source is (paragraph prefix tokens[:t], target tokens[t]). Query-space
candidates: "own" (the model's final hidden state at t-1), "own_tgt" (own
concatenated with the target token's input embedding), "emb"
(Qwen3-Embedding-0.6B of the decoded prefix plus target word).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from kpsa.adoption.mlp_energy import MLPEnergyCapture, direct_energy

ROOT = Path.home() / "main/data/kpsa-demo"
OUT = ROOT / "language"
WIKI = ROOT / "wiki/20231101.en"
MODEL = "Qwen/Qwen3-0.6B-Base"
EMBED = "Qwen/Qwen3-Embedding-0.6B"
MAX_TOKENS = 128
CANDIDATES = ("own", "own_tgt", "emb")
SCALES = (0.05, 0.1, 0.25)
PROTOTYPES = 1000


def shard(i):
    return WIKI / f"train-{i:05d}-of-00041.parquet"


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


def paragraphs(shard_index, tok, min_tokens, limit, seed, one_per_article=False, skip_articles=0):
    """Prose paragraphs (no headings or lists), truncated to MAX_TOKENS tokens."""
    table = pq.read_table(shard(shard_index), columns=["title", "url", "text"])
    order = list(range(table.num_rows))
    random.Random(seed).shuffle(order)
    order = order[skip_articles:]
    titles, urls, texts = (table.column(c) for c in ("title", "url", "text"))
    out = []
    for row in order:
        text = texts[row].as_py()
        paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 200 and p.strip()[-1] in ".!?\"')"]
        if one_per_article:
            paras = paras[:1]
        batch = tok(paras, add_special_tokens=False)["input_ids"] if paras else []
        for para, ids in zip(paras, batch):
            if len(ids) < min_tokens:
                continue
            ids = ids[:MAX_TOKENS]
            out.append({"title": titles[row].as_py(), "url": urls[row].as_py(), "ids": ids})
            if len(out) >= limit:
                return out
    return out


def stage_corpus(args):
    tok = tokenizer()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    sources = paragraphs(0, tok, 24, args.sources + args.construction + args.heldout, seed=1)
    rng.shuffle(sources)
    sets = {
        "sources": sources[: args.sources],
        "construction": sources[args.sources : args.sources + args.construction],
        "heldout": sources[args.sources + args.construction :],
    }
    for name, items in sets.items():
        for item in items:
            item["t"] = rng.randint(4, len(item["ids"]) - 1)
        (OUT / f"{name}.json").write_text(json.dumps(items))
        print(name, len(items))
    demo = paragraphs(2, tok, 60, args.demo, seed=2, one_per_article=True)
    (OUT / "demo_passages.json").write_text(json.dumps(demo))
    print("demo", len(demo))


class Models:
    def __init__(self, embed=True):
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.tok = tokenizer()
        self.lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.mlps = [layer.mlp for layer in self.lm.model.layers]
        for mlp in self.mlps:
            for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                lin.weight.requires_grad_(True)
        if embed:
            self.etok = AutoTokenizer.from_pretrained(EMBED, padding_side="left")
            self.emb = AutoModel.from_pretrained(EMBED, dtype=torch.bfloat16).cuda().eval()

    def batch_inputs(self, contexts):
        width = max(len(c) for c in contexts)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(contexts), width), pad, dtype=torch.long)
        mask = torch.zeros((len(contexts), width), dtype=torch.long)
        for i, c in enumerate(contexts):
            ids[i, width - len(c) :] = torch.tensor(c)
            mask[i, width - len(c) :] = 1
        return ids.cuda(), mask.cuda()

    def loss_and_reps(self, contexts, targets, grad=True):
        ids, mask = self.batch_inputs(contexts)
        target = torch.tensor(targets, device="cuda")
        with torch.set_grad_enabled(grad):
            hidden = self.lm.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1]
            logits = self.lm.lm_head(hidden).float()
            loss = F.cross_entropy(logits, target, reduction="none")
        own = F.normalize(hidden.detach().float(), dim=-1)
        tgt = F.normalize(self.lm.model.embed_tokens.weight[target].float(), dim=-1)
        reps = {"own": own, "own_tgt": torch.cat((own, tgt), -1) / math.sqrt(2)}
        return loss, logits.detach(), reps

    @torch.no_grad()
    def embed_texts(self, texts):
        batch = self.etok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        out = self.emb(**batch).last_hidden_state[:, -1]
        return F.normalize(out.float(), dim=-1)

    def texts(self, items):
        return [self.tok.decode(it["ids"][: it["t"] + 1]) for it in items]


def batches(items, size, sort=True):
    order = sorted(range(len(items)), key=lambda i: items[i]["t"]) if sort else list(range(len(items)))
    for start in range(0, len(order), size):
        yield [items[i] for i in order[start : start + size]], order[start : start + size]


def stage_check(args):
    m = Models(embed=False)
    items = json.loads((OUT / "heldout.json").read_text())[:3]
    capture = MLPEnergyCapture(m.mlps, "swiglu")
    m.lm.zero_grad(set_to_none=True)
    loss, _, _ = m.loss_and_reps([it["ids"][: it["t"]] for it in items], [it["ids"][it["t"]] for it in items])
    loss.sum().backward()
    got = capture.compute([1])["energy"]
    capture.enabled = False
    m.lm.float()
    for i, it in enumerate(items):
        m.lm.zero_grad(set_to_none=True)
        loss, _, _ = m.loss_and_reps([it["ids"][: it["t"]]], [it["ids"][it["t"]]])
        loss.sum().backward()
        ref = direct_energy(m.mlps, "swiglu")
        g = got[i]
        corr = torch.corrcoef(torch.stack((g.flatten(), ref.flatten())))[0, 1].item()
        k = 86
        top_ref = ref.flatten().topk(k).indices
        top_got = g.flatten().topk(k).indices
        overlap = len(set(top_ref.tolist()) & set(top_got.tolist())) / k
        rel = ((g.sum() - ref.sum()) / ref.sum()).item()
        print(f"sample {i}: loss {loss.item():.3f} corr {corr:.5f} top-86 overlap {overlap:.3f} total rel {rel:.3e}")


def reps_for(m, items, size=64):
    out = {c: [] for c in CANDIDATES}
    for chunk, _ in batches(items, size, sort=False):
        _, _, reps = m.loss_and_reps([it["ids"][: it["t"]] for it in chunk], [it["ids"][it["t"]] for it in chunk], grad=False)
        for c in ("own", "own_tgt"):
            out[c].append(reps[c].cpu())
        out["emb"].append(m.embed_texts(m.texts(chunk)).cpu())
    return {c: torch.cat(v) for c, v in out.items()}


def kmeans_gpu(x, k, iters=40, seed=0):
    """Lloyd's k-means on the GPU with k-means++ seeding (unit-norm data)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = x.cuda()
    first = int(torch.randint(len(x), (1,), generator=g))
    centers = [x[first]]
    d2 = (x - x[first]).square().sum(1)
    for _ in range(1, k):
        idx = int(torch.multinomial((d2 / d2.sum()).cpu(), 1, generator=g))
        centers.append(x[idx])
        d2 = torch.minimum(d2, (x - x[idx]).square().sum(1))
    centers = torch.stack(centers)
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).float()[:, None]
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
    return centers.cpu()


def fit_prototypes(reps):
    maps = {}
    for c, values in reps.items():
        protos = F.normalize(kmeans_gpu(values.float(), PROTOTYPES), dim=-1)
        sub = values[torch.randperm(len(values), generator=torch.Generator().manual_seed(0))[:4000]].float()
        d = torch.pdist(sub).square()
        maps[c] = {"protos": protos, "median": float(d[d > 0].median())}
        print(c, "median sq dist", maps[c]["median"], flush=True)
    return maps


def responses(z, pmap, scale):
    """Prototype RBF responses, L2-normalised per row as in the paper's map."""
    log_r = -torch.cdist(z, pmap["protos"].to(z.device)).square() / (2 * scale**2 * pmap["median"])
    return F.normalize(torch.exp(log_r - log_r.max(dim=1, keepdim=True).values), dim=1)


def profile_pass(m, capture, items, size, pmaps, atlas=None, keep_profiles=False, keep_reps=False):
    n_feat = len(m.mlps) * capture.n_features
    profiles = np.zeros((len(items), n_feat), np.float16) if keep_profiles else None
    total_profile = torch.zeros(n_feat, device="cuda", dtype=torch.float64)
    kept = {c: torch.zeros(len(items), 1024 if c != "own_tgt" else 2048, dtype=torch.float16) for c in CANDIDATES} if keep_reps else None
    losses = np.zeros(len(items), np.float32)
    started, done = time.time(), 0
    for chunk, idx in batches(items, size):
        m.lm.zero_grad(set_to_none=True)
        loss, _, reps = m.loss_and_reps([it["ids"][: it["t"]] for it in chunk], [it["ids"][it["t"]] for it in chunk])
        loss.sum().backward()
        energy = capture.compute([1])["energy"].flatten(1)
        prof = energy / energy.sum(1, keepdim=True)
        reps["emb"] = m.embed_texts(m.texts(chunk))
        total_profile += prof.sum(0).double()
        if atlas is not None:
            for c in CANDIDATES:
                for s in SCALES:
                    atlas[(c, s)].addmm_(prof.T, responses(reps[c], pmaps[c], s))
        idx_t = torch.tensor(idx)
        if keep_profiles:
            profiles[idx] = prof.cpu().numpy().astype(np.float16)
        if keep_reps:
            for c in CANDIDATES:
                kept[c][idx_t] = reps[c].cpu().half()
        losses[idx] = loss.detach().cpu().numpy()
        done += len(chunk)
        if done % (size * 100) < size:
            rate = done / (time.time() - started)
            print(f"{done}/{len(items)} {rate:.1f}/s eta {(len(items) - done) / rate / 60:.1f} min", flush=True)
    return profiles, total_profile, kept, losses


def stage_atlas(args):
    m = Models()
    capture = MLPEnergyCapture(m.mlps, "swiglu")
    capture.enabled = False
    construction = json.loads((OUT / "construction.json").read_text())
    if (OUT / "prototype_maps.pt").exists():
        pmaps = torch.load(OUT / "prototype_maps.pt")
    else:
        print("construction reps", flush=True)
        pmaps = fit_prototypes(reps_for(m, construction))
        torch.save(pmaps, OUT / "prototype_maps.pt")
    capture.enabled = True
    n_feat = len(m.mlps) * capture.n_features
    atlas = {(c, s): torch.zeros(n_feat, PROTOTYPES, device="cuda") for c in CANDIDATES for s in SCALES}
    heldout = json.loads((OUT / "heldout.json").read_text())
    print("heldout profiles", flush=True)
    profiles, _, reps, losses = profile_pass(m, capture, heldout, args.batch, pmaps, keep_profiles=True, keep_reps=True)
    np.save(OUT / "heldout_profiles.npy", profiles)
    torch.save(reps, OUT / "heldout_reps.pt")
    sources = json.loads((OUT / "sources.json").read_text())[: args.limit or None]
    print("sources", len(sources), flush=True)
    _, total, reps, losses = profile_pass(m, capture, sources, args.batch, pmaps, atlas=atlas, keep_reps=True)
    n = len(sources)
    torch.save({f"{c}|{s}": (a / n).cpu() for (c, s), a in atlas.items()}, OUT / "atlases.pt")
    torch.save((total / n).float().cpu(), OUT / "constant_profile.pt")
    torch.save(reps, OUT / "source_reps.pt")
    np.save(OUT / "source_losses.npy", losses)
    print("atlas done", flush=True)


def stage_eval(args):
    pmaps = torch.load(OUT / "prototype_maps.pt")
    atlases = torch.load(OUT / "atlases.pt")
    constant = torch.load(OUT / "constant_profile.pt").cuda()
    exact = torch.from_numpy(np.load(OUT / "heldout_profiles.npy")).cuda().float()
    reps = torch.load(OUT / "heldout_reps.pt")
    k = 86
    best = exact.topk(k, dim=1).values.sum(1)

    def capture_of(scores):
        top = scores.topk(k, dim=1).indices
        return (exact.gather(1, top).sum(1) / best).mean().item()

    results = {"constant": capture_of(constant.expand_as(exact))}
    for key, atlas in atlases.items():
        c, s = key.split("|")
        phi = responses(reps[c].cuda().float(), pmaps[c], float(s))
        results[key] = capture_of(phi @ atlas.cuda().T)
    for key, value in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{key:20s} {value:.4f}")
    (OUT / "eval.json").write_text(json.dumps(results, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["corpus", "check", "atlas", "eval"])
    parser.add_argument("--sources", type=int, default=300_000)
    parser.add_argument("--construction", type=int, default=20_000)
    parser.add_argument("--heldout", type=int, default=3_000)
    parser.add_argument("--demo", type=int, default=1_500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    {"corpus": stage_corpus, "check": stage_check, "atlas": stage_atlas, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    main()
