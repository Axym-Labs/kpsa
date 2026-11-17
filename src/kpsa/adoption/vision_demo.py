"""Precompute the vision half of the KPSA demo (ViT-B/16 AugReg, DINOv2-S query space).

Stages:
  atlas     prototype map on DINOv2-S, atlas from the 25k ImageNet-train
            source profiles, bandwidth chosen on 1k held-out validation images
  score     classify gallery candidates (for selection)
  queries   per gallery image: prediction, KPSA ranking, activation maps,
            exact intervention effects, layer mass, atlas neighbours
  features  feature cards: top ImageNet-validation images per feature with maps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kpsa.adoption.vision import balanced_indices

ROOT = Path.home() / "main/data/kpsa-demo"
PROFILES = ROOT / "profiles"
WORK = ROOT / "vision_demo"
SITE = Path.home() / "main/pinned/axym-website/public/demo/kpsa/vision"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
BUDGETS = [(0.02, 7), (0.05, 18), (0.1, 37)]
SCALES = (0.05, 0.1, 0.25)
# Interventions on the top-ranked set: multiply the features' activations (fc2 inputs).
FACTORS = (0.5, 0.0, 3.0, 10.0)
NEIGHBORS = 6
CROP = 224 / 248


def labels():
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    return [info.index_to_description(i).split(",")[0] for i in range(1000)]


def center_square(image: Image.Image, size: int) -> Image.Image:
    """The region the classifier sees (resize 248, center crop 224), at `size` px."""
    w, h = image.size
    side = min(w, h) * CROP
    left, top = (w - side) / 2, (h - side) / 2
    return image.resize((size, size), Image.BICUBIC, box=(left, top, left + side, top + side))


def save_jpg(image, path, size, quality=82):
    path.parent.mkdir(parents=True, exist_ok=True)
    center_square(image, size).save(path, quality=quality, optimize=True)


class Encoder:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

    @torch.no_grad()
    def __call__(self, images, batch=128):
        out = []
        for i in range(0, len(images), batch):
            chunk = [im if isinstance(im, Image.Image) else Image.open(im).convert("RGB") for im in images[i : i + batch]]
            px = self.proc(images=chunk, return_tensors="pt")["pixel_values"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.model(pixel_values=px)
            out.append(F.normalize(o.pooler_output.float(), dim=1))
        return torch.cat(out)


def responses(z, protos, bw2):
    log_r = -torch.cdist(z, protos).square() / (2 * bw2)
    return F.normalize(torch.exp(log_r - log_r.max(1, keepdim=True).values), dim=1)


def stage_atlas(args):
    from sklearn.cluster import KMeans

    from kpsa.adoption import vision_profiles as vp

    WORK.mkdir(parents=True, exist_ok=True)
    enc = Encoder()
    meta = json.loads((PROFILES / "demo_atlas_vision/meta.json").read_text())
    src_paths = meta["files"]
    src = enc(src_paths)
    construction = enc([str(p) for p, _ in balanced_indices("train", range(25, 30))])
    km = KMeans(n_clusters=1000, init="k-means++", n_init=1, max_iter=100, random_state=0).fit(construction.cpu().numpy())
    protos = F.normalize(torch.from_numpy(km.cluster_centers_).float().cuda(), dim=1)
    d = torch.pdist(construction[:4000]).square()
    median = float(d[d > 0].median())
    if not (PROFILES / "demo_heldout_vision/profiles.npy").exists():
        vp.run("augreg2", "validation", range(49, 50), 64, "demo_heldout_vision", False, 37, 16)
    held_meta = json.loads((PROFILES / "demo_heldout_vision/meta.json").read_text())
    held = enc(held_meta["files"])
    exact = torch.from_numpy(np.load(PROFILES / "demo_heldout_vision/profiles.npy")).cuda().float()
    profiles = np.load(PROFILES / "demo_atlas_vision/profiles.npy", mmap_mode="r")
    k = 37
    best = exact.topk(k, 1).values.sum(1)

    def capture(scores):
        return (exact.gather(1, scores.topk(k, 1).indices).sum(1) / best).mean().item()

    constant = torch.zeros(exact.shape[1], device="cuda")
    atlases = {}
    for s in SCALES:
        phi = responses(src, protos, s**2 * median)
        atlas = torch.zeros(exact.shape[1], 1000, device="cuda")
        for i in range(0, len(src), 2000):
            p = torch.from_numpy(np.asarray(profiles[i : i + 2000])).cuda().float()
            atlas.addmm_(p.T, phi[i : i + 2000])
            if s == SCALES[0]:
                constant += p.sum(0)
        atlases[s] = atlas / len(src)
    results = {"constant": capture(constant.expand_as(exact))}
    for s, atlas in atlases.items():
        results[s] = capture(responses(held, protos, s**2 * median) @ atlas.T)
    print(results)
    scale = max(SCALES, key=lambda s: results[s])
    torch.save({"atlas": atlases[scale].half().cpu(), "protos": protos.cpu(), "bw2": scale**2 * median,
                "src": src.half().cpu(), "src_files": src_paths, "src_labels": [int(Path(p).parent.name) for p in src_paths],
                "eval": {str(k_): v for k_, v in results.items()}, "scale": scale}, WORK / "atlas.pt")


def load_classifier():
    import timm

    model = timm.create_model(MODEL, pretrained=True).cuda().eval()
    cfg = timm.data.resolve_model_data_config(model)
    return model, timm.data.create_transform(**cfg, is_training=False)


class Ablate:
    """Capture fc2 inputs and optionally zero features per batch row."""

    def __init__(self, model):
        self.masks = None
        self.acts = {}
        self.handles = [b.mlp.fc2.register_forward_pre_hook(self._hook(i)) for i, b in enumerate(model.blocks)]

    def _hook(self, layer):
        def hook(module, args):
            x = args[0]
            if self.masks is not None and layer in self.masks:  # per-row multipliers
                x = x * self.masks[layer][:, None, :]
            self.acts[layer] = x.detach()
            return (x,)

        return hook


def stage_score(args):
    model, transform = load_classifier()
    names = labels()
    cands = json.loads((ROOT / "gallery/candidates.json").read_text())
    for c in cands:
        x = transform(Image.open(c["path"]).convert("RGB"))[None].cuda()
        with torch.no_grad():
            p = model(x).softmax(-1)[0]
        top = p.topk(3)
        c["pred"] = [(names[i], round(float(v), 3)) for v, i in zip(top.values, top.indices)]
    (ROOT / "gallery/candidates_scored.json").write_text(json.dumps(cands, indent=1))
    for c in cands:
        print(c["group"], "|", Path(c["path"]).name, "|", c["pred"][0], c["license"])


@torch.no_grad()
def stage_queries(args):
    model, transform = load_classifier()
    ablate = Ablate(model)
    names = labels()
    atlas = torch.load(WORK / "atlas.pt")
    A = atlas["atlas"].cuda().float()
    protos, bw2 = atlas["protos"].cuda(), atlas["bw2"]
    src = atlas["src"].cuda().float()
    phi_src = responses(src, protos, bw2)
    gen = torch.Generator(device="cuda").manual_seed(0)
    enc = Encoder()
    gallery = json.loads((ROOT / "gallery/gallery.json").read_text())
    index = {"model": "ViT-B/16 (AugReg, ImageNet-1k)", "queryModel": "DINOv2-S image embedding (self-supervised)", "layers": 12,
             "featuresPerLayer": 3072, "atlasSources": len(src),
             "budgets": [{"pct": p, "k": k} for p, k in BUDGETS], "factors": list(FACTORS),
             "groups": list(dict.fromkeys(g["group"] for g in gallery)), "images": []}
    needed = set()
    for g in gallery:
        image = Image.open(g["path"]).convert("RGB")
        gid = g["id"]
        save_jpg(image, SITE / f"img/{gid}.jpg", 448)
        save_jpg(image, SITE / f"img/{gid}_thumb.jpg", 160)
        index["images"].append({"id": gid, "group": g["group"], "src": f"vision/img/{gid}.jpg",
                                "thumb": f"vision/img/{gid}_thumb.jpg",
                                "credit": {"author": g["author"], "license": g["license"], "url": g["page"]}})
        x = transform(image)[None].cuda()
        ablate.masks = None
        probs = model(x).softmax(-1)[0]
        c = int(probs.argmax())
        acts = {l: a[0, 1:].float() for l, a in ablate.acts.items()}
        z = enc([image])
        scores = (responses(z, protos, bw2) @ A.T)[0]
        scores = scores / scores.sum()
        top = scores.topk(37)
        feats = [(int(i) // 3072, int(i) % 3072, float(v)) for v, i in zip(top.values, top.indices)]
        n_sets = len(BUDGETS) * len(FACTORS)
        rows = len(feats) + n_sets
        masks = {l: torch.ones(rows, 3072, device="cuda") for l in range(12)}
        for r, (l, f, _) in enumerate(feats):
            masks[l][r, f] = 0
        for j, (_, k) in enumerate(BUDGETS):
            for a, factor in enumerate(FACTORS):
                for l, f, _ in feats[:k]:
                    masks[l][len(feats) + j * len(FACTORS) + a, f] = factor
        ablate.masks = masks
        p_ab = model(x.expand(rows, -1, -1, -1)).softmax(-1)[:, c]
        ablate.masks = None
        out_feats = []
        for r, (l, f, s) in enumerate(feats):
            m = acts[l][:, f].clamp_min(0)
            m = (m / m.max() * 255).round() if m.max() > 0 else m * 0
            out_feats.append({"id": f"{l}_{f}", "l": l, "f": f, "s": round(s, 6),
                              "map": [int(v) for v in m.tolist()], "deact": {"p": round(float(p_ab[r]), 4)}})
            needed.add((l, f))
        mass = [0.0] * 12
        for l, _, s in feats:
            mass[l] += s
        tot = sum(mass)
        w = (responses(z, protos, bw2) @ phi_src.T)[0]
        w = w / w.sum()
        sorted_w = w.sort(descending=True).values
        half = int((sorted_w.cumsum(0) < 0.5).sum()) + 1
        # highest weights first, ties (sources sharing one prototype) in a fixed random order
        order = torch.argsort(w + torch.rand(w.shape, generator=gen, device="cuda") * 1e-9, descending=True)[:NEIGHBORS]
        neighbors = []
        for i in order.tolist():
            save_jpg(Image.open(atlas["src_files"][i]).convert("RGB"), SITE / f"src/{i}.jpg", 128)
            neighbors.append({"thumb": f"vision/src/{i}.jpg", "label": names[atlas["src_labels"][i]], "w": round(float(w[i]), 5)})
        top5 = probs.topk(5)
        q = {"id": gid,
             "prediction": {"label": names[c], "p": round(float(probs[c]), 4),
                            "top5": [[names[int(i)], round(float(v), 4)] for v, i in zip(top5.values, top5.indices)]},
             "features": out_feats,
             "sets": [[round(float(p_ab[len(feats) + j * len(FACTORS) + a]), 4) for a in range(len(FACTORS))] for j in range(len(BUDGETS))],
             "layerMass": [round(m / tot, 4) for m in mass],
             "neighbors": neighbors, "neighborsHalf": half}
        (SITE / "q").mkdir(parents=True, exist_ok=True)
        (SITE / f"q/{gid}.json").write_text(json.dumps(q, separators=(",", ":")))
        print(gid, names[c], round(float(probs[c]), 3), "sets", q["sets"], "half", half, flush=True)
    (SITE / "index.json").write_text(json.dumps(index, indent=1))
    torch.save(sorted(needed), WORK / "needed.pt")


@torch.no_grad()
def stage_features(args):
    model, transform = load_classifier()
    ablate = Ablate(model)
    names = labels()
    needed = torch.load(WORK / "needed.pt")
    by_layer = {}
    for l, f in needed:
        by_layer.setdefault(l, []).append(f)
    items = balanced_indices("validation", range(50))
    k = 8
    best_v = {l: torch.full((k, len(fs)), -1e9, device="cuda") for l, fs in by_layer.items()}
    best_i = {l: torch.zeros((k, len(fs)), dtype=torch.long, device="cuda") for l, fs in by_layer.items()}
    from kpsa.adoption.vision import SupervisedSet

    loader = torch.utils.data.DataLoader(SupervisedSet(items, transform), batch_size=128, num_workers=16)
    offset = 0
    for x, _ in loader:
        model(x.cuda())
        for l, fs in by_layer.items():
            a = ablate.acts[l][:, 1:, fs].float().amax(1)
            ids = torch.arange(offset, offset + len(x), device="cuda")[:, None].expand_as(a)
            v = torch.cat((best_v[l], a))
            i = torch.cat((best_i[l], ids))
            top = v.topk(k, 0)
            best_v[l], best_i[l] = top.values, i.gather(0, top.indices)
        offset += len(x)
    cards = {}
    images = set()
    for l, fs in by_layer.items():
        for j, f in enumerate(fs):
            cards[(l, f)] = [(int(best_i[l][r, j]), float(best_v[l][r, j])) for r in range(k)]
            images.update(i for i, _ in cards[(l, f)])
    maps = {}
    image_list = sorted(images)
    for start in range(0, len(image_list), 128):
        chunk = image_list[start : start + 128]
        pil = [Image.open(items[i][0]).convert("RGB") for i in chunk]
        model(torch.stack([transform(p) for p in pil]).cuda())
        for r, i in enumerate(chunk):
            save_jpg(pil[r], SITE / f"fimg/{i}.jpg", 128, quality=78)
            for l in by_layer:
                maps[(i, l)] = ablate.acts[l][r, 1:].half().cpu()
    (SITE / "f").mkdir(parents=True, exist_ok=True)
    for (l, f), tops in cards.items():
        peak = max(v for _, v in tops)
        entries = []
        for i, v in tops:
            m = maps[(i, l)][:, f].float().clamp_min(0)
            m = (m / peak * 255).clamp(0, 255).round()
            entries.append({"thumb": f"vision/fimg/{i}.jpg", "label": names[items[i][1]], "act": round(v, 3),
                            "map": [int(t) for t in m.tolist()]})
        (SITE / f"f/{l}_{f}.json").write_text(json.dumps({"id": f"{l}_{f}", "top": entries}, separators=(",", ":")))
    print("cards", len(cards), "images", len(image_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["atlas", "score", "queries", "features"])
    args = parser.parse_args()
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = True
    {"atlas": stage_atlas, "score": stage_score, "queries": stage_queries, "features": stage_features}[args.stage](args)


if __name__ == "__main__":
    main()
