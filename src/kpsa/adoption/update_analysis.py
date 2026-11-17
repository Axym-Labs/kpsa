"""Research update: ImageNet classes and parameter-group sensitivity.

Reads the 50-images-per-class validation profiles of the supervised ViT-B/16
(update_sup) and DINO ViT-B/16 (update_dino) and computes the statistics of
the five questions. Writes stats.json; figures are drawn by update_figures.py.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROFILES = Path.home() / "main/data/kpsa-demo/profiles"
OUT = Path.home() / "main/workspace/kpsa-paper-internal/05_adoption_demo/artifacts/update"
MODELS = {"supervised": "update_sup", "dino": "update_dino"}
L, F = 12, 3072
LEVELS = (250, 50, 10)  # maximum supergroup size
SEED = 0


def wordnet_paths():
    os.environ.setdefault("NLTK_DATA", str(Path.home() / "main/data/kpsa-demo/nltk_data"))
    from nltk.corpus import wordnet as wn
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    syns = [wn.synset_from_pos_and_offset("n", int(info.index_to_label_name(i)[1:])) for i in range(1000)]
    names = [info.index_to_description(i).split(",")[0] for i in range(1000)]
    return [s.hypernym_paths()[0] for s in syns], names


def groups_by_size(paths, max_size):
    """Split the WordNet tree top-down until every group has <= max_size classes."""
    groups = []

    def split(members, depth):
        if len(members) <= max_size:
            groups.append(members)
            return
        children = {}
        for c in members:
            key = paths[c][depth].name() if depth < len(paths[c]) else f"leaf:{c}"
            children.setdefault(key, []).append(c)
        if len(children) == 1:
            split(members, depth + 1)
            return
        for child in children.values():
            split(child, depth + 1)

    split(list(range(1000)), 1)
    label = np.zeros(1000, np.int64)
    for g, members in enumerate(groups):
        label[members] = g
    return label


def lca_depth(paths):
    """Depth of the deepest common WordNet ancestor for every class pair."""
    n = len(paths)
    out = np.zeros((n, n), np.float32)
    names = [[s.name() for s in p] for p in paths]
    for i in range(n):
        for j in range(i, n):
            d = 0
            for a, b in zip(names[i], names[j]):
                if a != b:
                    break
                d += 1
            out[i, j] = out[j, i] = d
    return out


def class_sums(u, y, n_classes=1000):
    s = torch.zeros(n_classes, u.shape[1], device=u.device)
    s.index_add_(0, y, u)
    return s


def stability(u, y, group):
    """Mean pairwise cosine: same class / same supergroup, other class / other supergroup."""
    counts = torch.bincount(y, minlength=1000).float()
    s = class_sums(u, y)
    sq = (s * s).sum(1)
    within_class = ((sq - counts) / (counts * (counts - 1))).cpu().numpy()
    g = torch.as_tensor(group, device=u.device)
    sg = torch.zeros(int(g.max()) + 1, u.shape[1], device=u.device)
    sg.index_add_(0, g, s)
    g_counts = torch.zeros(len(sg), device=u.device).index_add_(0, g, counts)
    total = s.sum(0)
    pairs_same_g = (g_counts**2).sum() - (counts**2).sum()
    same_g = ((sg * sg).sum() - sq.sum()) / pairs_same_g
    n = counts.sum()
    other = ((total * total).sum() - (sg * sg).sum()) / (n**2 - (g_counts**2).sum())
    return float(np.mean(within_class)), within_class, float(same_g), float(other)


def kl_to_prior(q, prior):
    q = q.clamp_min(1e-30)
    return (q * (q / prior[:, None]).log2()).sum(0)


def selectivity_from_sums(m, y, level_labels):
    """Mass-weighted information (bits) a feature's sensitivity carries about classes and supergroups.

    m: (1000, F) class sums of profiles. For each supergroup level, only classes
    whose supergroup has at least two classes enter, and the class-level value
    is recomputed on the same subset, so the two are directly comparable.
    """
    counts = torch.bincount(y, minlength=1000).float()

    def info(q_sub, prior):
        q_sub = q_sub / q_sub.sum(0, keepdim=True).clamp_min(1e-30)
        return q_sub, kl_to_prior(q_sub, prior)

    def weighted(v, mass):
        w = mass / mass.sum()
        return float((w * v).sum())

    mass = m.sum(0)
    _, kl = info(m, counts / counts.sum())
    out = {"class": weighted(kl, mass)}
    for name, lab in level_labels.items():
        sizes = np.bincount(lab)
        keep = torch.as_tensor(sizes[lab] >= 2, device=m.device)
        sub = m[keep]
        c_sub = counts[keep]
        _, kl_c = info(sub, c_sub / c_sub.sum())
        g = torch.as_tensor(lab, device=m.device)[keep]
        _, g = torch.unique(g, return_inverse=True)
        qg = torch.zeros(int(g.max()) + 1, sub.shape[1], device=m.device).index_add_(0, g, sub)
        prior = torch.zeros(len(qg), device=m.device).index_add_(0, g, c_sub)
        _, kl_g = info(qg, prior / prior.sum())
        sub_mass = sub.sum(0)
        out[name] = weighted(kl_g, sub_mass)
        out[name + "_classes"] = weighted(kl_c, sub_mass)
    return out


def entropy_eff(p, dim):
    p = p / p.sum(dim, keepdim=True)
    h = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(dim)
    return h.exp()


def spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def analyse(name, paths, levels, lca):
    root = PROFILES / MODELS[name]
    y = torch.from_numpy(np.load(root / "label.npy").astype(np.int64)).cuda()
    P = torch.from_numpy(np.load(root / "profiles.npy")).cuda()
    top_maps = torch.from_numpy(np.load(root / "top_maps.npy")).cuda()
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    y_shuf = y[torch.randperm(len(y), generator=gen).cuda()]
    res = {"n": len(y)}
    if name == "supervised":
        pred = np.load(root / "pred.npy")
        res["accuracy"] = float((pred == np.load(root / "label.npy")).mean())
    res["loss_mean"] = float(np.load(root / "loss.npy").mean())
    iu = np.triu_indices(1000, 1)
    lca_flat = lca[iu]

    q1 = {"same_class": [], "same_group": [], "other_group": [], "class_spread": []}
    q2, q2_null = [], []
    q4 = []
    layer_mass = torch.zeros(len(y), L, device="cuda")
    for layer in list(range(L)) + ["all"]:
        if layer == "all":
            # chunked: class sums of sqrt(profile) and of profile over all groups
            u_sums = torch.zeros(1000, P.shape[1], device="cuda")
            m = torch.zeros(1000, P.shape[1], device="cuda")
            m_null = torch.zeros(1000, P.shape[1], device="cuda")
            for i in range(0, len(y), 4000):
                chunk = P[i : i + 4000].float()
                u_sums.index_add_(0, y[i : i + 4000], chunk.sqrt())
                m.index_add_(0, y[i : i + 4000], chunk)
                m_null.index_add_(0, y_shuf[i : i + 4000], chunk)
            counts = torch.bincount(y, minlength=1000).float()
            sq = (u_sums * u_sums).sum(1)
            per_class = ((sq - counts) / (counts * (counts - 1))).cpu().numpy()
            g = torch.as_tensor(levels["50"], device="cuda")
            sg = torch.zeros(int(g.max()) + 1, P.shape[1], device="cuda").index_add_(0, g, u_sums)
            g_counts = torch.zeros(len(sg), device="cuda").index_add_(0, g, counts)
            tot = u_sums.sum(0)
            same_g = float(((sg * sg).sum() - sq.sum()) / ((g_counts**2).sum() - (counts**2).sum()))
            other = float(((tot * tot).sum() - (sg * sg).sum()) / (counts.sum() ** 2 - (g_counts**2).sum()))
            same_c = float(np.mean(per_class))
            s = u_sums
            del sg
        else:
            sub = P[:, layer * F : (layer + 1) * F].float()
            layer_mass[:, layer] = sub.sum(1)
            dist = sub / sub.sum(1, keepdim=True).clamp_min(1e-30)
            u = dist.sqrt()
            same_c, per_class, same_g, other = stability(u, y, levels["50"])
            m = class_sums(dist, y)
            m_null = class_sums(dist, y_shuf)
            s = class_sums(u, y)
            del sub, dist, u
        q1["same_class"].append(same_c)
        q1["same_group"].append(same_g)
        q1["other_group"].append(other)
        q1["class_spread"].append([float(np.percentile(per_class, p)) for p in (10, 25, 50, 75, 90)])
        q2.append(selectivity_from_sums(m, y, levels))
        q2_null.append(selectivity_from_sums(m_null, y_shuf, levels))
        s = s / s.norm(dim=1, keepdim=True)
        sim = (s @ s.T).cpu().numpy()
        q4.append(spearman(sim[iu], lca_flat))
        if layer == "all":
            eff_classes = entropy_eff(m, 0)
            fmass = (m.sum(0) / m.sum()).cpu().numpy()
            q5_feature_classes = (eff_classes.cpu().numpy(), fmass)
            m_all = m
        else:
            del m, m_null
        torch.cuda.empty_cache()
    res["q1"] = q1
    res["q2"] = q2
    res["q2_null"] = q2_null
    res["q4_rho"] = q4
    # Q3: depth and space.
    lm = layer_mass / layer_mass.sum(1, keepdim=True)
    depth = (lm * torch.arange(1, L + 1, device="cuda")).sum(1)
    patches = top_maps[:, 1:].clamp_min(0)
    patches = patches / patches.sum(1, keepdim=True).clamp_min(1e-30)
    grid = patches.reshape(-1, 14, 14)
    row = (grid.sum(2) * torch.arange(14, device="cuda")).sum(1)
    col = (grid.sum(1) * torch.arange(14, device="cuda")).sum(1)
    cls_share = top_maps[:, 0] / top_maps.sum(1).clamp_min(1e-30)

    def eta2(v):
        counts = torch.bincount(y, minlength=1000).float()
        means = torch.zeros(1000, device="cuda").index_add_(0, y, v) / counts
        total = ((v - v.mean()) ** 2).sum()
        between = (counts * (means - v.mean()) ** 2).sum()
        null_means = torch.zeros(1000, device="cuda").index_add_(0, y_shuf, v) / counts
        null = (counts * (null_means - v.mean()) ** 2).sum()
        return float(between / total), float(null / total), means.cpu().numpy()

    res["q3"] = {}
    for key, v in {"depth": depth, "row": row, "col": col, "cls_share": cls_share}.items():
        e, e_null, means = eta2(v)
        res["q3"][key] = {"eta2": e, "eta2_null": e_null, "class_means": means.tolist()}
    cm = torch.zeros(1000, L, device="cuda").index_add_(0, y, lm) / 50
    res["q3"]["class_layer_mass"] = cm.cpu().numpy().tolist()
    res["q3"]["class_maps"] = (torch.zeros(1000, 196, device="cuda").index_add_(0, y, patches) / 50).cpu().numpy().tolist()
    # Q5: sparsity.
    top1pct = int(0.01 * P.shape[1])
    eff_img, img_top1 = [], []
    for i in range(0, len(y), 4000):
        chunk = P[i : i + 4000].float()
        eff_img.append(entropy_eff(chunk, 1).cpu().numpy())
        img_top1.append(chunk.topk(top1pct, 1).values.sum(1).cpu().numpy())
    eff_img, img_top1 = np.concatenate(eff_img), np.concatenate(img_top1)
    eff_class = entropy_eff(m_all, 1).cpu().numpy()
    share_top1 = (m_all.topk(top1pct, 1).values.sum(1) / m_all.sum(1)).cpu().numpy()
    res["q5"] = {"eff_groups_image": eff_img.tolist(), "eff_groups_class": eff_class.tolist(),
                 "class_top1pct_share": share_top1.tolist(), "image_top1pct_share": img_top1.tolist(),
                 "eff_classes_feature": q5_feature_classes[0].tolist(), "feature_mass": q5_feature_classes[1].tolist()}
    print(name, json.dumps({k: res[k] for k in ("n", "loss_mean")}),
          "q1 all", [round(q1[k][-1], 3) for k in ("same_class", "same_group", "other_group")],
          "q2 class", [round(a["class"] - b["class"], 3) for a, b in zip(q2, q2_null)],
          "q4", [round(r, 3) for r in q4],
          "q3", {k: round(res["q3"][k]["eta2"], 3) for k in ("depth", "row", "col", "cls_share")}, flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths, names = wordnet_paths()
    levels = {str(s): groups_by_size(paths, s) for s in LEVELS}
    for k, v in levels.items():
        print("level", k, "groups", len(set(v)), Counter(Counter(v.tolist()).values()).most_common(3))
    lca = lca_depth(paths)
    stats = {"names": names, "levels": {k: v.tolist() for k, v in levels.items()}}
    for name in MODELS:
        if (PROFILES / MODELS[name] / "top_maps.npy").exists():
            stats[name] = analyse(name, paths, levels, lca)
            torch.cuda.empty_cache()
    (OUT / "stats.json").write_text(json.dumps(stats))


if __name__ == "__main__":
    main()
