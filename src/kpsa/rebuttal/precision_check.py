"""Do rankings and intervention utility survive matched precision? (R4 Q1)

For the E1 test queries (N = 6,400, paper KPSA prototype map):
  - activation attribution computed under bf16 autocast vs the fp32 reference:
    top-k overlap and exact zero-deactivation gap recovery;
  - KPSA with an fp32 DINOv2 forward vs the bf16 forward used in the paper:
    top-k overlap and gap recovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..kernel_sensitivity import fit_prototype_response_map, median_squared_distance
from .vision_cache import MODEL
from .vision_retrieval_eval import BatchedZeroAblation, bootstrap_gap, causal_assay, counts_for

FRACTIONS = (0.0002, 0.0005, 0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import timm
    from torchvision.datasets import ImageFolder
    from transformers import AutoImageProcessor, AutoModel

    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.load(args.cache, weights_only=False)
    counts = counts_for(cache["scope_sizes"], FRACTIONS)
    root = Path.home() / "main/data/imagenet-1k-256/validation"
    raw = ImageFolder(root)
    encoder = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).cuda().eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
    construction = cache["features"]["prototype"]
    pmap = fit_prototype_response_map(construction, 800, bandwidth_squared=median_squared_distance(construction) * 0.025**2,
                                      seed=91_027 + 800)
    S = cache["source_profiles"].cuda()
    atlas = pmap.transform(cache["features"]["source"]).float().cuda().T @ S / len(S)

    model = timm.create_model(MODEL, pretrained=True).cuda().eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    dataset = ImageFolder(root, transform=transform)
    fc2 = [b.mlp.fc2 for b in model.blocks]
    captured = {}
    for i, m in enumerate(fc2):
        m.register_forward_pre_hook(lambda _m, inp, i=i: captured.__setitem__(i, inp[0]))

    def attribution_bf16(image, label):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(image[None].cuda()).float()
        alt = logits[0].clone(); alt[label] = -torch.inf
        obj = logits[0, label] - alt.max()
        acts = [captured[i] for i in range(12)]
        grads = torch.autograd.grad(obj, acts)
        return torch.cat([(g.float() * a.float()).sum((0, 1)) for g, a in zip(grads, acts)])

    ablation = BatchedZeroAblation(model)
    rec = {m: {f: [] for f in FRACTIONS} for m in ("constant", "attr_fp32", "attr_bf16", "kpsa_bf16enc", "kpsa_fp32enc")}
    overlap = {k: {f: [] for f in FRACTIONS} for k in ("attribution", "kpsa")}
    const = S.mean(0)
    for qname in ("query49", "query47"):
        zq_bf16 = cache["features"][qname]
        for row in cache[f"{qname}_correct"].nonzero().flatten().tolist():
            index = int(cache["roles"][qname][row])
            image, label = dataset[index]
            pixels = processor(images=[raw[index][0]], return_tensors="pt")["pixel_values"].cuda()
            with torch.no_grad():
                out = encoder(pixel_values=pixels)
            z32 = F.normalize(out.pooler_output.float(), dim=1).cpu()
            per = {
                "constant": const,
                "attr_fp32": cache[f"{qname}_attribution"][row].cuda(),
                "attr_bf16": attribution_bf16(image, int(label)).detach(),
                "kpsa_bf16enc": (pmap.transform(zq_bf16[row:row + 1]).float().cuda() @ atlas)[0],
                "kpsa_fp32enc": (pmap.transform(z32).float().cuda() @ atlas)[0],
            }
            sels, keys = [], []
            tops = {m: torch.topk(s.clamp_min(0) if m != "attr_bf16" else s.clamp_min(0), max(counts)).indices for m, s in per.items()}
            for f, c in zip(FRACTIONS, counts):
                overlap["attribution"][f].append(len(set(tops["attr_fp32"][:c].tolist()) & set(tops["attr_bf16"][:c].tolist())) / c)
                overlap["kpsa"][f].append(len(set(tops["kpsa_bf16enc"][:c].tolist()) & set(tops["kpsa_fp32enc"][:c].tolist())) / c)
                for m in per:
                    sels.append(tops[m][:c]); keys.append((m, f))
            with torch.no_grad():
                deg, _ = causal_assay(model, ablation, image, int(label), sels)
            for (m, f), d in zip(keys, deg.tolist()):
                rec[m][f].append(d)
    result = {"feature_counts": counts, "topk_overlap": {k: {f"{f:g}": float(np.mean(v[f])) for f in FRACTIONS} for k, v in overlap.items()},
              "gap_vs_fp32_attribution": {}}
    for m in rec:
        result["gap_vs_fp32_attribution"][m] = {
            f"{f:g}": bootstrap_gap(np.asarray(rec[m][f]), np.asarray(rec["constant"][f]), np.asarray(rec["attr_fp32"][f]))
            for f in FRACTIONS}
    result["n"] = len(rec["constant"][FRACTIONS[0]])
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
