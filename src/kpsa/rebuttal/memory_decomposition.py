"""Where does the submission's 10x peak-memory ratio come from? (R4 W1)

Re-measures peak allocated memory of the submission's own code paths
(vision_cost_accuracy_v8) in a fresh process per variant:

  paper_attr_isolated   submission attribution batch, nothing else on the GPU
  paper_attr_lingering  same, with the KPSA objects the submission kept alive
                        (DINOv2-S, both atlases, prototypes) still allocated
  paper_kpsa_bf16       submission KPSA batch path (bf16 DINOv2, CPU preprocessing)
  paper_kpsa_fp32       same path without autocast

Peak includes everything allocated when the batch runs (as in the submission).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .vision_cost_bench import MODEL, load_images


def peak_of(fn, repeats=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    peaks = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    return max(peaks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import timm
    from transformers import AutoImageProcessor, AutoModel

    from .. import vision_cost_accuracy_v8 as paper
    from ..parameter_groups import ParameterPartition
    from ..vision_causal_refined import mlp_feature_layout

    rec = {"variant": args.variant}
    if args.variant.startswith("paper_attr"):
        keep = []
        if args.variant == "paper_attr_lingering":
            enc = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).cuda().eval()
            g = torch.Generator(device="cuda").manual_seed(0)
            keep = [enc, torch.randn(1152, 800, generator=g, device="cuda"),
                    torch.randn(36864, 800, generator=g, device="cuda"),
                    F.normalize(torch.randn(800, 384, generator=g, device="cuda"), dim=1)]
            rec["lingering_bytes"] = int(sum(p.numel() * p.element_size() for p in enc.parameters())
                                         + sum(t.numel() * t.element_size() for t in keep[1:]))
        model = timm.create_model(MODEL, pretrained=True).cuda().eval()
        partition = ParameterPartition(model, "swiglu")
        layout = mlp_feature_layout(model, partition)
        images, labels = load_images(32)
        rec["peak_bytes"] = peak_of(lambda: paper.optimized_zero_activation_attribution_batch(
            model, images, labels, partition, layout))
        del keep
    else:
        from torchvision.datasets import ImageFolder

        enc = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).cuda().eval()
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
        root = Path.home() / "main/data/imagenet-1k-256/validation"
        raw = ImageFolder(root)
        classes = torch.linspace(0, 999, 200).round().long()
        imgs = [raw[int(c) * 50 + 49][0] for c in classes[:32]]
        g = torch.Generator(device="cuda").manual_seed(0)
        atlas = torch.randn(36864, 800, generator=g, device="cuda")
        protos = F.normalize(torch.randn(800, 384, generator=g, device="cuda"), dim=1)
        if args.variant == "paper_kpsa_bf16":
            fn = lambda: paper._atlas_query_batch(imgs, atlas, enc, proc, protos, 0.01)
        else:
            def fn():
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda()
                with torch.no_grad():
                    z = F.normalize(enc(pixel_values=px).pooler_output.float(), dim=1)
                d = torch.cdist(z, protos).square()
                lr = -d / 0.02
                r = F.normalize(torch.exp(lr - lr.max(1, keepdim=True).values), dim=1)
                torch.topk(atlas @ r.T, 96, dim=0)
        rec["peak_bytes"] = peak_of(fn)
    rec["peak_gib"] = rec["peak_bytes"] / 2**30
    with args.output.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
