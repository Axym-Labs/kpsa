"""Isolated, precision-matched cost benchmark for the vision setting (R4 W1/Q1).

One measurement per process (`--method`), so peak memory reflects only that
method's model, stored tensors, and activations. Both sides start from the
same pre-transformed GPU image batch (preprocessing excluded for both; the
paper's KPSA path had included CPU preprocessing, the baseline had not) and
both keep ranking on the GPU. `--precision` applies to every network forward
(and backward) in the measured path.

Methods
  attribution      action-matched gradient-times-activation (ViT-B/16), top-k on GPU
  attribution_cpu  same, with the submission's per-layer CPU transfer (for reference)
  kpsa_proto       DINOv2-S + 800 prototype responses + (G x 800) atlas + top-k
  kpsa_exact       DINOv2-S + exact RBF against N sources + (N x G) product + top-k
  knn              DINOv2-S + cosine to N sources + mean of k stored profiles + top-k
  ridge_z          DINOv2-S + (385 x G) linear map + top-k
  class_average    stored (classes x G) table indexed by label + top-k (no encoder)
  encoder          DINOv2-S forward alone
  profile_loop     exact per-source profile, paper loop (batch 1 backward, full partition)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
GROUPS = 36864


def load_images(n, which="vit"):
    """n distinct ImageNet validation images, pre-transformed (offset 49 of 200 classes)."""
    from torchvision.datasets import ImageFolder

    root = Path.home() / "main/data/imagenet-1k-256/validation"
    if which == "vit":
        import timm

        cfg = timm.data.resolve_model_data_config(timm.create_model(MODEL, pretrained=False))
        transform = timm.data.create_transform(**cfg, is_training=False)
        ds = ImageFolder(root, transform=transform)
    else:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
        ds = ImageFolder(root, transform=lambda im: processor(images=[im], return_tensors="pt")["pixel_values"][0])
    classes = torch.linspace(0, 999, 200).round().long()
    idx = (classes * 50 + 49).tolist()[:n]
    images = torch.stack([ds[i][0] for i in idx])
    labels = torch.tensor([ds.targets[i] for i in idx])
    return images, labels


def autocast(precision):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False)


class Encoder:
    def __init__(self, precision):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).cuda().eval()
        for p in self.model.parameters():  # frozen like the ViT (no autocast weight cache)
            p.requires_grad_(False)
        self.precision = precision

    @torch.no_grad()
    def __call__(self, pixels):
        with autocast(self.precision):
            out = self.model(pixel_values=pixels)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return F.normalize(pooled.float(), dim=1)


def build(method, precision, n, k, topk):
    """Return (callable(batch_pixels, labels), static description)."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    if method in ("attribution", "attribution_cpu", "attribution_multi", "attribution_multi_vjp"):
        import timm

        from ..parameter_groups import ParameterPartition
        from ..vision_causal_refined import mlp_feature_layout

        model = timm.create_model(MODEL, pretrained=True).cuda().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        layout = mlp_feature_layout(model, ParameterPartition(model, "swiglu"))
        captured = {}

        def capture(name):
            def hook(_m, inputs):
                captured[name] = inputs[0]
            return hook

        for item in layout:
            item["module"].register_forward_pre_hook(capture(item["name"]))
        names = [item["name"] for item in layout]

        def run(pixels, labels):
            pixels = pixels.requires_grad_(False)
            with autocast(precision):
                # Activations must require grad although parameters do not.
                x = model.patch_embed(pixels)
                x = model._pos_embed(x)
                x = model.norm_pre(x)
                x.requires_grad_(True)
                for block in model.blocks:
                    x = block(x)
                x = model.norm(x)
                logits = model.forward_head(x).float()
            rows = torch.arange(len(labels), device=logits.device)
            alt = logits.clone()
            alt[rows, labels] = -torch.inf
            objective = (logits[rows, labels] - alt.max(1).values).sum()
            acts = [captured[nm] for nm in names]
            captured.clear()  # no activation outlives its own query batch
            grads = None if method.startswith("attribution_multi") else torch.autograd.grad(objective, acts)
            if method == "attribution":
                scores = torch.cat([(g.float() * a.float()).sum(1) for g, a in zip(grads, acts)], 1)
                return torch.topk(scores, topk, dim=1).indices
            if method == "attribution_multi_vjp":
                # Same M functionals, all vector-Jacobian products in one batched backward.
                tops = logits.topk(k, dim=1).indices
                objs = []
                for j in range(k):
                    lab = tops[:, j]
                    alt_j = logits.clone()
                    alt_j[rows, lab] = -torch.inf
                    objs.append((logits[rows, lab] - alt_j.max(1).values).sum())
                objs = torch.stack(objs)
                g_all = torch.autograd.grad(objs, acts, grad_outputs=torch.eye(k, device=objs.device),
                                            is_grads_batched=True)
                sc = torch.cat([(g.float() * a.float()[None]).sum(2) for g, a in zip(g_all, acts)], 2)
                return torch.topk(sc, topk, dim=2).indices
            if method == "attribution_multi":
                # M functionals: margins of the M highest-scoring classes, one backward each.
                tops = logits.topk(k, dim=1).indices
                out = []
                for j in range(k):
                    lab = tops[:, j]
                    alt_j = logits.clone()
                    alt_j[rows, lab] = -torch.inf
                    obj_j = (logits[rows, lab] - alt_j.max(1).values).sum()
                    g_j = torch.autograd.grad(obj_j, acts, retain_graph=j < k - 1)
                    sc = torch.cat([(g.float() * a.float()).sum(1) for g, a in zip(g_j, acts)], 1)
                    out.append(torch.topk(sc, topk, dim=1).indices)
                return out
            scores = torch.zeros(len(labels), GROUPS, dtype=torch.float32, device="cpu")
            for i, (g, a) in enumerate(zip(grads, acts)):
                scores[:, i * 3072:(i + 1) * 3072] = (g.float() * a.float()).sum(1).cpu()
            return torch.topk(scores, topk, dim=1).indices

        return run, {"stored_bytes": 0}
    if method == "direct_multi":
        # Per-query direct-gradient profiles (the Application 2 reference) for M
        # functionals: one forward, M backward passes through the exact batched
        # group-energy engine (no parameter gradients are formed).
        import timm

        from ..parameter_groups import ParameterPartition
        from .batched_profiles import BatchedPartitionEnergy, freeze

        model = freeze(timm.create_model(MODEL, pretrained=True).cuda().eval())
        partition = ParameterPartition(model, "swiglu")
        engine = BatchedPartitionEnergy(model, partition)
        from ..vision_causal_refined import mlp_feature_layout, mlp_feature_scope

        scope_idx = mlp_feature_scope(partition, mlp_feature_layout(model, partition)).nonzero().flatten().cuda()

        def run(pixels, labels):
            x = pixels.clone().requires_grad_(True)
            engine.out = torch.zeros(len(x), partition.n_groups, dtype=torch.float64, device=x.device)
            with autocast(precision):
                logits = model(x).float()
            rows = torch.arange(len(x), device=x.device)
            tops = logits.topk(k, dim=1).indices
            out = []
            for j in range(k):
                engine.out = torch.zeros(len(x), partition.n_groups, dtype=torch.float64, device=x.device)
                alt = logits.clone()
                alt[rows, tops[:, j]] = -torch.inf
                obj = (logits[rows, tops[:, j]] - alt.max(1).values).sum()
                torch.autograd.grad(obj, x, retain_graph=j < k - 1)
                e = engine.out
                out.append(torch.topk((e / e.sum(1, keepdim=True))[:, scope_idx], topk, dim=1).indices)
            engine.out = None
            return out

        return run, {"stored_bytes": 0}
    if method == "class_average":
        table = torch.rand(200, GROUPS, generator=gen, device="cuda")

        def run(pixels, labels):
            return torch.topk(table[labels % 200], topk, dim=1).indices

        return run, {"stored_bytes": table.numel() * 4}
    encoder = Encoder(precision)
    if method == "encoder":
        return (lambda pixels, labels: encoder(pixels)), {"stored_bytes": 0}
    if method == "kpsa_multi":
        # M atlases (one per functional) sharing one query representation.
        protos = F.normalize(torch.randn(800, 384, generator=gen, device="cuda"), dim=1)
        atlases = torch.rand(k, GROUPS, 800, generator=gen, device="cuda")
        bw2 = 0.01

        def run(pixels, labels):
            z = encoder(pixels)
            logr = -torch.cdist(z, protos).square() / (2 * bw2)
            r = F.normalize(torch.exp(logr - logr.max(1, keepdim=True).values), dim=1)
            return [torch.topk(r @ atlases[j].T, topk, dim=1).indices for j in range(k)]

        return run, {"stored_bytes": (atlases.numel() + protos.numel()) * 4}
    if method == "kpsa_proto":
        protos = F.normalize(torch.randn(800, 384, generator=gen, device="cuda"), dim=1)
        atlas = torch.rand(GROUPS, 800, generator=gen, device="cuda")
        bw2 = 0.01

        def run(pixels, labels):
            z = encoder(pixels)
            logr = -torch.cdist(z, protos).square() / (2 * bw2)
            r = F.normalize(torch.exp(logr - logr.max(1, keepdim=True).values), dim=1)
            return torch.topk(r @ atlas.T, topk, dim=1).indices

        return run, {"stored_bytes": (atlas.numel() + protos.numel()) * 4}
    zs = F.normalize(torch.randn(n, 384, generator=gen, device="cuda"), dim=1)
    if method == "ridge_z":
        w = torch.randn(384, GROUPS, generator=gen, device="cuda")
        b = torch.randn(GROUPS, generator=gen, device="cuda")

        def run(pixels, labels):
            return torch.topk(encoder(pixels) @ w + b, topk, dim=1).indices

        return run, {"stored_bytes": (w.numel() + b.numel()) * 4}
    profiles = torch.rand(n, GROUPS, generator=gen, device="cuda")
    if method == "kpsa_exact":
        def run(pixels, labels):
            z = encoder(pixels)
            kern = torch.exp(-(2 - 2 * z @ zs.T).clamp_min(0) / (2 * 0.01))
            return torch.topk(kern @ profiles, topk, dim=1).indices

        return run, {"stored_bytes": (profiles.numel() + zs.numel()) * 4}
    if method == "knn":
        def run(pixels, labels):
            z = encoder(pixels)
            idx = torch.topk(z @ zs.T, k, dim=1).indices
            est = profiles[idx].mean(1) if k > 1 else profiles[idx[:, 0]]
            return torch.topk(est, topk, dim=1).indices

        return run, {"stored_bytes": (profiles.numel() + zs.numel()) * 4}
    raise ValueError(method)


def measure(run, pixels, labels, batch, warmup, repeats):
    batches = [(pixels[i:i + batch], labels[i:i + batch]) for i in range(0, len(pixels) - batch + 1, batch)]
    for i in range(warmup):
        run(*batches[i % len(batches)])
    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(*batches[i % len(batches)])
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    times = torch.tensor(times)
    return {
        "ms_per_batch_median": float(times.median()), "ms_per_batch_mean": float(times.mean()),
        "ms_per_batch_min": float(times.min()),
        "images_per_second_median": batch / float(times.median()) * 1000,
        "images_per_second_best": batch / float(times.min()) * 1000,
        "peak_reserved_bytes": peak_reserved,
        "static_allocated_bytes": static, "peak_allocated_bytes": peak,
    }


def profile_loop(precision, count):
    """Exact per-source profile cost (paper loop, full partition)."""
    import timm

    from ..parameter_groups import ParameterPartition
    from ..representation_sensitivity import partition_gradient_energy

    model = timm.create_model(MODEL, pretrained=True).cuda().eval()
    partition = ParameterPartition(model, "swiglu")
    pixels, labels = load_images(count + 4)
    pixels, labels = pixels.cuda(), labels.cuda()

    def one(i):
        model.zero_grad(set_to_none=True)
        with autocast(precision):
            logits = model(pixels[i:i + 1]).float()
        alt = logits[0].clone()
        alt[labels[i]] = -torch.inf
        (logits[0, labels[i]] - alt.max()).backward()
        e = partition_gradient_energy(partition)
        return e / e.sum()

    for i in range(4):
        one(i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for i in range(4, count + 4):
        one(i)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    return {"seconds_per_source": seconds / count, "sources": count,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--n", type=int, default=6400)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--topk", type=int, default=37)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    record = vars(args).copy()
    record["output"] = str(args.output)
    record["gpu"] = torch.cuda.get_device_name()
    record["torch"] = torch.__version__
    if args.method == "construction_extras":
        from ..kernel_sensitivity import fit_prototype_response_map, median_squared_distance
        gen = torch.Generator().manual_seed(0)
        construction = F.normalize(torch.randn(800, 384, generator=gen), dim=1)
        # Warm-up: first k-means fit and first cuBLAS call outside the timers.
        fit_prototype_response_map(construction[:64], 16, bandwidth_squared=1.0, seed=0)
        warm = torch.rand(64, GROUPS, device="cuda")
        (warm.T @ warm[:, :64]).sum().item()
        torch.cuda.synchronize()
        t = time.perf_counter()
        pmap = fit_prototype_response_map(construction, 800,
                                          bandwidth_squared=median_squared_distance(construction) * 0.025**2, seed=0)
        record["prototype_fit_seconds"] = time.perf_counter() - t
        zs = F.normalize(torch.randn(args.n, 384, generator=gen), dim=1)
        profiles = torch.rand(args.n, GROUPS, generator=gen).cuda()
        torch.cuda.synchronize()
        t = time.perf_counter()
        phi = pmap.transform(zs).float().cuda()
        atlas = phi.T @ profiles / args.n
        torch.cuda.synchronize()
        record["atlas_accumulation_seconds"] = time.perf_counter() - t
        record["atlas_shape"] = list(atlas.shape)
    elif args.method == "profile_loop":
        record.update(profile_loop(args.precision, 64))
    else:
        which = "vit" if args.method.startswith(("attribution", "direct")) else "dino"
        pixels, labels = load_images(args.batch, which)  # one resident input batch for every method
        pixels, labels = pixels.cuda(), labels.cuda()
        torch.cuda.reset_peak_memory_stats()
        run, static = build(args.method, args.precision, args.n, args.k, args.topk)
        record.update(static)
        record.update(measure(run, pixels, labels, args.batch, args.warmup, args.repeats))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps({k: record[k] for k in record if k in (
        "method", "precision", "n", "k", "images_per_second_median", "peak_allocated_bytes",
        "seconds_per_source", "prototype_fit_seconds", "atlas_accumulation_seconds")}), flush=True)


if __name__ == "__main__":
    main()
