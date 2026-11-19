"""Exact batched per-sample group energies for the timm ViT-B/16 partition.

Atlas construction in the paper runs one backward pass per source image and
reads squared parameter gradients. The same per-sample energies follow from
activation gradients alone, because every parameter tensor of the ViT acts
linearly on a per-token input:

  Linear row j:      ||sum_t d_tj x_t||^2 = d_j^T (X X^T) d_j
  Linear column j:   ||sum_t x_tj d_t||^2 = x_j^T (D D^T) x_j
  bias element j:    (sum_t d_tj)^2
  LayerNorm weight:  (sum_t d_tj xhat_tj)^2,   bias: (sum_t d_tj)^2
  patch conv row j:  as a Linear row on unfolded 16x16 patches
  pos_embed:         ||d||^2 of the block-0 input; cls_token: ||d_0||^2

with X the layer input and D the gradient at the layer output for one sample.
A batch of independent per-sample functionals is therefore profiled with one
batched backward that never forms parameter gradients. The result is the
complete partition energy, identical (up to float rounding) to the loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedPartitionEnergy:
    def __init__(self, model, partition):
        self.model = model
        self.partition = partition
        self.specs = {spec.name: spec for spec in partition.slices}
        self.out = None
        self.handles = []
        if not (model.global_pool == "token" and isinstance(model.fc_norm, nn.Identity)
                and isinstance(model.norm_pre, nn.Identity) and model.cls_token is not None
                and not model.no_embed_class and model.reg_token is None):
            raise ValueError("unsupported ViT configuration")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear(name, module)))
            elif isinstance(module, nn.LayerNorm):
                self.handles.append(module.register_forward_hook(self._layernorm(name, module)))
        self.handles.append(model.patch_embed.proj.register_forward_hook(self._conv("patch_embed.proj")))
        self.handles.append(model.blocks[0].register_forward_pre_hook(self._embed()))

    def _add(self, pname, values):
        spec = self.specs[pname]
        if values.shape[1] != spec.count:
            raise RuntimeError(f"{pname}: {values.shape} vs {spec.count}")
        self.out[:, spec.offset:spec.offset + spec.count] += values.double()

    def _linear_energy(self, pname, bname, x, d):
        """x: (B,T,in), d: (B,T,out)."""
        spec = self.specs[pname]
        if spec.axis == 0:
            gram = x @ x.transpose(1, 2)
            self._add(pname, (d * (gram @ d)).sum(1))
        else:
            gram = d @ d.transpose(1, 2)
            self._add(pname, (x * (gram @ x)).sum(1))
        if bname in self.specs:
            self._add(bname, d.sum(1).square())

    def _linear(self, name, module):
        def hook(_m, inputs, output):
            if self.out is None:
                return
            x_native = inputs[0].detach()  # native dtype; cast only inside the gradient hook

            def grad_hook(grad):
                x = x_native.float()
                if x.ndim == 2:
                    x = x[:, None]
                d = grad.detach().float()
                if d.ndim == 2:
                    d = d[:, None]
                self._linear_energy(f"{name}.weight", f"{name}.bias", x, d)

            output.register_hook(grad_hook)
        return hook

    def _layernorm(self, name, module):
        def hook(_m, inputs, output):
            if self.out is None:
                return
            x_native = inputs[0].detach()

            def grad_hook(grad):
                xhat = F.layer_norm(x_native.float(), module.normalized_shape, eps=module.eps)
                d = grad.detach().float()
                self._add(f"{name}.weight", (d * xhat).sum(1).square())
                self._add(f"{name}.bias", d.sum(1).square())

            output.register_hook(grad_hook)
        return hook

    def _conv(self, name):
        def hook(module, inputs, output):
            if self.out is None:
                return
            x_native = inputs[0].detach()

            def grad_hook(grad):
                patches = F.unfold(x_native.float(), module.kernel_size, stride=module.stride)
                x = patches.transpose(1, 2)  # (B, P, 3*16*16)
                d = grad.detach().float().flatten(2).transpose(1, 2)  # (B, P, C)
                self._linear_energy(f"{name}.weight", f"{name}.bias", x, d)

            output.register_hook(grad_hook)
        return hook

    def _embed(self):
        def hook(_m, inputs):
            if self.out is None:
                return
            x = inputs[0]

            def grad_hook(grad):
                d = grad.detach().float()
                self._add("pos_embed", d.square().sum((1, 2))[:, None])
                self._add("cls_token", d[:, 0].square().sum(1)[:, None])

            x.register_hook(grad_hook)
        return hook

    def __call__(self, images, labels):
        """Per-sample margin energies (B, n_groups), float64, on the model device."""
        device = next(self.model.parameters()).device
        images = images.to(device).requires_grad_(True)
        labels = labels.to(device)
        self.out = torch.zeros(len(images), self.partition.n_groups, dtype=torch.float64, device=device)
        try:
            logits = self.model(images).float()
            rows = torch.arange(len(labels), device=device)
            alt = logits.clone()
            alt[rows, labels] = -torch.inf
            objective = (logits[rows, labels] - alt.max(1).values).sum()
            torch.autograd.grad(objective, images)
            return self.out
        finally:
            self.out_ready = self.out
            self.out = None


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    """Validate against the paper loop and time both construction paths."""
    import argparse
    import json
    import time
    from pathlib import Path

    import timm

    from ..parameter_groups import ParameterPartition
    from ..representation_sensitivity import partition_gradient_energy
    from .vision_cost_bench import MODEL, load_images

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--contended", action="store_true", help="label the run as sharing the GPU")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = timm.create_model(MODEL, pretrained=True).cuda().eval()
    partition = ParameterPartition(model, "swiglu")
    images, labels = load_images(200)

    # Reference: the paper's per-image loop.
    ref = []
    for i in range(16):
        model.zero_grad(set_to_none=True)
        logits = model(images[i:i + 1].cuda()).float()
        alt = logits[0].clone()
        alt[labels[i]] = -torch.inf
        (logits[0, labels[i]] - alt.max()).backward()
        ref.append(partition_gradient_energy(partition).double().cpu())
    model.zero_grad(set_to_none=True)
    ref = torch.stack(ref)
    freeze(model)
    engine = BatchedPartitionEnergy(model, partition)
    got = engine(images[:16], labels[:16]).cpu()
    rel_total = ((got.sum(1) - ref.sum(1)).abs() / ref.sum(1)).max().item()
    share_ref = ref / ref.sum(1, keepdim=True)
    share_got = got / got.sum(1, keepdim=True)
    group_rel = ((share_got - share_ref).abs().sum(1)).max().item()  # L1 distance of shares
    big = share_ref > 1e-6
    max_rel_big = ((share_got - share_ref).abs()[big] / share_ref[big]).max().item()
    record = {"validation": {"images": 16, "max_rel_err_total_energy": rel_total,
                             "max_l1_share_distance": group_rel,
                             "max_rel_err_groups_with_share_gt_1e-6": max_rel_big}}
    print(json.dumps(record), flush=True)
    timings = {}
    images, labels = images[:192].cuda(), labels[:192].cuda()  # resident: no upload inside the timer
    for b in args.batches:
        for _ in range(2):
            engine(images[:b], labels[:b])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        reps = max(1, 768 // b)
        for r in range(reps):
            e = engine(images[(r * b) % 192:(r * b) % 192 + b], labels[(r * b) % 192:(r * b) % 192 + b])
            (e / e.sum(1, keepdim=True)).float()  # profiles stay on the GPU for atlas accumulation
        torch.cuda.synchronize()
        sec = (time.perf_counter() - t) / (reps * b)
        timings[str(b)] = {"seconds_per_source": sec, "sources_per_second": 1 / sec,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        print(b, timings[str(b)], flush=True)
    record["batched_timing"] = timings
    record["gpu_contended"] = args.contended
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
