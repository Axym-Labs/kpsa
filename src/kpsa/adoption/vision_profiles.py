"""Per-sample MLP-feature sensitivity profiles of vision models under their training loss.

Models: "augreg_in1k" (supervised ViT-B/16, ImageNet-1k NLL), "augreg2" (the
paper's ViT-B/16, NLL), and "dino" (DINO ViT-B/16 student, self-distillation
loss on seeded multi-crop views against the frozen teacher).

Writes, for N samples, the normalised profile over all 12 x 3,072 coupled MLP
features, the total energy, the loss, and optional token-bin maps on the 14 x 14
image grid (bin 0 is CLS): per-layer totals and the sample's top-k groups.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kpsa.adoption.mlp_energy import MLPEnergyCapture, direct_energy
from kpsa.adoption.vision import (
    N_BINS,
    DINOSet,
    SupervisedSet,
    balanced_indices,
    dino_forward_loss,
    load_dino,
    load_supervised,
    supervised_bins,
    supervised_loss,
)

TIMM = {
    "augreg_in1k": "vit_base_patch16_224.augreg_in1k",
    "augreg2": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
}
OUT = Path.home() / "main/data/kpsa-demo/profiles"


def dino_center(nets, n_per_class=10, batch=64, workers=16):
    path = OUT / "dino_center.pt"
    if path.exists():
        return torch.load(path).cuda()
    items = balanced_indices("train", range(n_per_class))
    loader = torch.utils.data.DataLoader(DINOSet(items, seed=7), batch_size=batch, num_workers=workers)
    teacher, head = nets["teacher"]
    total, count = None, 0
    with torch.no_grad():
        for globals_, _, _, _ in loader:
            out = head(teacher(globals_.cuda().flatten(0, 1))).float()
            total = out.sum(0) if total is None else total + out.sum(0)
            count += out.shape[0]
    center = (total / count)[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(center.cpu(), path)
    return center


def build(model_name, split, offsets, batch):
    if model_name == "dino":
        nets = load_dino()
        center = dino_center(nets)
        mlps = [b.mlp for b in nets["student"][0].blocks]
        dataset = DINOSet(balanced_indices(split, offsets))
        params = list(nets["student"][0].parameters()) + list(nets["student"][1].parameters())

        def step(data):
            globals_, locals_, bins, labels = data
            loss, _ = dino_forward_loss(nets, globals_.cuda(), locals_.cuda(), center)
            return loss, None, [2, locals_.shape[1]], bins.cuda()

        zero = lambda: [setattr(p, "grad", None) for p in params]  # noqa: E731
        modules = [nets["student"][0], nets["student"][1]]
    else:
        model, transform = load_supervised(TIMM[model_name])
        mlps = [b.mlp for b in model.blocks]
        dataset = SupervisedSet(balanced_indices(split, offsets), transform)

        def step(data):
            images, labels = data
            labels = labels.cuda()
            loss, logits = supervised_loss(model, images.cuda(), labels)
            return loss, logits.argmax(1), [1], supervised_bins(len(labels), labels.device)

        zero = lambda: model.zero_grad(set_to_none=True)  # noqa: E731
        modules = [model]
    return mlps, dataset, step, zero, modules


def check(model_name, n=3):
    """Compare batched Gram energies with single-sample .grad energies."""
    mlps, dataset, step, zero, modules = build(model_name, "validation", [0], 1)
    capture = MLPEnergyCapture(mlps, "gelu")
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(n)), batch_size=n)
    data = next(iter(loader))
    zero()
    loss, _, layout, bins = step(data)
    loss.sum().backward()
    batched = capture.compute(layout, bins, N_BINS, top_k=37)
    capture.enabled = False
    for i in range(n):
        one = tuple(t[i : i + 1] for t in data[:-1]) + (data[-1][i : i + 1],)
        zero()
        loss, *_ = step(one)
        loss.sum().backward()
        ref = direct_energy(mlps, "gelu")
        got = batched["energy"][i]
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        corr = torch.corrcoef(torch.stack((got.flatten(), ref.flatten())))[0, 1].item()
        map_sum = (batched["layer_maps"][i].sum(-1) - got.sum(-1)).abs().max().item() / got.sum(-1).abs().max().item()
        print(f"sample {i}: loss {loss.item():.4f} max rel err {rel:.2e} corr {corr:.6f} map-sum err {map_sum:.2e}")
    zero()


def run(model_name, split, offsets, batch, out_name, maps, top_k, workers):
    mlps, dataset, step, zero, _ = build(model_name, split, offsets, batch)
    capture = MLPEnergyCapture(mlps, "gelu")
    out = OUT / out_name
    out.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    features = len(mlps) * capture.n_features
    profiles = np.lib.format.open_memmap(out / "profiles.npy", "w+", np.float16, (n, features))
    total = np.zeros(n, np.float32)
    losses = np.zeros(n, np.float32)
    preds = np.full(n, -1, np.int32)
    labels = np.array([lab for _, lab in dataset.items], np.int32)
    if maps:
        layer_maps = np.lib.format.open_memmap(out / "layer_maps.npy", "w+", np.float16, (n, len(mlps), N_BINS))
        top_maps = np.zeros((n, N_BINS), np.float32)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=workers, pin_memory=True)
    started, done = time.time(), 0
    for data in loader:
        zero()
        loss, pred, layout, bins = step(data)
        loss.sum().backward()
        result = capture.compute(layout, bins if maps else None, N_BINS, top_k=top_k if maps else 0)
        energy = result["energy"].flatten(1)
        tot = energy.sum(1)
        b = energy.shape[0]
        profiles[done : done + b] = (energy / tot[:, None]).cpu().numpy().astype(np.float16)
        total[done : done + b] = tot.cpu().numpy()
        losses[done : done + b] = loss.detach().cpu().numpy()
        if pred is not None:
            preds[done : done + b] = pred.cpu().numpy()
        if maps:
            layer_maps[done : done + b] = (result["layer_maps"] / tot[:, None, None]).cpu().numpy().astype(np.float16)
            top_maps[done : done + b] = (result["top_map"] / tot[:, None]).cpu().numpy()
        done += b
        if done % (batch * 50) < batch:
            rate = done / (time.time() - started)
            print(f"{done}/{n} {rate:.1f}/s eta {(n - done) / rate / 60:.1f} min", flush=True)
    zero()
    profiles.flush()
    np.save(out / "total.npy", total)
    np.save(out / "loss.npy", losses)
    np.save(out / "pred.npy", preds)
    np.save(out / "label.npy", labels)
    if maps:
        layer_maps.flush()
        np.save(out / "top_maps.npy", top_maps)
    meta = dict(model=model_name, split=split, offsets=list(offsets), n=n, features=features,
                layers=len(mlps), top_k=top_k, seconds=time.time() - started,
                files=[str(p) for p, _ in dataset.items])
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"done {n} in {(time.time() - started) / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["augreg_in1k", "augreg2", "dino"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--offsets", default="0:50")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out")
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--top-k", type=int, default=37)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.check:
        check(args.model)
        return
    start, stop = map(int, args.offsets.split(":"))
    run(args.model, args.split, range(start, stop), args.batch, args.out or f"{args.model}_{args.split}",
        args.maps, args.top_k, args.workers)


if __name__ == "__main__":
    main()
