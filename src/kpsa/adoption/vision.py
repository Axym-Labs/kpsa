"""Vision models, losses and data for the KPSA demo and research update."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGENET = Path.home() / "main/data/imagenet-1k-256"
DINO_CKPT = Path.home() / "main/data/kpsa-demo/ckpt/dino_vitbase16_pretrain_full_checkpoint.pth"
GRID = 14
N_BINS = 1 + GRID * GRID  # CLS + 14 x 14 image grid


def class_files(split: str):
    root = IMAGENET / split
    classes = sorted(p.name for p in root.iterdir())
    return [sorted((root / c).iterdir()) for c in classes]


def balanced_indices(split: str, offsets) -> list[tuple[Path, int]]:
    """(path, label) for the given within-class offsets, class-major."""
    files = class_files(split)
    return [(files[c][o], c) for c in range(len(files)) for o in offsets]


# ----------------------------------------------------------------------------
# Supervised classifier: NLL of the labelled class.


def load_supervised(name: str, device="cuda"):
    import timm

    model = timm.create_model(name, pretrained=True).to(device).eval()
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform


class SupervisedSet(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def supervised_bins(batch: int, device) -> torch.Tensor:
    return torch.arange(N_BINS, device=device).expand(batch, -1).contiguous()


def supervised_loss(model, images, labels):
    logits = model(images).float()
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss, logits


# ----------------------------------------------------------------------------
# DINO: student/teacher self-distillation loss on seeded multi-crop views.


class DINOHeadPlain(torch.nn.Module):
    """DINO projection head with the weight-normalised last layer folded."""

    def __init__(self, in_dim=768, out_dim=65536, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, bottleneck),
        )
        self.last_layer = torch.nn.Linear(bottleneck, out_dim, bias=False)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1, p=2)
        return self.last_layer(x)


def load_dino(device="cuda"):
    from kpsa.adoption.dino_vit import vit_base

    ckpt = torch.load(DINO_CKPT, map_location="cpu", weights_only=False)
    nets = {}
    for role in ("student", "teacher"):
        state = {k.removeprefix("module."): v for k, v in ckpt[role].items()}
        backbone = vit_base(patch_size=16)
        head = DINOHeadPlain()
        backbone.load_state_dict(
            {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
        )
        head.load_state_dict(
            {k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")}
        )
        nets[role] = (backbone.to(device).eval(), head.to(device).eval())
    for p in list(nets["teacher"][0].parameters()) + list(nets["teacher"][1].parameters()):
        p.requires_grad_(False)
    return nets


DINO_ARGS = dict(
    global_scale=(0.25, 1.0),
    local_scale=(0.05, 0.25),
    n_local=10,
    teacher_temp=0.07,
    student_temp=0.1,
)
_NORMALIZE = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_JITTER = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)


def _dino_view(image, size, scale, blur_p, solarize_p):
    """One DINO view; returns the tensor and its crop box and flip."""
    top, left, h, w = transforms.RandomResizedCrop.get_params(image, scale, (3 / 4, 4 / 3))
    view = TF.resized_crop(image, top, left, h, w, [size, size], TF.InterpolationMode.BICUBIC)
    flip = random.random() < 0.5
    if flip:
        view = TF.hflip(view)
    if random.random() < 0.8:
        view = _JITTER(view)
    if random.random() < 0.2:
        view = TF.rgb_to_grayscale(view, num_output_channels=3)
    if random.random() < blur_p:
        view = view.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
    if random.random() < solarize_p:
        view = ImageOps.solarize(view)
    return _NORMALIZE(TF.to_tensor(view)), (top, left, h, w, flip)


def _view_bins(box, image_size, size):
    """Image-grid bin of every token (CLS first) of a view."""
    top, left, h, w, flip = box
    width, height = image_size
    g = size // 16
    centers = (torch.arange(g) + 0.5) / g
    u = centers.flip(0) if flip else centers
    cols = ((left + u * w) / width * GRID).floor().clamp(0, GRID - 1).long()
    rows = ((top + centers * h) / height * GRID).floor().clamp(0, GRID - 1).long()
    patch_bins = 1 + rows[:, None] * GRID + cols[None, :]
    return torch.cat((torch.zeros(1, dtype=torch.long), patch_bins.flatten()))


class DINOSet(torch.utils.data.Dataset):
    def __init__(self, items, seed=20261116):
        self.items, self.seed = items, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")
        random.seed(self.seed + i)
        torch.manual_seed(self.seed + i)
        a = DINO_ARGS
        views, bins = [], []
        specs = [(224, a["global_scale"], 1.0, 0.0), (224, a["global_scale"], 0.1, 0.2)]
        specs += [(96, a["local_scale"], 0.5, 0.0)] * a["n_local"]
        for size, scale, blur, sol in specs:
            view, box = _dino_view(image, size, scale, blur, sol)
            views.append(view)
            bins.append(_view_bins(box, image.size, size))
        globals_ = torch.stack(views[:2])
        locals_ = torch.stack(views[2:])
        return globals_, locals_, torch.cat(bins), label


def dino_forward_loss(nets, globals_, locals_, center):
    """Per-sample DINO loss. Inputs are (B, 2, ...) and (B, n_local, ...)."""
    student, s_head = nets["student"]
    teacher, t_head = nets["teacher"]
    batch = globals_.shape[0]
    a = DINO_ARGS
    with torch.no_grad():
        t_out = t_head(teacher(globals_.flatten(0, 1))).float()
        t_prob = F.softmax((t_out - center) / a["teacher_temp"], -1).reshape(batch, 2, -1)
    g_cls = student(globals_.flatten(0, 1))
    l_cls = student(locals_.flatten(0, 1))
    s_out = torch.cat(
        (s_head(g_cls).float().reshape(batch, 2, -1), s_head(l_cls).float().reshape(batch, a["n_local"], -1)),
        1,
    )
    log_p = F.log_softmax(s_out / a["student_temp"], -1)
    n_views = log_p.shape[1]
    loss = torch.zeros(batch, device=globals_.device)
    terms = 0
    for iq in range(2):
        for v in range(n_views):
            if v == iq:
                continue
            loss = loss + (-(t_prob[:, iq] * log_p[:, v]).sum(-1))
            terms += 1
    return loss / terms, t_out
