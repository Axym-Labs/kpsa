"""Reversible continuous SwiGLU feature multipliers, with no masking runner."""

from __future__ import annotations

import math

import torch

from .domain_applications import tensor_ranks


class FeatureGates:
    def __init__(self, model):
        self.model = model
        self.layers = [
            m for n, m in model.named_modules() if n.endswith(".mlp.down_proj")
        ]
        if not self.layers:
            raise ValueError("SwiGLU down projections are required")
        self.values = [
            torch.ones(m.in_features, device=m.weight.device, requires_grad=True)
            for m in self.layers
        ]
        self.handles = []

    def __enter__(self):
        for module, value in zip(self.layers, self.values):
            self.handles.append(
                module.register_forward_pre_hook(
                    lambda m, args, gate=value: (
                        args[0] * gate.to(args[0].dtype),
                        *args[1:],
                    )
                )
            )
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def gradients(self):
        return torch.stack([v.grad.detach().float() for v in self.values])

    def zero_grad(self):
        for value in self.values:
            value.grad = None

    @torch.no_grad()
    def set(self, values):
        if not bool(torch.isfinite(values).all() and (values > 0).all()):
            raise ValueError(
                "continuous scaling requires finite, strictly positive multipliers"
            )
        if tuple(values.shape) != (len(self.values), self.values[0].numel()):
            raise ValueError("gate shape does not match model")
        for gate, value in zip(self.values, values):
            gate.copy_(value)


def continuous_scales(scores, strength):
    """Positive, smooth-in-strength multipliers; equal rank budgets per layer.

    No selection, threshold or zeroing. Non-tied methods share precisely the
    same distribution of log multipliers, preventing intervention-size confounds.
    A flat profile implies identity rather than an arbitrary tie-breaking order.
    """
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError("finite layer-by-feature scores required")
    if scores.shape[1] == 1:
        return torch.ones_like(scores)
    ranks = torch.stack([tensor_ranks(row) for row in scores])
    direction = 2 * ranks / (scores.shape[1] - 1) - 1
    return (float(strength) * direction).exp()


def intervention_scales(scores, shape, method, strength):
    if method == "identity":
        return torch.ones(shape)
    if method == "uniform":
        rms = math.sqrt((shape[1] + 1) / (3 * (shape[1] - 1)))
        return torch.full(shape, strength * rms).exp()
    return continuous_scales(scores.reshape(shape), strength)


def grouped_scores(gradients, block_size):
    """OPG of a shared continuous multiplier includes cross-feature terms."""
    if gradients.shape[1] % block_size:
        raise ValueError("feature count must be divisible by multiplier block size")
    return gradients.reshape(gradients.shape[0], -1, block_size).sum(-1).square()
