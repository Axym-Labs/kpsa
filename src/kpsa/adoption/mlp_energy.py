"""Exact per-sample gradient energy of coupled MLP feature groups.

A coupled feature j of an MLP block owns row j of every input projection
(fc1, or gate_proj and up_proj), the matching input bias, and column j of the
output projection. For one sample whose loss L depends only on its own tokens,

    ||dL/dW_in[j]||^2 + (dL/db_in[j])^2 = d_j^T (X X^T + 1) d_j,
    ||dL/dW_out[:, j]||^2               = a_j^T (Y Y^T) a_j,

where X holds the input tokens of the block, d_j the gradient at the input
projection's output, a_j the input of the output projection and Y the
gradient at its output. The per-token terms c_j(t) = d_j(t) [(X X^T + 1) d_j](t)
+ a_j(t) [(Y Y^T) a_j](t) sum exactly to the group energy, which gives an
additive decomposition over tokens. Several forward calls (e.g. DINO crops)
can belong to one sample; their tokens are concatenated before the Gram
products, because the gradient sums over all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class _Record:
    x: torch.Tensor | None = None
    deltas: list = field(default_factory=list)
    a: torch.Tensor | None = None
    dy: torch.Tensor | None = None


class MLPEnergyCapture:
    """Hooks MLP blocks of kind "gelu" (fc1/fc2) or "swiglu" (gate/up/down)."""

    def __init__(self, mlps, kind: str):
        if kind not in {"gelu", "swiglu"}:
            raise ValueError(kind)
        self.kind = kind
        self.mlps = list(mlps)
        self.records: list[list[_Record]] = [[] for _ in self.mlps]
        self.handles = []
        self.enabled = True
        for layer, mlp in enumerate(self.mlps):
            inputs = [mlp.fc1] if kind == "gelu" else [mlp.gate_proj, mlp.up_proj]
            output = mlp.fc2 if kind == "gelu" else mlp.down_proj
            self.bias = kind == "gelu" and inputs[0].bias is not None
            for position, module in enumerate(inputs):
                self.handles.append(
                    module.register_forward_hook(self._input_hook(layer, position))
                )
            self.handles.append(output.register_forward_hook(self._output_hook(layer)))

    def _input_hook(self, layer, position):
        def hook(module, args, out):
            if not self.enabled or not out.requires_grad:
                return
            if position == 0:
                self.records[layer].append(_Record(x=args[0].detach()))
            record = self.records[layer][-1]
            slot = len(record.deltas)
            record.deltas.append(None)

            def grad_hook(grad):
                record.deltas[slot] = grad.detach()

            out.register_hook(grad_hook)

        return hook

    def _output_hook(self, layer):
        def hook(module, args, out):
            if not self.enabled or not out.requires_grad:
                return
            record = self.records[layer][-1]
            record.a = args[0].detach()

            def grad_hook(grad):
                record.dy = grad.detach()

            out.register_hook(grad_hook)

        return hook

    def clear(self):
        self.records = [[] for _ in self.mlps]

    def remove(self):
        for handle in self.handles:
            handle.remove()

    @property
    def n_features(self) -> int:
        mlp = self.mlps[0]
        return (mlp.fc1 if self.kind == "gelu" else mlp.gate_proj).out_features

    def compute(
        self,
        crops_per_sample: list[int],
        token_bins: torch.Tensor | None = None,
        n_bins: int = 0,
        top_k: int = 0,
        keep_last_token_only: bool = False,
    ):
        """Return per-sample energies (B, L, F) and optional token-bin maps.

        crops_per_sample[c] is the number of consecutive batch rows of forward
        call c that belong to one sample (calls are sample-major).
        token_bins (B, T_total) maps every concatenated token to a bin; the
        maps are (B, L, n_bins) per-layer totals and (B, n_bins) for the
        sample's top_k groups.
        """
        layers = len(self.mlps)
        energies = []
        layer_contrib = []
        for layer in range(layers):
            records = self.records[layer]
            if len(records) != len(crops_per_sample):
                raise RuntimeError("forward calls do not match the crop layout")
            batch = records[0].x.shape[0] // crops_per_sample[0]

            def cat(name):
                parts = []
                for record, k in zip(records, crops_per_sample):
                    value = getattr(record, name)
                    parts.append(value.float().reshape(batch, -1, value.shape[-1]))
                return torch.cat(parts, 1)

            x = cat("x")
            a = cat("a")
            dy = cat("dy")
            deltas = [
                torch.cat(
                    [
                        r.deltas[i].float().reshape(batch, -1, r.deltas[i].shape[-1])
                        for r in records
                    ],
                    1,
                )
                for i in range(len(records[0].deltas))
            ]
            gram_x = x @ x.transpose(1, 2)
            if self.bias:
                gram_x = gram_x + 1.0
            gram_y = dy @ dy.transpose(1, 2)
            contrib = a * (gram_y @ a)
            for delta in deltas:
                contrib = contrib + delta * (gram_x @ delta)
            energies.append(contrib.sum(1))
            if token_bins is not None:
                layer_contrib.append(contrib)
            del x, a, dy, deltas, gram_x, gram_y
        energy = torch.stack(energies, 1)
        out = {"energy": energy}
        if token_bins is not None:
            batch = energy.shape[0]
            maps = energy.new_zeros(batch, layers, n_bins)
            for layer, contrib in enumerate(layer_contrib):
                maps[:, layer].scatter_add_(1, token_bins, contrib.sum(-1))
            out["layer_maps"] = maps
            if top_k:
                flat = energy.reshape(batch, -1)
                top = flat.topk(top_k, dim=1).indices
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask.scatter_(1, top, True)
                mask = mask.reshape(batch, layers, -1)
                top_map = energy.new_zeros(batch, n_bins)
                for layer, contrib in enumerate(layer_contrib):
                    selected = (contrib * mask[:, layer, None, :]).sum(-1)
                    top_map.scatter_add_(1, token_bins, selected)
                out["top_map"] = top_map
        self.clear()
        return out


def direct_energy(mlps, kind: str) -> torch.Tensor:
    """Group energies from .grad after a single-sample backward (reference)."""
    rows = []
    for mlp in mlps:
        if kind == "gelu":
            e = mlp.fc1.weight.grad.float().square().sum(1)
            if mlp.fc1.bias is not None:
                e = e + mlp.fc1.bias.grad.float().square()
            e = e + mlp.fc2.weight.grad.float().square().sum(0)
        else:
            e = (
                mlp.gate_proj.weight.grad.float().square().sum(1)
                + mlp.up_proj.weight.grad.float().square().sum(1)
                + mlp.down_proj.weight.grad.float().square().sum(0)
            )
        rows.append(e)
    return torch.stack(rows)
