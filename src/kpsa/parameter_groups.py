"""Parameter-group partitions used by KPSA experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GroupSlice:
    name: str
    parameter: torch.nn.Parameter
    axis: int | None
    offset: int
    count: int

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.axis is None:
            return tensor.float().square().sum().reshape(1)
        dims = tuple(i for i in range(tensor.ndim) if i != self.axis)
        return (
            tensor.float().square().sum(dim=dims) if dims else tensor.float().square()
        )

    def expand(self, values: torch.Tensor) -> torch.Tensor:
        local = values[self.offset : self.offset + self.count]
        shape = [1] * self.parameter.ndim
        if self.axis is not None:
            shape[self.axis] = self.count
        return local.reshape(shape)


class ParameterPartition:
    """Rows, complete tensors, or shared SwiGLU gate/up/down feature groups.

    Every parameter is owned exactly once; tied parameters use named_parameters'
    deduplication. Attention and embeddings use rows in the feature partition.
    """

    def __init__(self, model, kind: str, *, cache_sizes=True):
        if kind not in {"row", "tensor", "swiglu"}:
            raise ValueError(kind)
        self.kind = kind
        self.slices = []
        shared = {}
        shared_counts = {}
        n = 0
        for name, parameter in model.named_parameters():
            axis = None if kind == "tensor" else 0
            key = None
            if kind == "swiglu" and ".mlp." in name:
                prefix = name.split(".mlp.")[0]
                is_input = any(
                    marker in name
                    for marker in (".gate_proj.", ".up_proj.", ".fc1.")
                )
                is_output = any(
                    marker in name for marker in (".down_proj.", ".fc2.")
                )
                if parameter.ndim == 2 and (is_input or is_output):
                    key = prefix
                    axis = 1 if is_output else 0
                elif parameter.ndim == 1 and is_input:
                    # The input-projection bias belongs to the same MLP feature.
                    key = prefix
                    axis = 0
            count = 1 if axis is None else parameter.shape[axis]
            offset = shared.get(key, n) if key else n
            if offset == n:
                if key:
                    shared[key] = n
                    shared_counts[key] = count
                n += count
            elif key and shared_counts[key] != count:
                raise ValueError(
                    f"incompatible shared feature widths for {key}: "
                    f"{shared_counts[key]} vs {count}"
                )
            self.slices.append(GroupSlice(name, parameter, axis, offset, count))
        self.n_groups = n
        device = next(model.parameters()).device
        # Shape-only models support CPU partition audits without allocating weights.
        if device.type == "meta":
            device = torch.device("cpu")
        self.device = device
        self._sizes = self._make_sizes() if cache_sizes else None
        self.n_parameters = sum(spec.parameter.numel() for spec in self.slices)
        if int(self.sizes.double().sum()) != self.n_parameters:
            raise RuntimeError("partition does not cover every parameter exactly once")

    def _make_sizes(self):
        sizes = torch.zeros(self.n_groups, device=self.device)
        for spec in self.slices:
            sizes[spec.offset : spec.offset + spec.count] += (
                spec.parameter.numel() / spec.count
            )
        return sizes

    @property
    def sizes(self):
        return self._sizes if self._sizes is not None else self._make_sizes()

    @property
    def metadata_bytes(self):
        """Persistent tensor metadata; shapes/offsets use O(number of tensors)."""
        return (
            0
            if self._sizes is None
            else self._sizes.numel() * self._sizes.element_size()
        )

    @torch.no_grad()
    def gradient_scores(self) -> torch.Tensor:
        sizes = self.sizes
        sums = torch.zeros_like(sizes)
        for spec in self.slices:
            if spec.parameter.grad is not None:
                sums[spec.offset : spec.offset + spec.count] += spec.reduce(
                    spec.parameter.grad
                )
        return sums / sizes

    @torch.no_grad()
    def scale_parameters(self, scales: torch.Tensor):
        for spec in self.slices:
            spec.parameter.mul_(spec.expand(scales))

    @torch.no_grad()
    def weight_energy(self):
        sizes = self.sizes
        sums = torch.zeros_like(sizes)
        for spec in self.slices:
            sums[spec.offset : spec.offset + spec.count] += spec.reduce(spec.parameter)
        return sums / sizes

    def regularization(self, anchor: dict, importance: torch.Tensor):
        penalty = torch.zeros((), device=self.device)
        for spec in self.slices:
            delta = spec.parameter - anchor[spec.name]
            penalty = penalty + (delta.square() * spec.expand(importance)).sum()
        return penalty / self.n_parameters

