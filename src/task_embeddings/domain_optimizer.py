"""Online task sketches and implicit parameter partitions, without dense ID maps."""

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


def partition_hierarchy(
    partition: ParameterPartition, resolution: str
) -> tuple[torch.Tensor, list[str]]:
    """Map fine groups to exact bundle, sublayer, or layer aggregates."""
    if resolution not in {"bundle", "sublayer", "layer"}:
        raise ValueError(resolution)

    def layer_prefix(name: str) -> tuple[str, int]:
        parts = name.split(".")
        if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
            return ".".join(parts[:2]), 2
        if (
            len(parts) > 2
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[2].isdigit()
        ):
            return ".".join(parts[:3]), 3
        return name, len(parts)

    def label_for(name: str) -> str:
        prefix, consumed = layer_prefix(name)
        parts = name.split(".")
        if resolution == "layer":
            return prefix
        if resolution == "sublayer":
            return ".".join(parts[: consumed + 1]) if consumed < len(parts) else prefix
        if ".mlp." in name:
            return name.split(".mlp.", 1)[0] + ".mlp.coupled_feature"
        return name

    labels: list[str] = []
    ids: dict[str, int] = {}
    assignment = torch.full((partition.n_groups,), -1, dtype=torch.long)
    assigned_labels: list[str | None] = [None] * partition.n_groups
    for spec in partition.slices:
        label = label_for(spec.name)
        if label not in ids:
            ids[label] = len(labels)
            labels.append(label)
        for group in range(spec.offset, spec.offset + spec.count):
            previous = assigned_labels[group]
            if previous is not None and previous != label:
                raise ValueError(
                    f"fine group {group} crosses {resolution} labels: "
                    f"{previous!r} and {label!r}"
                )
            assigned_labels[group] = label
            assignment[group] = ids[label]
    if bool((assignment < 0).any()):
        raise RuntimeError("partition hierarchy left fine groups unassigned")
    return assignment, labels


class OnlineTaskSketch:
    """Exponentially weighted linear regression sufficient statistics.

    full uses a one-hot task basis, mean uses a constant, and TBE/JL use
    a constant plus task features. Neither compact method stores an M*T atlas.
    Updates use only the current batch. A ridge handles initially unseen tasks.
    """

    def __init__(
        self,
        n_groups,
        features,
        representation,
        beta=0.99,
        seed=0,
        ridge=1e-3,
        score_link="linear",
    ):
        self.beta, self.ridge = beta, ridge
        if score_link not in {"linear", "log"}:
            raise ValueError(score_link)
        self.score_link = score_link
        self.clipped_fraction = 0.0
        tasks = len(features)
        self.representation = representation
        if representation == "full":
            phi = torch.eye(tasks, device=features.device)
        elif representation == "mean":
            phi = torch.ones(tasks, 1, device=features.device)
        else:
            if representation == "jl":
                generator = torch.Generator(device=features.device).manual_seed(seed)
                features = torch.randn(
                    features.shape, generator=generator, device=features.device
                )
            elif representation == "permuted":
                generator = torch.Generator(device=features.device).manual_seed(seed)
                features = features[
                    torch.randperm(tasks, generator=generator, device=features.device)
                ]
            elif representation != "tbe":
                raise ValueError(representation)
            phi = torch.cat(
                (torch.ones(tasks, 1, device=features.device), features), dim=1
            )
        self.phi = phi
        self.cross = torch.zeros(n_groups, phi.shape[1], device=features.device)
        self.gram = torch.zeros(phi.shape[1], phi.shape[1], device=features.device)
        self.steps = 0

    @torch.no_grad()
    def update(self, scores, task):
        if self.score_link == "log":
            scores = scores.clamp_min(1e-30).log()
        x = self.phi[task]
        self.cross.mul_(self.beta).add_(scores[:, None] * x, alpha=1 - self.beta)
        self.gram.mul_(self.beta).add_(torch.outer(x, x), alpha=1 - self.beta)
        self.steps += 1

    @torch.no_grad()
    def query(self, task):
        if self.representation in {"full", "mean"}:
            column = task if self.representation == "full" else 0
            result = self.cross[:, column] / self.gram[column, column].clamp_min(1e-12)
        else:
            regularized = self.gram + self.ridge * self.gram.trace().clamp_min(
                1e-8
            ) * torch.eye(self.gram.shape[0], device=self.gram.device)
            result = self.cross @ torch.linalg.solve(regularized, self.phi[task])
        if self.score_link == "log":
            return result.clamp(-69, 40).exp()
        self.clipped_fraction = float((result < 0).float().mean())
        return result.clamp_min(0)

    @property
    def bytes(self):
        return sum(
            t.numel() * t.element_size() for t in (self.phi, self.cross, self.gram)
        )

    def state_dict(self):
        """Independent CPU snapshot, including the actual JL/task basis."""
        return {
            "representation": self.representation,
            "beta": self.beta,
            "ridge": self.ridge,
            "score_link": self.score_link,
            "steps": self.steps,
            **{
                name: getattr(self, name).detach().cpu().clone()
                for name in ("phi", "cross", "gram")
            },
        }

    @classmethod
    def from_state_dict(cls, payload, device="cpu"):
        bank = cls(
            payload["cross"].shape[0],
            payload["phi"][:, 1:].to(device),
            payload["representation"],
            beta=payload["beta"],
            ridge=payload["ridge"],
            score_link=payload["score_link"],
        )
        for name in ("phi", "cross", "gram"):
            value = payload[name].to(device)
            if value.shape != getattr(bank, name).shape:
                raise ValueError(f"inconsistent online sketch {name} shape")
            setattr(bank, name, value.clone())
        bank.steps = payload["steps"]
        return bank


class OnlineGroupAdam(torch.optim.Optimizer):
    """Grouped second moments with optional Adam momentum (no dense state at beta1=0)."""

    def __init__(
        self,
        partition,
        features,
        representation,
        estimator="raw",
        lr=3e-4,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.01,
        seed=0,
        score_link="linear",
    ):
        super().__init__([s.parameter for s in partition.slices], {"lr": lr})
        self.partition = partition
        self.bank = OnlineTaskSketch(
            partition.n_groups,
            features,
            representation,
            beta2,
            seed,
            score_link=score_link,
        )
        self.estimator = estimator
        self.beta1, self.eps, self.weight_decay = beta1, eps, weight_decay
        self.steps = 0

    @torch.no_grad()
    def step(self, *, task=0, residual_norm=1.0):
        scores = self.partition.gradient_scores()
        observed = (
            scores
            if self.estimator == "raw"
            else scores
            / torch.as_tensor(residual_norm, device=scores.device).clamp_min(1e-12)
        )
        self.bank.update(observed, task)
        estimate = self.bank.query(task)
        sizes = self.partition.sizes
        observed_mean = (scores * sizes).sum() / sizes.sum()
        estimate_mean = (estimate * sizes).sum() / sizes.sum()
        estimate = estimate * observed_mean / estimate_mean.clamp_min(1e-30)
        # Explicit global damping, same for all representations and estimators.
        denominator = (0.9 * estimate + 0.1 * observed_mean).sqrt().add_(self.eps)
        self.steps += 1
        lr = self.param_groups[0]["lr"]
        for spec in self.partition.slices:
            p = spec.parameter
            if p.grad is None:
                continue
            if self.beta1 == 0:
                m = p.grad
            else:
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                m.lerp_(p.grad, 1 - self.beta1)
            p.mul_(1 - lr * self.weight_decay)
            p.addcdiv_(
                m, spec.expand(denominator), value=-lr / (1 - self.beta1**self.steps)
            )

    @property
    def persistent_bytes(self):
        return (
            self.bank.bytes
            + self.partition.metadata_bytes
            + sum(
                value.numel() * value.element_size()
                for state in self.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        )


class OnlineGroupAdafactor(OnlineGroupAdam):
    """Adafactor's update kernel with grouped/sketched arithmetic second moments.

    Matches torch.optim.Adafactor's defaults for decay, parameter scaling,
    clipping and epsilon. Only the variance representation is replaced. This
    controlled implementation requires dense gradient coverage on each step;
    it does not introduce the global recalibration/damping used by GroupAdam.
    """

    def __init__(self, partition, features, representation, **kwargs):
        if kwargs.get("estimator", "raw") != "raw":
            raise ValueError("matched Adafactor requires raw squared-gradient units")
        if kwargs.get("score_link", "linear") != "linear":
            raise ValueError("matched Adafactor requires arithmetic second moments")
        super().__init__(partition, features, representation, beta1=0.0, **kwargs)

    @torch.no_grad()
    def step(self, *, task=0, residual_norm=1.0):
        if any(spec.parameter.grad is None for spec in self.partition.slices):
            raise ValueError("matched Adafactor requires every parameter's gradient")
        self.steps += 1
        self.bank.beta = 1 - self.steps**-0.8
        self.bank.update(self.partition.gradient_scores(), task)
        variance = self.bank.query(task)
        lr = self.param_groups[0]["lr"]
        relative_step = min(lr, self.steps**-0.5)
        for spec in self.partition.slices:
            p = spec.parameter
            eps = torch.finfo(p.dtype).eps
            parameter_rms = p.norm() / p.numel() ** 0.5
            alpha = max(0.001, parameter_rms.item()) * relative_step
            denominator = spec.expand(variance).clamp_min(eps**2).sqrt()
            update = p.grad / denominator
            clipping = max(1.0, (update.norm() / update.numel() ** 0.5).item())
            p.mul_(1 - lr * self.weight_decay)
            p.add_(update, alpha=-alpha / clipping)
