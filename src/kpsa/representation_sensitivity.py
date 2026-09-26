"""Canonical representation-conditioned parameter-group sensitivity statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


@dataclass(frozen=True)
class SensitivityAtlas:
    """Additive sufficient statistics and their group-conditional embedding."""

    mass: torch.Tensor
    joint_moment: torch.Tensor
    embedding: torch.Tensor
    examples: int
    weight_mode: str = "normalized"


SensitivityWeightMode = Literal["raw", "normalized"]


def cold_fold_predictions(
    task_profiles: torch.Tensor,
    features: torch.Tensor,
    *,
    seed: int,
    folds: int = 4,
) -> dict[str, torch.Tensor]:
    """Build query profiles using gradient profiles from the other folds only."""
    if task_profiles.ndim != 2 or features.ndim != 2:
        raise ValueError("profiles and features must be matrices")
    if task_profiles.shape[1] != features.shape[0] or folds < 2:
        raise ValueError("task axes must match and at least two folds are required")
    profiles = task_profiles.detach().double().cpu()
    semantic = F.normalize(features.detach().double().cpu(), dim=1)
    generator = torch.Generator().manual_seed(seed)
    affine_semantic = F.normalize(
        torch.cat((torch.ones(len(semantic), 1), semantic), dim=1), dim=1
    )
    distance_sq = torch.cdist(semantic, semantic).square()
    positive = distance_sq[distance_sq > 0]
    bandwidth_sq = positive.median() if len(positive) else torch.tensor(1.0)
    kernel = torch.exp(-distance_sq / (2 * bandwidth_sq.clamp_min(1e-12)))
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    rbf_semantic = eigenvectors * eigenvalues.clamp_min(0).sqrt()[None]
    jl = F.normalize(
        torch.randn(features.shape, generator=generator, dtype=torch.float64), dim=1
    )
    permuted = semantic[torch.randperm(len(semantic), generator=generator)]
    methods = (
        "semantic",
        "affine_semantic",
        "rbf_semantic",
        "scalar_mass",
        "jl",
        "permuted",
        "nearest",
        "random",
    )
    result = {name: torch.empty_like(profiles) for name in methods}
    result["random"] = torch.rand(
        profiles.shape, generator=generator, dtype=torch.float64
    )
    for fold in range(folds):
        queries = list(range(fold, profiles.shape[1], folds))
        observed = [task for task in range(profiles.shape[1]) if task not in queries]
        for name, representation in (
            ("semantic", semantic),
            ("affine_semantic", affine_semantic),
            ("rbf_semantic", rbf_semantic),
            ("jl", jl),
            ("permuted", permuted),
        ):
            joint = profiles[:, observed] @ representation[observed] / len(observed)
            result[name][:, queries] = joint @ representation[queries].T
        result["scalar_mass"][:, queries] = profiles[:, observed].mean(1, keepdim=True)
        similarity = semantic[queries] @ semantic[observed].T
        nearest = torch.as_tensor(observed)[similarity.argmax(1)]
        result["nearest"][:, queries] = profiles[:, nearest]
    return result


def group_relative_shares(group_energy: torch.Tensor) -> torch.Tensor:
    """Normalize squared-gradient energy across groups for every example."""
    if group_energy.ndim != 2 or not group_energy.is_floating_point():
        raise ValueError("group energy must be a floating [examples, groups] tensor")
    if not torch.isfinite(group_energy).all() or (group_energy < 0).any():
        raise ValueError("group energy must be finite and nonnegative")
    total = group_energy.sum(dim=1, keepdim=True)
    if (total <= 0).any():
        raise ValueError("every example must have positive total energy")
    return group_energy / total


def sensitivity_weights(
    group_energy: torch.Tensor,
    mode: SensitivityWeightMode = "normalized",
) -> torch.Tensor:
    """Return either raw squared-gradient energy or group-relative shares."""
    if mode == "normalized":
        return group_relative_shares(group_energy)
    if mode != "raw":
        raise ValueError("weight mode must be 'raw' or 'normalized'")
    if group_energy.ndim != 2 or not group_energy.is_floating_point():
        raise ValueError("group energy must be a floating [examples, groups] tensor")
    if not torch.isfinite(group_energy).all() or (group_energy < 0).any():
        raise ValueError("group energy must be finite and nonnegative")
    if (group_energy.sum(dim=1) <= 0).any():
        raise ValueError("every example must have positive total energy")
    return group_energy


@torch.no_grad()
def partition_gradient_energy(partition) -> torch.Tensor:
    """Return squared-gradient norms for every group in a complete partition."""
    return partition.gradient_scores() * partition.sizes


def query_atlas(joint_moment: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Evaluate ``Q_G(z) = (A_G e_G) dot z`` for one or many queries."""
    if joint_moment.ndim != 2 or query.ndim not in {1, 2}:
        raise ValueError("expected a [groups, dimension] atlas and vector/matrix query")
    if joint_moment.shape[1] != query.shape[-1]:
        raise ValueError("atlas and query representation dimensions do not match")
    query = query.to(device=joint_moment.device, dtype=joint_moment.dtype)
    return joint_moment @ (query if query.ndim == 1 else query.T)


def profile_ranking_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    top_fraction: float = 0.05,
) -> dict[str, object]:
    """Compare one predicted and measured group-importance profile per query."""
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target profiles must share shape [groups, queries]")
    if not 0 < top_fraction <= 1 or predicted.shape[0] < 2:
        raise ValueError("top fraction must be in (0, 1] and at least two groups are needed")
    estimate = predicted.detach().double().cpu()
    truth = target.detach().double().cpu()
    if not torch.isfinite(estimate).all() or not torch.isfinite(truth).all():
        raise ValueError("profiles must be finite")
    count = max(1, math.ceil(top_fraction * len(estimate)))
    spearman, topk, ndcg, cosine = [], [], [], []
    for query in range(estimate.shape[1]):
        prediction = estimate[:, query]
        measured = truth[:, query]
        use_cuda = len(estimate) >= 32_768 and torch.cuda.is_available()
        if use_cuda:
            prediction = prediction.cuda()
            measured = measured.cuda()

            def ranks(values: torch.Tensor) -> torch.Tensor:
                ordered, indices = values.sort()
                _, inverse, counts = torch.unique_consecutive(
                    ordered, return_inverse=True, return_counts=True
                )
                ends = counts.cumsum(0)
                averages = (ends + ends - counts - 1).double() / 2
                output = torch.empty_like(values)
                output.scatter_(0, indices, averages[inverse])
                return output

            left, right = ranks(prediction), ranks(measured)
            left, right = left - left.mean(), right - right.mean()
            denominator = left.norm() * right.norm()
            rho = float(torch.dot(left, right) / denominator) if denominator else float("nan")
        else:
            rho = (
                float("nan")
                if prediction.std() == 0 or measured.std() == 0
                else float(spearmanr(prediction.numpy(), measured.numpy()).statistic)
            )
        spearman.append(rho)
        predicted_top = set(torch.topk(prediction, count).indices.tolist())
        target_top = set(torch.topk(measured, count).indices.tolist())
        topk.append(len(predicted_top & target_top) / count)
        ordering = torch.argsort(prediction, descending=True)
        ideal = torch.argsort(measured, descending=True)
        discounts = 1 / torch.log2(
            torch.arange(len(estimate), dtype=torch.float64, device=prediction.device)
            + 2
        )
        denominator = float((measured[ideal].clamp_min(0) * discounts).sum())
        ndcg.append(
            float((measured[ordering].clamp_min(0) * discounts).sum()) / denominator
            if denominator > 0
            else float("nan")
        )
        norm = float(prediction.norm() * measured.norm())
        cosine.append(float(torch.dot(prediction, measured)) / norm if norm else float("nan"))

    def finite_mean(values: list[float]) -> float:
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        "groups": predicted.shape[0],
        "queries": predicted.shape[1],
        "top_count": count,
        "spearman": spearman,
        "topk_recall": topk,
        "ndcg": ndcg,
        "cosine": cosine,
        "mean_spearman": finite_mean(spearman),
        "mean_topk_recall": finite_mean(topk),
        "mean_ndcg": finite_mean(ndcg),
        "mean_cosine": finite_mean(cosine),
    }


class SensitivityAtlasAccumulator:
    """Stream samples into a raw or normalized kernel-mean estimator."""

    def __init__(
        self,
        groups: int,
        dimension: int,
        *,
        weight_mode: SensitivityWeightMode = "normalized",
    ):
        if groups < 1 or dimension < 1:
            raise ValueError("groups and representation dimension must be positive")
        if weight_mode not in {"raw", "normalized"}:
            raise ValueError("weight mode must be 'raw' or 'normalized'")
        self._mass_sum = torch.zeros(groups, dtype=torch.float64)
        self._joint_sum = torch.zeros(groups, dimension, dtype=torch.float64)
        self._examples = 0
        self.weight_mode = weight_mode

    def update(
        self, group_energy: torch.Tensor, representation: torch.Tensor
    ) -> None:
        if representation.ndim != 2:
            raise ValueError("representation must have shape [examples, dimension]")
        if group_energy.shape[0] != representation.shape[0]:
            raise ValueError("energy and representation must describe the same examples")
        if group_energy.shape[1] != self._mass_sum.numel():
            raise ValueError("group count does not match accumulator")
        if representation.shape[1] != self._joint_sum.shape[1]:
            raise ValueError("representation dimension does not match accumulator")
        if not torch.isfinite(representation).all():
            raise ValueError("representation must be finite")
        weights = sensitivity_weights(group_energy, self.weight_mode)
        self.update_weights(weights, representation)

    def update_weights(
        self, weights: torch.Tensor, representation: torch.Tensor
    ) -> None:
        """Accumulate already-computed nonnegative sensitivity weights."""
        if representation.ndim != 2:
            raise ValueError("representation must have shape [examples, dimension]")
        if weights.ndim != 2 or weights.shape[0] != representation.shape[0]:
            raise ValueError("weights and representation must share the example axis")
        if weights.shape[1] != self._mass_sum.numel():
            raise ValueError("group count does not match accumulator")
        if representation.shape[1] != self._joint_sum.shape[1]:
            raise ValueError("representation dimension does not match accumulator")
        if (
            not weights.is_floating_point()
            or not torch.isfinite(weights).all()
            or bool((weights < 0).any())
        ):
            raise ValueError("weights must be finite and nonnegative")
        if not torch.isfinite(representation).all():
            raise ValueError("representation must be finite")
        weights = weights.detach().to("cpu", torch.float64)
        features = representation.detach().to("cpu", torch.float64)
        self._mass_sum.add_(weights.sum(dim=0))
        self._joint_sum.add_(weights.T @ features)
        self._examples += len(weights)

    def compute(self) -> SensitivityAtlas:
        if self._examples == 0:
            raise ValueError("cannot compute an empty sensitivity atlas")
        mass = self._mass_sum / self._examples
        joint = self._joint_sum / self._examples
        embedding = torch.where(
            mass[:, None] > 0,
            joint / mass[:, None].clamp_min(torch.finfo(joint.dtype).tiny),
            torch.zeros_like(joint),
        )
        return SensitivityAtlas(
            mass, joint, embedding, self._examples, weight_mode=self.weight_mode
        )


def coarsen_atlas(
    mass: torch.Tensor,
    joint_moment: torch.Tensor,
    assignment: torch.Tensor,
    *,
    groups: int | None = None,
) -> SensitivityAtlas:
    """Merge disjoint fine groups exactly using their additive statistics."""
    if mass.ndim != 1 or joint_moment.ndim != 2:
        raise ValueError("mass and joint moment must have shapes [G] and [G, D]")
    if len(mass) != len(joint_moment) or assignment.shape != mass.shape:
        raise ValueError("mass, moment, and assignment must share the fine-group axis")
    assignment = assignment.to(device=mass.device, dtype=torch.long)
    if groups is None:
        groups = int(assignment.max()) + 1
    if groups < 1 or (assignment < 0).any() or (assignment >= groups).any():
        raise ValueError("assignment contains an invalid coarse-group index")
    coarse_mass = torch.zeros(groups, dtype=mass.dtype, device=mass.device)
    coarse_joint = torch.zeros(
        groups,
        joint_moment.shape[1],
        dtype=joint_moment.dtype,
        device=joint_moment.device,
    )
    coarse_mass.index_add_(0, assignment, mass)
    coarse_joint.index_add_(0, assignment.to(joint_moment.device), joint_moment)
    embedding = torch.where(
        coarse_mass[:, None] > 0,
        coarse_joint
        / coarse_mass[:, None].to(coarse_joint.dtype).clamp_min(
            torch.finfo(coarse_joint.dtype).tiny
        ),
        torch.zeros_like(coarse_joint),
    )
    return SensitivityAtlas(coarse_mass, coarse_joint, embedding, examples=0)


def coarsen_profiles(
    profiles: torch.Tensor,
    assignment: torch.Tensor,
    *,
    groups: int | None = None,
) -> torch.Tensor:
    """Sum normalized fine-group profiles into an exact coarser partition."""
    if profiles.ndim != 2 or assignment.shape != (profiles.shape[0],):
        raise ValueError("profiles and assignment must share the fine-group axis")
    assignment = assignment.to(device=profiles.device, dtype=torch.long)
    if groups is None:
        groups = int(assignment.max()) + 1
    if groups < 1 or bool((assignment < 0).any()) or bool((assignment >= groups).any()):
        raise ValueError("assignment contains an invalid coarse-group index")
    coarse = torch.zeros(
        groups,
        profiles.shape[1],
        dtype=profiles.dtype,
        device=profiles.device,
    )
    coarse.index_add_(0, assignment, profiles)
    return coarse
