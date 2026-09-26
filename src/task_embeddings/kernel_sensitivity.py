"""Kernel mean representations of parameter-group sensitivity distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn.functional as F

from .representation_sensitivity import SensitivityAtlas, SensitivityAtlasAccumulator


class KernelFeatureMap(Protocol):
    """Common interface for finite-dimensional kernel feature maps."""

    @property
    def dimension(self) -> int: ...

    def transform(self, values: torch.Tensor) -> torch.Tensor: ...


def _matrix(values: torch.Tensor, name: str) -> torch.Tensor:
    result = values.detach().to("cpu", torch.float64)
    if result.ndim != 2 or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _normalized(values: torch.Tensor) -> torch.Tensor:
    values = _matrix(values, "representations")
    if bool((values.norm(dim=1) <= 0).any()):
        raise ValueError("representation rows must have positive norm")
    return F.normalize(values, dim=1)


def median_squared_distance(values: torch.Tensor) -> float:
    """Median nonzero squared pairwise distance for normalized vectors."""
    normalized = _normalized(values)
    distances = torch.pdist(normalized).square()
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("at least two distinct representations are required")
    return float(positive.median())


def rbf_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bandwidth_squared: float,
) -> torch.Tensor:
    """RBF kernel on L2-normalized representation rows."""
    if bandwidth_squared <= 0:
        raise ValueError("bandwidth_squared must be positive")
    left = _normalized(left)
    right = _normalized(right)
    distances = torch.cdist(left, right).square()
    return torch.exp(-distances / (2 * bandwidth_squared))


@dataclass(frozen=True)
class LinearKernelMap:
    input_dimension: int
    normalize_inputs: bool = False

    @property
    def dimension(self) -> int:
        return self.input_dimension

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        values = _matrix(values, "representations")
        if values.shape[1] != self.input_dimension:
            raise ValueError("representation dimension does not match feature map")
        return _normalized(values) if self.normalize_inputs else values


@dataclass(frozen=True)
class NystromRBFMap:
    landmarks: torch.Tensor
    inverse_root: torch.Tensor
    bandwidth_squared: float
    landmark_method: str

    @property
    def dimension(self) -> int:
        return len(self.landmarks)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        kernel = rbf_kernel(
            values,
            self.landmarks,
            bandwidth_squared=self.bandwidth_squared,
        )
        return kernel @ self.inverse_root


@dataclass(frozen=True)
class TensorSketchPolynomialMap:
    """TensorSketch feature map for ``(offset + cosine_similarity) ** 2``."""

    input_dimension: int
    output_dimension: int
    offset: float
    hashes: torch.Tensor
    signs: torch.Tensor

    @property
    def dimension(self) -> int:
        return self.output_dimension

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        values = _normalized(values)
        if values.shape[1] != self.input_dimension:
            raise ValueError("representation dimension does not match feature map")
        augmented = torch.cat(
            (
                torch.full(
                    (len(values), 1),
                    self.offset**0.5,
                    dtype=values.dtype,
                ),
                values,
            ),
            dim=1,
        )
        sketches = []
        for hashes, signs in zip(self.hashes, self.signs):
            sketch = torch.zeros(
                len(values), self.output_dimension, dtype=values.dtype
            )
            sketch.scatter_add_(
                1,
                hashes[None].expand(len(values), -1),
                augmented * signs,
            )
            sketches.append(torch.fft.rfft(sketch, dim=1))
        return torch.fft.irfft(
            sketches[0] * sketches[1],
            n=self.output_dimension,
            dim=1,
        )


@dataclass(frozen=True)
class PrototypeResponseMap:
    """Fixed local-response coordinates defined by shared encoder prototypes."""

    prototypes: torch.Tensor
    bandwidth_squared: float

    @property
    def dimension(self) -> int:
        return len(self.prototypes)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        values = _normalized(values)
        prototypes = _normalized(self.prototypes)
        log_response = -torch.cdist(values, prototypes).square() / (
            2 * self.bandwidth_squared
        )
        # Rowwise rescaling prevents underflow and disappears after L2
        # normalization, so it leaves the represented direction unchanged.
        response = torch.exp(
            log_response - log_response.max(dim=1, keepdim=True).values
        )
        return F.normalize(response, dim=1)


def fit_prototype_response_map(
    reference: torch.Tensor,
    dimension: int,
    *,
    bandwidth_squared: float,
    seed: int = 23_041,
) -> PrototypeResponseMap:
    """Fit sensitivity-independent k-means++ coordinates in encoder space."""
    from sklearn.cluster import KMeans

    values = _normalized(reference)
    if not 1 < dimension <= len(values):
        raise ValueError("prototype dimension must fit the construction set")
    if bandwidth_squared <= 0:
        raise ValueError("bandwidth_squared must be positive")
    estimator = KMeans(
        n_clusters=dimension,
        init="k-means++",
        n_init=1,
        max_iter=100,
        random_state=seed,
    ).fit(values.numpy())
    prototypes = F.normalize(
        torch.from_numpy(estimator.cluster_centers_).to(torch.float64), dim=1
    )
    return PrototypeResponseMap(prototypes, bandwidth_squared)


def fit_nystrom_rbf(
    reference: torch.Tensor,
    rank: int,
    *,
    bandwidth_squared: float | None = None,
    landmark_method: Literal["kmeans++", "even"] = "kmeans++",
    seed: int = 23_041,
    eigenvalue_floor: float = 1e-8,
) -> NystromRBFMap:
    """Fit a deterministic global Nyström map with data-adaptive landmarks."""
    values = _normalized(reference)
    if not 1 <= rank <= len(values):
        raise ValueError("rank must lie between one and the reference-set size")
    if eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive")
    if bandwidth_squared is None:
        bandwidth_squared = median_squared_distance(values)
    if landmark_method == "kmeans++":
        from sklearn.cluster import KMeans

        estimator = KMeans(
            n_clusters=rank,
            init="k-means++",
            n_init=1,
            max_iter=100,
            random_state=seed,
        ).fit(values.numpy())
        landmarks = F.normalize(
            torch.from_numpy(estimator.cluster_centers_).to(torch.float64), dim=1
        )
    elif landmark_method == "even":
        indices = torch.linspace(0, len(values) - 1, rank).round().long().unique()
        landmarks = values[indices]
    else:
        raise ValueError("unknown landmark method")
    landmark_kernel = rbf_kernel(
        landmarks,
        landmarks,
        bandwidth_squared=bandwidth_squared,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(landmark_kernel)
    inverse_root = (
        eigenvectors * eigenvalues.clamp_min(eigenvalue_floor).rsqrt()
    ) @ eigenvectors.T
    return NystromRBFMap(
        landmarks=landmarks,
        inverse_root=inverse_root,
        bandwidth_squared=float(bandwidth_squared),
        landmark_method=landmark_method,
    )


def _anchor_net_indices(
    values: torch.Tensor,
    rank: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Select data landmarks with the two-level Anchor-Net construction."""
    from scipy.stats import qmc

    points = _normalized(values)
    dimensions = points.shape[1]
    lower = points.min(0).values
    upper = points.max(0).values
    span = (upper - lower).clamp_min(1e-12)
    coarse_count = max(1, min(rank, rank // 4))
    coarse_unit = torch.from_numpy(
        qmc.Halton(dimensions, scramble=True, seed=seed).random(coarse_count)
    ).to(torch.float64)
    coarse = lower + coarse_unit * span
    assignments = torch.cdist(points, coarse, p=float("inf")).argmin(1)
    clusters = [
        (assignments == index).nonzero().flatten()
        for index in range(coarse_count)
    ]
    clusters = [indices for indices in clusters if len(indices)]

    cluster_bounds = []
    log_volumes = []
    for indices in clusters:
        current = points[indices]
        current_lower = current.min(0).values
        current_upper = current.max(0).values
        cluster_bounds.append((current_lower, current_upper))
        local_span = (current_upper - current_lower).clamp_min(span * 1e-6)
        log_volumes.append(local_span.log().sum())
    log_volumes = torch.stack(log_volumes)
    weights = torch.softmax(log_volumes - log_volumes.max(), dim=0)
    counts = torch.ones(len(clusters), dtype=torch.long)
    remaining = rank - len(clusters)
    if remaining > 0:
        fractional = weights * remaining
        counts += fractional.floor().long()
        remainder = rank - int(counts.sum())
        if remainder:
            order = (fractional - fractional.floor()).argsort(descending=True)
            counts[order[:remainder]] += 1

    selected = []
    for cluster, ((current_lower, current_upper), count) in enumerate(
        zip(cluster_bounds, counts.tolist())
    ):
        unit = torch.from_numpy(
            qmc.Halton(
                dimensions,
                scramble=True,
                seed=seed + cluster + 1,
            ).random(count)
        ).to(torch.float64)
        anchors = current_lower + unit * (current_upper - current_lower)
        nearest = torch.cdist(anchors, points, p=float("inf")).argmin(1)
        selected.extend(nearest.tolist())

    selected = list(dict.fromkeys(selected))
    if len(selected) < rank:
        chosen = torch.zeros(len(points), dtype=torch.bool)
        chosen[selected] = True
        if selected:
            min_distance = torch.cdist(
                points, points[selected], p=float("inf")
            ).min(1).values
        else:
            min_distance = torch.full((len(points),), torch.inf)
        while len(selected) < rank:
            min_distance[chosen] = -torch.inf
            index = int(min_distance.argmax())
            selected.append(index)
            chosen[index] = True
            distance = torch.cdist(
                points, points[index : index + 1], p=float("inf")
            )[:, 0]
            min_distance = torch.minimum(min_distance, distance)
    return torch.tensor(selected[:rank], dtype=torch.long)


def fit_anchor_nystrom_rbf(
    reference: torch.Tensor,
    rank: int,
    *,
    bandwidth_squared: float | None = None,
    seed: int = 23_041,
    eigenvalue_floor: float = 1e-8,
) -> NystromRBFMap:
    """Fit an RBF Nyström map using deterministic Anchor-Net landmarks."""
    values = _normalized(reference)
    if not 1 <= rank <= len(values):
        raise ValueError("rank must lie between one and the reference-set size")
    if bandwidth_squared is None:
        bandwidth_squared = median_squared_distance(values)
    indices = _anchor_net_indices(values, rank, seed=seed)
    landmarks = values[indices]
    landmark_kernel = rbf_kernel(
        landmarks,
        landmarks,
        bandwidth_squared=bandwidth_squared,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(landmark_kernel)
    inverse_root = (
        eigenvectors * eigenvalues.clamp_min(eigenvalue_floor).rsqrt()
    ) @ eigenvectors.T
    return NystromRBFMap(
        landmarks=landmarks,
        inverse_root=inverse_root,
        bandwidth_squared=float(bandwidth_squared),
        landmark_method="anchor_net",
    )


def fit_tensor_sketch_polynomial(
    input_dimension: int,
    output_dimension: int,
    *,
    offset: float = 1.0,
    seed: int = 23_041,
) -> TensorSketchPolynomialMap:
    """Create a fixed degree-2 polynomial TensorSketch feature map."""
    if input_dimension < 1 or output_dimension < 1:
        raise ValueError("input and output dimensions must be positive")
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    generator = torch.Generator().manual_seed(seed)
    hashes = torch.randint(
        output_dimension,
        (2, input_dimension + 1),
        generator=generator,
    )
    signs = (
        torch.randint(
            2,
            (2, input_dimension + 1),
            generator=generator,
            dtype=torch.int64,
        ).to(torch.float64)
        * 2
        - 1
    )
    return TensorSketchPolynomialMap(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        offset=float(offset),
        hashes=hashes,
        signs=signs,
    )


@dataclass(frozen=True)
class KernelMeanIndex:
    """Finite-dimensional per-group kernel mean representation."""

    atlas: SensitivityAtlas
    feature_map: KernelFeatureMap

    def query(self, representations: torch.Tensor, *, mass_weighted: bool) -> torch.Tensor:
        query_features = self.feature_map.transform(representations)
        stored = self.atlas.joint_moment if mass_weighted else self.atlas.embedding
        return stored @ query_features.T


def fit_kernel_mean_index(
    group_energy: torch.Tensor,
    representations: torch.Tensor,
    feature_map: KernelFeatureMap,
    *,
    weight_mode: Literal["raw", "normalized"] = "normalized",
) -> KernelMeanIndex:
    """Build a compact kernel index directly from per-example group energies."""
    features = feature_map.transform(representations)
    accumulator = SensitivityAtlasAccumulator(
        groups=group_energy.shape[1],
        dimension=feature_map.dimension,
        weight_mode=weight_mode,
    )
    accumulator.update(group_energy, features)
    return KernelMeanIndex(accumulator.compute(), feature_map)


def fit_weighted_kernel_mean_index(
    weights: torch.Tensor,
    representations: torch.Tensor,
    feature_map: KernelFeatureMap,
    *,
    weight_mode: str = "precomputed",
) -> KernelMeanIndex:
    """Build a compact index from already-computed raw or normalized weights."""
    features = feature_map.transform(representations)
    accumulator = SensitivityAtlasAccumulator(
        groups=weights.shape[1],
        dimension=feature_map.dimension,
    )
    accumulator.update_weights(weights, features)
    atlas = accumulator.compute()
    atlas = SensitivityAtlas(
        atlas.mass,
        atlas.joint_moment,
        atlas.embedding,
        atlas.examples,
        weight_mode=weight_mode,
    )
    return KernelMeanIndex(atlas, feature_map)


@dataclass(frozen=True)
class EmpiricalRBFKernelMeanIndex:
    """Exact full-reference RBF index used as the expensive fidelity target."""

    weights: torch.Tensor
    reference: torch.Tensor
    bandwidth_squared: float

    @property
    def mass(self) -> torch.Tensor:
        return self.weights.mean(dim=1)

    def query(self, representations: torch.Tensor, *, mass_weighted: bool) -> torch.Tensor:
        kernel = rbf_kernel(
            self.reference,
            representations,
            bandwidth_squared=self.bandwidth_squared,
        )
        numerator = self.weights @ kernel
        if mass_weighted:
            return numerator / self.weights.shape[1]
        return numerator / self.weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(self.weights.dtype).tiny
        )


def fit_empirical_rbf_index(
    weights: torch.Tensor,
    reference: torch.Tensor,
    *,
    bandwidth_squared: float | None = None,
) -> EmpiricalRBFKernelMeanIndex:
    """Fit the exact empirical RBF kernel mean from [groups, examples] weights."""
    weights = _matrix(weights, "weights")
    reference = _normalized(reference)
    if weights.shape[1] != len(reference):
        raise ValueError("weights and references must share the example axis")
    if bool((weights < 0).any()):
        raise ValueError("weights must be nonnegative")
    if bandwidth_squared is None:
        bandwidth_squared = median_squared_distance(reference)
    return EmpiricalRBFKernelMeanIndex(weights, reference, float(bandwidth_squared))
