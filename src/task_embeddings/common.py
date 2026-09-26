from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

EPS = 1e-12
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def arc_artifact_dir(
    arc: str,
    setting: str | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve generated artifacts into the private sibling repository."""
    internal_root = project_root.with_name(f"{project_root.name}_internal")
    path = internal_root / arc / "artifacts"
    return path / setting if setting else path


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_accelerator_peak_memory(device: torch.device) -> None:
    """Start a run-level CUDA peak-memory measurement when CUDA is in use."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def accelerator_peak_memory(device: torch.device) -> dict[str, int]:
    """Return explicit run-level CUDA peaks, with zero-valued CPU fields."""
    if device.type != "cuda":
        return {
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_reserved_bytes": 0,
        }
    torch.cuda.synchronize(device)
    return {
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def normalized_rows(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def jl_task_representation(n_tasks: int, dimension: int, seed: int) -> torch.Tensor:
    """Deterministic, row-normalized Gaussian task features."""
    if n_tasks < 1 or dimension < 1:
        raise ValueError("n_tasks and dimension must be positive")
    generator = torch.Generator().manual_seed(seed)
    return normalized_rows(torch.randn(n_tasks, dimension, generator=generator))


def build_task_atlas(
    sensitivities: torch.Tensor,
    task_ids: torch.Tensor,
    n_tasks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a prevalence-invariant module-by-task sensitivity atlas.

    The task columns are conditional means, so unequal task counts do not
    change the intended task-balanced reference distribution.
    """
    values = sensitivities.detach().float().cpu()
    tasks = task_ids.detach().long().cpu()
    if values.ndim != 2 or tasks.ndim != 1 or values.shape[0] != tasks.numel():
        raise ValueError(
            "expected sensitivities [samples, modules] and task_ids [samples]"
        )
    if tasks.numel() and (int(tasks.min()) < 0 or int(tasks.max()) >= n_tasks):
        raise ValueError("task id outside declared task range")
    sums = torch.zeros(values.shape[1], n_tasks, dtype=values.dtype)
    counts = torch.zeros(n_tasks, dtype=values.dtype)
    sums.index_add_(1, tasks, values.T)
    counts.index_add_(0, tasks, torch.ones_like(tasks, dtype=values.dtype))
    if bool((counts == 0).any()):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"balanced atlas requires every task; missing {missing}")
    conditional_means = sums / counts[None, :]
    amplitude = conditional_means.mean(dim=1)
    atlas = conditional_means / conditional_means.sum(dim=1, keepdim=True).clamp_min(
        EPS
    )
    return atlas, amplitude


def _linear_centered_kernel_alignment(
    first: torch.Tensor, second: torch.Tensor
) -> float:
    """Linear CKA without materializing either module-by-module Gram matrix."""
    first_centered = first - first.mean(dim=0, keepdim=True)
    second_centered = second - second.mean(dim=0, keepdim=True)
    cross = first_centered.T @ second_centered
    first_covariance = first_centered.T @ first_centered
    second_covariance = second_centered.T @ second_centered
    denominator = first_covariance.norm() * second_covariance.norm()
    if denominator <= EPS:
        return float("nan")
    return float(cross.square().sum() / denominator)


def source_fidelity(
    atlas: torch.Tensor,
    task_representation: torch.Tensor,
    *,
    k_neighbors: int = 10,
    top_fraction: float = 0.05,
    max_pairs: int = 200_000,
    max_knn_queries: int = 512,
) -> dict[str, float]:
    """Measure how well ``E = P Phi`` preserves the full atlas ``P``."""
    reference = atlas.detach().float().cpu()
    representation = normalized_rows(task_representation.detach().float().cpu())
    if reference.ndim != 2 or representation.ndim != 2:
        raise ValueError("atlas and task representation must be matrices")
    if reference.shape[1] != representation.shape[0]:
        raise ValueError("atlas task axis must match task representation rows")
    embedding = reference @ representation

    n_modules = reference.shape[0]
    n_pairs = n_modules * (n_modules - 1) // 2
    if n_pairs <= max_pairs:
        pair_indices = torch.triu_indices(n_modules, n_modules, offset=1)
        first_indices, second_indices = pair_indices[0], pair_indices[1]
    else:
        generator = torch.Generator().manual_seed(0)
        first_indices = torch.randint(0, n_modules, (max_pairs,), generator=generator)
        second_indices = torch.randint(0, n_modules, (max_pairs,), generator=generator)
        equal = first_indices == second_indices
        second_indices[equal] = (second_indices[equal] + 1) % n_modules
    ref_dist = (reference[first_indices] - reference[second_indices]).norm(dim=1)
    emb_dist = (embedding[first_indices] - embedding[second_indices]).norm(dim=1)
    valid = ref_dist > EPS
    distortion = (emb_dist[valid] / ref_dist[valid] - 1.0).abs()

    k = min(max(1, k_neighbors), max(1, n_modules - 1))
    if n_modules <= 1:
        knn = float("nan")
        query_count = 0
    else:
        query_count = min(n_modules, max_knn_queries)
        query_indices = torch.linspace(0, n_modules - 1, query_count).long().unique()
        query_count = query_indices.numel()
        ref_distances = torch.cdist(reference[query_indices], reference)
        emb_distances = torch.cdist(embedding[query_indices], embedding)
        ref_distances[torch.arange(query_count), query_indices] = float("inf")
        emb_distances[torch.arange(query_count), query_indices] = float("inf")
        ref_neighbors = ref_distances.topk(k, largest=False).indices
        emb_neighbors = emb_distances.topk(k, largest=False).indices
        overlap = [
            len(set(a.tolist()) & set(b.tolist())) / k
            for a, b in zip(ref_neighbors, emb_neighbors)
        ]
        knn = float(np.mean(overlap))

    candidate_scores = embedding @ representation.T
    ranking_correlations = [
        _safe_corr(
            candidate_scores[:, task].numpy(), reference[:, task].numpy(), "spearman"
        )
        for task in range(reference.shape[1])
    ]
    top_count = min(n_modules, max(1, math.ceil(top_fraction * n_modules)))
    top_overlaps = []
    for task in range(reference.shape[1]):
        candidate_top = set(
            torch.topk(candidate_scores[:, task], top_count).indices.tolist()
        )
        reference_top = set(torch.topk(reference[:, task], top_count).indices.tolist())
        top_overlaps.append(len(candidate_top & reference_top) / top_count)

    return {
        "dimension": int(representation.shape[1]),
        "distance_distortion_mean": float(distortion.mean())
        if distortion.numel()
        else float("nan"),
        "distance_distortion_median": float(distortion.median())
        if distortion.numel()
        else float("nan"),
        "distance_distortion_p95": float(torch.quantile(distortion, 0.95))
        if distortion.numel()
        else float("nan"),
        "distance_pairs": int(valid.sum()),
        "centered_kernel_alignment": _linear_centered_kernel_alignment(
            reference, embedding
        ),
        "knn_preservation": knn,
        "knn_queries": int(query_count),
        "ranking_spearman_mean": float(np.nanmean(ranking_correlations)),
        "topk_overlap_mean": float(np.mean(top_overlaps)),
        "topk_fraction": top_fraction,
    }


def task_aligned_importance(
    embedding: torch.Tensor,
    task_representation: torch.Tensor,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    """Common ``A_m [e_m^T phi_t]_+`` score used by all applications."""
    affinity = query_scores(embedding, task_representation).clamp_min(0)
    return amplitude.detach().float().cpu()[:, None] * affinity


def conditional_importance(
    atlas: torch.Tensor,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    """Recover per-task conditional means from a balanced normalized atlas.

    ``build_task_atlas`` defines ``amplitude`` as the mean conditional score
    across tasks and normalizes every atlas row to sum to one. Consequently the
    unnormalized conditional score is ``T * amplitude * atlas``.
    """
    values = atlas.detach().float().cpu()
    scale = amplitude.detach().float().cpu()
    if values.ndim != 2 or scale.ndim != 1 or values.shape[0] != scale.numel():
        raise ValueError("expected atlas [modules, tasks] and amplitude [modules]")
    return values * scale[:, None] * values.shape[1]


def _validated_task_folds(
    n_tasks: int, folds: Sequence[Sequence[int]] | None
) -> list[list[int]]:
    if folds is None:
        return [[task] for task in range(n_tasks)]
    result: list[list[int]] = []
    seen: set[int] = set()
    for fold in folds:
        heldout = [int(task) for task in fold]
        if not heldout:
            raise ValueError("task folds may not be empty")
        if len(set(heldout)) != len(heldout):
            raise ValueError("a task may appear only once within a fold")
        if any(task < 0 or task >= n_tasks for task in heldout):
            raise ValueError("task fold contains an out-of-range task")
        overlap = seen.intersection(heldout)
        if overlap:
            raise ValueError(f"tasks occur in more than one fold: {sorted(overlap)}")
        seen.update(heldout)
        result.append(heldout)
    return result


def cross_validated_task_scores(
    importance: torch.Tensor,
    task_representation: torch.Tensor,
    *,
    folds: Sequence[Sequence[int]] | None = None,
    rectify: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Query held-out task scores without using their importance columns.

    For each fold ``H``, this computes the canonical index
    ``E_train = I_train Phi_train`` and queries it with ``Phi_H``. The target
    columns are used only by the caller for evaluation. Orthogonal held-out
    queries (notably one-hot task IDs) are marked unqueryable instead of being
    assigned an artificial tied ranking.
    """
    target = importance.detach().float().cpu()
    representation = normalized_rows(task_representation.detach().float().cpu())
    if target.ndim != 2 or representation.ndim != 2:
        raise ValueError("importance and task representation must be matrices")
    if target.shape[1] != representation.shape[0]:
        raise ValueError("importance task axis must match representation rows")
    n_tasks = target.shape[1]
    predictions = torch.full_like(target, float("nan"))
    queryable = torch.zeros(n_tasks, dtype=torch.bool)
    for heldout in _validated_task_folds(n_tasks, folds):
        train_mask = torch.ones(n_tasks, dtype=torch.bool)
        train_mask[heldout] = False
        if not bool(train_mask.any()):
            continue
        train_representation = representation[train_mask]
        query_representation = representation[heldout]
        affinities = train_representation @ query_representation.T
        fold_queryable = affinities.abs().amax(dim=0) > EPS
        embedding = target[:, train_mask] @ train_representation
        fold_predictions = embedding @ query_representation.T
        if rectify:
            fold_predictions = fold_predictions.clamp_min(0)
        for local_index, task in enumerate(heldout):
            if bool(fold_queryable[local_index]):
                predictions[:, task] = fold_predictions[:, local_index]
                queryable[task] = True
    return predictions, queryable


def _continuous_ndcg(candidate: torch.Tensor, target: torch.Tensor) -> float:
    if candidate.numel() != target.numel() or not candidate.numel():
        return float("nan")
    gains = target.clamp_min(0)
    ideal = torch.sort(gains, descending=True).values
    discount = torch.log2(torch.arange(gains.numel(), dtype=torch.float32) + 2.0)
    ideal_dcg = (ideal / discount).sum()
    if ideal_dcg <= EPS:
        return float("nan")
    order = torch.argsort(candidate, descending=True)
    return float((gains[order] / discount).sum() / ideal_dcg)


def cross_validated_task_query_fidelity(
    importance: torch.Tensor,
    task_representation: torch.Tensor,
    *,
    folds: Sequence[Sequence[int]] | None = None,
    top_fraction: float = 0.05,
    target_importance: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score inductive task queries against independently measured columns."""
    source = importance.detach().float().cpu()
    target = (
        source
        if target_importance is None
        else target_importance.detach().float().cpu()
    )
    if target.shape != source.shape:
        raise ValueError("source and target importance must share one shape")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0, 1]")
    predictions, queryable = cross_validated_task_scores(
        source, task_representation, folds=folds
    )
    top_count = max(1, math.ceil(target.shape[0] * top_fraction))
    records: list[dict[str, Any]] = []
    for task in range(target.shape[1]):
        if not bool(queryable[task]):
            records.append(
                {
                    "task": task,
                    "queryable": False,
                    "spearman": float("nan"),
                    "ndcg": float("nan"),
                    "topk_recall": float("nan"),
                }
            )
            continue
        predicted = predictions[:, task]
        truth = target[:, task]
        predicted_top = set(torch.topk(predicted, top_count).indices.tolist())
        truth_top = set(torch.topk(truth, top_count).indices.tolist())
        records.append(
            {
                "task": task,
                "queryable": True,
                "spearman": _safe_corr(predicted.numpy(), truth.numpy(), "spearman"),
                "ndcg": _continuous_ndcg(predicted, truth),
                "topk_recall": len(predicted_top & truth_top) / top_count,
            }
        )
    eligible = [record for record in records if record["queryable"]]

    def macro(key: str) -> float:
        finite = [record[key] for record in eligible if math.isfinite(record[key])]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        "dimension": int(task_representation.shape[1]),
        "queryable_tasks": len(eligible),
        "total_tasks": target.shape[1],
        "spearman_mean": macro("spearman"),
        "ndcg_mean": macro("ndcg"),
        "topk_recall_mean": macro("topk_recall"),
        "topk_fraction": top_fraction,
        "per_task": records,
    }


def calibrate_importance_matrices(
    methods: Mapping[str, torch.Tensor],
    *,
    reference_key: str,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Match whole-matrix RMS norms without erasing layer/task structure."""
    if reference_key not in methods:
        raise KeyError(f"unknown reference method {reference_key!r}")
    reference = methods[reference_key].detach().float().cpu()
    reference_rms = float(reference.square().mean().sqrt())
    if not math.isfinite(reference_rms) or reference_rms <= EPS:
        raise ValueError("reference importance must have a finite nonzero RMS")
    calibrated: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name, matrix in methods.items():
        values = matrix.detach().float().cpu()
        if values.shape != reference.shape:
            raise ValueError("all importance matrices must share one shape")
        native_rms = float(values.square().mean().sqrt())
        valid = math.isfinite(native_rms) and native_rms > EPS
        multiplier = reference_rms / native_rms if valid else float("nan")
        calibrated[name] = values * multiplier if valid else values.clone()
        metadata[name] = {
            "valid": valid,
            "native_rms": native_rms,
            "multiplier": multiplier,
            "calibrated_rms": (
                float(calibrated[name].square().mean().sqrt()) if valid else native_rms
            ),
            "reference_key": reference_key,
        }
    return calibrated, metadata


def tensor_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_json(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return tensor_json(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return tensor_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): tensor_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_json(v) for v in value]
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete snapshot, keeping the old result on failed writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                tensor_json(dict(payload)),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class WallTimer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - self.start


class OnlineAccumulator:
    """Task-directed full/late/EMA accumulators kept on the training device."""

    def __init__(
        self,
        n_modules: int,
        representations: Mapping[str, torch.Tensor],
        device: torch.device,
        *,
        beta: float = 0.99,
        late_fraction: float = 0.7,
        total_steps: int,
    ) -> None:
        self.representations = {
            k: normalized_rows(v.to(device=device, dtype=torch.float32))
            for k, v in representations.items()
        }
        self.n_modules = n_modules
        self.device = device
        self.beta = beta
        self.late_start = int(total_steps * late_fraction)
        self.states: dict[str, dict[str, list[torch.Tensor]]] = {}
        self.normalizers: dict[str, float] = {}
        variants = (
            "full_raw",
            "late_raw",
            "ema_raw",
            "late_ema_raw",
            "full_ief",
            "late_ief",
        )
        for variant in variants:
            self.normalizers[variant] = 0.0
            self.states[variant] = {}
            for name, rep in self.representations.items():
                self.states[variant][name] = [
                    torch.zeros(n_modules, rep.shape[1], device=device),
                    torch.zeros(n_modules, device=device),
                ]

    @torch.no_grad()
    def _update_state(
        self,
        variant: str,
        scores: torch.Tensor,
        task: int,
        decay: float | None,
    ) -> None:
        if decay is None:
            self.normalizers[variant] += 1.0
        else:
            self.normalizers[variant] = decay * self.normalizers[variant] + 1.0 - decay
        for name, rep in self.representations.items():
            phi = rep[task]
            numerator, denominator = self.states[variant][name]
            if decay is None:
                numerator.add_(scores[:, None] * phi[None, :])
                denominator.add_(scores)
            else:
                numerator.mul_(decay).add_(
                    scores[:, None] * phi[None, :], alpha=1.0 - decay
                )
                denominator.mul_(decay).add_(scores, alpha=1.0 - decay)

    @torch.no_grad()
    def update(
        self,
        scores: torch.Tensor,
        task: int,
        residual_norm_sq: torch.Tensor | float,
        step: int,
    ) -> None:
        scores = scores.detach().to(dtype=torch.float32)
        residual = torch.as_tensor(
            residual_norm_sq, device=scores.device, dtype=torch.float32
        )
        ief_scores = scores / residual.clamp_min(EPS)
        self._update_state("full_raw", scores, task, None)
        self._update_state("ema_raw", scores, task, self.beta)
        self._update_state("full_ief", ief_scores, task, None)
        if step >= self.late_start:
            self._update_state("late_raw", scores, task, None)
            self._update_state("late_ema_raw", scores, task, self.beta)
            self._update_state("late_ief", ief_scores, task, None)

    def finalize(self) -> dict[str, dict[str, torch.Tensor]]:
        result: dict[str, dict[str, torch.Tensor]] = {}
        for variant, reps in self.states.items():
            result[variant] = {}
            for name, (numerator, denominator) in reps.items():
                result[variant][name] = (
                    (numerator / denominator[:, None].clamp_min(EPS)).detach().cpu()
                )
        return result

    def amplitudes(self) -> dict[str, torch.Tensor]:
        """Return each variant's mean (or bias-corrected EMA) module score."""
        result: dict[str, torch.Tensor] = {}
        first_representation = next(iter(self.representations))
        for variant, reps in self.states.items():
            denominator = reps[first_representation][1]
            scale = max(self.normalizers[variant], EPS)
            result[variant] = (denominator / scale).detach().cpu()
        return result


class PosthocAccumulator:
    def __init__(
        self,
        n_modules: int,
        representations: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        self.representations = {
            k: normalized_rows(v.to(device=device, dtype=torch.float32))
            for k, v in representations.items()
        }
        self.n_modules = n_modules
        self.device = device
        self.stats = {
            statistic: {
                name: [
                    torch.zeros(n_modules, rep.shape[1], device=device),
                    torch.zeros(n_modules, device=device),
                ]
                for name, rep in self.representations.items()
            }
            for statistic in ("ief", "raw", "activation", "actgrad")
        }
        self.count_by_task = torch.zeros(
            next(iter(self.representations.values())).shape[0], device=device
        )

    @torch.no_grad()
    def update(self, task: int, **statistics: torch.Tensor) -> None:
        self.count_by_task[task] += 1
        for statistic, scores in statistics.items():
            scores = scores.detach().to(device=self.device, dtype=torch.float32)
            for name, rep in self.representations.items():
                numerator, denominator = self.stats[statistic][name]
                numerator.add_(scores[:, None] * rep[task][None, :])
                denominator.add_(scores)

    def finalize(
        self,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        result: dict[str, dict[str, torch.Tensor]] = {}
        amplitudes: dict[str, torch.Tensor] = {}
        n_total = self.count_by_task.sum().clamp_min(1)
        for statistic, reps in self.stats.items():
            result[statistic] = {}
            for name, (numerator, denominator) in reps.items():
                result[statistic][name] = (
                    (numerator / denominator[:, None].clamp_min(EPS)).detach().cpu()
                )
            amplitudes[statistic] = (reps[next(iter(reps))][1] / n_total).detach().cpu()
        return result, amplitudes


def query_scores(embedding: torch.Tensor, representation: torch.Tensor) -> torch.Tensor:
    representation = normalized_rows(representation.float().cpu())
    return embedding.float().cpu() @ representation.T


def _safe_corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) < 1e-12 or np.std(b[good]) < 1e-12:
        return float("nan")
    if kind == "spearman":
        return float(spearmanr(a[good], b[good]).statistic)
    return float(pearsonr(a[good], b[good]).statistic)


def embedding_fidelity(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    representation: torch.Tensor,
    *,
    top_fraction: float = 0.05,
) -> dict[str, float]:
    cand = candidate.float().cpu()
    ref = reference.float().cpu()
    row_cos = torch.nn.functional.cosine_similarity(cand, ref, dim=1).numpy()
    cand_scores = query_scores(cand, representation).numpy()
    ref_scores = query_scores(ref, representation).numpy()
    correlations = [
        _safe_corr(cand_scores[:, t], ref_scores[:, t], "spearman")
        for t in range(ref_scores.shape[1])
    ]
    k = max(1, math.ceil(top_fraction * ref.shape[0]))
    overlaps = []
    for task in range(ref_scores.shape[1]):
        a = set(np.argpartition(cand_scores[:, task], -k)[-k:].tolist())
        b = set(np.argpartition(ref_scores[:, task], -k)[-k:].tolist())
        overlaps.append(len(a & b) / k)
    return {
        "module_cosine_mean": float(np.nanmean(row_cos)),
        "module_cosine_median": float(np.nanmedian(row_cos)),
        "task_ranking_spearman_mean": float(np.nanmean(correlations)),
        "topk_overlap_mean": float(np.mean(overlaps)),
        "topk_fraction": top_fraction,
    }


def cluster_fidelity(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    cand_labels = candidate.argmax(dim=1).cpu().numpy()
    ref_labels = reference.argmax(dim=1).cpu().numpy()
    return {
        "adjusted_rand": float(adjusted_rand_score(ref_labels, cand_labels)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(ref_labels, cand_labels)
        ),
    }


def kmeans_cluster_fidelity(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    n_clusters: int,
) -> dict[str, float]:
    candidate_labels = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=0
    ).fit_predict(candidate.float().cpu().numpy())
    reference_labels = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=0
    ).fit_predict(reference.float().cpu().numpy())
    return {
        "adjusted_rand": float(adjusted_rand_score(reference_labels, candidate_labels)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(reference_labels, candidate_labels)
        ),
    }


def layer_balanced_indices(
    scores: torch.Tensor,
    layer_sizes: Sequence[int],
    fraction: float,
) -> list[torch.Tensor]:
    """Top-scoring indices independently within each layer."""
    scores = scores.detach().cpu()
    selected: list[torch.Tensor] = []
    offset = 0
    for size in layer_sizes:
        count = max(1, math.ceil(size * fraction))
        local = torch.topk(scores[offset : offset + size], count).indices
        selected.append(local)
        offset += size
    if offset != scores.numel():
        raise ValueError(f"layer sizes sum to {offset}, expected {scores.numel()}")
    return selected


def random_layer_balanced_scores(layer_sizes: Sequence[int], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.cat([torch.rand(size, generator=generator) for size in layer_sizes])


@dataclass
class TangentModule:
    name: str
    gradients: torch.Tensor  # [samples, parameter-dimension]


def tangent_kernel_metrics(
    modules: Sequence[TangentModule],
    task_ids: torch.Tensor,
    representations: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Compare task kernels with exact cosine kernels and amplitude kernels."""
    task_ids = task_ids.cpu().long()
    n = task_ids.numel()
    upper = torch.triu_indices(n, n, offset=1)
    exact_cosines: list[np.ndarray] = []
    exact_kernels: list[torch.Tensor] = []
    amplitudes: list[torch.Tensor] = []
    for module in modules:
        q = module.gradients.float().cpu()
        kernel = q @ q.T
        norms = q.norm(dim=1).clamp_min(EPS)
        cosine = kernel / (norms[:, None] * norms[None, :])
        exact_cosines.append(cosine[upper[0], upper[1]].numpy())
        exact_kernels.append(kernel)
        amplitudes.append(norms)
    exact_flat = np.concatenate(exact_cosines)
    output: dict[str, dict[str, float]] = {}
    for name, rep in representations.items():
        phi = normalized_rows(rep.float().cpu())[task_ids]
        task_kernel = phi @ phi.T
        pred_flat = np.tile(task_kernel[upper[0], upper[1]].numpy(), len(modules))
        errors = []
        for kernel, amp in zip(exact_kernels, amplitudes):
            approx = amp[:, None] * task_kernel * amp[None, :]
            errors.append(
                float((kernel - approx).norm() / kernel.norm().clamp_min(EPS))
            )
        output[name] = {
            "rho_pearson": _safe_corr(pred_flat, exact_flat, "pearson"),
            "rho_spearman": _safe_corr(pred_flat, exact_flat, "spearman"),
            "relative_frobenius_mean": float(np.mean(errors)),
        }
    const = np.ones_like(exact_flat)
    output["constant"] = {
        "rho_pearson": _safe_corr(const, exact_flat, "pearson"),
        "rho_spearman": _safe_corr(const, exact_flat, "spearman"),
        "relative_frobenius_mean": float("nan"),
    }
    return output


def select_evenly(layer_sizes: Sequence[int], per_layer: int) -> list[int]:
    indices: list[int] = []
    offset = 0
    for size in layer_sizes:
        local = np.linspace(0, size - 1, min(per_layer, size), dtype=int)
        indices.extend((local + offset).tolist())
        offset += size
    return indices
