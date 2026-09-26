"""Stability and semantic-structure diagnostics for a sample-level vision atlas."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from .common import save_json
from .domain_optimizer import ParameterPartition
from .representation_sensitivity import profile_ranking_metrics
from .vision_causal_refined import mlp_feature_layout, mlp_feature_scope
from .vision_sample_refined import sample_predictions


def _means(metrics):
    return {
        key: metrics[f"mean_{key}"]
        for key in ("spearman", "topk_recall", "ndcg", "cosine")
    }


def _bootstrap(values, *, draws=10_000, seed=23_041):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.choice(values, (draws, len(values)), replace=True).mean(1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "n": len(values),
    }


def _subset_columns(samples_per_class, replicate, *, classes=200, available=4):
    start = replicate * samples_per_class
    offsets = torch.arange(start, start + samples_per_class)
    return (torch.arange(classes)[:, None] * available + offsets[None]).flatten()


def atlas_size_diagnostics(source_profiles, source_features, target_features, target_profiles):
    rows = []
    predictions = {}
    replicates = {1: 4, 2: 2, 4: 1}
    method_names = {
        "sample_dinov2_rbf_scale_0.1": "dinov2_rbf",
        "sample_nearest": "nearest_sample",
        "scalar_mass": "scalar_mass",
    }
    for samples_per_class, count in replicates.items():
        for replicate in range(count):
            columns = _subset_columns(samples_per_class, replicate)
            current = sample_predictions(
                source_profiles[:, columns],
                source_features[columns],
                target_features,
                rbf_scales=(0.1,),
            )
            predictions[(samples_per_class, replicate)] = current
            for source_name, output_name in method_names.items():
                metrics = profile_ranking_metrics(
                    current[source_name], target_profiles
                )
                rows.append(
                    {
                        "samples_per_class": samples_per_class,
                        "replicate": replicate,
                        "method": output_name,
                        **_means(metrics),
                    }
                )
    stability = []
    for samples_per_class, count in replicates.items():
        for first, second in combinations(range(count), 2):
            for source_name, output_name in method_names.items():
                metrics = profile_ranking_metrics(
                    predictions[(samples_per_class, first)][source_name],
                    predictions[(samples_per_class, second)][source_name],
                )
                stability.append(
                    {
                        "samples_per_class": samples_per_class,
                        "replicate_pair": [first, second],
                        "method": output_name,
                        **_means(metrics),
                    }
                )
    return rows, stability


def semantic_structure(
    source_profiles,
    source_features,
    scope,
    selected_class_indices,
    *,
    groups=10_000,
    clusters=32,
    seed=23_041,
):
    source_profiles = source_profiles.float()
    source_features = F.normalize(source_features.float(), dim=1)
    class_scores = source_profiles.reshape(len(source_profiles), 200, 4).mean(2)
    mass = class_scores.mean(1)
    eligible = scope.nonzero().flatten()
    chosen = eligible[torch.topk(mass[eligible], min(groups, len(eligible))).indices]
    local_scores = class_scores[chosen]
    local_distribution = local_scores / local_scores.sum(1, keepdim=True).clamp_min(1e-30)
    entropy = -(local_distribution * local_distribution.clamp_min(1e-30).log()).sum(1)
    effective_classes = entropy.exp()
    top_share = local_distribution.max(1).values

    class_features = F.normalize(source_features.reshape(200, 4, -1).mean(1), dim=1)
    top_classes = torch.topk(local_scores, 5, dim=1).indices
    top_features = class_features[top_classes]
    similarity = top_features @ top_features.transpose(1, 2)
    triangle = torch.triu_indices(5, 5, offset=1)
    coherence = similarity[:, triangle[0], triangle[1]].mean(1)
    generator = torch.Generator().manual_seed(seed)
    random_classes = torch.rand(len(chosen), 200, generator=generator).topk(5, dim=1).indices
    random_features = class_features[random_classes]
    random_similarity = random_features @ random_features.transpose(1, 2)
    random_coherence = random_similarity[:, triangle[0], triangle[1]].mean(1)

    group_embedding = F.normalize(local_scores @ class_features, dim=1)
    nearest_classes = (group_embedding @ class_features.T).topk(5, dim=1).indices
    top_one_match = nearest_classes[:, 0].eq(top_classes[:, 0]).float()
    top_five_match = nearest_classes.eq(top_classes[:, :1]).any(1).float()

    embedding_np = group_embedding.numpy()
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        batch_size=2048,
        n_init=10,
        random_state=seed,
    ).fit(embedding_np)
    labels = torch.from_numpy(kmeans.labels_)
    cluster_rows = []
    for cluster in range(clusters):
        members = labels.eq(cluster)
        centroid = F.normalize(group_embedding[members].mean(0), dim=0)
        nearest = (class_features @ centroid).topk(5).indices
        cluster_rows.append(
            {
                "cluster": cluster,
                "groups": int(members.sum()),
                "mean_cosine_to_centroid": float(
                    (group_embedding[members] @ centroid).mean()
                ),
                "nearest_class_indices": selected_class_indices[nearest].tolist(),
            }
        )
    return {
        "selection": {
            "groups": len(chosen),
            "rule": "largest scalar mass among coupled MLP-feature groups",
        },
        "effective_class_count": {
            "mean": float(effective_classes.mean()),
            "median": float(effective_classes.median()),
            "q10": float(effective_classes.quantile(0.1)),
            "q90": float(effective_classes.quantile(0.9)),
        },
        "largest_class_share": {
            "mean": float(top_share.mean()),
            "median": float(top_share.median()),
        },
        "top5_class_semantic_coherence": {
            "observed": _bootstrap(coherence.numpy(), seed=seed),
            "random_class_sets": _bootstrap(random_coherence.numpy(), seed=seed + 1),
            "paired_difference": _bootstrap(
                (coherence - random_coherence).numpy(), seed=seed + 2
            ),
        },
        "embedding_nearest_class_alignment": {
            "top1_sensitivity_class_is_nearest": float(top_one_match.mean()),
            "top1_sensitivity_class_is_in_nearest5": float(top_five_match.mean()),
        },
        "clusters": cluster_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k"
    )
    args = parser.parse_args()
    import timm

    payload = torch.load(args.atlas, map_location="cpu", weights_only=True)
    retained = torch.load(args.profiles, map_location="cpu", weights_only=True)
    model = timm.create_model(args.model, pretrained=False).eval()
    partition = ParameterPartition(model, "swiglu")
    if partition.n_groups != len(payload["source_profiles"]):
        raise ValueError("atlas and model partition have different group counts")
    scope = mlp_feature_scope(partition, mlp_feature_layout(model, partition))
    rows, stability = atlas_size_diagnostics(
        payload["source_profiles"],
        payload["source_features"],
        payload["target_features"],
        payload["target_profiles"],
    )
    result = {
        "protocol": {
            "atlas_samples_per_class": [1, 2, 4],
            "classes": 200,
            "rbf_scale": 0.1,
            "stability": "independent within-class atlas subsets",
        },
        "atlas_size_retrieval": rows,
        "independent_subset_stability": stability,
        "semantic_structure": semantic_structure(
            payload["source_profiles"],
            payload["source_features"],
            scope,
            retained["selected_class_indices"],
        ),
    }
    save_json(args.output, result)


if __name__ == "__main__":
    main()
