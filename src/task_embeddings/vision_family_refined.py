"""WordNet-family and semantic-direction queries for ImageNet sensitivity."""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json
from .qwen_sensitivity import cold_fold_predictions
from .representation_sensitivity import profile_ranking_metrics


def read_wordnet_nouns(archive: Path):
    parents, names = {}, {}
    with zipfile.ZipFile(archive) as handle:
        lines = handle.read("wordnet/data.noun").decode().splitlines()
    for line in lines:
        if not line or line[0].isspace():
            continue
        fields = line.split()
        offset = int(fields[0])
        names[offset] = fields[4]
        word_count = int(fields[3], 16)
        cursor = 4 + 2 * word_count
        pointer_count = int(fields[cursor])
        cursor += 1
        local = []
        for _ in range(pointer_count):
            symbol, target, part, _source = fields[cursor : cursor + 4]
            cursor += 4
            if symbol in {"@", "@i"} and part == "n":
                local.append(int(target))
        parents[offset] = local
    return parents, names


def longest_paths(nodes: list[int], parents: dict[int, list[int]]):
    @lru_cache(None)
    def path(node: int) -> tuple[int, ...]:
        local = parents.get(node, [])
        if not local:
            return (node,)
        return max((path(parent) for parent in local), key=len) + (node,)

    return [path(node) for node in nodes]


def adaptive_family_ids(paths: list[tuple[int, ...]], maximum_size: int = 15):
    """Choose the shallowest non-singleton ancestor under a size ceiling."""
    counts = Counter(ancestor for path in paths for ancestor in path)
    families = []
    for path in paths:
        candidates = [
            ancestor for ancestor in path if 2 <= counts[ancestor] <= maximum_size
        ]
        families.append(candidates[0] if candidates else path[-1])
    return families


def family_tensors(profiles, features, families):
    kept = [family for family, count in Counter(families).items() if count >= 2]
    family_profiles, family_features, members = [], [], []
    for family in kept:
        indices = torch.tensor([i for i, value in enumerate(families) if value == family])
        family_profiles.append(profiles[:, indices].mean(1))
        family_features.append(F.normalize(features[indices].mean(0), dim=0))
        members.append(indices.tolist())
    return torch.stack(family_profiles, 1), torch.stack(family_features), kept, members


def pair_holdout_direction_predictions(source, features, combinations, *, seed=23_041):
    """Predict contrasts with both endpoint families excluded together."""
    semantic = F.normalize(features.detach().double().cpu(), dim=1)
    affine = F.normalize(torch.cat((torch.ones(len(semantic), 1), semantic), 1), dim=1)
    distances = torch.cdist(semantic, semantic).square()
    positive = distances[distances > 0]
    bandwidth = positive.median() if len(positive) else torch.tensor(1.0)
    kernel = torch.exp(-distances / (2 * bandwidth.clamp_min(1e-12)))
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    rbf = eigenvectors * eigenvalues.clamp_min(0).sqrt()[None]
    generator = torch.Generator().manual_seed(seed)
    jl = F.normalize(torch.randn(semantic.shape, generator=generator, dtype=torch.float64), dim=1)
    permuted = semantic[torch.randperm(len(semantic), generator=generator)]
    representations = {
        "semantic": semantic,
        "affine_semantic": affine,
        "rbf_semantic": rbf,
        "jl": jl,
        "permuted": permuted,
    }
    estimates = {name: [] for name in (*representations, "scalar_mass", "nearest")}
    for left, right in combinations.tolist():
        observed = [
            index for index in range(source.shape[1]) if index not in {left, right}
        ]
        for name, representation in representations.items():
            joint = source[:, observed] @ representation[observed] / len(observed)
            estimates[name].append(joint @ (representation[left] - representation[right]))
        estimates["scalar_mass"].append(torch.zeros(source.shape[0], dtype=source.dtype))
        similarities = semantic[[left, right]] @ semantic[observed].T
        nearest = torch.as_tensor(observed)[similarities.argmax(1)]
        estimates["nearest"].append(source[:, nearest[0]] - source[:, nearest[1]])
    estimates["source_onehot_reference"] = [
        source[:, int(left)] - source[:, int(right)] for left, right in combinations
    ]
    return {method: torch.stack(values, 1) for method, values in estimates.items()}


def pair_holdout_direction_metrics(source, target, features, *, seed=23_041, pairs=64):
    """Score pair-holdout semantic directions against measured contrasts."""
    combinations = torch.combinations(torch.arange(target.shape[1]), r=2)
    order = torch.randperm(len(combinations), generator=torch.Generator().manual_seed(seed))
    combinations = combinations[order[: min(pairs, len(order))]]
    truth = torch.stack(
        [target[:, int(left)] - target[:, int(right)] for left, right in combinations],
        1,
    )
    estimates = pair_holdout_direction_predictions(
        source, features, combinations, seed=seed
    )
    estimates["direct_gradient_oracle"] = truth
    result = {
        method: profile_ranking_metrics(values, truth)
        for method, values in estimates.items()
    }
    return result, combinations.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wordnet", type=Path, default=Path("/home/davwis/nltk_data/corpora/wordnet.zip")
    )
    parser.add_argument("--maximum-family-size", type=int, default=15)
    args = parser.parse_args()

    import timm

    payload = torch.load(args.profiles, weights_only=True, mmap=True)
    synset_file = Path(timm.__file__).parent / "data/_info/imagenet_synsets.txt"
    synsets = synset_file.read_text().splitlines()
    selected = payload["selected_class_indices"].tolist()
    nodes = [int(synsets[index][1:]) for index in selected]
    parents, names = read_wordnet_nouns(args.wordnet)
    paths = longest_paths(nodes, parents)
    families = adaptive_family_ids(paths, args.maximum_family_size)
    features = payload["feature_maps"]["frozen_input_encoder"].double()
    studies = {}
    metadata = None
    for functional in ("class_logit", "margin", "loss"):
        source, family_features, family_nodes, members = family_tensors(
            payload[functional]["source_profiles"].double(), features, families
        )
        target, _, _, _ = family_tensors(
            payload[functional]["target_profiles"].double(), features, families
        )
        predictions = cold_fold_predictions(source, family_features, seed=23_041, folds=4)
        predictions["source_onehot_reference"] = source
        predictions["direct_gradient_oracle"] = target
        absolute = {
            method: profile_ranking_metrics(values, target)
            for method, values in predictions.items()
        }
        directions, pairs = pair_holdout_direction_metrics(
            source, target, family_features
        )
        studies[functional] = {
            "absolute_family_queries": absolute,
            "family_direction_queries": directions,
        }
        if metadata is None:
            metadata = {
                "family_nodes": family_nodes,
                "family_names": [names.get(node, str(node)) for node in family_nodes],
                "member_columns": members,
                "member_class_indices": [[selected[i] for i in local] for local in members],
                "direction_pairs": pairs,
            }
    result = {
        "protocol": {
            "taxonomy": "WordNet noun hypernyms for official ImageNet-1k synsets",
            "family_rule": "shallowest ancestor with 2..maximum_family_size selected descendants",
            "maximum_family_size": args.maximum_family_size,
            "singleton_families_excluded": True,
            "representation": "full normalized frozen DINOv2-Small class means",
            "cold_folds": 4,
        },
        "families": metadata,
        "studies": studies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
