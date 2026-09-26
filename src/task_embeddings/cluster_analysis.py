from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import arc_artifact_dir, kmeans_cluster_fidelity, save_json


def compare(path: Path, representation: str, n_clusters: int) -> dict:
    arrays = np.load(path)
    reference = torch.from_numpy(arrays[f"post_ief_{representation}"])
    return {
        variant: kmeans_cluster_fidelity(
            torch.from_numpy(arrays[f"online_{variant}_{representation}"]),
            reference,
            n_clusters,
        )
        for variant in ("late_raw", "late_ief")
    }


def run(args: argparse.Namespace) -> None:
    root = Path(args.artifact_dir)
    payload = {
        "controlled_seed1": compare(
            root / "controlled/controlled_embeddings_seed1.npz", "latent4", 4
        ),
        "vision_seed1": compare(
            root / "vision_corrected/vision_embeddings_seed1.npz", "jl32", 20
        ),
        "vision_seed2": compare(
            root / "vision_corrected/vision_embeddings_seed2.npz", "jl32", 20
        ),
        "language_seed1": compare(
            root / "language/language_embeddings_seed1.npz", "semantic4", 6
        ),
    }
    output = root / "cluster_fidelity.json"
    save_json(output, payload)
    print(f"CLUSTER_RESULT={output}")
    print(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default=str(arc_artifact_dir("01_exploratory"))
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
