import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from kpsa.common import (
    OnlineAccumulator,
    accelerator_peak_memory,
    arc_artifact_dir,
    build_task_atlas,
    calibrate_importance_matrices,
    conditional_importance,
    cross_validated_task_query_fidelity,
    cross_validated_task_scores,
    jl_task_representation,
    kmeans_cluster_fidelity,
    layer_balanced_indices,
    normalized_rows,
    save_json,
    source_fidelity,
    task_aligned_importance,
    tensor_json,
)


class CommonTests(unittest.TestCase):
    def test_json_publish_is_atomic_and_failed_replace_preserves_previous_result(self):
        with TemporaryDirectory() as root:
            target = Path(root) / "result.json"
            save_json(target, {"step": 1})
            previous = target.read_text()
            with (
                patch("kpsa.common.os.replace", side_effect=OSError),
                self.assertRaises(OSError),
            ):
                save_json(target, {"step": 2})
            self.assertEqual(target.read_text(), previous)
            self.assertEqual(list(Path(root).iterdir()), [target])

    def test_tensor_json_replaces_nonfinite_values_with_null(self):
        self.assertEqual(
            tensor_json({"values": torch.tensor([float("nan"), float("inf"), 1.0])}),
            {"values": [None, None, 1.0]},
        )

    def test_cpu_peak_memory_record_is_explicitly_zero(self):
        self.assertEqual(
            accelerator_peak_memory(torch.device("cpu")),
            {
                "peak_cuda_allocated_bytes": 0,
                "peak_cuda_reserved_bytes": 0,
            },
        )

    def test_kmeans_cluster_fidelity_is_one_for_identical_embeddings(self):
        embedding = torch.tensor([[-2.0, 0.0], [-1.5, 0.0], [1.5, 0.0], [2.0, 0.0]])
        metrics = kmeans_cluster_fidelity(embedding, embedding, n_clusters=2)
        self.assertAlmostEqual(metrics["adjusted_rand"], 1.0)
        self.assertAlmostEqual(metrics["normalized_mutual_info"], 1.0)

    def test_layer_balancing(self):
        scores = torch.arange(10.0)
        selected = layer_balanced_indices(scores, [4, 6], 0.25)
        self.assertEqual(selected[0].tolist(), [3])
        self.assertEqual(set(selected[1].tolist()), {4, 5})

    def test_online_embedding_is_weighted_task_mean(self):
        rep = torch.eye(2)
        acc = OnlineAccumulator(
            1, {"onehot": rep}, torch.device("cpu"), total_steps=2, late_fraction=0
        )
        acc.update(torch.tensor([1.0]), 0, 1.0, 0)
        acc.update(torch.tensor([3.0]), 1, 1.0, 1)
        got = acc.finalize()["full_raw"]["onehot"][0]
        torch.testing.assert_close(got, torch.tensor([0.25, 0.75]))

    def test_online_accumulator_exposes_mean_module_amplitude(self):
        acc = OnlineAccumulator(
            2,
            {"onehot": torch.eye(2)},
            torch.device("cpu"),
            total_steps=2,
            late_fraction=0,
        )
        acc.update(torch.tensor([1.0, 2.0]), 0, 1.0, 0)
        acc.update(torch.tensor([3.0, 6.0]), 1, 1.0, 1)
        torch.testing.assert_close(
            acc.amplitudes()["full_raw"], torch.tensor([2.0, 4.0])
        )

    def test_mse_normalization_recovers_jacobian_row_norm(self):
        layer = torch.nn.Linear(3, 1, bias=False)
        x = torch.tensor([[1.0, -2.0, 0.5]])
        y = torch.tensor([3.0])
        pred = layer(x).squeeze()
        residual = pred - y
        loss = 0.5 * residual.square()
        loss.backward()
        normalized = layer.weight.grad.square().sum() / residual.detach().square()
        torch.testing.assert_close(normalized.squeeze(), x.square().sum())

    def test_normalized_rows(self):
        x = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(normalized_rows(x), torch.tensor([[0.6, 0.8]]))

    def test_arc_artifacts_resolve_to_internal_sibling(self):
        project = Path("/tmp/workspace/kpsa")
        got = arc_artifact_dir("02_exploratory", "controlled", project_root=project)
        self.assertEqual(
            got,
            Path(
                "/tmp/workspace/kpsa-internal/02_exploratory/artifacts/controlled"
            ),
        )

    def test_build_task_atlas_uses_per_task_means_before_row_normalization(self):
        # Module 0 means are [2, 1]; module 1 means are [1, 3]. The unequal
        # sample counts must not turn these into prevalence-weighted profiles.
        sensitivities = torch.tensor([[1.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
        tasks = torch.tensor([0, 0, 1])
        atlas, amplitude = build_task_atlas(sensitivities, tasks, n_tasks=2)
        torch.testing.assert_close(
            atlas, torch.tensor([[2 / 3, 1 / 3], [1 / 4, 3 / 4]])
        )
        torch.testing.assert_close(amplitude, torch.tensor([1.5, 2.0]))

    def test_source_fidelity_is_exact_for_onehot(self):
        atlas = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
        metrics = source_fidelity(
            atlas, torch.eye(2), k_neighbors=1, top_fraction=1 / 3
        )
        self.assertAlmostEqual(metrics["distance_distortion_mean"], 0.0, places=6)
        self.assertAlmostEqual(metrics["centered_kernel_alignment"], 1.0, places=6)
        self.assertAlmostEqual(metrics["knn_preservation"], 1.0, places=6)
        self.assertAlmostEqual(metrics["ranking_spearman_mean"], 1.0, places=6)
        self.assertAlmostEqual(metrics["topk_overlap_mean"], 1.0, places=6)

    def test_source_fidelity_scales_by_sampling_module_pairs(self):
        generator = torch.Generator().manual_seed(4)
        atlas = torch.rand(6000, 4, generator=generator)
        atlas = atlas / atlas.sum(dim=1, keepdim=True)
        metrics = source_fidelity(
            atlas,
            torch.eye(4),
            max_pairs=10_000,
            max_knn_queries=32,
        )
        self.assertEqual(metrics["distance_pairs"], 10_000)
        self.assertEqual(metrics["knn_queries"], 32)
        self.assertAlmostEqual(metrics["centered_kernel_alignment"], 1.0, places=5)

    def test_jl_projection_is_reproducible_and_has_requested_shape(self):
        first = jl_task_representation(10, 4, seed=7)
        second = jl_task_representation(10, 4, seed=7)
        self.assertEqual(first.shape, (10, 4))
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(first).all())

    def test_task_aligned_importance_keeps_amplitude_separate(self):
        atlas = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        amplitude = torch.tensor([2.0, 4.0])
        got = task_aligned_importance(atlas, torch.eye(2), amplitude)
        torch.testing.assert_close(got, torch.tensor([[1.5, 0.5], [1.0, 3.0]]))

    def test_cross_validated_query_never_uses_heldout_atlas_column(self):
        importance = torch.tensor([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [1.0, 1.0, 3.0]])
        representation = normalized_rows(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        first, queryable = cross_validated_task_scores(
            importance, representation, folds=[[2]]
        )
        changed = importance.clone()
        changed[:, 2] = torch.tensor([999.0, -999.0, 500.0])
        second, _ = cross_validated_task_scores(changed, representation, folds=[[2]])
        torch.testing.assert_close(first[:, 2], second[:, 2])
        self.assertTrue(queryable[2])

    def test_onehot_cannot_query_a_task_whose_column_was_withheld(self):
        importance = torch.arange(12.0).reshape(4, 3)
        predictions, queryable = cross_validated_task_scores(
            importance, torch.eye(3), folds=[[0], [1], [2]]
        )
        self.assertFalse(bool(queryable.any()))
        self.assertTrue(bool(torch.isnan(predictions).all()))

    def test_cross_validated_query_metrics_reward_known_task_structure(self):
        representation = normalized_rows(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        importance = torch.tensor(
            [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0], [1.0, 1.0, 2.0], [3.0, 0.0, 3.0]]
        )
        metrics = cross_validated_task_query_fidelity(
            importance, representation, folds=[[2]], top_fraction=0.5
        )
        self.assertEqual(metrics["queryable_tasks"], 1)
        self.assertGreater(metrics["spearman_mean"], 0.9)
        self.assertAlmostEqual(metrics["topk_recall_mean"], 1.0)
        self.assertGreater(metrics["ndcg_mean"], 0.95)

    def test_cross_validated_metrics_can_use_an_independent_target_split(self):
        source = torch.tensor(
            [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0], [1.0, 1.0, 2.0], [3.0, 0.0, 3.0]]
        )
        target = source.clone()
        target[:, 2] = torch.tensor([0.0, 0.0, 10.0, 0.0])
        representation = normalized_rows(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        same_split = cross_validated_task_query_fidelity(
            source, representation, folds=[[2]], top_fraction=0.25
        )
        independent = cross_validated_task_query_fidelity(
            source,
            representation,
            folds=[[2]],
            top_fraction=0.25,
            target_importance=target,
        )
        self.assertEqual(same_split["topk_recall_mean"], 1.0)
        self.assertEqual(independent["topk_recall_mean"], 0.0)

    def test_conditional_importance_inverts_balanced_atlas_normalization(self):
        atlas = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        amplitude = torch.tensor([2.0, 4.0])
        torch.testing.assert_close(
            conditional_importance(atlas, amplitude),
            torch.tensor([[3.0, 1.0], [2.0, 6.0]]),
        )

    def test_global_importance_calibration_preserves_ratios(self):
        methods = {
            "reference": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "scaled": torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
            "zero": torch.zeros(2, 2),
        }
        calibrated, metadata = calibrate_importance_matrices(
            methods, reference_key="reference"
        )
        torch.testing.assert_close(calibrated["reference"], calibrated["scaled"])
        self.assertAlmostEqual(
            calibrated["scaled"][1, 1] / calibrated["scaled"][0, 0], 4.0
        )
        self.assertFalse(metadata["zero"]["valid"])
        self.assertEqual(float(calibrated["zero"].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
