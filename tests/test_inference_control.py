import importlib.util
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from task_embeddings import inference_control
from task_embeddings.controlled_v4 import HardControlledConfig, build_model


class InferenceControlTests(unittest.TestCase):
    def test_inference_control_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("task_embeddings.inference_control")
        )

    def test_task_gate_matrix_keeps_exact_global_budget_per_task(self):
        scores = torch.tensor([[4.0, 1.0], [3.0, 2.0], [2.0, 3.0], [1.0, 4.0]])

        gates = inference_control.task_gate_matrix(scores, retained_fraction=0.5)

        torch.testing.assert_close(gates.sum(dim=0), torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(gates[:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(gates[:, 1], torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_control_metrics_report_target_recovery_and_spillover(self):
        metrics = inference_control.normalized_control_metrics(
            baseline_losses=torch.tensor([1.0, 2.0]),
            null_losses=torch.tensor([5.0, 6.0]),
            controlled_losses=torch.tensor([2.0, 4.0]),
            target_task=0,
        )

        self.assertAlmostEqual(metrics["target_quality_recovery"], 0.75)
        self.assertAlmostEqual(metrics["off_target_spillover"], 0.5)

    def test_split_feature_gate_preserves_global_module_order(self):
        pieces = inference_control.split_feature_gate(
            torch.arange(6, dtype=torch.float32), [2, 1, 3]
        )

        self.assertEqual(
            [piece.tolist() for piece in pieces], [[0.0, 1.0], [2.0], [3.0, 4.0, 5.0]]
        )

    def test_tiny_run_executes_regular_grid_and_identity_gate(self):
        model_config = HardControlledConfig(
            seed=7,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=8,
            train_steps=1,
            refinement_steps=1,
        )
        model = build_model(model_config, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "configuration": asdict(model_config),
                    "states": {"trained": model.state_dict()},
                },
                checkpoint,
            )

            result = inference_control.run_inference_control(
                checkpoint,
                inference_control.InferenceControlConfig(
                    seed=7,
                    reference_per_task=1,
                    test_per_task=1,
                    retained_fractions=(1.0,),
                ),
                device=torch.device("cpu"),
            )

        self.assertEqual(len(result["methods"]), 8)
        self.assertLess(result["identity_max_abs_loss_error"], 1e-7)
        self.assertTrue(result["requires_distinct_task_queries"])
        self.assertEqual(result["mechanism"], "oracle_task_mlp_feature_gating")
        self.assertFalse(result["execution"]["claims_dense_flop_reduction"])


if __name__ == "__main__":
    unittest.main()
