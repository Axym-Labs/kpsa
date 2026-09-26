import importlib.util
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from task_embeddings import optimizer_experiment
from task_embeddings.controlled_v4 import HardControlledConfig, build_model


class OptimizerExperimentTests(unittest.TestCase):
    def test_optimizer_experiment_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("task_embeddings.optimizer_experiment")
        )

    def test_tiny_run_executes_full_tbe_jl_and_both_estimators(self):
        model_config = HardControlledConfig(
            seed=5,
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

            result = optimizer_experiment.run_optimizer_experiment(
                checkpoint,
                optimizer_experiment.OptimizerExperimentConfig(
                    seed=5,
                    reference_per_task=1,
                    test_per_task=1,
                    batch_size=4,
                    steps=1,
                    task_indices=(0,),
                    corruption_std=0.001,
                ),
                device=torch.device("cpu"),
            )

        methods = {record["method"] for record in result["records"]}
        self.assertEqual(len(methods), 9)
        self.assertIn("raw_opg/tbe", methods)
        self.assertIn("residual_normalized_opg/jl", methods)
        self.assertIn("adamw", methods)
        self.assertTrue(result["task_homogeneous_steps"])
        self.assertTrue(result["requires_distinct_task_queries"])
        self.assertEqual(
            result["preconditioner_calibration"],
            "match_weighted_profile_mean_to_current_gradient_second_moment",
        )
        self.assertEqual(result["resource_accounting"]["tbe_preconditioner"], 242)
        self.assertIn("corruption_damage", result["premise_gate"])
        self.assertIn("adamw_recovery", result["premise_gate"])
        self.assertIn("passed", result["premise_gate"])
        for record in result["records"]:
            self.assertIn("clean_target_half_mse", record)
            self.assertIn("recoverable_damage", record)
            self.assertIn("damage_recovery_fraction", record)


if __name__ == "__main__":
    unittest.main()
