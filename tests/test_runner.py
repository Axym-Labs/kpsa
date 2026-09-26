import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from task_embeddings import runner
from task_embeddings.continual_v4 import ContinualConfig


class RunnerTests(unittest.TestCase):
    def test_optimizer_output_paths_separate_methods_features_and_data_roles(self):
        variations = (
            {},
            {"method": "tbe"},
            {"feature_file": "token_features.pt"},
            {"role": "test"},
            {"data": Path("another_corpus")},
        )
        outputs = [
            runner.build_experiment_plan(
                "optimizer", "explore", seed=11, **({"data": Path("data")} | change)
            ).output
            for change in variations
        ]
        self.assertEqual(len(set(outputs)), len(variations))

    def test_optimizer_honors_selected_task_feature_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"features": torch.tensor([2.0])}, root / "task_features.pt")
            torch.save({"features": torch.tensor([7.0])}, root / "token_features.pt")
            plan = runner.build_experiment_plan(
                "optimizer",
                "explore",
                seed=11,
                data=root,
                feature_file="token_features.pt",
            )
            plan = replace(plan, output=root / "result.json")

            def train(corpus, features, output, config, **kwargs):
                return {
                    "feature_sum": float(
                        torch.load(features, weights_only=True)["features"].sum()
                    )
                }

            with patch("task_embeddings.domain_train.train", side_effect=train):
                result = runner.execute_experiment_plan(
                    plan, device=torch.device("cpu")
                )
            self.assertEqual(result["feature_sum"], 7.0)

    def test_runner_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("task_embeddings.runner"))

    def test_continual_profiles_change_scale_without_changing_protocol(self):
        explore = runner.build_experiment_plan(
            "continual", "explore", seed=7, order="forward"
        )
        paper = runner.build_experiment_plan(
            "continual", "paper", seed=7, order="forward"
        )

        self.assertEqual(explore.configuration.seed, paper.configuration.seed)
        self.assertLess(
            explore.configuration.steps_per_task,
            paper.configuration.steps_per_task,
        )
        self.assertLess(explore.configuration.n_modules, paper.configuration.n_modules)
        self.assertEqual(explore.control_repeats, 1)
        self.assertGreaterEqual(paper.control_repeats, 3)
        self.assertFalse(explore.claim_ready)
        self.assertFalse(paper.claim_ready)
        self.assertGreaterEqual(explore.configuration.n_tasks, 10)
        self.assertGreaterEqual(explore.configuration.steps_per_task, 40)

    def test_checkpoint_experiments_share_one_profile_interface(self):
        checkpoint = Path("model.pt")
        for experiment in (
            "query",
            "causal",
            "streaming",
            "inference_control",
            "optimizer_recovery",
        ):
            plan = runner.build_experiment_plan(
                experiment,
                "explore",
                seed=2,
                checkpoint=checkpoint,
            )
            self.assertEqual(plan.checkpoint, checkpoint)
            self.assertEqual(plan.profile, "explore")
            self.assertFalse(plan.claim_ready)

    def test_optimizer_default_is_scratch_and_rejects_checkpoint(self):
        plan = runner.build_experiment_plan(
            "optimizer",
            "explore",
            seed=2,
            data=Path("data"),
        )
        self.assertIsNone(plan.checkpoint)
        self.assertEqual(plan.configuration.task_mode, "multi")
        self.assertGreaterEqual(plan.configuration.steps, 2000)
        with self.assertRaises(ValueError):
            runner.build_experiment_plan(
                "optimizer",
                "explore",
                seed=2,
                data=Path("data"),
                checkpoint=Path("model.pt"),
            )

    def test_domain_application_profiles_keep_data_roles_explicit(self):
        plan = runner.build_experiment_plan(
            "domain_applications",
            "paper",
            seed=2,
            data=Path("data"),
            checkpoint=Path("model.pt"),
        )
        self.assertEqual(plan.configuration["role"], "validation")
        self.assertFalse(plan.claim_ready)

    def test_precision_run_can_disable_pruning_without_changing_other_budgets(self):
        plan = runner.build_experiment_plan(
            "domain_applications",
            "explore",
            seed=2,
            data=Path("data"),
            checkpoint=Path("model.pt"),
            skip_pruning=True,
        )
        self.assertEqual(plan.configuration["pruning_fractions"], ())
        self.assertEqual(plan.configuration["role"], "validation")

    def test_modern_language_control_uses_checkpoint_and_profile_archive(self):
        plan = runner.build_experiment_plan(
            "language_inference_control",
            "explore",
            seed=1,
            checkpoint=Path("qwen.pt"),
            profiles=Path("profiles.npz"),
        )

        self.assertEqual(plan.checkpoint, Path("qwen.pt"))
        self.assertEqual(plan.profiles, Path("profiles.npz"))
        self.assertEqual(plan.configuration.evaluation_start, 2)
        self.assertLess(plan.configuration.evaluation_per_task, 6)

        paper = runner.build_experiment_plan(
            "language_inference_control",
            "paper",
            seed=1,
            checkpoint=Path("qwen.pt"),
            profiles=Path("profiles.npz"),
        )
        self.assertFalse(paper.claim_ready)

    def test_paper_plan_writes_to_paper_arc_and_explore_to_current_arc(self):
        explore = runner.build_experiment_plan(
            "continual", "explore", seed=1, order="reverse"
        )
        paper = runner.build_experiment_plan(
            "continual", "paper", seed=1, order="reverse"
        )

        self.assertIn("04_queryable_mechanisms", str(explore.output))
        self.assertIn("kpsa-paper-internal", str(paper.output))

    def test_execute_plan_records_profile_and_writes_result(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = runner.build_experiment_plan(
                "continual", "explore", seed=4, order="forward"
            )
            plan = replace(
                plan,
                configuration=ContinualConfig(
                    seed=4,
                    input_dim=4,
                    n_tasks=3,
                    n_modules=8,
                    batch_size=4,
                    steps_per_task=1,
                    reference_per_task=4,
                    test_per_task=4,
                ),
                strengths=(2.0,),
                output=Path(directory) / "result.json",
            )

            result = runner.execute_experiment_plan(plan, device=torch.device("cpu"))

            self.assertEqual(result["run_profile"], "explore")
            self.assertFalse(result["claim_ready"])
            self.assertTrue(plan.output.exists())
            self.assertEqual(
                result["setting"], "thirty_task_compositional_regression_stream"
            )


if __name__ == "__main__":
    unittest.main()
