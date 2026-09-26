import unittest

import torch

from task_embeddings.domain_optimizer import ParameterPartition
from task_embeddings.domain_train import DomainConfig, make_model

try:
    from task_embeddings.qwen_sensitivity import (
        cold_fold_predictions,
        evaluate_precision_allocations,
        nonlinear_fold_predictions,
        profile_relative_sensitivity,
        run_qwen_study,
    )
except ModuleNotFoundError:
    cold_fold_predictions = None
    evaluate_precision_allocations = None
    nonlinear_fold_predictions = None
    profile_relative_sensitivity = None
    run_qwen_study = None
except ImportError:
    from task_embeddings.qwen_sensitivity import (
        cold_fold_predictions,
        profile_relative_sensitivity,
    )

    evaluate_precision_allocations = None
    run_qwen_study = None


class QwenSensitivityTest(unittest.TestCase):
    def test_nonlinear_predictions_do_not_read_heldout_profile_columns(self):
        profiles = torch.arange(32, dtype=torch.float64).reshape(4, 8) + 1
        features = torch.arange(16, dtype=torch.float64).reshape(8, 2) + 1
        changed = profiles.clone()
        changed[:, 0::4] = 1_000_000
        original = nonlinear_fold_predictions(profiles, features, seed=5)
        modified = nonlinear_fold_predictions(changed, features, seed=5)
        for method in original:
            torch.testing.assert_close(
                original[method][:, 0::4], modified[method][:, 0::4]
            )

    def test_profile_columns_are_complete_group_relative_distributions(self):
        self.assertIsNotNone(profile_relative_sensitivity)
        config = DomainConfig(
            hidden=8, layers=1, intermediate=16, heads=2, kv_heads=1
        )
        model = make_model(config, vocab=16, device="cpu")
        domains = [
            torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
            torch.tensor([[5, 4, 3, 2], [4, 3, 2, 1]]),
        ]
        partition = ParameterPartition(model, "swiglu")

        result = profile_relative_sensitivity(
            model, domains, partition, samples=1, start=0
        )

        self.assertEqual(result.task_profiles.shape, (partition.n_groups, 2))
        torch.testing.assert_close(
            result.task_profiles.sum(dim=0), torch.ones(2, dtype=torch.float64)
        )
        self.assertLess(result.max_partition_relative_error, 1e-5)

    def test_cold_predictions_do_not_read_heldout_profile_columns(self):
        self.assertIsNotNone(cold_fold_predictions)
        profiles = torch.arange(32, dtype=torch.float64).reshape(4, 8) + 1
        features = torch.arange(16, dtype=torch.float64).reshape(8, 2) + 1
        changed = profiles.clone()
        changed[:, 0::4] = 1_000_000

        original = cold_fold_predictions(profiles, features, seed=5)
        modified = cold_fold_predictions(changed, features, seed=5)

        for method in (
            "semantic",
            "affine_semantic",
            "rbf_semantic",
            "scalar_mass",
            "jl",
            "permuted",
            "nearest",
        ):
            torch.testing.assert_close(
                original[method][:, 0::4], modified[method][:, 0::4]
            )

    def test_precision_allocation_reports_scalar_and_vector_and_restores_model(self):
        self.assertIsNotNone(evaluate_precision_allocations)
        config = DomainConfig(
            hidden=8, layers=1, intermediate=16, heads=2, kv_heads=1
        )
        model = make_model(config, vocab=16, device="cpu")
        domains = [
            torch.tensor([[1 + task, 2, 3, 4], [2 + task, 3, 4, 5]]) % 16
            for task in range(4)
        ]
        partition = ParameterPartition(model, "swiglu")
        profiles = profile_relative_sensitivity(
            model, domains, partition, samples=1, start=0
        ).task_profiles
        features = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]]
        )
        predictions = cold_fold_predictions(profiles, features, seed=9, folds=2)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}

        result = evaluate_precision_allocations(
            model,
            partition,
            domains,
            predictions,
            direct_profiles=profiles,
            source_profiles=profiles.roll(1, dims=1),
            eval_blocks=1,
            fractions=(0.5,),
            low_bits=4,
            high_bits=8,
        )

        self.assertIn("semantic", {row["method"] for row in result["records"]})
        self.assertIn("scalar_mass", {row["method"] for row in result["records"]})
        self.assertIn(
            "direct_gradient_oracle", {row["method"] for row in result["records"]}
        )
        self.assertIn(
            "source_onehot_reference", {row["method"] for row in result["records"]}
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_study_separates_reference_and_target_profiles(self):
        self.assertIsNotNone(run_qwen_study)
        config = DomainConfig(
            hidden=8, layers=1, intermediate=16, heads=2, kv_heads=1
        )
        model = make_model(config, vocab=16, device="cpu")
        domains = [
            torch.tensor([[1 + task, 2, 3, 4], [2 + task, 3, 4, 5]]) % 16
            for task in range(4)
        ]
        corpus = {"domains": tuple(f"task_{i}" for i in range(4)), "train": domains}
        features = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]]
        )

        result, tensors = run_qwen_study(
            model,
            corpus,
            features,
            model_source="tiny-test",
            samples=1,
            folds=2,
            evaluation_role=None,
        )

        self.assertEqual(result["protocol"]["source_start"], 0)
        self.assertEqual(result["protocol"]["target_start"], 1)
        self.assertIn("semantic", result["cold_query"])
        self.assertIn("scalar_mass", result["cold_query"])
        self.assertEqual(tensors["source_profiles"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
