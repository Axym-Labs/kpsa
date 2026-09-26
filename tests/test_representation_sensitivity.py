import unittest

import torch
from torch import nn

from kpsa.parameter_groups import ParameterPartition

try:
    from kpsa.representation_sensitivity import (
        SensitivityAtlasAccumulator,
        coarsen_atlas,
        coarsen_profiles,
        cold_fold_predictions,
        group_relative_shares,
        sensitivity_weights,
    )
except ModuleNotFoundError:
    SensitivityAtlasAccumulator = None
    coarsen_atlas = None
    coarsen_profiles = None
    group_relative_shares = None
    sensitivity_weights = None
    cold_fold_predictions = None

try:
    from kpsa.representation_sensitivity import (
        partition_gradient_energy,
        profile_ranking_metrics,
        query_atlas,
    )
except ImportError:
    partition_gradient_energy = None
    profile_ranking_metrics = None
    query_atlas = None


class RepresentationSensitivityTest(unittest.TestCase):
    def test_cold_predictions_do_not_read_heldout_profile_columns(self):
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

    def test_coarsen_profiles_preserves_each_normalized_query(self):
        profiles = torch.tensor([[0.1, 0.2], [0.3, 0.1], [0.6, 0.7]])
        coarse = coarsen_profiles(profiles, torch.tensor([0, 0, 1]))
        torch.testing.assert_close(
            coarse, torch.tensor([[0.4, 0.3], [0.6, 0.7]])
        )
        torch.testing.assert_close(coarse.sum(0), profiles.sum(0))

    def test_per_example_shares_and_streaming_atlas_match_hand_calculation(self):
        self.assertIsNotNone(group_relative_shares)
        energies = torch.tensor([[1.0, 3.0], [4.0, 0.0]])
        representation = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        shares = group_relative_shares(energies)
        torch.testing.assert_close(
            shares, torch.tensor([[0.25, 0.75], [1.0, 0.0]])
        )
        torch.testing.assert_close(shares.sum(dim=1), torch.ones(2))

        accumulator = SensitivityAtlasAccumulator(groups=2, dimension=2)
        accumulator.update(energies[:1], representation[:1])
        accumulator.update(energies[1:], representation[1:])
        atlas = accumulator.compute()

        torch.testing.assert_close(
            atlas.mass, torch.tensor([0.625, 0.375], dtype=torch.float64)
        )
        torch.testing.assert_close(
            atlas.joint_moment,
            torch.tensor([[0.125, 0.5], [0.375, 0.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            atlas.embedding,
            torch.tensor([[0.2, 0.8], [1.0, 0.0]], dtype=torch.float64),
        )
        self.assertEqual(atlas.examples, 2)

    def test_raw_and_normalized_weights_are_the_two_supported_variants(self):
        energies = torch.tensor([[1.0, 3.0], [4.0, 0.0]])
        torch.testing.assert_close(sensitivity_weights(energies, "raw"), energies)
        torch.testing.assert_close(
            sensitivity_weights(energies, "normalized"),
            torch.tensor([[0.25, 0.75], [1.0, 0.0]]),
        )
        with self.assertRaisesRegex(ValueError, "raw.*normalized"):
            sensitivity_weights(energies, "partial")

        accumulator = SensitivityAtlasAccumulator(
            groups=2, dimension=1, weight_mode="raw"
        )
        accumulator.update(energies, torch.tensor([[1.0], [2.0]]))
        atlas = accumulator.compute()
        torch.testing.assert_close(
            atlas.mass, torch.tensor([2.5, 1.5], dtype=torch.float64)
        )
        torch.testing.assert_close(
            atlas.joint_moment, torch.tensor([[4.5], [1.5]], dtype=torch.float64)
        )
        self.assertEqual(atlas.weight_mode, "raw")

    def test_coarse_atlas_is_exact_sum_of_fine_sufficient_statistics(self):
        self.assertIsNotNone(coarsen_atlas)
        mass = torch.tensor([0.2, 0.3, 0.5])
        joint = torch.tensor([[0.2, 0.0], [0.0, 0.3], [0.25, 0.25]])
        assignment = torch.tensor([0, 0, 1])

        coarse = coarsen_atlas(mass, joint, assignment, groups=2)

        torch.testing.assert_close(coarse.mass, torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(
            coarse.joint_moment, torch.tensor([[0.2, 0.3], [0.25, 0.25]])
        )
        torch.testing.assert_close(
            coarse.embedding, torch.tensor([[0.4, 0.6], [0.5, 0.5]])
        )

    def test_zero_total_energy_is_rejected_instead_of_silently_losing_mass(self):
        self.assertIsNotNone(group_relative_shares)
        with self.assertRaisesRegex(ValueError, "positive total energy"):
            group_relative_shares(torch.tensor([[0.0, 0.0]]))

    def test_partition_energy_recovers_the_full_parameter_gradient_norm(self):
        self.assertIsNotNone(partition_gradient_energy)
        model = nn.Linear(3, 2, bias=False)
        model(torch.tensor([[1.0, -2.0, 0.5]])).sum().backward()
        partition = ParameterPartition(model, "row")

        energy = partition_gradient_energy(partition)
        expected = sum(
            parameter.grad.square().sum() for parameter in model.parameters()
        )

        torch.testing.assert_close(energy.sum(), expected)

    def test_query_uses_the_additive_joint_moment(self):
        self.assertIsNotNone(query_atlas)
        joint = torch.tensor([[0.5, -0.25], [0.0, 1.5]])
        query = torch.tensor([2.0, 4.0])

        torch.testing.assert_close(query_atlas(joint, query), torch.tensor([0.0, 6.0]))

    def test_ranking_metrics_are_one_for_exact_nonconstant_profiles(self):
        self.assertIsNotNone(profile_ranking_metrics)
        profiles = torch.tensor(
            [[4.0, 1.0], [3.0, 2.0], [2.0, 3.0], [1.0, 4.0]]
        )

        metrics = profile_ranking_metrics(profiles, profiles, top_fraction=0.5)

        self.assertAlmostEqual(metrics["mean_spearman"], 1.0)
        self.assertAlmostEqual(metrics["mean_topk_recall"], 1.0)
        self.assertAlmostEqual(metrics["mean_ndcg"], 1.0)
        self.assertAlmostEqual(metrics["mean_cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()
