import unittest

import torch

from kpsa.kernel_sensitivity import (
    LinearKernelMap,
    fit_anchor_nystrom_rbf,
    fit_empirical_rbf_index,
    fit_kernel_mean_index,
    fit_nystrom_rbf,
    fit_prototype_response_map,
    fit_tensor_sketch_polynomial,
    fit_weighted_kernel_mean_index,
    median_squared_distance,
    rbf_kernel,
)


class KernelSensitivityTest(unittest.TestCase):
    def test_linear_map_reduces_to_weighted_first_moment(self):
        energy = torch.tensor([[1.0, 3.0], [4.0, 0.0]])
        representations = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        index = fit_kernel_mean_index(
            energy,
            representations,
            LinearKernelMap(input_dimension=2),
            weight_mode="normalized",
        )

        torch.testing.assert_close(
            index.atlas.joint_moment,
            torch.tensor([[0.125, 0.5], [0.375, 0.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            index.query(torch.eye(2), mass_weighted=True),
            index.atlas.joint_moment,
        )
        torch.testing.assert_close(
            index.query(torch.eye(2), mass_weighted=False),
            index.atlas.embedding,
        )

    def test_exact_empirical_index_matches_direct_kernel_formula(self):
        weights = torch.tensor([[0.25, 1.0], [0.75, 0.0]], dtype=torch.float64)
        reference = torch.eye(2, dtype=torch.float64)
        query = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        bandwidth = median_squared_distance(reference)
        index = fit_empirical_rbf_index(
            weights, reference, bandwidth_squared=bandwidth
        )
        kernel = rbf_kernel(reference, query, bandwidth_squared=bandwidth)

        torch.testing.assert_close(
            index.query(query, mass_weighted=True), weights @ kernel / 2
        )
        torch.testing.assert_close(
            index.query(query, mass_weighted=False),
            weights @ kernel / weights.sum(dim=1, keepdim=True),
        )

    def test_nystrom_index_is_a_fixed_dimensional_kernel_mean(self):
        reference = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
        )
        weights = torch.tensor(
            [[1.0, 0.8], [0.0, 0.2], [0.1, 0.9], [0.7, 0.3]]
        )
        feature_map = fit_nystrom_rbf(
            reference, rank=2, landmark_method="kmeans++", seed=7
        )
        index = fit_weighted_kernel_mean_index(
            weights, reference, feature_map, weight_mode="normalized"
        )

        self.assertEqual(feature_map.dimension, 2)
        self.assertEqual(index.atlas.joint_moment.shape, (2, 2))
        self.assertEqual(index.query(reference[:1], mass_weighted=True).shape, (2, 1))
        self.assertEqual(index.atlas.weight_mode, "normalized")

    def test_prototype_map_has_fixed_finite_normalized_coordinates(self):
        reference = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        )
        feature_map = fit_prototype_response_map(
            reference,
            dimension=2,
            bandwidth_squared=0.05,
            seed=7,
        )

        transformed = feature_map.transform(reference)

        self.assertEqual(feature_map.dimension, 2)
        self.assertEqual(transformed.shape, (4, 2))
        torch.testing.assert_close(
            transformed.norm(dim=1), torch.ones(4, dtype=torch.float64)
        )
        self.assertTrue(torch.isfinite(transformed).all())

    def test_anchor_net_nystrom_returns_distinct_data_landmarks(self):
        generator = torch.Generator().manual_seed(17)
        reference = torch.randn(24, 6, generator=generator)
        feature_map = fit_anchor_nystrom_rbf(
            reference,
            rank=8,
            bandwidth_squared=0.5,
            seed=7,
        )

        self.assertEqual(feature_map.dimension, 8)
        self.assertEqual(feature_map.landmark_method, "anchor_net")
        self.assertEqual(len(torch.unique(feature_map.landmarks, dim=0)), 8)
        self.assertEqual(feature_map.transform(reference[:3]).shape, (3, 8))

    def test_tensor_sketch_approximates_degree_two_polynomial_kernel(self):
        generator = torch.Generator().manual_seed(19)
        values = torch.randn(12, 5, generator=generator)
        feature_map = fit_tensor_sketch_polynomial(
            input_dimension=5,
            output_dimension=4096,
            offset=1.0,
            seed=7,
        )
        features = feature_map.transform(values)
        normalized = torch.nn.functional.normalize(values.double(), dim=1)
        target = (1 + normalized @ normalized.T).square()

        self.assertEqual(features.shape, (12, 4096))
        self.assertLess(float((features @ features.T - target).abs().mean()), 0.12)

    def test_invalid_partial_normalization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw.*normalized"):
            fit_kernel_mean_index(
                torch.ones(2, 2),
                torch.eye(2),
                LinearKernelMap(input_dimension=2),
                weight_mode="partial",
            )


if __name__ == "__main__":
    unittest.main()
