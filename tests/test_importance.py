import importlib.util
import unittest

import torch

from task_embeddings import importance
from task_embeddings.experiment_core import Estimator


class ImportanceTests(unittest.TestCase):
    def test_importance_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("task_embeddings.importance"))

    def test_opg_trace_normalization_is_explicit(self):
        raw = torch.tensor([2.0, 8.0])

        torch.testing.assert_close(
            importance.transform_opg_trace(raw, Estimator.RAW_OPG), raw
        )
        torch.testing.assert_close(
            importance.transform_opg_trace(
                raw,
                Estimator.RESIDUAL_NORMALIZED_OPG,
                residual_norm_squared=torch.tensor(2.0),
            ),
            torch.tensor([1.0, 4.0]),
        )

    def test_legacy_profile_keys_are_canonicalized_without_duplication(self):
        profiles = importance.canonicalize_profile_mapping(
            {"raw_ef": torch.ones(2, 3), "ief": torch.full((2, 3), 2.0)}
        )

        self.assertEqual(set(profiles), {"raw_opg", "residual_normalized_opg"})
        torch.testing.assert_close(profiles["raw_opg"], torch.ones(2, 3))

    def test_representation_grid_uses_one_common_construction_path(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        coefficients = torch.tensor([[2.0, -1.0], [-0.5, 3.0]])
        atlas = torch.tensor([[4.0], [1.5]]) + coefficients @ features.T
        atlases = {
            "raw_opg": atlas,
            "residual_normalized_opg": 2.0 * atlas,
        }
        jl = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])

        scores = importance.build_regular_score_grid(atlases, features, jl_features=jl)

        self.assertEqual(len(scores), 8)
        torch.testing.assert_close(scores["raw_opg/full_atlas"], atlas)
        torch.testing.assert_close(scores["raw_opg/tbe"], atlas, atol=1e-4, rtol=1e-4)
        self.assertEqual(scores["raw_opg/jl"].shape, atlas.shape)
        expected_mean = atlas.mean(dim=1, keepdim=True).repeat(1, atlas.shape[1])
        torch.testing.assert_close(scores["raw_opg/mean_only"], expected_mean)


if __name__ == "__main__":
    unittest.main()
