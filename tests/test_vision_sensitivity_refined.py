import unittest

import torch
from torch import nn

from kpsa.parameter_groups import ParameterPartition

try:
    from kpsa.vision_sensitivity_refined import (
        profile_class_sensitivity,
        run_vision_profiles,
    )
except ModuleNotFoundError:
    profile_class_sensitivity = None
    run_vision_profiles = None
except ImportError:
    from kpsa.vision_sensitivity_refined import profile_class_sensitivity

    run_vision_profiles = None


class VisionSensitivityRefinedTest(unittest.TestCase):
    def test_class_profiles_cover_the_complete_parameter_partition(self):
        self.assertIsNotNone(profile_class_sensitivity)
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
        images = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 1.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ]
        )
        labels = torch.tensor([0, 1, 2])
        partition = ParameterPartition(model, "swiglu")

        result = profile_class_sensitivity(
            model, images, labels, partition, functional="class_logit"
        )

        self.assertEqual(result.task_profiles.shape, (partition.n_groups, 3))
        torch.testing.assert_close(
            result.task_profiles.sum(dim=0), torch.ones(3, dtype=torch.float64)
        )
        self.assertLess(result.max_partition_relative_error, 1e-6)

    def test_functional_study_reports_cold_vector_and_scalar_controls(self):
        self.assertIsNotNone(run_vision_profiles)
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 4))
        source = torch.eye(4).reshape(4, 1, 2, 2)
        target = source.roll(1, dims=-1)
        labels = torch.arange(4)
        features = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]]
        )

        result, tensors = run_vision_profiles(
            model, source, target, labels, features, folds=2, seed=13
        )

        self.assertEqual(
            set(result["functionals"]), {"class_logit", "margin", "loss"}
        )
        self.assertIn(
            "affine_semantic", result["functionals"]["class_logit"]["cold_query"]
        )
        self.assertIn(
            "scalar_mass", result["functionals"]["class_logit"]["cold_query"]
        )
        self.assertEqual(tensors["features"].shape, features.shape)


if __name__ == "__main__":
    unittest.main()
