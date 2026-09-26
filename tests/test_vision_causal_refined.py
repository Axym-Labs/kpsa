import unittest

import torch
from torch import nn

from kpsa.parameter_groups import ParameterPartition
from kpsa.vision_causal_refined import (
    mean_ablate_groups,
    mean_ablate_mlp_activations,
    mean_ablation_taylor,
    select_parameter_budget,
    zero_ablate_mlp_activations,
)


class VisionCausalRefinedTest(unittest.TestCase):
    def test_mean_ablation_taylor_matches_featurewise_first_order_loss(self):
        activation = torch.tensor([[[2.0, 4.0], [3.0, 1.0]]])
        gradient = torch.tensor([[[0.5, -1.0], [2.0, 3.0]]])
        mean = torch.tensor([1.0, 2.0])

        scores = mean_ablation_taylor(activation, gradient, mean)

        torch.testing.assert_close(scores, torch.tensor([4.5, -5.0]))

    def test_activation_mean_ablation_changes_only_selected_features(self):
        layer = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(3))
        layout = [{"name": "mlp", "offset": 0, "count": 3, "module": layer}]
        selected = torch.tensor([False, True, False])
        means = {"mlp": torch.tensor([10.0, 20.0, 30.0])}
        inputs = torch.ones(1, 3)
        with mean_ablate_mlp_activations(layout, selected, means):
            torch.testing.assert_close(layer(inputs), torch.tensor([[1.0, 20.0, 1.0]]))
        torch.testing.assert_close(layer(inputs), torch.ones(1, 3))

    def test_activation_zero_ablation_changes_only_selected_features(self):
        layer = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(3))
        layout = [{"name": "mlp", "offset": 0, "count": 3, "module": layer}]
        selected = torch.tensor([False, True, False])
        inputs = torch.ones(1, 3)
        with zero_ablate_mlp_activations(layout, selected):
            torch.testing.assert_close(layer(inputs), torch.tensor([[1.0, 0.0, 1.0]]))
        torch.testing.assert_close(layer(inputs), torch.ones(1, 3))

    def test_parameter_budget_selects_highest_scores(self):
        scores = torch.tensor([1.0, 4.0, 3.0, 2.0])
        sizes = torch.tensor([1.0, 2.0, 1.0, 1.0])
        selected, actual = select_parameter_budget(
            scores, sizes, torch.ones(4, dtype=torch.bool), 0.5
        )
        torch.testing.assert_close(selected, torch.tensor([False, True, True, False]))
        self.assertAlmostEqual(actual, 0.6)

    def test_mean_ablation_restores_selected_rows(self):
        layer = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]))
        partition = ParameterPartition(layer, "row")
        original = layer.weight.detach().clone()
        selected = torch.tensor([False, True, False])

        with mean_ablate_groups(partition, selected):
            torch.testing.assert_close(layer.weight[1], original.mean(0))
            torch.testing.assert_close(layer.weight[[0, 2]], original[[0, 2]])

        torch.testing.assert_close(layer.weight, original)


if __name__ == "__main__":
    unittest.main()
