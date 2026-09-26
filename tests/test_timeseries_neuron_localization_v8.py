import unittest

import torch
from torch import nn

from task_embeddings.timeseries_neuron_localization_v8 import (
    _method_scores,
    _summarize,
    activation_attribution,
    aggregate_feature_bundles,
    expand_bundle_selection,
    zero_activation_groups,
)


class _ToySensitivity:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = nn.Module()
        self.incoming = nn.Linear(2, 3, bias=False)
        self.outgoing = nn.Linear(3, 1, bias=False)
        self.model.add_module("incoming", self.incoming)
        self.model.add_module("outgoing", self.outgoing)
        with torch.no_grad():
            self.incoming.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
            )
            self.outgoing.weight.copy_(torch.tensor([[1.0, -1.0, 0.5]]))
        self.ff_pairs = [("layer", self.incoming, self.outgoing)]
        self.units_per_layer = 3
        self.groups = 3

    def _forward_components(self, x):
        hidden = torch.relu(self.incoming(x[None]))
        value = self.outgoing(hidden)[0, 0]
        return value, x

    @torch.no_grad()
    def functional(self, x):
        return float(self._forward_components(x)[0])


class TimeSeriesNeuronLocalizationTest(unittest.TestCase):
    def test_activation_attribution_matches_single_feature_zero_action(self):
        experiment = _ToySensitivity()
        x = torch.tensor([2.0, 1.0])

        baseline, _feature, scores = activation_attribution(
            experiment, x, "all"
        )

        for group in range(3):
            with zero_activation_groups(experiment, torch.tensor([group])):
                intervened = experiment.functional(x)
            self.assertAlmostEqual(scores[group].item(), abs(baseline - intervened))

    def test_kernel_methods_keep_matched_shapes(self):
        atlas = {
            "source_features": torch.eye(2),
            "source_profiles": torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
            "source_categories": torch.eye(2),
            "permutation": torch.tensor([1, 0]),
            "median_squared_distance": 1.0,
        }
        methods = _method_scores(
            atlas,
            torch.eye(2),
            torch.eye(2),
            scale=1.0,
        )

        self.assertEqual(set(methods), {
            "exact_rbf",
            "matched_shuffle",
            "nearest",
            "scalar",
            "temporal_categorical",
        })
        self.assertTrue(all(value.shape == (2, 2) for value in methods.values()))
        torch.testing.assert_close(
            methods["nearest"], atlas["source_profiles"].double()
        )

    def test_summary_reports_activation_gap_recovery(self):
        records = []
        for query, values in enumerate(((3.0, 1.0, 5.0), (5.0, 1.0, 9.0))):
            for method, value in zip(
                ("exact_rbf", "scalar", "activation_attribution"), values
            ):
                records.append(
                    {
                        "query": query,
                        "method": method,
                        "fraction": 0.1,
                        "absolute_functional_change": value,
                        "causal_effect": value,
                    }
                )
        summaries, comparisons = _summarize(
            records,
            ("exact_rbf", "scalar", "activation_attribution"),
            (0.1,),
        )

        self.assertEqual(summaries["0.1"]["exact_rbf"]["mean"], 4.0)
        self.assertAlmostEqual(
            comparisons["0.1"]["exact_rbf"]["activation_gap_recovered"],
            0.5,
        )

    def test_feature_bundles_stay_within_layers_and_expand(self):
        scores = torch.arange(16, dtype=torch.float64).reshape(8, 2)
        bundled = aggregate_feature_bundles(
            scores, units_per_layer=4, bundle_size=2
        )

        torch.testing.assert_close(
            bundled,
            torch.stack((scores[0:2].sum(0), scores[2:4].sum(0),
                         scores[4:6].sum(0), scores[6:8].sum(0))),
        )
        torch.testing.assert_close(
            expand_bundle_selection(
                torch.tensor([1, 2]), units_per_layer=4, bundle_size=2
            ),
            torch.tensor([2, 3, 4, 5]),
        )


if __name__ == "__main__":
    unittest.main()
