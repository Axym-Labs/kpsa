import unittest

import torch

from task_embeddings.applications import (
    causal_assay_gate,
    continual_learning_assay_gate,
    continual_learning_metrics,
    faithfulness_scores,
    layer_balanced_overlap,
    selective_effect,
)


class ApplicationMetricTests(unittest.TestCase):
    def test_selective_effect_compares_target_against_other_tasks(self):
        delta = torch.tensor([1.0, 4.0, 2.0])
        got = selective_effect(delta, target=1)
        self.assertAlmostEqual(got["target_effect"], 4.0)
        self.assertAlmostEqual(got["nontarget_effect"], 1.5)
        self.assertAlmostEqual(got["selectivity"], 2.5)

    def test_continual_metrics_for_higher_is_better(self):
        history = torch.tensor(
            [
                [0.8, float("nan"), float("nan")],
                [0.7, 0.9, float("nan")],
                [0.6, 0.8, 0.7],
            ]
        )
        got = continual_learning_metrics(history, higher_is_better=True)
        self.assertAlmostEqual(got["average_forgetting"], (0.2 + 0.1) / 2, places=6)
        self.assertTrue(torch.isnan(torch.tensor(got["per_task_forgetting"][-1])))
        self.assertEqual(got["forgetting_eligible_tasks"], 2)
        self.assertAlmostEqual(got["final_average"], 0.7, places=6)
        self.assertAlmostEqual(got["new_task_plasticity"], 0.8, places=6)

    def test_continual_metrics_for_lower_is_better(self):
        history = torch.tensor([[1.0, float("nan")], [1.5, 0.5]])
        got = continual_learning_metrics(history, higher_is_better=False)
        self.assertAlmostEqual(got["average_forgetting"], 0.5, places=6)
        self.assertTrue(torch.isnan(torch.tensor(got["per_task_forgetting"][-1])))
        self.assertEqual(got["forgetting_eligible_tasks"], 1)
        self.assertAlmostEqual(got["final_average"], 1.0, places=6)

    def test_continual_metrics_have_no_forgetting_for_single_task(self):
        got = continual_learning_metrics(torch.tensor([[0.8]]), higher_is_better=True)
        self.assertTrue(torch.isnan(torch.tensor(got["average_forgetting"])))
        self.assertTrue(torch.isnan(torch.tensor(got["per_task_forgetting"][0])))
        self.assertEqual(got["forgetting_eligible_tasks"], 0)

    def test_continual_metrics_reject_future_task_observations(self):
        with self.assertRaises(ValueError):
            continual_learning_metrics(
                torch.tensor([[0.8, 0.2], [0.7, 0.9]]), higher_is_better=True
            )

    def test_continual_metrics_report_acquisition_gain(self):
        history = torch.tensor([[0.8, float("nan")], [0.7, 0.9]])
        got = continual_learning_metrics(
            history,
            higher_is_better=True,
            pre_update_per_task=torch.tensor([0.2, 0.4]),
        )
        self.assertAlmostEqual(got["average_acquisition_gain"], 0.55, places=6)
        torch.testing.assert_close(
            torch.tensor(got["per_task_acquisition_gain"]), torch.tensor([0.6, 0.5])
        )

    def test_faithfulness_scores_normalize_keep_and_drop_curves(self):
        got = faithfulness_scores(
            kept_divergence=torch.tensor([0.8, 0.2]),
            dropped_divergence=torch.tensor([0.2, 0.7]),
            null_divergence=1.0,
        )
        torch.testing.assert_close(got["sufficiency"], torch.tensor([0.2, 0.8]))
        torch.testing.assert_close(got["necessity"], torch.tensor([0.2, 0.7]))

    def test_causal_gate_requires_attainable_mask_to_beat_random(self):
        failed = causal_assay_gate(
            null_divergence=1.0,
            attainable_sufficiency=0.51,
            random_sufficiency_p95=0.50,
            minimum_span=0.1,
            minimum_gap=0.2,
        )
        self.assertFalse(failed["passed"])
        self.assertIn("attainable_reference_not_above_random", failed["reasons"])
        passed = causal_assay_gate(
            null_divergence=1.0,
            attainable_sufficiency=0.8,
            random_sufficiency_p95=0.5,
            minimum_span=0.1,
            minimum_gap=0.2,
        )
        self.assertTrue(passed["passed"])

    def test_continual_gate_requires_forgetting_and_learning(self):
        negligible = continual_learning_assay_gate(
            no_protection_forgetting=0.001,
            no_protection_acquisition_gain=0.5,
            minimum_forgetting=0.05,
            minimum_acquisition_gain=0.1,
        )
        self.assertFalse(negligible["passed"])
        self.assertIn("negligible_forgetting", negligible["reasons"])
        no_learning = continual_learning_assay_gate(
            no_protection_forgetting=0.2,
            no_protection_acquisition_gain=0.01,
            minimum_forgetting=0.05,
            minimum_acquisition_gain=0.1,
        )
        self.assertFalse(no_learning["passed"])
        self.assertIn("insufficient_acquisition", no_learning["reasons"])

    def test_layer_balanced_overlap_uses_global_identity(self):
        first = [torch.tensor([0]), torch.tensor([0])]
        second = [torch.tensor([0]), torch.tensor([1])]
        self.assertAlmostEqual(layer_balanced_overlap(first, second, [2, 2]), 0.5)


if __name__ == "__main__":
    unittest.main()
