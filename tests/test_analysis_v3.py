import unittest

import torch

from task_embeddings.analysis_v3 import (
    bootstrap_mean_ci,
    global_circuit,
    summarize_task_records,
)


class AnalysisV3Tests(unittest.TestCase):
    def test_bootstrap_interval_is_deterministic_and_contains_mean(self):
        first = bootstrap_mean_ci([1.0, 2.0, 3.0], draws=1000, seed=9)
        second = bootstrap_mean_ci([1.0, 2.0, 3.0], draws=1000, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 2.0)
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])

    def test_task_summary_averages_repeated_measurements_within_task(self):
        records = [
            {"method": "a", "task": 0, "value": 1.0},
            {"method": "a", "task": 0, "value": 3.0},
            {"method": "a", "task": 1, "value": 4.0},
            {"method": "b", "task": 0, "value": 10.0},
        ]
        rows = summarize_task_records(records, "value", ("method",), draws=200)
        by_method = {row["method"]: row for row in rows}
        self.assertAlmostEqual(by_method["a"]["mean"], 3.0)
        self.assertEqual(by_method["a"]["n_tasks"], 2)
        self.assertAlmostEqual(by_method["b"]["mean"], 10.0)
        self.assertEqual(by_method["b"]["ci_low"], 10.0)
        self.assertEqual(by_method["b"]["ci_high"], 10.0)

    def test_global_circuit_preserves_layer_identity(self):
        selected = [torch.tensor([0, 2]), torch.tensor([1])]
        self.assertEqual(global_circuit(selected, [3, 2]), {0, 2, 4})


if __name__ == "__main__":
    unittest.main()
