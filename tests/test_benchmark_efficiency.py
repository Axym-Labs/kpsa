import unittest

import torch

from task_embeddings.benchmark_efficiency import (
    benchmark_loops,
    embedding_storage_bytes,
)


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_benchmark_reports_finite_nonnegative_timings(self):
        value = torch.tensor(0.0)

        def baseline():
            nonlocal value
            value = value + 1

        def profiled():
            nonlocal value
            value = value + 2

        result = benchmark_loops(
            baseline, profiled, torch.device("cpu"), warmup=1, iterations=3
        )
        self.assertGreaterEqual(result["baseline_seconds"], 0)
        self.assertGreaterEqual(result["profiled_seconds"], 0)
        self.assertEqual(result["iterations"], 3)

    def test_embedding_storage_counts_float32_coordinates_and_denominator(self):
        self.assertEqual(embedding_storage_bytes(10, [4, 6]), 10 * (4 + 6 + 2) * 4)


if __name__ == "__main__":
    unittest.main()
