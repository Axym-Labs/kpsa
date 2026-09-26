import unittest

import torch

from task_embeddings.streaming_v5 import (
    TaskBlockedTBEAccumulator,
    fit_linear_tbe,
    query_linear_tbe,
    tbe_resource_counts,
)


class StreamingV5Tests(unittest.TestCase):
    def test_task_blocked_accumulator_matches_full_task_means(self):
        features = torch.tensor([[-1.0], [0.0], [1.0]])
        task_samples = [
            torch.tensor([[7.0, 24.0], [9.0, 22.0]]),
            torch.tensor([[10.0, 20.0]]),
            torch.tensor([[11.0, 18.0], [12.0, 16.0], [13.0, 17.0]]),
        ]
        task_means = torch.stack([samples.mean(dim=0) for samples in task_samples], 1)

        accumulator = TaskBlockedTBEAccumulator(n_modules=2, dimension=1)
        for feature, samples in zip(features, task_samples):
            accumulator.begin_task(feature)
            for sample in samples:
                accumulator.update(sample)
            accumulator.end_task()
        streaming = accumulator.finalize(ridge=0.0)
        posthoc = fit_linear_tbe(task_means, features, ridge=0.0)

        torch.testing.assert_close(streaming.module_mean, posthoc.module_mean)
        torch.testing.assert_close(streaming.feature_mean, posthoc.feature_mean)
        torch.testing.assert_close(streaming.coefficients, posthoc.coefficients)
        torch.testing.assert_close(
            query_linear_tbe(streaming, features),
            query_linear_tbe(posthoc, features),
        )

    def test_resource_counts_separate_storage_from_construction_state(self):
        counts = tbe_resource_counts(n_modules=100, n_tasks=20, dimension=4)

        self.assertEqual(counts["full_atlas_stored_floats"], 2000)
        self.assertEqual(counts["tbe_stored_floats"], 584)
        self.assertLess(
            counts["streaming_construction_state_floats"],
            counts["posthoc_construction_state_floats"],
        )


if __name__ == "__main__":
    unittest.main()
