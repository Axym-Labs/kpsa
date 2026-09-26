import unittest

import torch

from task_embeddings.language_mcq_atlas_v8 import (
    premise_query_indices,
    prototype_responses,
)


class LanguageMCQAtlasV8Test(unittest.TestCase):
    def test_prototype_responses_are_finite_and_normalized(self):
        values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        responses = prototype_responses(
            values, codebook, bandwidth_squared=1e-4
        )

        self.assertTrue(torch.isfinite(responses).all())
        torch.testing.assert_close(responses.norm(dim=1), torch.ones(2, dtype=torch.float64))
        self.assertEqual(responses.argmax(1).tolist(), [0, 1])

    def test_premise_query_indices_deduplicates_method_rows(self):
        payload = {
            "records": [
                {"query": 1, "dataset_index": 9},
                {"query": 0, "dataset_index": 4},
                {"query": 1, "dataset_index": 9},
                {"query": 0, "dataset_index": 4},
            ]
        }

        self.assertEqual(premise_query_indices(payload, 2), [4, 9])


if __name__ == "__main__":
    unittest.main()
