import math
import unittest

import torch

from task_embeddings.paper_causal_v5 import causal_ranking_records


class PaperCausalV5Tests(unittest.TestCase):
    def test_residual_causal_ranking_removes_shared_module_effect(self):
        effects = torch.tensor(
            [
                [100.0, 100.0, 100.0],
                [50.0, 50.0, 50.0],
                [10.0, 10.0, -10.0],
                [1.0, 1.0, 21.0],
            ]
        )
        baseline = effects[:, :2].mean(dim=1, keepdim=True).repeat(1, 3)
        method_scores = {
            "tbe_linear": effects.clone(),
            "mean_only": baseline.clone(),
            "observed_opg_oracle": effects.clone(),
        }

        records = causal_ranking_records(
            method_scores,
            baseline,
            effects,
            effects,
            module_indices=torch.arange(4),
            train_tasks=[0, 1],
            target_tasks=[2],
        )
        by_method = {record["method"]: record for record in records}

        self.assertAlmostEqual(by_method["tbe_linear"]["residual_spearman"], 1.0)
        self.assertTrue(math.isnan(by_method["mean_only"]["residual_spearman"]))
        self.assertGreater(by_method["mean_only"]["raw_spearman"], 0.7)


if __name__ == "__main__":
    unittest.main()
