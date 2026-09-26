import unittest

import torch

from task_embeddings.language_behavior_refined import behavior_margin_from_logits


class LanguageBehaviorRefinedTest(unittest.TestCase):
    def test_correct_token_margin_uses_only_jointly_valid_positions(self):
        logits = torch.zeros(1, 3, 6)
        correct = torch.tensor([1, 2, -1])
        alternatives = torch.tensor([[3, 4, -1], [4, 5, -1]])
        logits[0, 0, 1] = 4
        logits[0, 1, 2] = 2
        logits[0, 0, 3:5] = 1
        logits[0, 1, 4:6] = 0
        self.assertAlmostEqual(
            float(behavior_margin_from_logits(logits, correct, alternatives)),
            2.5,
        )


if __name__ == "__main__":
    unittest.main()
