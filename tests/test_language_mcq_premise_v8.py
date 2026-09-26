import unittest

import torch

from task_embeddings.language_mcq_premise_v8 import (
    answer_margin_from_logits,
    format_mmlu_prompt,
)


class LanguageMCQPremiseV8Test(unittest.TestCase):
    def test_prompt_contains_all_choices_and_answer_instruction(self):
        prompt = format_mmlu_prompt(
            {
                "question": "Which value?",
                "choices": ["one", "two", "three", "four"],
                "answer": 1,
            }
        )
        self.assertIn("Question: Which value?", prompt)
        self.assertIn("A. one", prompt)
        self.assertIn("D. four", prompt)
        self.assertTrue(prompt.endswith("Answer with only the letter."))

    def test_answer_margin_uses_strongest_incorrect_choice(self):
        logits = torch.zeros(2, 3, 8)
        candidates = torch.tensor([1, 2, 3, 4])
        logits[0, -1, candidates] = torch.tensor([2.0, 5.0, 1.0, 0.0])
        logits[1, -1, candidates] = torch.tensor([1.0, 2.0, 7.0, 3.0])
        answers = torch.tensor([1, 2])

        margin = answer_margin_from_logits(logits, answers, candidates)

        torch.testing.assert_close(margin, torch.tensor([3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
