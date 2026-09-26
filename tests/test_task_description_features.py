import unittest
from types import SimpleNamespace

import torch

from task_embeddings.task_description_features import (
    encode_task_inputs,
    last_token_pool,
    task_descriptions,
)


class TaskDescriptionFeaturesTest(unittest.TestCase):
    def test_input_features_exclude_reserved_profile_positions(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 4)

            def forward(self, input_ids, attention_mask, use_cache=False):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        domains = [
            torch.tensor([[i + task, 2, 3, -1] for i in range(10)])
            for task in range(2)
        ]
        features = encode_task_inputs(
            Encoder(), domains, pad_token_id=0, samples=4, excluded_profile_samples=2
        )
        self.assertEqual(features.shape, (2, 4))
        torch.testing.assert_close(features.norm(dim=1), torch.ones(2))

    def test_multilingual_domains_receive_language_descriptions(self):
        self.assertEqual(
            task_descriptions(["de_DE", "ja_JP"]),
            ["Text written naturally in German.", "Text written naturally in Japanese."],
        )

    def test_last_token_pool_handles_left_and_right_padding(self):
        hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        left = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        torch.testing.assert_close(last_token_pool(hidden, left), hidden[:, -1])
        torch.testing.assert_close(
            last_token_pool(hidden, right), torch.stack((hidden[0, 1], hidden[1, 2]))
        )


if __name__ == "__main__":
    unittest.main()
