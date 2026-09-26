import torch

from task_embeddings.language_mechinterp_strengthening_v8 import (
    nearest_prediction,
    select_correct_rank_block,
)


def test_select_correct_rank_block_is_balanced_and_skips_prior_ranks():
    records = [
        {"index": index, "subject": subject, "correct": correct}
        for index, (subject, correct) in enumerate(
            [
                ("a", False),
                ("a", True),
                ("a", False),
                ("a", True),
                ("a", True),
                ("b", False),
                ("b", True),
                ("b", True),
                ("b", False),
                ("b", True),
            ]
        )
    ]
    assert select_correct_rank_block(records, start_rank=1, count=2) == [3, 4, 7, 9]


def test_nearest_prediction_uses_the_most_similar_source_per_query():
    profiles = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    similarity = torch.tensor([[0.8, 0.1], [0.2, 0.3], [0.1, 0.9]])
    expected = torch.tensor([[1.0, 3.0], [10.0, 30.0]])
    torch.testing.assert_close(nearest_prediction(profiles, similarity), expected)
