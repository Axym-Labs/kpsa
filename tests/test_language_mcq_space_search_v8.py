from task_embeddings.language_mcq_space_search_v8 import (
    choose_candidate,
    representation_text,
    round_robin_indices,
)


class _Dataset(dict):
    pass


def test_round_robin_indices_balance_subject_prefixes():
    dataset = _Dataset(subject=["a", "a", "a", "b", "b", "c", "c"])
    assert round_robin_indices(dataset) == [0, 3, 5, 1, 4, 6, 2]


def test_answer_conditioned_representation_adds_requested_direction():
    example = {"question": "Q?", "choices": ["x", "y", "z", "w"], "answer": 1}
    context = representation_text(example, answer_conditioned=False)
    answer = representation_text(example, answer_conditioned=True)
    assert "Requested output direction" not in context
    assert "B (y)" in answer


def test_choose_candidate_requires_both_positive_intervals_when_available():
    def candidate(name, scalar, permutation, lower):
        return {
            "space": name,
            "causal_vs_scalar": {"mean": scalar, "lower_95": lower},
            "causal_vs_permutation": {"mean": permutation, "lower_95": lower},
        }

    selected, gated = choose_candidate(
        [candidate("ungated", 3.0, 3.0, -0.1), candidate("gated", 1.0, 0.8, 0.1)]
    )
    assert gated
    assert selected["space"] == "gated"
