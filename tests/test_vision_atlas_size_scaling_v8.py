import torch

from kpsa.vision_atlas_size_scaling_v8 import (
    balanced_pairing_permutation,
    offset_major_indices,
    select_default,
)


def test_offset_major_indices_have_balanced_prefixes():
    representation = torch.tensor([0, 1, 50, 51, 100, 101])
    assert offset_major_indices(representation, range(3)).tolist() == [
        0,
        50,
        100,
        1,
        51,
        101,
        2,
        52,
        102,
    ]


def test_balanced_pairing_permutation_stays_within_offset_blocks():
    permutation = balanced_pairing_permutation(12, 4, seed=7)
    assert sorted(permutation.tolist()) == list(range(12))
    for block in range(3):
        values = permutation[block * 4 : (block + 1) * 4]
        assert sorted(values.tolist()) == list(range(block * 4, (block + 1) * 4))


def test_select_default_uses_smallest_gated_candidate_within_best_95_percent():
    def entry(mean, lower):
        summary = {"mean": mean, "lower_95": lower, "upper_95": mean + 0.1}
        return {
            "paired_causal_vs_scalar": {
                "prototype_rbf": {"0.1": summary, "0.2": summary}
            },
            "paired_causal_vs_matched_permutation": {
                "prototype_rbf": {"0.1": summary, "0.2": summary}
            },
        }

    selected = select_default(
        {"200": entry(0.8, 0.1), "400": entry(0.96, 0.1), "800": entry(1.0, 0.1)},
        (0.1, 0.2),
    )
    assert selected["source_examples"] == 400
