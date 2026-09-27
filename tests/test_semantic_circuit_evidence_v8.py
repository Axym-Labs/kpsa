import torch

from kpsa.semantic_circuit_evidence_v8 import _selection_grid


def test_selection_grid_respects_scopes_and_matches_layer_counts():
    generator = torch.Generator().manual_seed(17)
    layers, units, count = 12, 32, 20
    groups = layers * units
    grid = _selection_grid(
        torch.rand(groups, generator=generator),
        torch.rand(groups, generator=generator),
        torch.rand(groups, generator=generator),
        count=count,
        units_per_layer=units,
        late_layers=3,
    )

    assert bool((grid["semantic_nonlate"] < 9 * units).all())
    assert bool((grid["semantic_late"] >= 9 * units).all())
    for semantic, control in (
        ("semantic_full", "constant_full_matched"),
        ("semantic_nonlate", "constant_nonlate_matched"),
        ("semantic_late", "constant_late_matched"),
    ):
        assert len(grid[semantic]) == count
        assert len(torch.unique(grid[semantic])) == count
        torch.testing.assert_close(
            torch.bincount(grid[semantic] // units, minlength=layers),
            torch.bincount(grid[control] // units, minlength=layers),
        )
