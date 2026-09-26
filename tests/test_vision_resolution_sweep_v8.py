import torch
from torch import nn

from kpsa.parameter_groups import ParameterPartition
from kpsa.vision_causal_refined import mlp_feature_layout
from kpsa.vision_resolution_sweep_v8 import (
    aggregate_groups,
    expand_selection,
    nested_feature_assignment,
)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(2, 4)
        self.mlp.fc2 = nn.Linear(4, 2)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])


def test_nested_assignment_respects_layers_and_bundle_width():
    model = TinyModel()
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    assignment, labels = nested_feature_assignment(partition, layout, 2)
    scoped = assignment >= 0
    assert len(labels) == 4
    assert assignment[scoped].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_aggregate_and_expand_are_additive():
    model = TinyModel()
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    assignment, labels = nested_feature_assignment(partition, layout, 2)
    values = torch.arange(partition.n_groups, dtype=torch.float32)
    coarse = aggregate_groups(values, assignment, len(labels))
    scoped_values = values[assignment >= 0]
    torch.testing.assert_close(
        coarse, scoped_values.reshape(len(labels), 2).sum(1)
    )
    selected = torch.tensor([False, True, True, False])
    expanded = expand_selection(selected, assignment)
    assert expanded[assignment >= 0].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
