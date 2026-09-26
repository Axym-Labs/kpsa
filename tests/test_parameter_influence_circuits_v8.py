import torch
from torch import nn

from task_embeddings.domain_optimizer import ParameterPartition
from task_embeddings.parameter_influence_circuits_v8 import (
    additive_group_noise,
    scoped_parameter_rms,
)


def test_additive_group_noise_changes_shared_group_and_restores_exactly():
    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Module()])
            self.blocks[0].mlp = nn.Module()
            self.blocks[0].mlp.fc1 = nn.Linear(2, 3)
            self.blocks[0].mlp.fc2 = nn.Linear(3, 2, bias=False)

    model = TinyMLP()
    partition = ParameterPartition(model, "swiglu")
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    selected = torch.zeros(partition.n_groups, dtype=torch.bool)
    first = next(
        spec for spec in partition.slices if spec.name == "blocks.0.mlp.fc1.weight"
    )
    selected[first.offset + 1] = True

    with additive_group_noise(partition, selected, 0.25, seed=7):
        mlp = model.blocks[0].mlp
        assert not torch.equal(
            mlp.fc1.weight[1], original["blocks.0.mlp.fc1.weight"][1]
        )
        assert not torch.equal(mlp.fc1.bias[1], original["blocks.0.mlp.fc1.bias"][1])
        assert not torch.equal(
            mlp.fc2.weight[:, 1], original["blocks.0.mlp.fc2.weight"][:, 1]
        )
        torch.testing.assert_close(
            mlp.fc1.weight[[0, 2]], original["blocks.0.mlp.fc1.weight"][[0, 2]]
        )

    for name, value in model.named_parameters():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_antithetic_group_noise_has_exact_opposite_displacement():
    layer = nn.Linear(3, 2, bias=False)
    partition = ParameterPartition(layer, "row")
    selected = torch.tensor([True, False])
    original = layer.weight.detach().clone()
    with additive_group_noise(partition, selected, 0.1, seed=11, sign=1):
        positive = layer.weight.detach().clone() - original
    with additive_group_noise(partition, selected, 0.1, seed=11, sign=-1):
        negative = layer.weight.detach().clone() - original
    torch.testing.assert_close(positive, -negative)
    torch.testing.assert_close(layer.weight, original, rtol=0, atol=0)


def test_scoped_parameter_rms_uses_only_selected_rows():
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 4.0], [10.0, 10.0]]))
    partition = ParameterPartition(layer, "row")
    scope = torch.tensor([True, False])
    assert scoped_parameter_rms(partition, scope) == (25 / 2) ** 0.5
