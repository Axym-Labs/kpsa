import torch
from torch import nn

from kpsa.parameter_groups import ParameterPartition


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].mlp = nn.Module()
        self.blocks[0].mlp.fc1 = nn.Linear(3, 4)
        self.blocks[0].mlp.fc2 = nn.Linear(4, 3, bias=False)


def test_partition_conserves_gradient_and_weight_energy():
    model = TinyMLP()
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    gradient_energy = sum(
        parameter.grad.square().sum() for parameter in model.parameters()
    )
    weight_energy = sum(parameter.square().sum() for parameter in model.parameters())

    for kind in ("row", "tensor", "swiglu"):
        partition = ParameterPartition(model, kind)
        torch.testing.assert_close(
            (partition.gradient_scores() * partition.sizes).sum(), gradient_energy
        )
        torch.testing.assert_close(
            (partition.weight_energy() * partition.sizes).sum(), weight_energy
        )
        assert partition.n_parameters == sum(
            parameter.numel() for parameter in model.parameters()
        )


def test_swiglu_partition_couples_input_and_output_features():
    partition = ParameterPartition(TinyMLP(), "swiglu")
    fc1 = next(spec for spec in partition.slices if spec.name.endswith("fc1.weight"))
    fc2 = next(spec for spec in partition.slices if spec.name.endswith("fc2.weight"))

    assert fc1.offset == fc2.offset
    assert fc1.count == fc2.count == 4


def test_uncached_group_sizes_match_cached_sizes_without_persistent_tensor():
    model = TinyMLP()
    cached = ParameterPartition(model, "swiglu")
    compact = ParameterPartition(model, "swiglu", cache_sizes=False)

    assert compact.metadata_bytes == 0
    torch.testing.assert_close(compact.sizes, cached.sizes)
