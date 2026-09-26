import pytest
import torch

from kpsa.protein_neuron_localization_v8 import EsmSensitivity


def toy_experiment():
    experiment = EsmSensitivity.__new__(EsmSensitivity)
    experiment.device = torch.device("cpu")
    experiment.units_per_layer = 3
    incoming = torch.nn.Linear(2, 3)
    outgoing = torch.nn.Linear(3, 2, bias=False)
    experiment.ff_pairs = [("encoder.0", incoming, outgoing)]
    return experiment, incoming, outgoing


def test_outgoing_parameter_noise_is_deterministic_scoped_and_restored():
    experiment, incoming, outgoing = toy_experiment()
    incoming_before = incoming.weight.detach().clone()
    outgoing_before = outgoing.weight.detach().clone()

    def perturbed():
        with experiment.additive_parameter_noise(
            torch.tensor([1]),
            0.1,
            component="outgoing",
            seed=17,
            sign=1,
        ):
            assert torch.equal(incoming.weight, incoming_before)
            assert torch.equal(outgoing.weight[:, 0], outgoing_before[:, 0])
            assert torch.equal(outgoing.weight[:, 2], outgoing_before[:, 2])
            return outgoing.weight.detach().clone()

    first = perturbed()
    assert torch.equal(outgoing.weight, outgoing_before)
    second = perturbed()
    assert torch.equal(first, second)
    assert torch.allclose((first[:, 1] - outgoing_before[:, 1]).abs(), torch.full((2,), 0.1))


def test_parameter_rms_uses_only_requested_scoped_component():
    experiment, _incoming, outgoing = toy_experiment()
    with torch.no_grad():
        outgoing.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    observed = experiment.parameter_rms(torch.tensor([1]), "outgoing")
    expected = torch.tensor([2.0, 5.0]).square().mean().sqrt().item()
    assert observed == pytest.approx(expected)
