import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from task_embeddings.domain_gates import FeatureGates, grouped_scores


class FeatureGateTests(unittest.TestCase):
    def model(self):
        return Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                use_cache=False,
            )
        ).eval()

    def test_identity_removal_and_gradients(self):
        model = self.model()
        x = torch.tensor([[1, 2, 3]])
        reference = model(x).logits.detach()
        with FeatureGates(model) as gates:
            torch.testing.assert_close(model(x).logits, reference, rtol=0, atol=0)
            model(x).logits.sum().backward()
            self.assertEqual(gates.gradients().shape, (2, 16))
            self.assertGreater(float(gates.gradients().abs().sum()), 0)
            gates.set(torch.full((2, 16), 0.8))
            self.assertFalse(torch.equal(model(x).logits, reference))
        torch.testing.assert_close(model(x).logits, reference, rtol=0, atol=0)

    def test_joint_gate_gradient_is_summed_before_squaring(self):
        gradient = torch.tensor([[1.0, -1.0, 2.0, 3.0]])
        torch.testing.assert_close(
            grouped_scores(gradient, 2), torch.tensor([[0.0, 25.0]])
        )

    def test_gate_gradient_matches_down_weight_scaling_direction(self):
        model = self.model()
        with FeatureGates(model) as gates:
            model(torch.tensor([[1, 2, 3]])).logits.square().sum().backward()
            expected = torch.stack(
                [
                    (layer.weight.detach() * layer.weight.grad).sum(0)
                    for layer in gates.layers
                ]
            )
            torch.testing.assert_close(gates.gradients(), expected)

    def test_gate_profiles_measure_actual_multiplier_and_restore_parameter_flags(self):
        from task_embeddings.domain_scaling import gate_profiles

        model = self.model()
        first = next(model.parameters())
        first.requires_grad_(False)
        flags = [p.requires_grad for p in model.parameters()]
        data = [torch.tensor([[1, 2, 3], [3, 2, 1]])]
        result = gate_profiles(model, data, samples=2)
        self.assertEqual(result["raw"].shape, (32, 1))
        self.assertEqual(result["mean_gradient"].shape, (32, 1))
        self.assertTrue(torch.isfinite(result["normalized"]).all())
        self.assertGreater(float(result["raw"].sum()), 0)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        self.assertTrue(all(not m._forward_pre_hooks for m in model.modules()))

    def test_calibration_directional_diagnostic_retains_gradient_sign(self):
        from task_embeddings.domain_scaling import gate_profiles
        from task_embeddings.domain_train import batch_loss

        model = self.model()
        batch = torch.tensor([[1, 2, 3]])
        with FeatureGates(model) as gates:
            loss, _ = batch_loss(model, batch)
            loss.backward()
            expected = gates.gradients().flatten().clone()
        actual = gate_profiles(model, [batch], samples=1)
        torch.testing.assert_close(actual["mean_gradient"][:, 0], expected)
        torch.testing.assert_close(actual["raw"][:, 0], expected.square())

    def test_scaling_interface_rejects_masks_and_nonfinite_values(self):
        model = self.model()
        with FeatureGates(model) as gates:
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    gates.set(torch.full((2, 16), value))


if __name__ == "__main__":
    unittest.main()
