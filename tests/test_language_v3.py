import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from task_embeddings.language_v3 import (
    QwenFeatureProbe,
    causal_residual_norm_sq,
    encode_instruction,
)


class LanguageV3Tests(unittest.TestCase):
    def make_model(self) -> QwenFeatureProbe:
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        return QwenFeatureProbe(Qwen3ForCausalLM(config))

    def test_probe_scores_every_swiglu_triplet(self):
        model = self.make_model()
        inputs = torch.randint(0, 64, (2, 6))
        labels = inputs.clone()
        output = model(input_ids=inputs, labels=labels)
        output.loss.backward()
        self.assertEqual(model.layer_sizes, [32, 32])
        self.assertEqual(model.block_grad_norms().shape, (64,))
        self.assertTrue(torch.isfinite(model.block_grad_norms()).all())

    def test_down_projection_hook_intervenes_on_swiglu_features(self):
        model = self.make_model().eval()
        inputs = torch.randint(0, 64, (1, 6))
        baseline = model(input_ids=inputs).logits
        selected = [torch.arange(32), torch.empty(0, dtype=torch.long)]
        model.set_intervention(selected, mode="zero")
        zeroed = model(input_ids=inputs).logits
        self.assertFalse(torch.allclose(baseline, zeroed))
        model.set_intervention(
            selected, mode="mean", means=[torch.ones(32), torch.zeros(32)]
        )
        meaned = model(input_ids=inputs).logits
        self.assertFalse(torch.allclose(zeroed, meaned))
        model.set_intervention(selected, mode="retain")
        self.assertEqual(model(input_ids=inputs).logits.shape, baseline.shape)

    def test_probe_supports_reversible_continuous_feature_gates(self):
        model = self.make_model().eval()
        inputs = torch.randint(0, 64, (1, 6))
        baseline = model(input_ids=inputs).logits

        model.set_feature_gates([torch.ones(32), torch.ones(32)])
        torch.testing.assert_close(model(input_ids=inputs).logits, baseline)
        model.set_feature_gates([torch.zeros(32), torch.ones(32)])
        self.assertFalse(torch.allclose(model(input_ids=inputs).logits, baseline))
        model.set_feature_gates(None)
        torch.testing.assert_close(model(input_ids=inputs).logits, baseline)

    def test_residual_norm_matches_cross_entropy_logit_gradient(self):
        logits = torch.randn(2, 4, 7, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 2], [-100, 1, 0, -100]])
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100
        )
        (gradient,) = torch.autograd.grad(loss, logits)
        torch.testing.assert_close(
            causal_residual_norm_sq(logits.detach(), labels), gradient.square().sum()
        )

    def test_capture_state_can_be_cleared(self):
        model = self.make_model()
        inputs = torch.randint(0, 64, (1, 4))
        model(input_ids=inputs, capture=True).logits.sum().backward()
        self.assertEqual(len(model.captured_activations()), 2)
        model.clear_capture()
        with self.assertRaises(RuntimeError):
            model.captured_activations()

    def test_instruction_encoding_masks_prompt_and_padding(self):
        class ToyTokenizer:
            eos_token_id = 9
            pad_token_id = 0

            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [1 + (ord(character) % 7) for character in text]

        encoded = encode_instruction(ToyTokenizer(), "abc", "x", max_length=8)
        self.assertEqual(encoded["input_ids"].shape, (8,))
        self.assertEqual(encoded["attention_mask"].tolist()[-1], 1)
        self.assertEqual(encoded["labels"].tolist()[-1], 9)
        valid = encoded["labels"] != -100
        self.assertEqual(int(valid.sum()), 3)  # leading space, answer token, EOS
        self.assertTrue(bool((encoded["labels"][~valid] == -100).all()))

    def test_gradient_scale_preserves_cross_layer_importance(self):
        model = self.make_model()
        for parameter in model.module_owned_parameters():
            parameter.grad = torch.ones_like(parameter)
        importance = torch.cat((torch.ones(32), torch.full((32,), 3.0)))
        model.apply_feature_gradient_scale(importance, 1.0)
        self.assertAlmostEqual(float(model.mlps[0].gate_proj.weight.grad[0, 0]), 0.5)
        self.assertAlmostEqual(float(model.mlps[1].gate_proj.weight.grad[0, 0]), 0.25)
        self.assertNotIn(
            id(model.model.lm_head.weight),
            {id(parameter) for parameter in model.module_owned_parameters()},
        )


if __name__ == "__main__":
    unittest.main()
