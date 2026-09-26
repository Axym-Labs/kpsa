import unittest

import torch
from transformers import T5Config, T5ForConditionalGeneration

from task_embeddings.language import (
    T5NeuronProbe,
    format_glue_example,
    sequence_ce_residual_norm_sq,
)


def tiny_t5() -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_kv=8,
        d_ff=24,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return T5ForConditionalGeneration(config)


class LanguageModelTests(unittest.TestCase):
    def test_glue_formatting_preserves_task_semantics(self):
        prompt, answer = format_glue_example(
            "rte",
            {
                "sentence1": "A cat sleeps.",
                "sentence2": "An animal sleeps.",
                "label": 0,
            },
        )
        self.assertIn("rte", prompt)
        self.assertIn("premise", prompt)
        self.assertEqual(answer, "true")
        _, negative = format_glue_example(
            "rte",
            {
                "sentence1": "A cat sleeps.",
                "sentence2": "No animal sleeps.",
                "label": 1,
            },
        )
        self.assertEqual(negative, "false")

    def test_sequence_residual_norm_matches_cross_entropy_gradient(self):
        logits = torch.randn(2, 3, 7, requires_grad=True)
        labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100
        )
        (gradient,) = torch.autograd.grad(loss, logits)
        torch.testing.assert_close(
            sequence_ce_residual_norm_sq(logits.detach(), labels),
            gradient.square().sum(),
        )

    def test_probe_scores_all_encoder_and_decoder_mlp_neurons(self):
        wrapper = T5NeuronProbe(tiny_t5())
        inputs = torch.randint(2, 31, (2, 5))
        labels = torch.randint(2, 31, (2, 3))
        output = wrapper(input_ids=inputs, labels=labels)
        output.loss.backward()
        scores = wrapper.block_grad_norms()
        self.assertEqual(scores.shape, (wrapper.n_modules,))
        self.assertEqual(wrapper.n_modules, 4 * 24)
        self.assertTrue(torch.isfinite(scores).all())

    def test_neuron_ablation_changes_logits(self):
        wrapper = T5NeuronProbe(tiny_t5()).eval()
        inputs = torch.randint(2, 31, (2, 5))
        labels = torch.randint(2, 31, (2, 3))
        with torch.no_grad():
            baseline = wrapper(input_ids=inputs, labels=labels).logits
            wrapper.set_ablation([torch.tensor([0]) for _ in wrapper.layer_sizes])
            ablated = wrapper(input_ids=inputs, labels=labels).logits
            wrapper.set_ablation(None)
        self.assertFalse(torch.equal(baseline, ablated))


if __name__ == "__main__":
    unittest.main()
