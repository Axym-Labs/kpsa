import unittest

import torch

from task_embeddings.vision import (
    CifarResNet18,
    ClassHomogeneousBatchSampler,
    GradientMeanBuffer,
    cross_entropy_residual_norm_sq,
)


class VisionModelTests(unittest.TestCase):
    def test_gradient_buffer_applies_mean_of_task_gradients(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        buffer = GradientMeanBuffer([parameter])
        parameter.grad = torch.tensor([1.0, 3.0])
        buffer.add()
        parameter.grad = torch.tensor([3.0, 5.0])
        buffer.add()
        buffer.apply_mean()
        torch.testing.assert_close(parameter.grad, torch.tensor([2.0, 4.0]))
        self.assertEqual(buffer.count, 0)

    def test_cross_entropy_residual_norm_matches_logit_gradient(self):
        logits = torch.tensor([[1.0, -0.5, 0.25], [0.1, 0.2, 0.3]], requires_grad=True)
        targets = torch.tensor([0, 2])
        loss = torch.nn.functional.cross_entropy(logits, targets)
        (grad,) = torch.autograd.grad(loss, logits)
        expected = grad.square().sum()
        torch.testing.assert_close(
            cross_entropy_residual_norm_sq(logits.detach(), targets), expected
        )

    def test_homogeneous_sampler_is_balanced_and_each_batch_has_one_class(self):
        labels = [0, 0, 0, 1, 1, 1]
        sampler = ClassHomogeneousBatchSampler(labels, batch_size=4, steps=6, seed=7)
        seen = []
        for batch in sampler:
            batch_labels = {labels[index] for index in batch}
            self.assertEqual(len(batch_labels), 1)
            seen.append(next(iter(batch_labels)))
        self.assertEqual(seen.count(0), seen.count(1))

    def test_selected_channel_blocks_produce_one_score_per_channel(self):
        model = CifarResNet18(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        loss = model(x).sum()
        loss.backward()
        scores = model.block_grad_norms()
        self.assertEqual(scores.shape, (model.n_modules,))
        self.assertEqual(sum(model.layer_sizes), model.n_modules)
        self.assertTrue(torch.isfinite(scores).all())

    def test_layer_balanced_ablation_changes_output(self):
        model = CifarResNet18(num_classes=10).eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            baseline = model(x)
            selected = [torch.tensor([0]) for _ in model.layer_sizes]
            model.set_ablation(selected)
            ablated = model(x)
            model.set_ablation(None)
        self.assertFalse(torch.equal(baseline, ablated))


if __name__ == "__main__":
    unittest.main()
