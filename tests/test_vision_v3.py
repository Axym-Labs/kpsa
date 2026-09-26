import unittest

import torch
from transformers import Dinov2Config, Dinov2Model

from task_embeddings.vision_v3 import (
    Dinov2FeatureProbe,
    continual_class_groups,
    freeze_interpolated_position_embeddings,
)


class VisionV3Tests(unittest.TestCase):
    def make_model(self) -> Dinov2FeatureProbe:
        config = Dinov2Config(
            image_size=16,
            patch_size=4,
            num_channels=3,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            mlp_ratio=2,
        )
        return Dinov2FeatureProbe(Dinov2Model(config), num_classes=5)

    def test_probe_scores_every_mlp_feature_as_disjoint_rows_and_columns(self):
        model = self.make_model()
        logits = model(torch.randn(2, 3, 16, 16))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertEqual(model.layer_sizes, [32, 32])
        self.assertEqual(model.block_grad_norms().shape, (64,))
        self.assertTrue(torch.isfinite(model.block_grad_norms()).all())

    def test_position_embedding_freeze_removes_nondeterministic_gradient_path(self):
        backbone = self.make_model().backbone
        self.assertTrue(backbone.embeddings.position_embeddings.requires_grad)
        freeze_interpolated_position_embeddings(backbone)
        self.assertFalse(backbone.embeddings.position_embeddings.requires_grad)

    def test_feature_hook_supports_zero_mean_and_retain_interventions(self):
        model = self.make_model().eval()
        images = torch.randn(2, 3, 16, 16)
        baseline = model(images)
        selected = [torch.arange(32), torch.empty(0, dtype=torch.long)]
        model.set_intervention(selected, mode="zero")
        zeroed = model(images)
        self.assertFalse(torch.allclose(baseline, zeroed))
        model.set_intervention(
            selected, mode="mean", means=[torch.ones(32), torch.zeros(32)]
        )
        meaned = model(images)
        self.assertFalse(torch.allclose(zeroed, meaned))
        model.set_intervention(selected, mode="retain")
        retained = model(images)
        self.assertEqual(retained.shape, baseline.shape)

    def test_capture_state_can_be_cleared(self):
        model = self.make_model()
        model(torch.randn(1, 3, 16, 16), capture=True).sum().backward()
        self.assertEqual(len(model.captured_activations()), 2)
        model.clear_capture()
        with self.assertRaises(RuntimeError):
            model.captured_activations()

    def test_continual_groups_cover_classes_with_controlled_relatedness(self):
        fine_to_coarse = [fine // 5 for fine in range(100)]
        groups = continual_class_groups(fine_to_coarse)
        for name in ("related", "dissimilar"):
            flattened = [fine for group in groups[name] for fine in group]
            self.assertEqual(sorted(flattened), list(range(100)))
            self.assertEqual(len(groups[name]), 20)
        for group in groups["related"]:
            self.assertEqual(len({fine_to_coarse[fine] for fine in group}), 1)
        for group in groups["dissimilar"]:
            self.assertEqual(len({fine_to_coarse[fine] for fine in group}), 5)

    def test_gradient_scale_preserves_cross_layer_importance(self):
        model = self.make_model()
        for parameter in model.module_owned_parameters():
            parameter.grad = torch.ones_like(parameter)
        importance = torch.cat((torch.ones(32), torch.full((32,), 3.0)))
        model.apply_feature_gradient_scale(importance, 1.0)
        self.assertAlmostEqual(float(model.mlps[0].fc1.weight.grad[0, 0]), 0.5)
        self.assertAlmostEqual(float(model.mlps[1].fc1.weight.grad[0, 0]), 0.25)
        self.assertNotIn(
            id(model.classifier.weight),
            {id(parameter) for parameter in model.module_owned_parameters()},
        )


if __name__ == "__main__":
    unittest.main()
