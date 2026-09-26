import copy
import unittest

import torch

from task_embeddings.controlled_v3 import (
    ControlledConfig,
    ControlledTransformer,
    GatedFeedForward,
    make_task_batch,
    primitive_targets,
    run_experiment,
    task_mixtures,
)


class ControlledV3Tests(unittest.TestCase):
    def test_primitive_targets_match_hand_computed_sequence_properties(self):
        tokens = torch.tensor([[0, 1, 2, 3], [2, 2, 2, 2]])
        got = primitive_targets(tokens, token_modulus=4)
        expected = torch.tensor(
            [
                [1 / 3, -1.0, 1.0, -1.0],
                [-1.0, -1.0, 0.0, 1.0],
            ]
        )
        torch.testing.assert_close(got, expected)

    def test_task_mixtures_define_sixteen_normalized_mixtures(self):
        mixtures = task_mixtures()
        self.assertEqual(mixtures.shape, (16, 4))
        torch.testing.assert_close(mixtures.sum(dim=1), torch.ones(16))

    def test_gated_feature_intervention_can_zero_or_retain_features(self):
        layer = GatedFeedForward(d_model=2, d_ff=2)
        with torch.no_grad():
            layer.gate.weight.fill_(1.0)
            layer.gate.bias.zero_()
            layer.up.weight.copy_(torch.eye(2))
            layer.up.bias.zero_()
            layer.down.weight.copy_(torch.eye(2))
            layer.down.bias.zero_()
        x = torch.tensor([[[1.0, 2.0]]])
        baseline = layer(x)
        layer.set_intervention(torch.tensor([0]), mode="zero")
        zeroed = layer(x)
        self.assertNotEqual(
            float(baseline[0, 0, 0].detach()), float(zeroed[0, 0, 0].detach())
        )
        self.assertEqual(float(zeroed[0, 0, 0].detach()), 0.0)
        layer.set_intervention(torch.tensor([1]), mode="retain")
        retained = layer(x)
        self.assertEqual(float(retained[0, 0, 0].detach()), 0.0)

    def test_continuous_feature_gate_is_differentiable(self):
        layer = GatedFeedForward(d_model=2, d_ff=3)
        gate = torch.ones(3, requires_grad=True)
        layer.set_feature_gate(gate)
        layer(torch.ones(1, 1, 2)).sum().backward()
        self.assertIsNotNone(gate.grad)
        self.assertTrue(bool(torch.isfinite(gate.grad).all()))
        layer.set_feature_gate(None)

    def test_module_gradient_scores_cover_disjoint_gated_features(self):
        model = ControlledTransformer(
            token_modulus=8,
            sequence_length=4,
            n_tasks=16,
            d_model=16,
            n_layers=2,
            n_heads=4,
            d_ff=24,
        )
        tokens = torch.randint(0, 8, (3, 4))
        tasks = torch.tensor([0, 1, 2])
        model(tokens, tasks).sum().backward()
        scores = model.block_grad_norms()
        self.assertEqual(model.layer_sizes, [24, 24])
        self.assertEqual(scores.shape, (48,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(bool((scores > 0).any()))

    def test_captured_state_can_be_cleared_before_model_replication(self):
        model = ControlledTransformer(
            token_modulus=8,
            sequence_length=4,
            n_tasks=16,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=12,
        )
        model(
            torch.randint(0, 8, (1, 4)), torch.tensor([0]), capture=True
        ).sum().backward()
        model.clear_capture()
        cloned = copy.deepcopy(model)
        self.assertIsInstance(cloned, ControlledTransformer)

    def test_gradient_scale_uses_one_global_calibrated_importance(self):
        model = ControlledTransformer(
            token_modulus=8,
            sequence_length=4,
            n_tasks=2,
            d_model=4,
            n_layers=2,
            n_heads=2,
            d_ff=2,
        )
        for parameter in model.module_owned_parameters():
            parameter.grad = torch.ones_like(parameter)
        model.apply_feature_gradient_scale(torch.tensor([1.0, 3.0, 7.0, 15.0]), 1.0)
        expected = [torch.tensor([0.5, 0.25]), torch.tensor([0.125, 0.0625])]
        for block, scale in zip(model.blocks, expected):
            torch.testing.assert_close(block.mlp.gate.bias.grad, scale)
            torch.testing.assert_close(block.mlp.up.bias.grad, scale)
            torch.testing.assert_close(
                block.mlp.down.weight.grad,
                scale[None, :].expand_as(block.mlp.down.weight),
            )

    def test_module_owned_parameters_exclude_unprotected_paths(self):
        model = ControlledTransformer(
            token_modulus=8,
            sequence_length=4,
            n_tasks=2,
            d_model=4,
            n_layers=1,
            n_heads=2,
            d_ff=3,
        )
        owned = {id(parameter) for parameter in model.module_owned_parameters()}
        self.assertIn(id(model.blocks[0].mlp.gate.weight), owned)
        self.assertIn(id(model.blocks[0].mlp.down.weight), owned)
        self.assertNotIn(id(model.blocks[0].attention.in_proj_weight), owned)
        self.assertNotIn(id(model.output.weight), owned)

    def test_tiny_experiment_covers_all_three_applications(self):
        config = ControlledConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=12,
            sequence_length=4,
            train_steps=2,
            batch_size=4,
            reference_per_task=1,
            test_per_task=1,
            intervention_fractions=(0.5,),
            pruning_fractions=(0.5,),
            application_tasks=1,
            cl_tasks=2,
            cl_steps_per_task=1,
            recovery_steps=0,
        )
        result = run_experiment(config, device=torch.device("cpu"))
        self.assertEqual(result["setting"], "controlled_transformer")
        self.assertIn("source_fidelity", result)
        self.assertIn("interpretability", result)
        self.assertIn("pruning", result)
        self.assertIn("continual_learning", result)
        self.assertEqual(result["model"]["n_layers"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression")
    def test_task_batch_accepts_gpu_mixture_with_cpu_generator(self):
        config = ControlledConfig(sequence_length=4, token_modulus=8)
        tokens, targets, tasks = make_task_batch(
            config,
            task_mixtures().cuda(),
            task=0,
            batch_size=2,
            generator=torch.Generator().manual_seed(3),
            device=torch.device("cuda"),
        )
        self.assertEqual(tokens.device.type, "cuda")
        self.assertEqual(targets.device.type, "cuda")
        self.assertEqual(tasks.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
