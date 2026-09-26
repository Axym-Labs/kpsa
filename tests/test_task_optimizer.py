import importlib.util
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from task_embeddings import task_optimizer
from task_embeddings.controlled_v3 import ControlledTransformer
from task_embeddings.language_v3 import QwenFeatureProbe


class TaskOptimizerTests(unittest.TestCase):
    def test_task_optimizer_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("task_embeddings.task_optimizer"))

    def test_linear_provider_queries_without_materializing_task_atlas(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        atlas = (
            torch.tensor([[2.0], [5.0]])
            + torch.tensor([[3.0, 1.0], [-2.0, 4.0]]) @ features.T
        )

        provider = task_optimizer.LinearTaskScores.from_atlas(atlas, features)

        torch.testing.assert_close(provider.query(2), atlas[:, 2], atol=1e-4, rtol=1e-4)
        self.assertEqual(provider.stored_floats, 2 * 3 + 3 * 2 + 2)

    def test_task_indexed_adamw_broadcasts_and_globally_calibrates_preconditioner(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        provider = task_optimizer.MatrixTaskScores(torch.tensor([[4.0], [1.0]]))
        optimizer = task_optimizer.TaskIndexedAdamW(
            [task_optimizer.ParameterGroupSpec(parameter, torch.tensor([0, 1]))],
            provider,
            lr=0.1,
            beta1=0.0,
            eps=0.0,
            weight_decay=0.0,
        )
        parameter.grad = torch.tensor([2.0, 2.0])

        optimizer.step(task_ids=torch.tensor([0, 0]))

        expected = torch.tensor(
            [1.0 - 0.2 / (4.0 / 2.5 * 4.0) ** 0.5, 1.0 - 0.2 / (1.0 / 2.5 * 4.0) ** 0.5]
        )
        torch.testing.assert_close(parameter, expected)

    def test_task_indexed_adamw_is_invariant_to_global_score_units(self):
        parameters = [
            torch.nn.Parameter(torch.tensor([1.0, 1.0])),
            torch.nn.Parameter(torch.tensor([1.0, 1.0])),
        ]
        for parameter, scale in zip(parameters, (1.0, 1e6)):
            optimizer = task_optimizer.TaskIndexedAdamW(
                [task_optimizer.ParameterGroupSpec(parameter, torch.tensor([0, 1]))],
                task_optimizer.MatrixTaskScores(scale * torch.tensor([[4.0], [1.0]])),
                lr=0.1,
                beta1=0.0,
                eps=1e-12,
                weight_decay=0.0,
            )
            parameter.grad = torch.tensor([2.0, 2.0])
            optimizer.step(task_ids=0)

        torch.testing.assert_close(parameters[0], parameters[1])

    def test_task_indexed_adamw_rejects_mixed_task_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = task_optimizer.TaskIndexedAdamW(
            [task_optimizer.ParameterGroupSpec(parameter, torch.tensor([0]))],
            task_optimizer.MatrixTaskScores(torch.ones(1, 2)),
        )
        parameter.grad = torch.ones_like(parameter)

        with self.assertRaisesRegex(ValueError, "task-homogeneous"):
            optimizer.step(task_ids=torch.tensor([0, 1]))

    def test_resource_counts_separate_preconditioner_and_total_state(self):
        counts = task_optimizer.optimizer_resource_counts(
            n_parameters=20_480,
            n_groups=512,
            n_tasks=30,
            dimension=6,
        )

        self.assertEqual(counts["consolidated_diagonal_preconditioner"], 20_480)
        self.assertEqual(counts["tbe_preconditioner"], 3_770)
        self.assertAlmostEqual(counts["tbe_vs_consolidated_reduction"], 0.81591796875)
        self.assertEqual(counts["adamw_total_state"], 40_960)
        self.assertEqual(counts["task_indexed_tbe_total_state"], 24_250)
        self.assertAlmostEqual(counts["tbe_vs_full_group_atlas_ratio"], 15_360 / 3_770)
        self.assertAlmostEqual(
            counts["task_indexed_tbe_vs_adamw_ratio"], 40_960 / 24_250
        )
        self.assertAlmostEqual(
            counts["task_indexed_tbe_vs_full_group_atlas_ratio"],
            (20_480 + 15_360) / 24_250,
        )

    def test_transformer_group_mapping_covers_every_owned_parameter_once(self):
        model = ControlledTransformer(
            token_modulus=8,
            sequence_length=4,
            n_tasks=3,
            d_model=8,
            n_layers=2,
            n_heads=2,
            d_ff=4,
        )

        specs = task_optimizer.controlled_transformer_group_specs(model)

        self.assertEqual(
            {id(spec.parameter) for spec in specs},
            {id(parameter) for parameter in model.module_owned_parameters()},
        )
        self.assertEqual(
            sorted(
                torch.cat([spec.group_ids.flatten() for spec in specs])
                .unique()
                .tolist()
            ),
            list(range(model.n_modules)),
        )

    def test_group_mapping_supports_modern_qwen_swiglu_blocks(self):
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        model = QwenFeatureProbe(Qwen3ForCausalLM(config))

        specs = task_optimizer.transformer_mlp_group_specs(model)

        self.assertEqual(
            {id(spec.parameter) for spec in specs},
            {id(parameter) for parameter in model.module_owned_parameters()},
        )
        self.assertEqual(
            sorted(
                torch.cat([spec.group_ids.flatten() for spec in specs])
                .unique()
                .tolist()
            ),
            list(range(model.n_modules)),
        )

    def test_relative_provider_makes_scores_positive_without_extra_atlas(self):
        base = task_optimizer.MeanTaskScores(torch.tensor([0.0, 2.0]), n_tasks=3)
        provider = task_optimizer.RelativeTaskScores(base, strength=1.0)

        torch.testing.assert_close(provider.query(2), torch.tensor([1.0, 3.0]))
        self.assertEqual(provider.stored_floats, 2)


if __name__ == "__main__":
    unittest.main()
