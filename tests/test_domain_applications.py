import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from task_embeddings import domain_applications
from task_embeddings.domain_applications import budget_mask, query_scores
from task_embeddings.domain_optimizer import ParameterPartition
from task_embeddings.domain_train import evaluate_details


class DomainApplicationTests(unittest.TestCase):
    def test_precision_budget_excludes_unchanged_vectors(self):
        model = torch.nn.ModuleDict(
            {
                "model": torch.nn.ModuleDict(
                    {
                        "layers": torch.nn.ModuleList([torch.nn.Linear(4, 3)]),
                        "norm": torch.nn.LayerNorm(3),
                        "embedding": torch.nn.Embedding(5, 4),
                    }
                )
            }
        )
        for kind in ("row", "tensor", "swiglu"):
            part = ParameterPartition(model, kind)
            all_matrices = domain_applications.precision_scope(part, "all")
            core_matrices = domain_applications.precision_scope(part, "transformer")
            self.assertEqual(int(part.sizes[all_matrices].sum()), 32)
            self.assertEqual(int(part.sizes[core_matrices].sum()), 12)
            if kind == "row":
                _, actual = domain_applications.scoped_budget_mask(
                    torch.ones(part.n_groups),
                    part.sizes,
                    0.5,
                    core_matrices,
                )
                # A 6-parameter cap fits one 4-parameter row, not two bias scalars.
                self.assertAlmostEqual(actual, 1 / 3)

    def test_cluster_baseline_pools_only_observed_task_profiles(self):
        features = torch.tensor([[-3.0], [-2.0], [2.0], [3.0], [-2.5], [2.5]])
        atlas = torch.tensor([[1.0, 3.0, 10.0, 14.0, 999.0, 999.0]])
        result = query_scores(atlas, features, "cluster", 7, observed=[0, 1, 2, 3])
        torch.testing.assert_close(
            result, torch.tensor([[2.0, 2.0, 12.0, 12.0, 2.0, 12.0]])
        )

    def test_cluster_shrinkage_zero_and_duplicate_features_reduce_to_source_mean(self):
        atlas = torch.tensor([[1.0, 3.0, 999.0]])
        for features, scale in (
            (torch.zeros(3, 1), 1.0),
            (torch.arange(3.0)[:, None], 0.0),
        ):
            actual = query_scores(
                atlas, features, "cluster", 7, observed=[0, 1], residual_scale=scale
            )
            torch.testing.assert_close(actual, torch.full((1, 3), 2.0))

    def test_skip_pruning_also_disables_token_frequency_control(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from task_embeddings.domain_train import DomainConfig

        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                use_cache=False,
            )
        )
        groups = ParameterPartition(model, "row").n_groups
        profiles = {"row": {"raw": torch.ones(groups, 4)}}

        def evaluate(model, domains, limit):
            return torch.ones(len(domains)), [{} for _ in domains]

        with TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / "model.pt"
            torch.save({}, checkpoint)
            blocks = torch.tensor([[1, 2, 3, 4]])
            torch.save(
                {"train": [blocks] * 4, "validation": [blocks] * 4},
                root / "corpus.pt",
            )
            torch.save(
                {
                    "features": torch.arange(4.0)[:, None],
                    "full_histograms": torch.ones(4, 16),
                },
                root / "token_features.pt",
            )
            with (
                patch.object(
                    domain_applications,
                    "load_checkpoint",
                    return_value=(model, DomainConfig()),
                ),
                patch.object(domain_applications, "profile", return_value=profiles),
                patch.object(
                    domain_applications, "evaluate_details", side_effect=evaluate
                ),
                patch.object(
                    ParameterPartition,
                    "scale_parameters",
                    side_effect=AssertionError("pruning ran despite empty fractions"),
                ),
                patch.object(torch.Tensor, "cuda", lambda tensor: tensor),
            ):
                result = domain_applications.run_applications(
                    checkpoint,
                    root,
                    root / "result.json",
                    samples=1,
                    eval_blocks=1,
                    causal_groups=0,
                    partition_kinds=("row",),
                    feature_file="token_features.pt",
                    methods=(),
                    pruning_fractions=(),
                    precision_fractions=(),
                )
            self.assertEqual(result["pruning"], [])

    def test_scoped_budget_never_modifies_frozen_groups(self):
        mask, actual = domain_applications.scoped_budget_mask(
            torch.tensor([-100.0, 1.0, 2.0]),
            torch.tensor([100.0, 2.0, 2.0]),
            0.5,
            torch.tensor([False, True, True]),
        )
        torch.testing.assert_close(mask, torch.tensor([1.0, 0.0, 1.0]))
        self.assertEqual(actual, 0.5)

    def test_tensor_ranks_average_ties(self):
        actual = domain_applications.tensor_ranks(torch.tensor([3.0, 1.0, 1.0, 2.0]))
        torch.testing.assert_close(actual, torch.tensor([3.0, 0.5, 0.5, 2.0]))

    def test_precision_candidates_leave_normalization_in_full_precision(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))
        with torch.no_grad():
            model[1].weight.copy_(torch.linspace(0.01, 2.0, 8))
        part = ParameterPartition(model, "row")
        low, high, _, _ = domain_applications.precision_candidates(part)
        for name, value in model.named_parameters():
            if value.ndim < 2:
                torch.testing.assert_close(low[name], value)
                torch.testing.assert_close(high[name], value)

    def test_evaluation_details_are_token_weighted_and_restore_mode(self):
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(vocab_size=3)
        model.eval()
        data = torch.tensor([[0, 0, 0], [0, 1, -1]])
        logits = torch.tensor(
            [[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        )

        def fake_loss(model, blocks):
            return torch.tensor(0.0), logits[: len(blocks)]

        with patch("task_embeddings.domain_train.batch_loss", side_effect=fake_loss):
            losses, details = evaluate_details(model, [data], limit=2)
        row = details[0]
        self.assertEqual(sum(row["token_counts"]), 3)
        self.assertAlmostEqual(float(losses[0]), sum(row["loss_sums"]) / 3, places=6)
        self.assertFalse(model.training)

    def test_cold_application_scores_withhold_entire_query_fold(self):
        atlas = torch.arange(32, dtype=torch.float32).reshape(4, 8) + 1
        features = torch.arange(8, dtype=torch.float32)[:, None]
        for method in ("tbe", "jl", "mean", "permuted", "cluster"):
            scores = domain_applications.application_scores(
                atlas, features, method, 7, query_mode="cold", score_link="log"
            )
            changed = atlas.clone()
            changed[:, [1, 5]] *= 100
            other = domain_applications.application_scores(
                changed, features, method, 7, query_mode="cold", score_link="log"
            )
            torch.testing.assert_close(scores[:, [1, 5]], other[:, [1, 5]])

    def test_symmetric_quantization_uses_stated_bit_grid(self):
        value = torch.tensor([[-1.0, 0.5, 1.0]])
        torch.testing.assert_close(
            domain_applications.quantize_weight(value, 2),
            torch.tensor([[-1.0, 0.0, 1.0]]),
        )
        torch.testing.assert_close(
            domain_applications.quantize_weight(value, 16), value
        )

    def test_quantization_groups_do_not_share_an_outlier_scale(self):
        value = torch.cat((torch.ones(1, 128), 100 * torch.ones(1, 128)), 1)
        quantized = domain_applications.quantize_weight(value, 2, group_size=128)
        torch.testing.assert_close(quantized, value)

    def test_parameter_budget_does_not_count_groups_as_parameters(self):
        mask, removed = budget_mask(
            torch.tensor([0.0, 1.0, 2.0]), torch.tensor([2.0, 3.0, 5.0]), 0.5
        )
        torch.testing.assert_close(mask, torch.tensor([0.0, 0.0, 1.0]))
        self.assertEqual(removed, 0.5)

    def test_budget_skips_oversized_group_instead_of_stopping(self):
        mask, used = budget_mask(
            torch.tensor([0.0, 1.0, 2.0]), torch.tensor([6.0, 3.0, 1.0]), 0.5
        )
        torch.testing.assert_close(mask, torch.tensor([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(used, 0.4)

    def test_budget_never_exceeds_cap_with_multiple_large_groups(self):
        mask, used = budget_mask(
            torch.arange(5.0), torch.tensor([3.0, 8.0, 2.0, 6.0, 1.0]), 0.3
        )
        torch.testing.assert_close(mask, torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))
        self.assertAlmostEqual(used, 0.3)

    def test_cold_start_fit_cannot_use_hidden_importance_column(self):
        features = torch.tensor([[-1.0], [0.0], [1.0]])
        first = torch.tensor([[1.0, 2.0, 999.0], [2.0, 3.0, 999.0]])
        second = first.clone()
        second[:, 2] = -10000
        for link in ("linear", "log"):
            a = query_scores(
                first, features, "tbe", 0, observed=[0, 1], score_link=link
            )
            b = query_scores(
                second, features, "tbe", 0, observed=[0, 1], score_link=link
            )
            torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
