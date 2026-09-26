import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from task_embeddings import domain_gates


class ContinuousScalingTests(unittest.TestCase):
    def test_gate_coordinate_screen_runs_end_to_end_on_a_tiny_model(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from task_embeddings.domain_scaling import run

        model = Qwen3ForCausalLM(
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
        with TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / "model.pt"
            torch.save({}, checkpoint)
            blocks = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]])
            torch.save(
                {
                    "train": [blocks, blocks.flip(1)],
                    "validation": [blocks.roll(1, 1), blocks.roll(2, 1)],
                    "domains": ["a", "b"],
                },
                root / "corpus.pt",
            )
            torch.save(
                {"features": torch.tensor([[-1.0], [1.0]])}, root / "token_features.pt"
            )
            cache = root / "profiles.pt"
            torch.save(
                {
                    "signature": {
                        "checkpoint": str(checkpoint.resolve()),
                        "corpus": str((root / "corpus.pt").resolve()),
                        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                        "samples": 1,
                    }
                },
                cache,
            )
            with patch(
                "task_embeddings.domain_scaling.load_checkpoint",
                return_value=(model, None),
            ):
                result = run(
                    checkpoint,
                    root,
                    cache,
                    root / "result.json",
                    eval_blocks=1,
                    strengths=(0.1,),
                    estimators=("raw", "normalized", "fisher"),
                    coordinate="gate",
                )
            self.assertTrue(result["complete"])
            self.assertEqual(result["identity_max_abs_error"], 0)
            self.assertEqual(result["restoration_max_abs_error"], 0)
            self.assertEqual(len(result["records"]), 22)
            for row in result["records"]:
                self.assertEqual(len(row["calibration_first_order_delta_nll"]), 2)
                self.assertTrue(
                    all(d["zero_count"] == 0 for d in row["multiplier_diagnostics"])
                )

    def test_uniform_control_needs_no_importance_and_matches_log_budget(self):
        self.assertTrue(hasattr(domain_gates, "intervention_scales"))
        shape = (2, 3)
        uniform = domain_gates.intervention_scales(None, shape, "uniform", 0.2)
        directed = domain_gates.continuous_scales(
            torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), 0.2
        )
        torch.testing.assert_close(
            uniform.log().square().mean(), directed.log().square().mean()
        )
        torch.testing.assert_close(
            domain_gates.intervention_scales(None, shape, "identity", 0.2),
            torch.ones(shape),
        )

    def test_scaling_is_continuous_positive_and_not_binary(self):
        self.assertTrue(hasattr(domain_gates, "continuous_scales"))
        fn = domain_gates.continuous_scales
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(fn(scores, 0.0), torch.ones_like(scores))
        expected = torch.tensor([[-0.2, 0.0, 0.2]]).exp()
        torch.testing.assert_close(fn(scores, 0.2), expected)
        torch.testing.assert_close(fn(scores, 0.1).square(), expected)
        self.assertGreater(float(fn(scores, 10.0).min()), 0)

    def test_flat_importance_does_not_invent_a_direction(self):
        self.assertTrue(hasattr(domain_gates, "continuous_scales"))
        torch.testing.assert_close(
            domain_gates.continuous_scales(torch.ones(2, 4), 0.3), torch.ones(2, 4)
        )

    def test_tied_scores_get_identical_scaling(self):
        self.assertTrue(hasattr(domain_gates, "continuous_scales"))
        actual = domain_gates.continuous_scales(torch.tensor([[1.0, 1.0, 2.0]]), 0.2)
        self.assertEqual(float(actual[0, 0]), float(actual[0, 1]))
        self.assertLess(float(actual[0, 0]), float(actual[0, 2]))


if __name__ == "__main__":
    unittest.main()
