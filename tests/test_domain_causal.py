import unittest

import torch

from task_embeddings import domain_causal
from task_embeddings.domain_causal import stable_kl


class CausalNumericsTests(unittest.TestCase):
    def test_multilingual_pilot_checks_shape_and_includes_gain_only_control(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from task_embeddings.domain_optimizer import ParameterPartition

        torch.manual_seed(53)
        model = self.model()
        part = ParameterPartition(model, "swiglu")
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        with TemporaryDirectory() as root:
            data = Path(root)
            checkpoint = data / "model.pt"
            corpus = {
                "parallel_examples": True,
                "domains": [f"language_{i}" for i in range(20)],
                "validation": [torch.randint(0, 16, (4, 5)) for _ in range(20)],
            }
            torch.save(corpus, data / "corpus.pt")
            torch.save({"features": torch.randn(20, 2)}, data / "token_features.pt")
            payload = {
                "checkpoint": str(checkpoint.resolve()),
                "data": str(data.resolve()),
                "samples": 2,
                "start": 0,
                "scores": {
                    "swiglu": {"raw": {"pruning": torch.rand(part.n_groups, 20) + 0.1}}
                },
            }
            torch.save(payload, data / "source.pt")
            payload["start"] = 2
            torch.save(payload, data / "target.pt")
            with patch.object(
                domain_causal, "load_checkpoint", return_value=(model, {})
            ):
                result = domain_causal.multilingual_bundle_audit(
                    checkpoint,
                    data,
                    data / "source.pt",
                    data / "target.pt",
                    data / "result.json",
                    groups_count=4,
                    width=4,
                    contexts=2,
                    directions=1,
                )
            self.assertEqual(result["identity_max_kl"], 0)
            self.assertEqual(len(result["shape_effect_repeatability"]), 5)
            self.assertEqual(result["data_role"], "validation")
            self.assertFalse(result["claim_ready"])
            self.assertIn("gain_only", {row["method"] for row in result["results"]})
            for row in result["results"]:
                if row["method"] in ("gain_only", "mean"):
                    self.assertEqual(row["shape_residual_causal_spearman"], [None] * 5)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, initial[k], atol=0, rtol=0)

    def model(self):
        from task_embeddings.domain_train import DomainConfig, make_model

        return make_model(
            DomainConfig(hidden=8, layers=2, intermediate=16, heads=2, kv_heads=1),
            16,
            device="cpu",
        ).eval()

    def test_feature_bundles_cover_only_mlp_parameters_and_preserve_mass(self):
        from task_embeddings.domain_optimizer import ParameterPartition

        self.assertTrue(hasattr(domain_causal, "mlp_bundles"))
        part = ParameterPartition(self.model(), "swiglu")
        bundles = domain_causal.mlp_bundles(part, 4)
        self.assertEqual(len(bundles), 8)
        self.assertEqual(sum(b["parameters"] for b in bundles), 2 * 3 * 8 * 16)
        atlas = torch.arange(part.n_groups * 3).reshape(-1, 3).float()
        actual = domain_causal.bundle_profiles(atlas, part.sizes, bundles)
        expected = torch.stack([atlas[b["start"] : b["stop"]].mean(0) for b in bundles])
        torch.testing.assert_close(actual, expected)

    def test_output_positions_ignore_padding_and_match_full_logits(self):
        self.assertTrue(hasattr(domain_causal, "sampled_log_probs"))
        model = self.model()
        blocks = torch.tensor([[1, 2, 3, 4, -1], [1, 3, 5, 7, 9]])
        actual, valid = domain_causal.sampled_log_probs(model, blocks, 2)
        full = model(
            input_ids=blocks[:, :-1].clamp_min(0), attention_mask=blocks[:, :-1] >= 0
        ).logits.log_softmax(-1)
        torch.testing.assert_close(actual[0], full[0, [0, 2]])
        torch.testing.assert_close(actual[1], full[1, [1, 3]])
        self.assertTrue(valid.all())

    def test_zero_probe_is_identity_and_nonzero_probe_restores_weights(self):
        from task_embeddings.domain_optimizer import ParameterPartition

        self.assertTrue(hasattr(domain_causal, "measure_bundles"))
        model = self.model()
        part = ParameterPartition(model, "swiglu")
        bundles = domain_causal.mlp_bundles(part, 4)[:2]
        data = [torch.tensor([[1, 2, 3, 4], [1, 3, 5, 7]])]
        original = {k: v.clone() for k, v in model.state_dict().items()}
        zero = domain_causal.measure_bundles(
            model, part, data, bundles, amplitude=0.0, directions=1, seed=7
        )
        torch.testing.assert_close(zero, torch.zeros_like(zero))
        actual = domain_causal.measure_bundles(
            model, part, data, bundles, amplitude=0.1, directions=1, seed=7
        )
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual >= 0).all())
        self.assertGreater(float(actual.sum()), 0)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, original[k], rtol=0, atol=0)

    def test_stable_kl_matches_double_precision_reference(self):
        p = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
        q = p + torch.tensor([[1e-5, -1e-5, 0.0]], dtype=torch.float64)
        expected = (p * (p.log() - q.log())).sum(-1)
        actual = stable_kl(p.log().float(), q.log().float())
        torch.testing.assert_close(actual.double(), expected, atol=1e-11, rtol=0.01)
        self.assertGreater(float(actual), 0.0)


if __name__ == "__main__":
    unittest.main()
