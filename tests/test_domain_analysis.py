import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from task_embeddings import domain_analysis


class DomainAnalysisTests(unittest.TestCase):
    def test_shape_metrics_compare_independent_backgrounds(self):
        self.assertTrue(hasattr(domain_analysis, "profile_shape_metrics"))
        background = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
        predicted = torch.tensor([[2.0], [1.0], [5.0], [7.0]])
        actual = domain_analysis.profile_shape_metrics(
            predicted,
            3 * predicted,
            background,
            3 * background,
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            [(0, 2), (2, 4)],
        )
        for values in actual.values():
            self.assertAlmostEqual(values[0], 1.0)

    def test_shape_normalization_removes_global_task_gains(self):
        self.assertTrue(hasattr(domain_analysis, "normalize_profile_shape"))
        values = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        actual = domain_analysis.normalize_profile_shape(
            values, torch.tensor([1.0, 1.0])
        )
        torch.testing.assert_close(
            actual, torch.tensor([[0.5, 0.5], [1.5, 1.5]], dtype=torch.float64)
        )

    def test_local_shape_normalization_removes_only_within_block_gains(self):
        self.assertTrue(hasattr(domain_analysis, "normalize_profile_shape"))
        values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [2.0, 8.0], [2.0, 8.0]])
        actual = domain_analysis.normalize_profile_shape(
            values, torch.ones(4), [(0, 2), (2, 4)]
        )
        torch.testing.assert_close(
            actual,
            torch.tensor(
                [[0.5, 0.5], [1.5, 1.5], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float64
            ),
        )

    def test_translation_bootstrap_uses_paired_documents_and_corpus_scores(self):
        self.assertTrue(hasattr(domain_analysis, "translation_paired_interval"))
        row = {
            "language": "a",
            "segment_ids": [1, 2, 3],
            "document_ids": ["doc1", "doc1", "doc2"],
            "references": [
                "a long enough sentence",
                "another complete example",
                "one more long example",
            ],
            "predictions": [
                "a long enough sentence",
                "another complete example",
                "one more long example",
            ],
        }
        other = {**row, "predictions": ["", "", ""]}
        result = domain_analysis.translation_paired_interval(
            [row, {**row, "language": "b"}],
            [other, {**other, "language": "b"}],
            repetitions=20,
        )
        self.assertEqual(result["documents"], 2)
        self.assertAlmostEqual(result["mean"], 100.0)
        self.assertAlmostEqual(result["ci95_low"], 100.0)
        self.assertAlmostEqual(result["ci95_high"], 100.0)

    def test_different_intervention_coordinates_cannot_be_pooled(self):
        base = {
            "seed": 1,
            "ranking": [],
            "causal": [],
            "pruning": [],
            "quantization": [],
            "resources": {},
        }
        for field, a, b in (
            ("importance_coordinate", "absolute", "relative"),
            ("intervention_scope", "all", "transformer"),
            ("residual_scale", 0.5, 1.0),
            ("method_scales", {"tbe": 0.5}, {"tbe": 1.0}),
        ):
            with self.subTest(field=field), TemporaryDirectory() as root:
                paths = [Path(root) / "a.json", Path(root) / "b.json"]
                for path, value in zip(paths, (a, b)):
                    path.write_text(json.dumps({**base, field: value}))
                with self.assertRaisesRegex(ValueError, "protocols"):
                    domain_analysis.summarize(paths)

    def test_document_bootstrap_preserves_paired_translations(self):
        row = {
            "indices": [0, 1, 2],
            "loss_sums": [2.0, 4.0, 6.0],
            "token_counts": [1, 2, 3],
        }
        reference = {**row, "loss_sums": [1.0, 2.0, 3.0]}
        result = domain_analysis.document_paired_interval(
            [row, row], [reference, reference], ["a", "a", "b"], repetitions=100
        )
        self.assertEqual(result["documents"], 2)
        self.assertAlmostEqual(result["mean"], 1.0)
        self.assertAlmostEqual(result["ci95_low"], 1.0)
        self.assertAlmostEqual(result["ci95_high"], 1.0)

    def test_tasks_are_averaged_before_independent_model_uncertainty(self):
        records = []
        for model, values in (("a", [1.0, 3.0]), ("b", [4.0, 6.0])):
            records.append(
                {
                    "model_source": model,
                    "seed": 1,
                    "ranking": [
                        {
                            "partition": "row",
                            "estimator": "raw",
                            "method": "tbe",
                            "task": i,
                            "residual_opg_spearman": v,
                        }
                        for i, v in enumerate(values)
                    ],
                }
            )
        result = domain_analysis.summarize_endpoint(
            records, "ranking", "residual_opg_spearman"
        )
        self.assertEqual(result[0]["n_independent_seeds"], 2)
        self.assertEqual(result[0]["mean"], 3.5)

    def test_repeated_controls_on_one_model_are_not_independent_seeds(self):
        base = {
            "model_source": "fixed-pretrained",
            "seed": 1,
            "ranking": [
                {
                    "partition": "row",
                    "estimator": "raw",
                    "method": "jl",
                    "task": 0,
                    "residual_opg_spearman": 0.2,
                }
            ],
        }
        result = domain_analysis.summarize_endpoint(
            [base, base], "ranking", "residual_opg_spearman"
        )
        self.assertEqual(result[0]["n_independent_seeds"], 1)
        self.assertIsNone(result[0]["ci95_low"])


if __name__ == "__main__":
    unittest.main()
