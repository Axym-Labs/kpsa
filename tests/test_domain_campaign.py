import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from task_embeddings import domain_campaign


class ValidationSelectionTests(unittest.TestCase):
    def test_matched_adafactor_screen_changes_only_variance_representation(self):
        configs = domain_campaign.optimizer_adafactor_kernel_screen()
        self.assertEqual(len(configs), 15)
        self.assertEqual(
            {c.method for c in configs},
            {
                "adafactor",
                "mean_adafactor",
                "full_adafactor",
                "tbe_adafactor",
                "jl_adafactor",
            },
        )
        self.assertEqual({c.lr for c in configs}, {0.01, 0.03, 0.1})
        self.assertTrue(all(c.role == "validation" for c in configs))
        self.assertTrue(
            all(c.partition == "row" and c.estimator == "raw" for c in configs)
        )
        self.assertTrue(all(c.score_link == "linear" for c in configs))
        self.assertTrue(all(c.seed == 11 and c.steps == 2000 for c in configs))

    def test_opt_in_divergence_is_recorded_as_a_failed_experiment(self):
        from unittest.mock import patch

        from task_embeddings.domain_train import DomainConfig

        with TemporaryDirectory() as root:
            directory = Path(root)
            config = DomainConfig(method="tbe_nomomentum")
            with patch.object(
                domain_campaign,
                "train",
                side_effect=FloatingPointError("nonfinite training at step 7"),
            ):
                domain_campaign.execute(
                    [config], directory, directory, allow_divergence=True
                )
            result = json.loads((directory / "summary.json").read_text())["runs"][0]
            self.assertEqual(result["status"], "diverged")
            self.assertEqual(result["configuration"]["method"], "tbe_nomomentum")
            self.assertIsNone(result["final"]["macro_nll"])
            self.assertFalse(result["claim_ready"])

    def test_momentum_free_screen_covers_matched_estimators_and_partitions(self):
        configs = domain_campaign.optimizer_momentum_free_screen()
        self.assertTrue(all(c.role == "validation" for c in configs))
        self.assertTrue(all(c.score_link == "linear" for c in configs))
        cells = {(c.method, c.partition, c.estimator) for c in configs}
        self.assertEqual(
            cells,
            {
                (method, partition, estimator)
                for method in (
                    "mean_nomomentum",
                    "full_nomomentum",
                    "tbe_nomomentum",
                    "jl_nomomentum",
                )
                for partition in ("row", "tensor", "swiglu")
                for estimator in ("raw", "normalized")
            },
        )
        for cell in cells:
            choices = [
                c for c in configs if (c.method, c.partition, c.estimator) == cell
            ]
            self.assertEqual(
                {c.lr for c in choices}, {0.0001, 0.0003, 0.001, 0.003, 0.01}
            )
        self.assertTrue(
            all(
                c.task_mode == "single"
                for c in configs
                if c.method == "mean_nomomentum"
            )
        )

    def test_campaign_rejects_resuming_with_different_task_features(self):
        from task_embeddings.common import save_json

        with TemporaryDirectory() as root:
            directory = Path(root)
            save_json(
                directory / "protocol.json", {"feature_file": "token_features.pt"}
            )
            with self.assertRaisesRegex(ValueError, "feature"):
                domain_campaign.execute([], directory, directory)

    def test_campaign_summary_uses_declared_runs_not_every_json_file(self):
        from task_embeddings.common import save_json

        with TemporaryDirectory() as root:
            directory = Path(root)
            save_json(directory / "selection.json", {"not": "an experiment"})
            domain_campaign.execute([], directory, directory)
            self.assertEqual(
                json.loads((directory / "summary.json").read_text())["runs"], []
            )

    def row(self, loss, role="validation"):
        return {"configuration": {"role": role}, "final": {"macro_nll": loss}}

    def test_selection_minimizes_validation_loss(self):
        rows = [self.row(3.0), self.row(2.0)]
        self.assertEqual(domain_campaign.validation_winner(rows), rows[1])

    def test_final_test_cannot_select_a_recipe(self):
        with self.assertRaises(ValueError):
            domain_campaign.validation_winner([self.row(1.0, "test")])

    def test_empty_or_nonfinite_selection_is_rejected(self):
        for rows in ([], [self.row(float("nan"))]):
            with self.assertRaises(ValueError):
                domain_campaign.validation_winner(rows)
