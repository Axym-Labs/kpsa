import importlib.util
import unittest

from task_embeddings import experiment_core


class ExperimentCoreTests(unittest.TestCase):
    def test_experiment_core_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("task_embeddings.experiment_core")
        )

    def test_profiles_separate_iteration_from_claim_ready_runs(self):
        explore = experiment_core.run_profile("explore")
        paper = experiment_core.run_profile("paper")

        self.assertEqual(explore.minimum_independent_seeds, 1)
        self.assertEqual(explore.control_repeats, 1)
        self.assertFalse(explore.claim_ready)
        self.assertGreaterEqual(paper.minimum_independent_seeds, 3)
        self.assertGreaterEqual(paper.control_repeats, 3)
        self.assertFalse(paper.claim_ready)

    def test_regular_grid_crosses_estimators_and_representations(self):
        cells = experiment_core.regular_comparison_grid()
        keys = {cell.key for cell in cells}

        self.assertEqual(len(cells), 8)
        self.assertIn("raw_opg/full_atlas", keys)
        self.assertIn("raw_opg/tbe", keys)
        self.assertIn("residual_normalized_opg/full_atlas", keys)
        self.assertIn("residual_normalized_opg/jl", keys)

    def test_legacy_estimator_names_are_read_compatible_only(self):
        self.assertEqual(
            experiment_core.canonical_estimator_name("ief"),
            experiment_core.Estimator.RESIDUAL_NORMALIZED_OPG,
        )
        self.assertEqual(
            experiment_core.canonical_estimator_name("raw_ef"),
            experiment_core.Estimator.RAW_OPG,
        )
        self.assertEqual(
            experiment_core.Estimator.RESIDUAL_NORMALIZED_OPG.value,
            "residual_normalized_opg",
        )

    def test_application_contracts_distinguish_queryable_state(self):
        continual = experiment_core.application_contract("continual_learning")
        retrieval = experiment_core.application_contract("cold_start_retrieval")
        control = experiment_core.application_contract("inference_control")

        self.assertFalse(continual.requires_distinct_task_queries)
        self.assertTrue(retrieval.requires_distinct_task_queries)
        self.assertTrue(control.requires_distinct_task_queries)
        self.assertIn("matched_acquisition_forgetting", continual.primary_metrics)
        self.assertIn("off_target_spillover", control.primary_metrics)


if __name__ == "__main__":
    unittest.main()
