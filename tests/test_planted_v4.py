import unittest

import torch

from task_embeddings.planted_v4 import (
    PlantedCircuitBenchmark,
    PlantedConfig,
    make_planted_data,
    run_planted_experiment,
)


class PlantedV4Tests(unittest.TestCase):
    def test_full_and_true_pure_circuit_reproduce_model_output(self):
        config = PlantedConfig(modules_per_primitive=3, background_modules=2)
        benchmark = PlantedCircuitBenchmark(config)
        data = make_planted_data(config, per_task=2, seed=3, noise=0.0)
        task = 0
        primitive_values = data[0][data[2] == task]
        task_ids = data[2][data[2] == task]
        full = benchmark.predict(primitive_values, task_ids)
        all_modules = torch.ones(config.n_modules, dtype=torch.bool)
        torch.testing.assert_close(
            benchmark.predict(primitive_values, task_ids, keep_mask=all_modules), full
        )
        true_modules = benchmark.module_primitive == task
        torch.testing.assert_close(
            benchmark.predict(primitive_values, task_ids, keep_mask=true_modules), full
        )

    def test_drop_and_keep_are_mechanistically_complementary(self):
        config = PlantedConfig(modules_per_primitive=2, background_modules=1)
        benchmark = PlantedCircuitBenchmark(config)
        data = make_planted_data(config, per_task=1, seed=4, noise=0.0)
        values, _, tasks = data
        mask = torch.zeros(config.n_modules, dtype=torch.bool)
        mask[:2] = True
        full = benchmark.predict(values, tasks)
        kept = benchmark.predict(values, tasks, keep_mask=mask)
        dropped = benchmark.predict(values, tasks, drop_mask=mask)
        torch.testing.assert_close(kept + dropped, full)

    def test_data_roles_are_reproducible_but_seed_distinct(self):
        config = PlantedConfig()
        first = make_planted_data(config, per_task=2, seed=11)
        repeated = make_planted_data(config, per_task=2, seed=11)
        other = make_planted_data(config, per_task=2, seed=12)
        torch.testing.assert_close(first[0], repeated[0])
        self.assertFalse(torch.equal(first[0], other[0]))

    def test_tiny_experiment_reports_gated_curves_without_cka(self):
        config = PlantedConfig(
            modules_per_primitive=2,
            background_modules=8,
            reference_per_task=3,
            calibration_per_task=4,
            test_per_task=5,
            circuit_fractions=(0.1, 0.5),
            random_masks=3,
            jl_repeats=2,
        )
        result = run_planted_experiment(config)
        self.assertIn("cross_validated_task_query", result)
        self.assertNotIn("cka", str(result).lower())
        self.assertEqual(
            result["cross_validated_task_query"]["onehot"]["queryable_tasks"], 0
        )
        self.assertTrue(result["causal_interpretability"]["application_enabled"])
        self.assertTrue(result["pruning"]["retention_identity_passed"])


if __name__ == "__main__":
    unittest.main()
