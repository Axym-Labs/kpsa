import math
import unittest

from task_embeddings.make_figures import finite_mean, mean_sem, selectivity_values


class FigureDataTests(unittest.TestCase):
    def test_finite_mean_returns_nan_without_warning_for_all_nan(self):
        self.assertTrue(math.isnan(finite_mean([float("nan")])))

    def test_selectivity_values_filters_method_fraction_and_field(self):
        records = [
            {"method": "a", "fraction": 0.05, "selective_drop": 1.0},
            {"method": "a", "fraction": 0.10, "selective_drop": 2.0},
            {"method": "b", "fraction": 0.05, "selective_drop": 3.0},
        ]
        self.assertEqual(
            selectivity_values(records, "a", 0.05, "selective_drop"), [1.0]
        )

    def test_mean_sem_is_zero_for_one_observation(self):
        mean, sem = mean_sem([2.0])
        self.assertEqual(mean, 2.0)
        self.assertEqual(sem, 0.0)


if __name__ == "__main__":
    unittest.main()
