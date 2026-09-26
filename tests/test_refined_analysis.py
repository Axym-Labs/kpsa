import unittest

try:
    from kpsa.refined_analysis import (
        paired_t_summary,
        summarize_precision_records,
    )
except ModuleNotFoundError:
    paired_t_summary = None
    summarize_precision_records = None
except ImportError:
    from kpsa.refined_analysis import paired_t_summary

    summarize_precision_records = None


class RefinedAnalysisTest(unittest.TestCase):
    def test_constant_paired_effect_has_zero_width_interval(self):
        self.assertIsNotNone(paired_t_summary)

        summary = paired_t_summary([0.25, 0.25, 0.25])

        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean"], 0.25)
        self.assertAlmostEqual(summary["lower_95"], 0.25)
        self.assertAlmostEqual(summary["upper_95"], 0.25)

    def test_precision_summary_pairs_methods_by_task_and_budget(self):
        self.assertIsNotNone(summarize_precision_records)
        records = []
        for method, values in (("scalar_mass", (1.0, 2.0)), ("affine_semantic", (0.5, 1.5))):
            for task, value in enumerate(values):
                records.append(
                    {
                        "method": method,
                        "task": task,
                        "requested_high_precision_scope_fraction": 0.1,
                        "nll_increase": value,
                        "ideal_packed_bits_per_parameter": 6.0,
                    }
                )

        summary = summarize_precision_records(records)

        effect = summary["paired_vs_scalar"]["affine_semantic"]["0.1"]
        self.assertAlmostEqual(effect["mean"], -0.5)
        self.assertAlmostEqual(effect["lower_95"], -0.5)
        self.assertEqual(summary["methods"]["scalar_mass"]["0.1"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
