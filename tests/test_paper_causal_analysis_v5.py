import unittest

from task_embeddings.paper_causal_analysis_v5 import aggregate_causal_runs


class PaperCausalAnalysisV5Tests(unittest.TestCase):
    def test_projection_repeats_are_nested_within_model_seed(self):
        runs = []
        for seed, tbe, jl in ((1, 0.5, (0.1, 0.3)), (2, 0.7, (0.5, 0.7))):
            records = [
                {
                    "method": "tbe_linear",
                    "task": 21,
                    "raw_spearman": tbe,
                    "residual_spearman": tbe,
                }
            ]
            records.extend(
                {
                    "method": f"jl_linear_seed{repeat}",
                    "task": 21,
                    "raw_spearman": value,
                    "residual_spearman": value,
                }
                for repeat, value in enumerate(jl)
            )
            runs.append(
                {
                    "seed": seed,
                    "module_sample": 8,
                    "records": records,
                    "resource_accounting": {},
                }
            )

        result = aggregate_causal_runs(runs)

        self.assertEqual(
            result["methods"]["jl_linear"]["residual_spearman"]["mean"], 0.4
        )
        self.assertEqual(
            result["methods"]["jl_linear"]["residual_spearman"]["n_independent_seeds"],
            2,
        )
        self.assertAlmostEqual(
            result["paired_differences"]["tbe_linear_minus_jl_linear"][
                "residual_spearman"
            ]["mean"],
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
