import unittest

from task_embeddings.paper_continual_analysis_v5 import aggregate_continual_runs


def _run(seed, order, tbe_value, jl_values):
    records = [
        {
            "method": "none",
            "strength": 0.0,
            "average_forgetting": 1.0,
            "average_acquisition_gain": 1.0,
            "final_average": 1.0,
        },
        {
            "method": "tbe_linear",
            "strength": 2.0,
            "average_forgetting": tbe_value,
            "average_acquisition_gain": 0.75,
            "final_average": tbe_value,
        },
    ]
    for repeat, value in enumerate(jl_values):
        records.append(
            {
                "method": f"jl_linear_seed{repeat}",
                "strength": 2.0,
                "average_forgetting": value,
                "average_acquisition_gain": 0.75,
                "final_average": value,
            }
        )
    return {
        "seed": seed,
        "order": order,
        "premise_gate": {"passed": True},
        "records": records,
        "resource_accounting": {"full_atlas_stored_floats": 100},
    }


class PaperContinualAnalysisV5Tests(unittest.TestCase):
    def test_canonical_estimator_names_pool_jl_repeats_without_pooling_estimators(self):
        for estimator in ("raw_opg", "residual_normalized_opg"):
            run = _run(1, [0, 1], 0.4, (0.2, 0.6))
            for record in run["records"]:
                record["method"] = (
                    record["method"]
                    .replace("tbe_linear", f"{estimator}/tbe")
                    .replace("jl_linear", f"{estimator}/jl")
                )
            result = aggregate_continual_runs([run], acquisition_ratios=(0.75,))
            self.assertAlmostEqual(
                result["matched_acquisition"][f"{estimator}/jl"]["0.75"]["mean"], 0.4
            )
            self.assertAlmostEqual(
                result["paired_differences"][f"{estimator}/tbe_minus_{estimator}/jl"][
                    "0.75"
                ]["mean"],
                0,
            )

    def test_paired_intervals_do_not_zip_different_missing_seeds(self):
        runs = [
            _run(1, [0], 0.4, (0.2,)),
            _run(2, [0], 0.6, (0.4,)),
            _run(3, [0], 0.8, (0.6,)),
        ]
        runs[0]["records"][1]["average_acquisition_gain"] = 0.9
        runs[2]["records"][2]["average_acquisition_gain"] = 0.9
        result = aggregate_continual_runs(runs, acquisition_ratios=(0.75,))
        paired = result["paired_differences"]["tbe_linear_minus_jl_linear"]["0.75"]
        self.assertEqual(paired["n_independent_seeds"], 1)
        self.assertAlmostEqual(paired["mean"], 0.2)

    def test_repeats_and_orders_are_averaged_within_model_seed(self):
        runs = [
            _run(1, [0, 1], 0.4, (0.2, 0.6)),
            _run(1, [1, 0], 0.4, (0.2, 0.6)),
            _run(2, [0, 1], 0.6, (0.4, 0.8)),
            _run(2, [1, 0], 0.6, (0.4, 0.8)),
        ]

        result = aggregate_continual_runs(runs, acquisition_ratios=(0.75,))

        tbe = result["matched_acquisition"]["tbe_linear"]["0.75"]
        jl = result["matched_acquisition"]["jl_linear"]["0.75"]
        self.assertEqual(tbe["mean"], 0.5)
        self.assertEqual(jl["mean"], 0.5)
        self.assertEqual(jl["n_independent_seeds"], 2)
        self.assertAlmostEqual(
            result["paired_differences"]["tbe_linear_minus_jl_linear"]["0.75"]["mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
