import unittest

from task_embeddings.analysis_v4 import (
    benefit_retention_summary,
    canonical_method_name,
    compact_utility_summary,
    mean_sd_ci,
    natural_seed_method_auc,
    natural_seed_method_mean,
    paired_benefit_retention,
    resource_accounting,
    structure_diagnostic_summary,
    within_primitive_spearman,
)


class AnalysisV4Tests(unittest.TestCase):
    def test_canonical_method_name_uses_provisional_task_basis_term(self):
        self.assertEqual(canonical_method_name("semantic6"), "tbe6")
        self.assertEqual(canonical_method_name("posthoc_ief__semantic6"), "tbe_ief")
        self.assertEqual(
            canonical_method_name("posthoc_ief__permuted_semantic6"),
            "permuted_task_basis_ief",
        )
        self.assertEqual(
            canonical_method_name("posthoc_ief__jl6_seed2"), "jl6_ief_seed2"
        )
        self.assertEqual(canonical_method_name("random"), "random")

    def test_compact_utility_is_paired_to_full_and_application_baseline(self):
        natural = []
        for seed in (1, 2, 3):
            causal_records = []
            pruning_records = []
            for method, value in (
                ("posthoc_ief__semantic6", 0.8),
                ("posthoc_ief__onehot", 1.0),
                ("random", 0.2),
            ):
                causal_records.append(
                    {
                        "method": method,
                        "requested_fraction": 0.1,
                        "premise_passed": True,
                        "sufficiency": value,
                    }
                )
                for fraction in (0.1, 1.0):
                    pruning_records.append(
                        {
                            "method": method,
                            "retained_fraction_requested": fraction,
                            "sufficiency": value,
                        }
                    )
            natural.append(
                {
                    "seed": seed,
                    "causal_interpretability": {"records": causal_records},
                    "pruning": {"records": pruning_records},
                }
            )
        continual = []
        for seed in (1, 2, 3):
            for order in ("forward", "reverse"):
                continual.append(
                    {
                        "seed": seed,
                        "task_order": order,
                        "methods": {
                            "none": {"summary": {"average_forgetting": 0.13}},
                            "posthoc_ief__semantic6": {
                                "summary": {"average_forgetting": 0.10}
                            },
                            "posthoc_ief__onehot": {
                                "summary": {"average_forgetting": 0.09}
                            },
                        },
                    }
                )
        result = compact_utility_summary(natural, continual)
        self.assertAlmostEqual(
            result["causal_10_percent"]["benefit_retained_vs_full_atlas"]["mean"],
            0.75,
        )
        self.assertAlmostEqual(
            result["pruning_curve_auc"]["benefit_retained_vs_full_atlas"]["mean"],
            0.75,
        )
        self.assertAlmostEqual(
            result["continual_forgetting"]["benefit_retained_vs_full_atlas"]["mean"],
            0.75,
        )

    def test_natural_seed_auc_averages_tasks_before_integrating_curve(self):
        records = []
        for fraction, values in (
            (0.1, (0.1, 0.3)),
            (0.5, (0.5, 0.7)),
            (1.0, (1.0, 1.0)),
        ):
            for task, value in enumerate(values):
                records.append(
                    {
                        "method": "candidate",
                        "retained_fraction_requested": fraction,
                        "task": task,
                        "sufficiency": value,
                    }
                )
        item = {"pruning": {"records": records}}
        self.assertAlmostEqual(natural_seed_method_auc(item, "candidate"), 0.56)

    def test_benefit_retention_summary_uses_paired_seed_ratios(self):
        result = benefit_retention_summary(
            baselines=[0.2, 0.3, 0.4],
            compact=[0.8, 0.9, 1.0],
            full=[1.0, 1.1, 1.2],
            higher_is_better=True,
        )
        self.assertAlmostEqual(result["mean"], 0.75)
        self.assertEqual(result["n_independent_seeds"], 3)
        for value in result["per_seed"]:
            self.assertAlmostEqual(value, 0.75)

    def test_structure_diagnostics_rank_predictors_within_each_primitive(self):
        natural = []
        for seed in (1, 2, 3):
            records = []
            for primitive in (0, 1):
                for rank in range(4):
                    records.append(
                        {
                            "primitive_circuit": primitive,
                            "necessity": float(rank),
                            "dropped_divergence": float(3 - rank),
                            "aligned": float(rank),
                            "reversed": float(3 - rank),
                        }
                    )
            natural.append(
                {
                    "cross_task_structure": {
                        "cross_task_records": records,
                        "pure_circuit_overlap": {
                            "mean_pairwise_overlap_fraction": 0.5,
                            "mean_task_agnostic_overlap_fraction": 0.75,
                            "union_fraction_of_modules": 0.4,
                        },
                    }
                }
            )
        result = structure_diagnostic_summary(
            natural, predictors=("aligned", "reversed")
        )
        self.assertEqual(result["necessity_correlations"]["aligned"]["mean"], 1.0)
        self.assertEqual(result["necessity_correlations"]["reversed"]["mean"], -1.0)
        self.assertEqual(
            result["dropped_divergence_correlations"]["aligned"]["mean"], -1.0
        )
        self.assertEqual(
            result["pure_circuit_overlap"]["mean_pairwise_overlap_fraction"]["mean"],
            0.5,
        )

    def test_resource_accounting_includes_task_feature_metadata(self):
        result = resource_accounting(n_modules=1536, n_tasks=30, n_features=6)
        self.assertEqual(result["full_atlas"]["stored_floats"], 46080)
        self.assertEqual(result["task_basis"]["embedding_floats"], 9216)
        self.assertEqual(result["task_basis"]["task_feature_floats"], 180)
        self.assertEqual(result["task_basis"]["stored_floats"], 9396)
        self.assertEqual(result["task_basis"]["bytes_float32"], 37584)
        self.assertAlmostEqual(result["storage_compression_ratio"], 4.9042145594)
        self.assertAlmostEqual(result["storage_reduction_fraction"], 0.79609375)
        self.assertEqual(result["seen_task_query_macs"]["full_atlas"], 0)
        self.assertEqual(result["seen_task_query_macs"]["task_basis"], 9216)
        self.assertEqual(
            result["dense_composition_query_macs"],
            {"full_atlas": 46080, "task_basis": 9216},
        )
        self.assertEqual(result["task_basis_build_macs"], 276480)

    def test_paired_benefit_retention_handles_both_metric_directions(self):
        self.assertAlmostEqual(
            paired_benefit_retention(
                baseline=0.2, compact=0.8, full=1.0, higher_is_better=True
            ),
            0.75,
        )
        self.assertAlmostEqual(
            paired_benefit_retention(
                baseline=0.13, compact=0.10, full=0.09, higher_is_better=False
            ),
            0.75,
        )
        self.assertIsNone(
            paired_benefit_retention(
                baseline=0.2, compact=0.8, full=0.2, higher_is_better=True
            )
        )

    def test_within_primitive_spearman_does_not_pool_primitive_intercepts(self):
        records = []
        for primitive in (0, 1):
            for rank in range(4):
                records.append(
                    {
                        "primitive_circuit": primitive,
                        "necessity": float(rank + 10 * primitive),
                        "aligned": float(rank),
                        "reversed": float(3 - rank),
                    }
                )
        aligned = within_primitive_spearman(records, "aligned")
        reversed_result = within_primitive_spearman(records, "reversed")
        self.assertEqual(aligned["per_primitive"], [1.0, 1.0])
        self.assertEqual(aligned["mean"], 1.0)
        self.assertEqual(reversed_result["per_primitive"], [-1.0, -1.0])
        self.assertEqual(reversed_result["mean"], -1.0)

    def test_uncertainty_uses_independent_seed_means(self):
        summary = mean_sd_ci([1.0, 2.0, 3.0])
        self.assertEqual(summary["n_independent_seeds"], 3)
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["sd"], 1.0)
        self.assertLess(summary["ci95_low"], 2.0)
        self.assertGreater(summary["ci95_high"], 2.0)

    def test_single_seed_is_explicitly_suggestive(self):
        summary = mean_sd_ci([4.0])
        self.assertEqual(summary["n_independent_seeds"], 1)
        self.assertIsNone(summary["sd"])
        self.assertIsNone(summary["ci95_low"])
        self.assertEqual(summary["evidence_strength"], "suggestive")

    def test_natural_seed_mean_keeps_jl_repeats_within_seed_and_gates_cells(self):
        item = {
            "causal_interpretability": {
                "records": [
                    {
                        "method": method,
                        "requested_fraction": 0.1,
                        "premise_passed": passed,
                        "sufficiency": value,
                    }
                    for method, passed, value in (
                        ("posthoc_ief__jl6_seed0", True, 0.4),
                        ("posthoc_ief__jl6_seed0", False, 0.9),
                        ("posthoc_ief__jl6_seed1", True, 0.6),
                        ("posthoc_ief__jl6_seed1", False, 1.0),
                    )
                ]
            }
        }
        self.assertAlmostEqual(
            natural_seed_method_mean(
                item,
                section="causal_interpretability",
                method="posthoc_ief__jl6_mean",
                fraction=0.1,
                metric="sufficiency",
                require_premise=True,
            ),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
