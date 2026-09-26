import unittest

from task_embeddings.paper_analysis_v5 import aggregate_query_runs


def _run(seed, tbe, jl_repeats, mean, oracle):
    ranking = {}
    causal = []
    values = {
        "tbe_linear_scale_0p75": [tbe],
        "mean_only": [mean],
        "observed_opg_oracle": [oracle],
        **{
            f"jl_linear_scale_{str(scale).replace('.', 'p')}_seed{index}": [value]
            for index, (scale, value) in enumerate(jl_repeats)
        },
    }
    for method, method_values in values.items():
        ranking[method] = {
            "raw": {
                "spearman_mean": method_values[0],
                "ndcg_mean": method_values[0],
                "topk_recall_mean": method_values[0],
            },
            "residual": {
                "spearman_mean": method_values[0],
                "ndcg_mean": method_values[0],
                "topk_recall_mean": method_values[0],
            },
        }
        for value in method_values:
            causal.append(
                {
                    "method": method,
                    "task": 21,
                    "sufficiency": value,
                    "necessity": value / 2,
                    "kept_half_mse": 1 - value,
                    "dropped_half_mse": value,
                }
            )
    return {
        "seed": seed,
        "splits": {
            "all_triples_out": {
                "scale_selections": {
                    "tbe": {"selected_scale": 0.75},
                    **{
                        f"jl_seed{index}": {"selected_scale": scale}
                        for index, (scale, _) in enumerate(jl_repeats)
                    },
                },
                "ranking": ranking,
                "causal_records": causal,
            }
        },
    }


class PaperAnalysisV5Tests(unittest.TestCase):
    def test_random_feature_repeats_are_averaged_within_model_seed(self):
        runs = [
            _run(
                1,
                tbe=5.0,
                jl_repeats=((0.25, 1.0), (0.75, 3.0)),
                mean=1.0,
                oracle=6.0,
            ),
            _run(
                2,
                tbe=7.0,
                jl_repeats=((0.25, 5.0), (0.75, 7.0)),
                mean=2.0,
                oracle=8.0,
            ),
        ]

        result = aggregate_query_runs(runs)

        self.assertEqual(result["causal"]["selected_jl"]["sufficiency"]["mean"], 4.0)
        self.assertEqual(
            result["causal"]["selected_jl"]["sufficiency"]["n_independent_seeds"],
            2,
        )
        self.assertEqual(
            result["paired_differences"]["selected_tbe_minus_selected_jl"][
                "sufficiency"
            ]["mean"],
            2.0,
        )
        self.assertEqual(
            result["ranking_paired_differences"]["selected_tbe_minus_selected_jl"][
                "residual_spearman_mean"
            ]["mean"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
