import unittest

import torch

from task_embeddings.continual_v4 import ContinualConfig
from task_embeddings.paper_continual_v5 import run_continual_curves


class PaperContinualV5Tests(unittest.TestCase):
    def test_tiny_curve_has_full_compact_and_task_agnostic_comparisons(self):
        config = ContinualConfig(
            seed=9,
            input_dim=4,
            n_tasks=3,
            n_modules=12,
            batch_size=8,
            steps_per_task=2,
            reference_per_task=8,
            test_per_task=8,
        )

        result = run_continual_curves(
            config,
            order=[2, 1, 0],
            strengths=(2.0,),
            control_repeats=1,
            device=torch.device("cpu"),
        )

        methods = {record["method"] for record in result["records"]}
        self.assertEqual(
            methods,
            {
                "none",
                "raw_opg/full_atlas",
                "raw_opg/tbe",
                "raw_opg/mean_only",
                "raw_opg/jl_seed0",
                "residual_normalized_opg/full_atlas",
                "residual_normalized_opg/tbe",
                "residual_normalized_opg/mean_only",
                "residual_normalized_opg/jl_seed0",
                "permuted_basis_linear_seed0",
            },
        )
        self.assertFalse(any("ief" in method.lower() for method in methods))
        self.assertEqual(result["order"], [2, 1, 0])
        self.assertEqual(result["resource_accounting"]["full_atlas_stored_floats"], 36)
        self.assertEqual(result["resource_accounting"]["tbe_stored_floats"], 108)
        self.assertEqual(
            result["resource_accounting"]["consolidated_parameter_diagonal_floats"],
            108,
        )
        self.assertEqual(
            result["resource_accounting"]["tbe_vs_consolidated_diagonal_ratio"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
