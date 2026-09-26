import unittest

import torch

from task_embeddings.controlled_assay_v4 import (
    build_method_scores,
    circuit_diagnostic_fields,
    cross_task_and_composition_study,
    pure_circuit_overlap_summary,
    split_global_mask,
    task_circuit_result,
)
from task_embeddings.controlled_v4 import (
    HardControlledConfig,
    build_model,
    hard_task_mixtures,
    make_program_dataset,
)


class ControlledAssayV4Tests(unittest.TestCase):
    def test_cross_task_study_exports_explanatory_predictors_and_overlap(self):
        config = HardControlledConfig(
            d_model=12,
            n_layers=1,
            n_heads=2,
            d_ff=6,
            train_steps=1,
            refinement_steps=1,
            batch_size=2,
        )
        model = build_model(config, torch.device("cpu"))
        mixtures = hard_task_mixtures()
        data = make_program_dataset(config, mixtures, per_task=1, seed=8, noise=0.0)
        generator = torch.Generator().manual_seed(9)
        task_basis = torch.rand(model.n_modules, 30, generator=generator)
        methods = {
            "posthoc_ief__semantic6": task_basis,
            "posthoc_ief__onehot": torch.rand(model.n_modules, 30, generator=generator),
            "task_agnostic_mean": task_basis.mean(dim=1, keepdim=True).repeat(1, 30),
        }
        result = cross_task_and_composition_study(
            model,
            methods,
            mixtures,
            data,
            torch.device("cpu"),
            fraction=0.2,
        )
        self.assertEqual(len(result["cross_task_records"]), 180)
        self.assertIn("task_basis_selected_score_mass", result["cross_task_records"][0])
        self.assertIn("full_variance", result["cross_task_records"][0])
        self.assertIn("pure_circuit_overlap", result)

    def test_circuit_diagnostics_measure_profile_and_selected_mass_alignment(self):
        task_basis = torch.tensor(
            [
                [4.0, 4.0, 0.0],
                [3.0, 1.0, 1.0],
                [2.0, 3.0, 3.0],
                [1.0, 2.0, 2.0],
            ]
        )
        full_atlas = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        mixtures = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mask = torch.tensor([True, True, False, False])
        result = circuit_diagnostic_fields(
            task_basis,
            full_atlas,
            mask,
            primitive=0,
            task=1,
            mixtures=mixtures,
        )
        self.assertEqual(result["task_cardinality"], 2)
        self.assertAlmostEqual(result["task_basis_profile_cosine"], 0.9, places=6)
        self.assertAlmostEqual(result["full_atlas_profile_cosine"], 0.5, places=6)
        self.assertAlmostEqual(result["task_basis_selected_score_mass"], 0.5)
        self.assertAlmostEqual(result["full_atlas_selected_score_mass"], 1.0)
        self.assertAlmostEqual(result["task_basis_topk_overlap"], 0.5)

    def test_pure_circuit_overlap_summary_detects_shared_generic_modules(self):
        task_basis = torch.tensor(
            [
                [4.0, 4.0],
                [3.0, 1.0],
                [2.0, 3.0],
                [1.0, 2.0],
            ]
        )
        task_agnostic = torch.tensor([4.0, 3.0, 2.0, 1.0])
        result = pure_circuit_overlap_summary(
            task_basis, task_agnostic, count=2, n_primitives=2
        )
        self.assertAlmostEqual(result["mean_pairwise_overlap_fraction"], 0.5)
        self.assertAlmostEqual(result["mean_task_agnostic_overlap_fraction"], 0.75)
        self.assertAlmostEqual(result["union_fraction_of_modules"], 0.75)

    def test_split_global_mask_preserves_exact_selected_modules(self):
        mask = torch.tensor([True, False, True, False, False, True])
        split = split_global_mask(mask, [2, 4])
        self.assertEqual(split[0].tolist(), [0])
        self.assertEqual(split[1].tolist(), [0, 3])
        with self.assertRaises(ValueError):
            split_global_mask(mask, [2, 3])

    def test_method_scores_include_isolating_contrasts_and_repeated_jl(self):
        generator = torch.Generator().manual_seed(4)
        profiles = {
            name: torch.rand(12, 30, generator=generator)
            for name in ("ief", "raw_ef", "activation", "activation_gradient")
        }
        methods, calibration = build_method_scores(
            profiles, hard_task_mixtures(), seed=7, jl_repeats=3
        )
        expected = {
            "posthoc_ief__onehot",
            "posthoc_ief__semantic6",
            "posthoc_ief__permuted_semantic6",
            "posthoc_raw_ef__semantic6",
            "posthoc_activation__semantic6",
            "posthoc_activation_gradient__semantic6",
            "task_agnostic_mean",
            "random",
        }
        self.assertTrue(expected <= methods.keys())
        self.assertEqual(
            sorted(
                name for name in methods if name.startswith("posthoc_ief__jl6_seed")
            ),
            [
                "posthoc_ief__jl6_seed0",
                "posthoc_ief__jl6_seed1",
                "posthoc_ief__jl6_seed2",
            ],
        )
        torch.testing.assert_close(methods["posthoc_ief__onehot"], profiles["ief"])
        self.assertEqual(methods["posthoc_ief__semantic6"].shape, (12, 30))
        self.assertEqual(set(methods), set(calibration))
        self.assertFalse(any("cka" in name.lower() for name in methods))

    def test_full_retention_is_behaviorally_identical(self):
        config = HardControlledConfig(
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=12,
            train_steps=1,
            refinement_steps=1,
            batch_size=2,
        )
        model = build_model(config, torch.device("cpu"))
        data = make_program_dataset(
            config, hard_task_mixtures(), per_task=2, seed=5, noise=0.0
        )
        result = task_circuit_result(
            model,
            data,
            task=0,
            mask=torch.ones(model.n_modules, dtype=torch.bool),
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(result["kept_divergence"], 0.0, places=12)
        self.assertAlmostEqual(result["sufficiency"], 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
