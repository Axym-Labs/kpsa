import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from task_embeddings.controlled_v4 import (
    HardControlledConfig,
    build_model,
    hard_task_mixtures,
    make_program_dataset,
)
from task_embeddings.domain_optimizer import ParameterPartition

try:
    from task_embeddings.controlled_sensitivity import (
        measure_controlled_atlas,
        parameter_hierarchy,
        run_controlled_study,
    )
except ModuleNotFoundError:
    measure_controlled_atlas = None
    parameter_hierarchy = None
    run_controlled_study = None
except ImportError:
    from task_embeddings.controlled_sensitivity import (
        measure_controlled_atlas,
        parameter_hierarchy,
    )

    run_controlled_study = None


class ControlledSensitivityTest(unittest.TestCase):
    def test_swiglu_partition_links_standard_vit_mlp_features(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 6)
                self.fc2 = nn.Linear(6, 4)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

        model = nn.Module()
        model.blocks = nn.ModuleList([Block()])
        partition = ParameterPartition(model, "swiglu")
        specs = {spec.name: spec for spec in partition.slices}

        self.assertEqual(specs["blocks.0.mlp.fc1.weight"].axis, 0)
        self.assertEqual(specs["blocks.0.mlp.fc1.bias"].axis, 0)
        self.assertEqual(specs["blocks.0.mlp.fc2.weight"].axis, 1)
        self.assertEqual(
            specs["blocks.0.mlp.fc1.weight"].offset,
            specs["blocks.0.mlp.fc2.weight"].offset,
        )
        self.assertEqual(
            specs["blocks.0.mlp.fc1.weight"].offset,
            specs["blocks.0.mlp.fc1.bias"].offset,
        )
        self.assertEqual(int(partition.sizes.sum()), sum(p.numel() for p in model.parameters()))

    def test_autograd_profiles_form_complete_group_relative_distributions(self):
        self.assertIsNotNone(measure_controlled_atlas)
        config = HardControlledConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            d_ff=16,
            train_steps=0,
            refinement_steps=0,
            batch_size=2,
        )
        model = build_model(config, torch.device("cpu"))
        mixtures = hard_task_mixtures()[:6]
        dataset = make_program_dataset(config, mixtures, per_task=1, seed=7, noise=0)

        result = measure_controlled_atlas(
            model,
            dataset,
            {"primitive": torch.eye(6)},
            functional="output",
        )

        torch.testing.assert_close(
            result.atlases["primitive"].mass.sum(),
            torch.tensor(1.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.task_profiles.sum(dim=0), torch.ones(6, dtype=torch.float64)
        )
        self.assertEqual(result.task_profiles.shape[0], result.partition.n_groups)

    def test_parameter_hierarchy_assigns_every_fine_group_once(self):
        self.assertIsNotNone(parameter_hierarchy)
        config = HardControlledConfig(
            d_model=16, n_layers=2, n_heads=4, d_ff=16
        )
        model = build_model(config, torch.device("cpu"))

        hierarchy = parameter_hierarchy(model, bundle_width=4)

        for assignment in hierarchy.assignments.values():
            self.assertEqual(assignment.shape, (hierarchy.partition.n_groups,))
            self.assertGreaterEqual(int(assignment.min()), 0)
            self.assertEqual(
                sorted(assignment.unique().tolist()),
                list(range(int(assignment.max()) + 1)),
            )

    def test_study_reports_cold_semantic_query_against_scalar_baseline(self):
        self.assertIsNotNone(run_controlled_study)
        config = HardControlledConfig(
            seed=3,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=8,
            train_steps=0,
            refinement_steps=0,
            batch_size=2,
        )
        model = build_model(config, torch.device("cpu"))
        payload = {
            "states": {"trained": model.state_dict()},
            "configuration": config.__dict__,
            "mixtures": hard_task_mixtures(),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.pt"
            torch.save(payload, checkpoint)
            result, tensors = run_controlled_study(
                checkpoint,
                device=torch.device("cpu"),
                reference_sizes=(1,),
                target_per_task=1,
            )

        self.assertIn("semantic6", result["cold_query"]["fine"])
        self.assertIn("scalar_mass", result["cold_query"]["fine"])
        self.assertIn("semantic6", tensors["prediction"])
        self.assertLess(result["fidelity"]["max_partition_relative_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
