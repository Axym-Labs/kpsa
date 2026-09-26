import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from task_embeddings.controlled_v4 import HardControlledConfig, build_model
from task_embeddings.paper_query_v5 import QueryExperimentConfig, run_controlled_query


class PaperQueryV5Tests(unittest.TestCase):
    def test_smoke_run_keeps_cold_start_methods_and_oracle_distinct(self):
        model_config = HardControlledConfig(
            seed=7,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            train_steps=1,
            refinement_steps=1,
        )
        model = build_model(model_config, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "configuration": asdict(model_config),
                    "states": {"trained": model.state_dict()},
                },
                checkpoint,
            )
            result = run_controlled_query(
                checkpoint,
                QueryExperimentConfig(
                    seed=7,
                    reference_per_task=1,
                    test_per_task=1,
                    repeats=1,
                    residual_scales=(1.0,),
                ),
                device=torch.device("cpu"),
            )

        self.assertEqual(
            set(result["splits"]), {"leave_one_triple_out", "all_triples_out"}
        )
        methods = result["splits"]["all_triples_out"]["ranking"]
        self.assertIn("tbe_linear_scale_1", methods)
        self.assertIn("observed_opg_oracle", methods)
        self.assertFalse(any("ief" in method.lower() for method in methods))
        self.assertEqual(
            len(result["splits"]["all_triples_out"]["causal_records"]),
            9 * len(methods),
        )
        self.assertGreater(
            result["resource_accounting"]["full_atlas_stored_floats"],
            result["resource_accounting"]["tbe_stored_floats"],
        )
        selections = result["splits"]["all_triples_out"]["scale_selections"]
        self.assertEqual(set(selections), {"tbe", "jl_seed0", "permuted_basis_seed0"})
        for selection in selections.values():
            self.assertEqual(selection["selected_scale"], 1.0)
            self.assertEqual(selection["task_indices"], list(range(21)))


if __name__ == "__main__":
    unittest.main()
