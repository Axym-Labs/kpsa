import unittest

import torch

from task_embeddings.controlled_v4 import (
    HardControlledConfig,
    build_model,
    hard_task_mixtures,
    make_program_dataset,
)
from task_embeddings.paper_streaming_v5 import compare_construction


class PaperStreamingV5Tests(unittest.TestCase):
    def test_streaming_construction_matches_posthoc_on_real_gradients(self):
        config = HardControlledConfig(
            seed=3,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            train_steps=1,
            refinement_steps=1,
        )
        model = build_model(config, torch.device("cpu"))
        features = hard_task_mixtures()
        dataset = make_program_dataset(config, features, per_task=1, seed=41)

        result = compare_construction(
            model, dataset, features, torch.device("cpu"), ridge=1e-6
        )

        self.assertLess(result["prediction_max_abs"], 1e-4)
        self.assertGreater(result["prediction_cosine"], 0.9999)
        self.assertLessEqual(result["prediction_cosine"], 1.0)
        self.assertLess(
            result["resource_accounting"]["streaming_conservative_peak_floats"],
            result["resource_accounting"]["posthoc_construction_state_floats"],
        )


if __name__ == "__main__":
    unittest.main()
