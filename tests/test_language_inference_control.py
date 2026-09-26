import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from task_embeddings import language_inference_control
from task_embeddings.language_v3 import EncodedTask, LanguageConfig, QwenFeatureProbe


class LanguageInferenceControlTests(unittest.TestCase):
    @staticmethod
    def _tasks(n_tasks: int = 6) -> list[EncodedTask]:
        return [
            EncodedTask(
                input_ids=torch.randint(0, 32, (3, 6)),
                attention_mask=torch.ones(3, 6, dtype=torch.long),
                labels=torch.randint(0, 32, (3, 6)),
            )
            for _ in range(n_tasks)
        ]

    def test_slice_encoded_task_preserves_aligned_fields(self):
        task = self._tasks(1)[0]

        selected = language_inference_control.slice_encoded_task(task, 1, 2)

        self.assertEqual(len(selected), 2)
        torch.testing.assert_close(selected.input_ids, task.input_ids[1:3])
        torch.testing.assert_close(selected.labels, task.labels[1:3])

    def test_tiny_qwen_run_uses_regular_grid_and_disjoint_evaluation(self):
        qwen_config = Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        model = QwenFeatureProbe(Qwen3ForCausalLM(qwen_config))
        language_config = LanguageConfig(
            seed=5,
            validation_per_task=3,
            reference_per_task=1,
            smoke=True,
        )
        tasks = self._tasks()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            embeddings = root / "profiles.npz"
            torch.save(
                {
                    "model": model.state_dict(),
                    "configuration": asdict(language_config),
                },
                checkpoint,
            )
            rng = np.random.default_rng(3)
            raw = np.abs(rng.normal(size=(model.n_modules, 6))).astype("float32")
            normalized = np.abs(rng.normal(size=(model.n_modules, 6))).astype("float32")
            raw /= raw.sum(axis=1, keepdims=True)
            normalized /= normalized.sum(axis=1, keepdims=True)
            np.savez_compressed(
                embeddings,
                atlas_raw=raw,
                amplitude_raw=np.ones(model.n_modules, dtype="float32"),
                atlas_ief=normalized,
                amplitude_ief=np.ones(model.n_modules, dtype="float32"),
                semantic_representation=rng.normal(size=(6, 4)).astype("float32"),
            )
            with (
                patch.object(
                    language_inference_control,
                    "load_model_and_tokenizer",
                    return_value=(model, None),
                ),
                patch.object(
                    language_inference_control,
                    "load_task_suite",
                    return_value=(tasks, tasks),
                ),
            ):
                result = language_inference_control.run_language_inference_control(
                    checkpoint,
                    embeddings,
                    language_inference_control.LanguageInferenceControlConfig(
                        seed=5,
                        evaluation_start=1,
                        evaluation_per_task=2,
                        retained_fractions=(1.0,),
                    ),
                    device=torch.device("cpu"),
                )

        self.assertEqual(len(result["methods"]), 8)
        self.assertEqual(result["data_roles"]["evaluation_indices"], [1, 2])
        self.assertLess(result["identity_max_abs_loss_error"], 1e-6)
        self.assertEqual(result["model_family"], "Qwen3")
        self.assertIn("too_few_tasks_for_compression", result["promotion_blockers"])


if __name__ == "__main__":
    unittest.main()
