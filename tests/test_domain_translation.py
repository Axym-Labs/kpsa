import unittest

from task_embeddings import domain_scaling


class TranslationSelectionTests(unittest.TestCase):
    def test_decoding_uses_explicit_non_thinking_sampling_recipe(self):
        recipe = domain_scaling.translation_generation_options()
        self.assertEqual(
            recipe,
            {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "max_new_tokens": 1024,
                "use_cache": True,
            },
        )

    def test_tokenized_prompt_is_a_flat_token_list_with_current_tokenizer_api(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            chat_template="{{ messages[0]['content'] }}",
        )
        tokens = domain_scaling.translation_prompt(
            tokenizer, "A sentence.", "French", tokenize=True
        )
        self.assertIsInstance(tokens, list)
        self.assertTrue(tokens)
        self.assertTrue(all(isinstance(t, int) for t in tokens))

    def test_translation_selection_covers_documents_before_repeating_them(self):
        self.assertTrue(hasattr(domain_scaling, "translation_examples"))
        docs = ["a"] * 1000 + ["b", "c"]
        manifest = {
            "segment_ids": {"train": [], "validation": list(range(1002))},
            "document_ids": {"train": [], "validation": docs},
        }
        table = {
            i: {
                "segment_id": i,
                "document_id": doc,
                "source": str(i),
                "target": str(i),
                "is_bad_source": False,
            }
            for i, doc in enumerate(docs)
        }
        rows = domain_scaling.translation_examples(manifest, [table], 3)[0]
        self.assertEqual({r["document_id"] for r in rows}, {"a", "b", "c"})

    def test_prompt_tokens_are_not_translation_targets(self):
        self.assertTrue(hasattr(domain_scaling, "conditional_tokens"))
        inputs, labels = domain_scaling.conditional_tokens([10, 11, 12], [20, 21], 2)
        self.assertEqual(inputs.tolist(), [[10, 11, 12, 20, 21]])
        self.assertEqual(labels.tolist(), [[-1, -1, 20, 21, 2]])

    def test_translation_selection_uses_aligned_validation_only(self):
        self.assertTrue(hasattr(domain_scaling, "translation_examples"))
        manifest = {
            "segment_ids": {"train": [0], "validation": [1, 2]},
            "document_ids": {"train": ["train"], "validation": ["a", "b"]},
        }
        tables = [
            {
                i: {
                    "segment_id": i,
                    "source": str(i),
                    "target": f"{lang}-{i}",
                    "is_bad_source": False,
                    "document_id": doc,
                }
                for i, doc in enumerate(["train", "a", "b"])
            }
            for lang in ("a", "b")
        ]
        rows = domain_scaling.translation_examples(manifest, tables, 2)
        self.assertEqual({r["segment_id"] for r in rows[0]}, {1, 2})
        self.assertEqual(
            [r["segment_id"] for r in rows[0]], [r["segment_id"] for r in rows[1]]
        )

    def test_translation_selection_rejects_document_leakage(self):
        self.assertTrue(hasattr(domain_scaling, "translation_examples"))
        manifest = {
            "segment_ids": {"train": [0], "validation": [1]},
            "document_ids": {"train": ["same"], "validation": ["same"]},
        }
        with self.assertRaises(ValueError):
            domain_scaling.translation_examples(manifest, [], 1)


if __name__ == "__main__":
    unittest.main()
