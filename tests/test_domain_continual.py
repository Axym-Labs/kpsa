import unittest

from task_embeddings.domain_continual import screen_configs, task_sequence


class ContinualProtocolTests(unittest.TestCase):
    def test_summed_linear_queries_do_not_test_task_conditioning(self):
        import torch

        from task_embeddings.domain_applications import query_scores

        generator = torch.Generator().manual_seed(2026)
        # Positive predictions avoid the only nonlinear step, clipping at zero.
        atlas = 10 + torch.rand(12, 20, generator=generator)
        features = torch.randn(20, 6, generator=generator)
        for method in ("full", "mean", "tbe", "jl"):
            actual = query_scores(atlas, features, method, 11)
            self.assertTrue((actual > 0).all())
            torch.testing.assert_close(actual.sum(1), atlas.sum(1))

    def test_screen_contains_all_representations_and_estimators(self):
        rows = screen_configs()
        for strength in (1e3, 1e4, 1e5):
            for estimator in ("raw", "normalized"):
                self.assertEqual(
                    {
                        r["method"]
                        for r in rows
                        if r.get("strength") == strength
                        and r.get("estimator") == estimator
                    },
                    {"diagonal", "full", "tbe", "jl", "mean"},
                )

    def test_twenty_domain_sequence_covers_every_task_once(self):
        sequence = task_sequence(20, 20, 11)
        self.assertEqual(sorted(sequence), list(range(20)))
        self.assertEqual(sequence, task_sequence(20, 20, 11))
        self.assertNotEqual(sequence, task_sequence(20, 20, 12))

    def test_legacy_sequence_is_preserved(self):
        import torch

        g = torch.Generator().manual_seed(1211)
        expected = torch.tensor([1, 3, 7, 10, 17, 19])[torch.randperm(6, generator=g)]
        self.assertEqual(task_sequence(20, 6, 11), expected.tolist())


if __name__ == "__main__":
    unittest.main()
