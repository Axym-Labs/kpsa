import unittest

import torch

from task_embeddings.controlled_v4 import (
    HardControlledConfig,
    global_topk_indices,
    hard_task_mixtures,
    make_program_batch,
    make_stratified_task_order,
    program_primitive_targets,
    training_phases,
)


class ControlledV4Tests(unittest.TestCase):
    def test_task_mixtures_cover_pure_pairs_and_triples(self):
        mixtures = hard_task_mixtures()
        self.assertEqual(mixtures.shape, (30, 6))
        torch.testing.assert_close(mixtures[:6], torch.eye(6))
        torch.testing.assert_close(mixtures.sum(dim=1), torch.ones(30))
        self.assertEqual((mixtures[6:21] > 0).sum(dim=1).unique().tolist(), [2])
        self.assertEqual((mixtures[21:] > 0).sum(dim=1).unique().tolist(), [3])

    def test_program_primitives_are_finite_and_bounded(self):
        payloads = torch.arange(2 * 6 * 8).reshape(2, 6, 8) % 16
        targets = program_primitive_targets(payloads, payload_modulus=16)
        self.assertEqual(targets.shape, (2, 6))
        self.assertTrue(bool(torch.isfinite(targets).all()))
        self.assertTrue(bool((targets.abs() <= 1).all()))

    def test_global_checksum_is_smooth_tokenwise_accumulation(self):
        payloads = torch.zeros(1, 6, 8, dtype=torch.long)
        targets = program_primitive_targets(payloads, payload_modulus=16)
        self.assertAlmostEqual(float(targets[0, 0]), 0.0, places=6)

    def test_program_batch_contains_each_field_tag_once(self):
        config = HardControlledConfig(payload_length=8, payload_modulus=16)
        tokens, targets, task_ids = make_program_batch(
            config,
            hard_task_mixtures(),
            task=3,
            batch_size=4,
            generator=torch.Generator().manual_seed(7),
            device=torch.device("cpu"),
            noise=0.0,
        )
        self.assertEqual(tokens.shape, (4, 54))
        self.assertEqual(targets.shape, (4,))
        self.assertEqual(task_ids.tolist(), [3, 3, 3, 3])
        tags = tokens[(tokens >= 16)]
        for tag in range(16, 22):
            self.assertEqual(int((tags == tag).sum()), 4)

    def test_training_order_balances_pure_and_composite_supervision(self):
        order = make_stratified_task_order(
            n_tasks=30,
            n_pure_tasks=6,
            steps=120,
            pure_fraction=0.5,
            generator=torch.Generator().manual_seed(9),
        )
        counts = torch.bincount(order, minlength=30)
        self.assertEqual(len(order), 120)
        self.assertEqual(int(counts[:6].sum()), 60)
        self.assertEqual(int(counts[6:].sum()), 60)
        self.assertLessEqual(int(counts[:6].max() - counts[:6].min()), 1)
        self.assertLessEqual(int(counts[6:].max() - counts[6:].min()), 1)
        self.assertFalse(torch.equal(order, order.sort().values))

    def test_training_order_rejects_missing_task_stratum(self):
        generator = torch.Generator().manual_seed(9)
        with self.assertRaises(ValueError):
            make_stratified_task_order(6, 6, 120, 0.5, generator)
        with self.assertRaises(ValueError):
            make_stratified_task_order(30, 6, 10, 1.1, generator)

    def test_training_phases_predeclare_low_rate_refinement(self):
        config = HardControlledConfig()
        self.assertEqual(
            training_phases(config),
            (
                ("primary", 24_000, 3e-4),
                ("refinement", 6_000, 1e-4),
            ),
        )

    def test_global_topk_indices_select_exact_count_across_layers(self):
        selected = global_topk_indices(
            torch.tensor([1.0, 9.0, 4.0, 8.0, 2.0]), [2, 3], count=3
        )
        self.assertEqual(sum(value.numel() for value in selected), 3)
        self.assertEqual(selected[0].tolist(), [1])
        self.assertEqual(set(selected[1].tolist()), {0, 1})
        self.assertEqual(
            sum(
                value.numel() for value in global_topk_indices(torch.ones(5), [2, 3], 0)
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
