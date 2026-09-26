import math
import unittest

import torch

from task_embeddings.query_v5 import (
    build_query_methods,
    centered_linear_task_scores,
    fold_training_means,
    query_ranking_metrics,
    select_residual_scale,
)


class QueryV5Tests(unittest.TestCase):
    def test_centered_linear_query_recovers_affine_heldout_scores(self):
        # A shared module intercept plus a task-feature-dependent residual.
        features = torch.tensor([[-1.0], [0.0], [1.0]])
        importance = torch.tensor(
            [
                [8.0, 10.0, 12.0],
                [23.0, 20.0, 17.0],
                [4.5, 5.0, 5.5],
            ]
        )

        predictions, queryable = centered_linear_task_scores(
            importance, features, folds=[[2]], ridge=0.0
        )

        torch.testing.assert_close(predictions[:, 2], importance[:, 2])
        self.assertTrue(queryable[2])

    def test_centered_linear_query_never_reads_heldout_importance(self):
        features = torch.tensor([[-1.0], [0.0], [1.0]])
        importance = torch.tensor(
            [[8.0, 10.0, 12.0], [23.0, 20.0, 17.0], [4.5, 5.0, 5.5]]
        )
        first, _ = centered_linear_task_scores(
            importance, features, folds=[[2]], ridge=0.0
        )
        changed = importance.clone()
        changed[:, 2] = torch.tensor([999.0, -500.0, 42.0])

        second, _ = centered_linear_task_scores(
            changed, features, folds=[[2]], ridge=0.0
        )

        torch.testing.assert_close(first[:, 2], second[:, 2])

    def test_fold_training_means_exclude_every_heldout_column(self):
        importance = torch.tensor([[1.0, 3.0, 100.0, 200.0]])

        means = fold_training_means(importance, folds=[[2, 3]])

        torch.testing.assert_close(means[:, 2], torch.tensor([2.0]))
        torch.testing.assert_close(means[:, 3], torch.tensor([2.0]))

    def test_residual_metric_removes_a_dominant_shared_ranking(self):
        source = torch.tensor(
            [
                [100.0, 100.0, 100.0],
                [50.0, 50.0, 50.0],
                [10.0, 10.0, 10.0],
                [1.0, 1.0, 1.0],
            ]
        )
        target = source.clone()
        target[:, 2] += torch.tensor([0.0, 0.0, -20.0, 20.0])
        prediction = source.clone()
        prediction[:, 2] = source[:, :2].mean(dim=1)
        queryable = torch.tensor([False, False, True])

        metrics = query_ranking_metrics(
            prediction,
            source,
            target,
            folds=[[2]],
            queryable=queryable,
            top_fraction=0.25,
        )

        self.assertGreater(metrics["raw"]["spearman_mean"], 0.7)
        self.assertTrue(math.isnan(metrics["residual"]["spearman_mean"]))
        self.assertTrue(math.isnan(metrics["residual"]["topk_recall_mean"]))

    def test_query_method_matrix_uses_no_heldout_scores_except_oracle(self):
        features = torch.tensor([[-1.0], [0.0], [1.0]])
        importance = torch.tensor(
            [[8.0, 10.0, 12.0], [23.0, 20.0, 17.0], [4.5, 5.0, 5.5]]
        )
        first = build_query_methods(
            importance,
            features,
            folds=[[2]],
            seed=11,
            repeats=2,
            residual_scales=(0.5, 1.0),
        )
        changed = importance.clone()
        changed[:, 2] = torch.tensor([999.0, -500.0, 42.0])
        second = build_query_methods(
            changed,
            features,
            folds=[[2]],
            seed=11,
            repeats=2,
            residual_scales=(0.5, 1.0),
        )

        self.assertFalse(any("ief" in name.lower() for name in first))
        self.assertEqual(set(first), set(second))
        for name in first:
            if name == "observed_opg_oracle":
                self.assertFalse(
                    torch.equal(first[name][0][:, 2], second[name][0][:, 2])
                )
            else:
                torch.testing.assert_close(first[name][0][:, 2], second[name][0][:, 2])

    def test_scale_selection_uses_training_task_topk_fidelity(self):
        features = torch.tensor([[-1.0], [0.0], [1.0]])
        source = torch.tensor(
            [
                [10.0, 10.0, 10.0],
                [6.0, 9.0, 12.0],
                [11.0, 8.0, 5.0],
            ]
        )
        target = source.clone()

        selection = select_residual_scale(
            source,
            target,
            features,
            task_indices=[0, 1, 2],
            residual_scales=(0.25, 0.5, 1.0),
            top_fraction=1 / 3,
            ridge=0.0,
        )

        self.assertEqual(selection["selected_scale"], 1.0)
        self.assertEqual(selection["selection_metric"], "raw_topk_recall")


if __name__ == "__main__":
    unittest.main()
