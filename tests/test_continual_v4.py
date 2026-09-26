import unittest

import torch

from task_embeddings.continual_v4 import (
    ContinualConfig,
    ModularRegressor,
    make_experiment_data,
    run_cl_method,
    run_continual_experiment,
)


class ContinualV4Tests(unittest.TestCase):
    def test_every_trainable_parameter_is_module_owned(self):
        model = ModularRegressor(input_dim=4, n_tasks=3, n_modules=5)
        owned = {id(parameter) for parameter in model.module_owned_parameters()}
        trainable = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        self.assertEqual(owned, trainable)

    def test_module_importance_is_broadcast_to_all_owned_elements(self):
        model = ModularRegressor(input_dim=4, n_tasks=3, n_modules=3)
        for parameter in model.module_owned_parameters():
            parameter.grad = torch.ones_like(parameter)
        model.apply_module_gradient_scale(torch.tensor([1.0, 3.0, 7.0]), 1.0)
        expected = torch.tensor([0.5, 0.25, 0.125])
        torch.testing.assert_close(model.hidden.weight.grad[:, 0], expected)
        torch.testing.assert_close(model.hidden.bias.grad, expected)
        torch.testing.assert_close(model.output.weight.grad[0], expected)

    def test_task_order_changes_only_sequence_not_task_data(self):
        config = ContinualConfig(
            seed=4,
            input_dim=4,
            n_tasks=3,
            n_modules=8,
            batch_size=4,
            steps_per_task=1,
            reference_per_task=4,
            test_per_task=4,
        )
        forward = make_experiment_data(config, order=[0, 1, 2])
        reverse = make_experiment_data(config, order=[2, 1, 0])
        self.assertEqual(reverse["order"], [2, 1, 0])
        for task in range(3):
            torch.testing.assert_close(
                forward["tests"][task][0], reverse["tests"][task][0]
            )
            forward_batch = forward["train_batches"][forward["order"].index(task)][0]
            reverse_batch = reverse["train_batches"][reverse["order"].index(task)][0]
            torch.testing.assert_close(forward_batch[0], reverse_batch[0])

    def test_zero_strength_replays_identical_trajectory_across_method_labels(self):
        config = ContinualConfig(
            seed=5,
            input_dim=4,
            n_tasks=3,
            n_modules=12,
            batch_size=8,
            steps_per_task=2,
            reference_per_task=8,
            test_per_task=8,
        )
        data = make_experiment_data(config)
        torch.manual_seed(config.seed)
        initial = ModularRegressor(
            config.input_dim, config.n_tasks, config.n_modules
        ).state_dict()
        first = run_cl_method(
            config,
            "posthoc_ief__semantic6",
            initial,
            data,
            strength=0.0,
            device=torch.device("cpu"),
        )
        second = run_cl_method(
            config,
            "posthoc_ief__jl6",
            initial,
            data,
            strength=0.0,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(first["history"], second["history"], equal_nan=True)
        for method in (
            "tbe_linear",
            "jl_linear",
            "permuted_basis_linear",
            "full_atlas",
            "mean_only",
        ):
            candidate = run_cl_method(
                config,
                method,
                initial,
                data,
                strength=0.0,
                device=torch.device("cpu"),
            )
            torch.testing.assert_close(
                first["history"], candidate["history"], equal_nan=True
            )

    def test_tiny_full_experiment_executes_representation_and_application_baselines(
        self,
    ):
        config = ContinualConfig(
            seed=6,
            input_dim=4,
            n_tasks=3,
            n_modules=12,
            batch_size=8,
            steps_per_task=2,
            reference_per_task=8,
            test_per_task=8,
            replay_per_task=2,
        )
        result = run_continual_experiment(
            config,
            strength_candidates=(0.5,),
            device=torch.device("cpu"),
            force_comparison=True,
        )
        expected = {
            "none",
            "posthoc_ief__semantic6",
            "posthoc_ief__onehot",
            "posthoc_ief__jl6",
            "posthoc_raw_ef__semantic6",
            "online_raw_ef__semantic6",
            "posthoc_activation__semantic6",
            "random",
            "ewc",
            "module_iewc__semantic6",
            "replay",
        }
        self.assertTrue(expected.issubset(set(result["methods"])))
        self.assertIn("posthoc_ief__jl6_seed1", result["methods"])
        self.assertIn("posthoc_ief__jl6_seed2", result["methods"])
        self.assertIn("premise_gate", result)
        for method in expected:
            self.assertEqual(
                result["methods"][method]["summary"]["forgetting_eligible_tasks"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
