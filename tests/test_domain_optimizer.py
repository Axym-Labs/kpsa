import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from task_embeddings.domain_optimizer import (
    OnlineGroupAdam,
    OnlineTaskSketch,
    ParameterPartition,
    partition_hierarchy,
)


class DomainOptimizerTests(unittest.TestCase):
    def test_partition_hierarchy_keeps_shared_mlp_features_together(self):
        partition = ParameterPartition(self.model(), "swiglu")
        bundle, bundle_labels = partition_hierarchy(partition, "bundle")
        _sublayer, sublayer_labels = partition_hierarchy(partition, "sublayer")
        layer, layer_labels = partition_hierarchy(partition, "layer")
        self.assertEqual(bundle.shape, (partition.n_groups,))
        self.assertLess(len(sublayer_labels), len(bundle_labels))
        self.assertLessEqual(len(layer_labels), len(sublayer_labels))
        self.assertEqual(int(bundle.min()), 0)
        self.assertEqual(int(layer.max()), len(layer_labels) - 1)

    def test_shape_derived_group_sizes_need_no_persistent_size_vector(self):
        model = self.model()
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        for kind in ("row", "tensor", "swiglu"):
            cached = ParameterPartition(model, kind)
            compact = ParameterPartition(model, kind, cache_sizes=False)
            self.assertEqual(compact.metadata_bytes, 0)
            self.assertEqual(cached.metadata_bytes, 4 * cached.n_groups)
            torch.testing.assert_close(compact.sizes, cached.sizes)
            torch.testing.assert_close(
                compact.gradient_scores(), cached.gradient_scores()
            )
            torch.testing.assert_close(compact.weight_energy(), cached.weight_energy())

    def test_grouped_adafactor_matches_reference_when_rows_determine_variance(self):
        from task_embeddings.domain_optimizer import OnlineGroupAdafactor

        # With column-constant gradients, Adafactor's factored matrix is exactly
        # the row second moment. Include a vector to check its unfactored path.
        first = torch.nn.Linear(3, 2)
        second = torch.nn.Linear(3, 2)
        with torch.no_grad():
            first.weight.copy_(torch.tensor([[0.1, 0.2, 0.3], [2.0, 3.0, 4.0]]))
            first.bias.copy_(torch.tensor([0.2, 2.0]))
        second.load_state_dict(first.state_dict())
        grouped = OnlineGroupAdafactor(
            ParameterPartition(first, "row"),
            torch.zeros(1, 1),
            "mean",
            lr=0.03,
            weight_decay=0.02,
        )
        reference = torch.optim.Adafactor(
            second.parameters(), lr=0.03, weight_decay=0.02
        )
        for scale in (1.0, -3.0, 40.0, 0.2):
            first.weight.grad = scale * torch.tensor([[1.0] * 3, [4.0] * 3])
            first.bias.grad = scale * torch.tensor([1.0, 3.0])
            second.weight.grad = first.weight.grad.clone()
            second.bias.grad = first.bias.grad.clone()
            grouped.step()
            reference.step()
            for p, q in zip(first.parameters(), second.parameters()):
                torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)
        self.assertEqual(len(grouped.state), 0)
        self.assertEqual(grouped.bank.cross.shape, (4, 1))

    def test_grouped_adafactor_factory_is_explicit_and_rejects_incompatible_units(self):
        from task_embeddings.domain_optimizer import OnlineGroupAdafactor
        from task_embeddings.domain_train import DomainConfig, optimizer_for

        for method in (
            "mean_adafactor",
            "full_adafactor",
            "tbe_adafactor",
            "jl_adafactor",
        ):
            optimizer = optimizer_for(
                self.model(), DomainConfig(method=method), torch.randn(3, 2)
            )
            self.assertIsInstance(optimizer, OnlineGroupAdafactor)
            self.assertEqual(optimizer.beta1, 0.0)
            self.assertEqual(optimizer.partition.metadata_bytes, 0)
            self.assertEqual(optimizer.persistent_bytes, optimizer.bank.bytes)
        for settings in ({"estimator": "normalized"}, {"score_link": "log"}):
            with self.assertRaises(ValueError):
                optimizer_for(
                    self.model(),
                    DomainConfig(method="tbe_adafactor", **settings),
                    torch.randn(3, 2),
                )

    def test_online_bank_snapshot_preserves_queries_and_can_continue_updates(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        for method in ("mean", "full", "tbe", "jl"):
            bank = OnlineTaskSketch(2, features, method, score_link="linear", beta=0.9)
            bank.update(torch.tensor([1.0, 4.0]), 0)
            bank.update(torch.tensor([3.0, 2.0]), 1)
            payload = bank.state_dict()
            restored = OnlineTaskSketch.from_state_dict(payload)
            for task in range(3):
                torch.testing.assert_close(bank.query(task), restored.query(task))
            bank.update(torch.tensor([2.0, 8.0]), 2)
            restored.update(torch.tensor([2.0, 8.0]), 2)
            self.assertEqual(restored.steps, 3)
            torch.testing.assert_close(bank.query(2), restored.query(2))
            self.assertEqual(payload["steps"], 2)
            self.assertFalse(torch.equal(payload["cross"], bank.cross))

    def test_zero_momentum_uses_current_gradient_without_allocating_dense_state(self):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        partition = ParameterPartition(model, "row")
        optimizer = OnlineGroupAdam(
            partition,
            torch.zeros(1, 1),
            "mean",
            beta1=0.0,
            lr=0.1,
            weight_decay=0.0,
            eps=0.0,
        )
        model.weight.grad = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        optimizer.step()
        # Row energies 1 and 9, mean 5; the shared 10% damping gives 1.4 and 8.6.
        expected = torch.tensor([[1 - 0.1 / 1.4**0.5] * 2, [1 - 0.3 / 8.6**0.5] * 2])
        torch.testing.assert_close(model.weight, expected)
        self.assertEqual(len(optimizer.state), 0)
        model.weight.grad = torch.tensor([[-1.0, -1.0], [-3.0, -3.0]])
        optimizer.step()
        torch.testing.assert_close(model.weight, torch.ones(2, 2))
        self.assertEqual(len(optimizer.state), 0)

    def test_momentum_free_factory_keeps_one_coefficient_per_pooled_group(self):
        from task_embeddings.domain_train import DomainConfig, optimizer_for

        for method in ("tbe_nomomentum", "jl_nomomentum", "full_nomomentum"):
            model = self.model()
            optimizer = optimizer_for(
                model,
                DomainConfig(method=method, task_mode="single"),
                torch.zeros(1, 1),
            )
            self.assertEqual(
                optimizer.bank.cross.shape, (optimizer.partition.n_groups, 1)
            )
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            optimizer.step()
            self.assertEqual(len(optimizer.state), 0)
            self.assertEqual(optimizer.bank.representation, "mean")

    def test_adammini_matches_schedule_and_uses_native_transformer_groups(self):
        from adam_mini import Adam_mini

        from task_embeddings.domain_train import (
            DomainConfig,
            optimizer_bytes,
            optimizer_for,
        )

        model = self.model()
        config = DomainConfig(method="adammini", lr=0.001, weight_decay=0.01)
        optimizer = optimizer_for(model, config, torch.zeros(2, 1))
        self.assertIsInstance(optimizer, Adam_mini)
        self.assertEqual(optimizer.wv_names, set())
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(set(map(id, owned)), set(map(id, model.parameters())))
        for group in optimizer.param_groups:
            self.assertEqual((group["beta1"], group["beta2"]), (0.9, 0.99))
            self.assertEqual(group["weight_decay"], 0.01)
            self.assertEqual(group["lr"], 0.001)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for _ in range(3):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            optimizer.step()
        for name, p in model.named_parameters():
            self.assertTrue(torch.isfinite(p).all())
            self.assertFalse(torch.equal(p, before[name]))
            if "q_proj" in name:
                self.assertEqual(optimizer.state[p]["vmean"].numel(), 2)
            elif "k_proj" in name or "v_proj" in name:
                self.assertEqual(optimizer.state[p]["vmean"].numel(), 1)
        parameters = sum(p.numel() for p in model.parameters())
        self.assertGreater(optimizer_bytes(optimizer), parameters * 4)
        self.assertLess(optimizer_bytes(optimizer), parameters * 8)

    def test_partition_metadata_needs_no_allocated_model_weights(self):
        reference = self.model()
        with torch.device("meta"):
            metadata_only = self.model()
        for kind in ("row", "tensor", "swiglu"):
            expected = ParameterPartition(reference, kind)
            actual = ParameterPartition(metadata_only, kind)
            self.assertEqual(actual.n_parameters, expected.n_parameters)
            self.assertEqual(actual.n_groups, expected.n_groups)
            torch.testing.assert_close(actual.sizes, expected.sizes)

    def test_optimizer_memory_counts_shared_state_storage_once(self):
        from types import SimpleNamespace

        from task_embeddings.domain_train import optimizer_bytes

        shared = torch.zeros(256)
        optimizer = SimpleNamespace(state={0: {"map": shared}, 1: {"map": shared}})
        self.assertEqual(optimizer_bytes(optimizer), 256 * 4)

    def test_adamw8bit_baseline_uses_matching_hyperparameters_and_all_weights(self):
        import bitsandbytes as bnb

        from task_embeddings.domain_train import DomainConfig, optimizer_for

        model = self.model()
        config = DomainConfig(method="adamw8bit", lr=0.001, weight_decay=0.01)
        optimizer = optimizer_for(model, config, torch.zeros(2, 1))
        self.assertIsInstance(optimizer, bnb.optim.AdamW8bit)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.99))
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(set(map(id, owned)), set(map(id, model.parameters())))

    def test_padding_does_not_change_loss_or_residual_norm(self):
        from task_embeddings.domain_train import batch_loss, logit_residual_norm

        model = self.model().eval()
        short = torch.tensor([[1, 2, 3, 4]])
        padded = torch.tensor([[1, 2, 3, 4, -1, -1]])
        a, x = batch_loss(model, short)
        b, y = batch_loss(model, padded)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(
            logit_residual_norm(x, short[:, 1:]), logit_residual_norm(y, padded[:, 1:])
        )

    def model(self):
        return Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                tie_word_embeddings=True,
            )
        )

    def test_partition_conserves_squared_gradient_mass_and_covers_tied_weights_once(
        self,
    ):
        model = self.model()
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        truth = sum(p.grad.square().sum() for p in model.parameters())
        for kind in ("row", "tensor", "swiglu"):
            partition = ParameterPartition(model, kind)
            torch.testing.assert_close(
                (partition.gradient_scores() * partition.sizes).sum(), truth
            )
            self.assertEqual(
                partition.n_parameters, sum(p.numel() for p in model.parameters())
            )
            for spec in partition.slices:
                self.assertEqual(
                    spec.expand(torch.ones(partition.n_groups)).numel(), spec.count
                )

    def test_single_task_full_and_tbe_are_equivalent_after_calibration(self):
        torch.manual_seed(0)
        first, second = self.model(), self.model()
        second.load_state_dict(first.state_dict())
        a = OnlineGroupAdam(ParameterPartition(first, "row"), torch.zeros(1, 1), "full")
        b = OnlineGroupAdam(ParameterPartition(second, "row"), torch.zeros(1, 1), "tbe")
        for p, q in zip(first.parameters(), second.parameters()):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        a.step()
        b.step()
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)

    def test_task_bank_does_not_conflate_distinct_tasks(self):
        bank = OnlineTaskSketch(2, torch.zeros(2, 1), "full", beta=0.9)
        bank.update(torch.tensor([1.0, 2.0]), 0)
        bank.update(torch.tensor([4.0, 3.0]), 1)
        torch.testing.assert_close(bank.query(0), torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(bank.query(1), torch.tensor([4.0, 3.0]))

    def test_log_bank_recovers_positive_task_specific_scores(self):
        bank = OnlineTaskSketch(
            2, torch.zeros(2, 1), "full", beta=0.9, score_link="log"
        )
        bank.update(torch.tensor([1.0, 4.0]), 0)
        bank.update(torch.tensor([9.0, 16.0]), 1)
        torch.testing.assert_close(bank.query(0), torch.tensor([1.0, 4.0]))
        torch.testing.assert_close(bank.query(1), torch.tensor([9.0, 16.0]))

    def test_group_weight_energy_conserves_parameter_squared_norm(self):
        model = self.model()
        expected = sum(p.square().sum() for p in model.parameters()).detach()
        for kind in ("row", "tensor", "swiglu"):
            part = ParameterPartition(model, kind)
            torch.testing.assert_close(
                (part.weight_energy() * part.sizes).sum(), expected
            )

    def test_zero_learning_rate_is_identity(self):
        model = self.model()
        part = ParameterPartition(model, "swiglu")
        initial = [p.detach().clone() for p in model.parameters()]
        opt = OnlineGroupAdam(part, torch.randn(3, 2), "tbe", lr=0.0)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step(task=2)
        for p, q in zip(model.parameters(), initial):
            torch.testing.assert_close(p, q)

    def test_muon_baseline_updates_every_parameter_including_embeddings(self):
        from task_embeddings.domain_train import DomainConfig, optimizer_for

        model = self.model()
        initial = [p.detach().clone() for p in model.parameters()]
        optimizer = optimizer_for(model, DomainConfig(method="muon"), torch.zeros(2, 1))
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(set(map(id, owned))), len(list(model.parameters())))
        self.assertEqual(len(owned), len(list(model.parameters())))
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()
        for p, old in zip(model.parameters(), initial):
            self.assertFalse(torch.equal(p, old))


if __name__ == "__main__":
    unittest.main()
