import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from task_embeddings.domain_optimizer import ParameterPartition
from task_embeddings.domain_oracles import collect, weighted_diagonal_scores
from task_embeddings.domain_train import DomainConfig, make_model


class DiagonalOracleTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA integration test")
    def test_joint_collector_preserves_mass_across_partitions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = DomainConfig(
                hidden=8, layers=1, intermediate=16, heads=2, kv_heads=1
            )
            model = make_model(config, 16, device="cpu")
            checkpoint = root / "model.pt"
            torch.save(
                {
                    "configuration": asdict(config),
                    "vocab_size": 16,
                    "model": model.state_dict(),
                },
                checkpoint,
            )
            torch.save(
                {
                    "domains": ["a", "b"],
                    "train": [torch.randint(0, 16, (2, 5)) for _ in range(2)],
                },
                root / "corpus.pt",
            )
            collect(
                checkpoint,
                root,
                root / "scores.pt",
                samples=1,
                kind="all",
                include_fisher=False,
            )
            result = torch.load(root / "scores.pt", weights_only=False)["scores"]
            self.assertEqual(set(result), {"row", "tensor", "swiglu"})
            totals = []
            for kind, scores in result.items():
                part = ParameterPartition(model, kind)
                totals.append((scores["raw"]["pruning"] * part.sizes[:, None]).sum(0))
            for total in totals[1:]:
                torch.testing.assert_close(total, totals[0])

    def test_weighting_precedes_group_averaging(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 3.0]]))
        model.weight.grad = torch.tensor([[2.0, 4.0]])
        part = ParameterPartition(model, "row")
        result = weighted_diagonal_scores(part, {"weight": torch.tensor([[0.5, 2.0]])})
        torch.testing.assert_close(result["absolute"], torch.tensor([10.0]))
        torch.testing.assert_close(result["pruning"], torch.tensor([74.0]))
        torch.testing.assert_close(result["quantization"], torch.tensor([17.0]))


if __name__ == "__main__":
    unittest.main()
