"""Training from scratch on real domains with matched batches and online state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from .common import save_json, seed_everything
from .domain_optimizer import OnlineGroupAdafactor, OnlineGroupAdam, ParameterPartition

OPTIMIZER_METHODS = (
    "adamw",
    "adamw8bit",
    "adammini",
    "adafactor",
    "muon",
    "full",
    "tbe",
    "jl",
    "mean",
    "permuted",
    "full_nomomentum",
    "tbe_nomomentum",
    "jl_nomomentum",
    "mean_nomomentum",
    "full_adafactor",
    "tbe_adafactor",
    "jl_adafactor",
    "mean_adafactor",
)


@dataclass(frozen=True)
class DomainConfig:
    seed: int = 11
    steps: int = 2000
    batch_size: int = 16
    hidden: int = 384
    layers: int = 8
    intermediate: int = 1024
    heads: int = 6
    kv_heads: int = 2
    lr: float = 3e-4
    method: str = "adamw"
    partition: str = "row"
    estimator: str = "raw"
    task_mode: str = "multi"
    eval_blocks: int = 32
    eval_every: int = 500
    weight_decay: float = 0.01
    role: str = "validation"
    score_link: str = "linear"


def make_model(config: DomainConfig, vocab: int, device="cuda"):
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=config.hidden,
        intermediate_size=config.intermediate,
        num_hidden_layers=config.layers,
        num_attention_heads=config.heads,
        num_key_value_heads=config.kv_heads,
        head_dim=config.hidden // config.heads,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).to(device)


def batch_loss(model, blocks):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=blocks.is_cuda):
        inputs = blocks[:, :-1]
        attention_mask = (inputs >= 0) if bool((inputs < 0).any()) else None
        logits = model(
            input_ids=inputs.clamp_min(0),
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), blocks[:, 1:].flatten(), ignore_index=-1
        )
    return loss, logits


@torch.no_grad()
def logit_residual_norm(logits, labels):
    # Exact squared derivative of mean token cross entropy w.r.t. logits.
    probs = logits.float().softmax(-1)
    valid = labels >= 0
    correct = probs.gather(-1, labels.clamp_min(0)[..., None]).squeeze(-1)
    return (
        (probs.square().sum(-1) - 2 * correct + 1) * valid
    ).sum() / valid.sum().clamp_min(1) ** 2


@torch.no_grad()
def evaluate_details(model, domain_blocks, limit=64, batch_size=32):
    was_training = model.training
    model.eval()
    if model.config.vocab_size > 65536:
        batch_size = min(batch_size, 2)
    losses = []
    details = []
    for data in domain_blocks:
        sums, counts = [], []
        indices = torch.randperm(
            len(data), generator=torch.Generator().manual_seed(90210)
        )[:limit]
        for blocks in data[indices].split(batch_size):
            blocks = blocks.to(next(model.parameters()).device).long()
            _, logits = batch_loss(model, blocks)
            labels = blocks[:, 1:]
            token_losses = torch.nn.functional.cross_entropy(
                logits.float().flatten(0, 1),
                labels.flatten(),
                ignore_index=-1,
                reduction="none",
            ).reshape(labels.shape)
            sums.extend(token_losses.sum(1).cpu().tolist())
            counts.extend((labels >= 0).sum(1).cpu().tolist())
        losses.append(sum(sums) / sum(counts))
        details.append(
            {"indices": indices.tolist(), "loss_sums": sums, "token_counts": counts}
        )
    model.train(was_training)
    return torch.tensor(losses), details


def evaluate(model, domain_blocks, limit=64, batch_size=32):
    return evaluate_details(model, domain_blocks, limit, batch_size)[0]


def optimizer_for(model, config, features):
    if config.method == "adammini":
        from adam_mini import Adam_mini

        optimizer = Adam_mini(
            model.named_parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
            dim=model.config.hidden_size,
            n_heads=model.config.num_attention_heads,
            n_kv_heads=model.config.num_key_value_heads,
            verbose=False,
        )
        # Authors' recommended whole-value block for short training runs.
        optimizer.wv_names = set()
        # Keep decay identical across baselines, including normalization weights.
        for group in optimizer.param_groups:
            group["weight_decay"] = config.weight_decay
        return optimizer
    if config.method == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
            min_8bit_size=4096,
        )
    if config.method == "muon":
        hidden = []
        other = []
        for name, p in model.named_parameters():
            (hidden if p.ndim == 2 and ".layers." in name else other).append(p)
        return OptimizerPair(
            torch.optim.Muon(
                hidden,
                lr=config.lr,
                weight_decay=config.weight_decay,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW(
                other,
                lr=config.lr,
                betas=(0.9, 0.99),
                weight_decay=config.weight_decay,
                fused=next(model.parameters()).is_cuda,
            ),
        )
    if config.method == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
            fused=True,
        )
    if config.method == "adafactor":
        return torch.optim.Adafactor(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    partition = ParameterPartition(
        model, config.partition, cache_sizes=not config.method.endswith("_adafactor")
    )
    if config.method.endswith("_adafactor"):
        return OnlineGroupAdafactor(
            partition,
            features,
            "mean" if len(features) == 1 else config.method.removesuffix("_adafactor"),
            estimator=config.estimator,
            lr=config.lr,
            weight_decay=config.weight_decay,
            seed=config.seed + 401,
            score_link=config.score_link,
        )
    no_momentum = config.method.endswith("_nomomentum")
    representation = config.method.removesuffix("_nomomentum")
    if no_momentum and len(features) == 1:
        # The pooled stream needs no task-feature coordinate or redundant slope.
        representation = "mean"
    return OnlineGroupAdam(
        partition,
        features,
        representation,
        estimator=config.estimator,
        lr=config.lr,
        weight_decay=config.weight_decay,
        seed=config.seed + 401,
        score_link=config.score_link,
        beta1=0.0 if no_momentum else 0.9,
    )


class OptimizerPair:
    """Muon for hidden matrices, AdamW for embeddings and scalar/vector weights."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [
            g for optimizer in optimizers for g in optimizer.param_groups
        ]

    @property
    def state(self):
        return {
            p: s for optimizer in self.optimizers for p, s in optimizer.state.items()
        }

    def zero_grad(self, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(**kwargs)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


def optimizer_bytes(optimizer):
    if isinstance(optimizer, OnlineGroupAdam):
        return optimizer.persistent_bytes
    # Quantization lookup maps can be shared across parameter states.
    storages = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                storages[(value.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def experiment_id(config):
    body = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def train(
    corpus_path: Path,
    features_path: Path,
    output: Path,
    config: DomainConfig,
    *,
    checkpoint: bool = False,
):
    seed_everything(config.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    corpus = torch.load(corpus_path, weights_only=False)
    feature_payload = torch.load(features_path, weights_only=False)
    features = feature_payload["features"].cuda()
    if config.task_mode == "single":
        features = torch.zeros(1, 1, device="cuda")
    model = make_model(config, corpus["vocab_size"])
    optimizer = optimizer_for(model, config, features)
    generator = torch.Generator().manual_seed(config.seed + 1000)
    order = []
    while len(order) < config.steps:
        order.extend(
            torch.randperm(len(corpus["domains"]), generator=generator).tolist()
        )
    train_data = [x.cuda().long() for x in corpus["train"]]
    initial = evaluate(model, corpus[config.role], config.eval_blocks)
    curves = [
        {
            "step": 0,
            "macro_nll": float(initial.mean()),
            "per_domain_nll": initial.tolist(),
        }
    ]
    torch.cuda.reset_peak_memory_stats()
    training_seconds = 0.0
    started = time.perf_counter()
    for step, task in enumerate(order[: config.steps]):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        local = train_data[task]
        indices = torch.randint(
            len(local), (config.batch_size,), generator=generator
        ).to(local.device)
        blocks = local[indices]
        optimizer.zero_grad(set_to_none=True)
        loss, logits = batch_loss(model, blocks)
        residual = (
            logit_residual_norm(logits.detach(), blocks[:, 1:])
            if config.estimator == "normalized"
            else 1.0
        )
        loss.backward()
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # A single batch schedule feeds every method, regardless of task labels.
        warmup = max(20, config.steps // 20)
        scale = min(1.0, (step + 1) / warmup) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / config.steps))
        )
        for group in optimizer.param_groups:
            group["lr"] = config.lr * scale
        if isinstance(optimizer, OnlineGroupAdam):
            optimizer.step(
                task=0 if config.task_mode == "single" else task, residual_norm=residual
            )
        else:
            optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - tick
        if not torch.isfinite(loss) or not torch.isfinite(gradnorm):
            raise FloatingPointError(
                f"nonfinite training at step {step}: {loss}, {gradnorm}"
            )
        if (step + 1) % config.eval_every == 0 or step + 1 == config.steps:
            values = evaluate(model, corpus[config.role], config.eval_blocks)
            record = {
                "step": step + 1,
                "macro_nll": float(values.mean()),
                "worst_domain_nll": float(values.max()),
                "per_domain_nll": values.tolist(),
                "training_seconds": training_seconds,
            }
            if isinstance(optimizer, OnlineGroupAdam):
                record["clipped_group_fraction"] = optimizer.bank.clipped_fraction
            curves.append(record)
            print(
                json.dumps(
                    {
                        "method": config.method,
                        "partition": config.partition,
                        "seed": config.seed,
                        **record,
                    }
                ),
                flush=True,
            )
    result = {
        "configuration": asdict(config),
        "experiment_id": experiment_id(config),
        "domains": corpus["domains"],
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "initialization": "random",
        "importance_source": "current training batches only",
        "feature_source": str(features_path.resolve()),
        "task_count": len(features),
        "task_switches": sum(a != b for a, b in pairwise(order[: config.steps])),
        "curve": curves,
        "final": curves[-1],
        "optimizer_persistent_bytes": optimizer_bytes(optimizer),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "training_seconds": training_seconds,
        "wall_seconds": time.perf_counter() - started,
        "tokens": config.steps * config.batch_size * corpus["length"],
        "claim_ready": False,
        "estimator_sampling": "squared minibatch mean gradient, not per-example EF",
    }
    result["evaluation_sampling"] = "fixed random blocks, seed 90210"
    if isinstance(optimizer, OnlineGroupAdam):
        result["grouped_optimizer"] = {
            "update_rule": type(optimizer).__name__,
            "beta1": optimizer.beta1,
            "groups": optimizer.partition.n_groups,
            "coefficients_per_group": optimizer.bank.cross.shape[1],
            "first_moment_bytes": sum(
                s["exp_avg"].numel() * s["exp_avg"].element_size()
                for s in optimizer.state.values()
                if "exp_avg" in s
            ),
            "bank_bytes": optimizer.bank.bytes,
            "group_size_bytes": optimizer.partition.metadata_bytes,
        }
    if config.method == "adammini":
        result["optimizer_implementation"] = {
            "package": "adam-mini==1.1.1",
            "value_partition": "whole tensor, authors' short-run recommendation",
            "weight_decay_scope": "all parameters, matched to other methods",
            "source": "https://github.com/zyushun/Adam-mini",
        }
    save_json(output, result)
    if checkpoint:
        online = (
            {
                "bank": optimizer.bank.state_dict(),
                "partition": config.partition,
                "estimator": config.estimator,
                "task_mode": config.task_mode,
                "domains": corpus["domains"],
                "feature_source": str(features_path.resolve()),
                "statistic": "Online squared clipped minibatch-mean gradients, before method-specific preconditioning; not per-example EF. Moments precede the final weight update.",
            }
            if isinstance(optimizer, OnlineGroupAdam)
            else None
        )
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "configuration": asdict(config),
                "vocab_size": corpus["vocab_size"],
                "online_importance": online,
            },
            output.with_suffix(".pt"),
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--feature-file", default="task_features.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--method",
        default="adamw",
        choices=OPTIMIZER_METHODS,
    )
    p.add_argument("--partition", default="row", choices=("row", "tensor", "swiglu"))
    p.add_argument("--estimator", default="raw", choices=("raw", "normalized"))
    p.add_argument("--task-mode", default="multi", choices=("multi", "single"))
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--score-link", choices=("linear", "log"), default="linear")
    p.add_argument("--eval-blocks", type=int, default=32)
    p.add_argument("--role", choices=("validation", "test"), default="validation")
    p.add_argument("--scale", choices=("screen", "medium"), default="screen")
    p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    kw = {
        "seed": args.seed,
        "steps": args.steps,
        "method": args.method,
        "partition": args.partition,
        "estimator": args.estimator,
        "task_mode": args.task_mode,
        "lr": args.lr,
        "eval_blocks": args.eval_blocks,
        "role": args.role,
        "score_link": args.score_link,
    }
    if args.scale == "medium":
        kw.update(
            hidden=512,
            layers=12,
            intermediate=1536,
            heads=8,
            kv_heads=2,
            batch_size=32,
            eval_every=2000,
        )
    if args.smoke:
        kw.update(
            hidden=128,
            layers=2,
            intermediate=256,
            heads=4,
            kv_heads=2,
            steps=10,
            eval_every=10,
            eval_blocks=2,
            batch_size=4,
        )
    train(
        args.data / "corpus.pt",
        args.data / args.feature_file,
        args.output,
        DomainConfig(**kw),
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
