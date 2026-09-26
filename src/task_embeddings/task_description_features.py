"""Frozen task-description representations for language sensitivity studies."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

LANGUAGE_NAMES = {
    "ar_SA": "Arabic",
    "bg_BG": "Bulgarian",
    "bn_IN": "Bengali",
    "cs_CZ": "Czech",
    "de_DE": "German",
    "el_GR": "Greek",
    "es_MX": "Spanish",
    "fa_IR": "Persian",
    "fr_FR": "French",
    "hi_IN": "Hindi",
    "id_ID": "Indonesian",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
    "ko_KR": "Korean",
    "nl_NL": "Dutch",
    "pl_PL": "Polish",
    "pt_BR": "Portuguese",
    "ru_RU": "Russian",
    "tr_TR": "Turkish",
    "zh_CN": "Chinese",
}


def task_descriptions(domains: list[str]) -> list[str]:
    """Describe either multilingual language tasks or Amazon review domains."""
    if all(domain in LANGUAGE_NAMES for domain in domains):
        return [
            f"Text written naturally in {LANGUAGE_NAMES[domain]}."
            for domain in domains
        ]
    return [
        "Amazon customer reviews for products in the "
        + domain.replace("_", " ")
        + " category."
        for domain in domains
    ]


def last_token_pool(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool the last non-padding token for left- or right-padded causal encoders."""
    if hidden_state.ndim != 3 or attention_mask.shape != hidden_state.shape[:2]:
        raise ValueError("hidden states and attention mask must share batch/sequence axes")
    if bool(attention_mask[:, -1].all()):
        return hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden_state[
        torch.arange(len(hidden_state), device=hidden_state.device), lengths
    ]


def encode_descriptions(
    model,
    tokenizer,
    descriptions: list[str],
    *,
    batch_size: int = 20,
) -> torch.Tensor:
    if not descriptions:
        raise ValueError("at least one task description is required")
    device = next(model.parameters()).device
    features = []
    model.eval()
    for start in range(0, len(descriptions), batch_size):
        batch = tokenizer(
            descriptions[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        batch = {name: value.to(device) for name, value in batch.items()}
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(**batch, output_hidden_states=True, use_cache=False)
        pooled = last_token_pool(output.hidden_states[-1], batch["attention_mask"])
        features.append(F.normalize(pooled.float(), dim=1).cpu())
    return torch.cat(features)


def encode_task_inputs(
    model,
    domains: list[torch.Tensor],
    *,
    pad_token_id: int,
    samples: int = 32,
    excluded_profile_samples: int = 8,
    batch_size: int = 16,
) -> torch.Tensor:
    """Average frozen encoder states from examples disjoint from profiling."""
    device = next(model.parameters()).device
    task_features = []
    model.eval()
    for task, data in enumerate(domains):
        order = torch.randperm(
            len(data), generator=torch.Generator().manual_seed(31_415 + task)
        )
        indices = order[
            excluded_profile_samples : excluded_profile_samples + samples
        ]
        if len(indices) != samples:
            raise ValueError("insufficient examples after excluding profile samples")
        pooled = []
        for batch_indices in indices.split(batch_size):
            input_ids = data[batch_indices].long()
            attention_mask = input_ids.ge(0)
            input_ids = input_ids.masked_fill(~attention_mask, pad_token_id).to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            pooled.append(
                F.normalize(
                    last_token_pool(output.last_hidden_state, attention_mask).float(),
                    dim=1,
                ).cpu()
            )
        task_features.append(F.normalize(torch.cat(pooled).mean(0), dim=0))
    return torch.stack(task_features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--source", choices=("descriptions", "inputs"), default="descriptions")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--excluded-profile-samples", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    corpus = torch.load(args.corpus, map_location="cpu", weights_only=False, mmap=True)
    domains = list(corpus["domains"])
    descriptions = task_descriptions(domains)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.source == "descriptions":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        ).cuda()
        features = encode_descriptions(model, tokenizer, descriptions)
        pooling = "last non-padding description token, L2 normalized"
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        ).cuda()
        features = encode_task_inputs(
            model,
            corpus["train"],
            pad_token_id=tokenizer.pad_token_id,
            samples=args.samples,
            excluded_profile_samples=args.excluded_profile_samples,
        )
        pooling = (
            f"mean of {args.samples} L2-normalized last-token input encodings; "
            f"first {args.excluded_profile_samples} deterministic profile positions excluded"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "domains": domains,
            "descriptions": descriptions,
            "model": args.model,
            "source": args.source,
            "pooling": pooling,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
