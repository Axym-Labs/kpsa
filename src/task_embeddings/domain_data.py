"""Reproducible bounded real-text corpus for multi-domain LM experiments.

The sample is a bounded prefix of each source file, not a representative
sample of Amazon customers. Product-disjoint roles and global text deduplication
prevent review/product leakage. Only training documents train the tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .common import save_json

DOMAINS = (
    "All_Beauty",
    "Toys_and_Games",
    "Cell_Phones_and_Accessories",
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "Electronics",
    "Arts_Crafts_and_Sewing",
    "Baby_Products",
    "Health_and_Household",
    "Office_Products",
    "Digital_Music",
    "Grocery_and_Gourmet_Food",
    "Sports_and_Outdoors",
    "Home_and_Kitchen",
    "Tools_and_Home_Improvement",
    "Pet_Supplies",
    "Video_Games",
    "Clothing_Shoes_and_Jewelry",
    "Patio_Lawn_and_Garden",
    "Automotive",
)


def role_for_product(product: str) -> str:
    number = (
        int(hashlib.sha256(("tbe-domain-v1:" + product).encode()).hexdigest()[:8], 16)
        % 10
    )
    return "validation" if number == 8 else "test" if number == 9 else "train"


def fetch_domain(domain: str, root: Path, count: int) -> Path:
    target = root / f"{domain}.jsonl"
    if target.exists():
        return target
    url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{domain}.jsonl"
    rows = []
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            row = json.loads(line)
            text = (str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()
            if len(text) < 120:
                continue
            product = str(row.get("parent_asin") or row.get("asin"))
            rows.append(
                {"text": text, "product": product, "role": role_for_product(product)}
            )
            if len(rows) >= count:
                break
    if len(rows) < count:
        raise RuntimeError(f"{domain}: only {len(rows)} qualifying documents")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(f"DATA {domain} {len(rows)}", flush=True)
    return target


def prepare(
    root: Path,
    documents_per_domain: int = 12000,
    length: int = 128,
    vocab_size: int = 8192,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(
                lambda name: fetch_domain(name, root / "raw", documents_per_domain),
                DOMAINS,
            )
        )
    rows_by_domain = []
    seen = set()
    duplicates = 0
    for path in paths:
        rows = []
        for line in path.read_text().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            key = hashlib.sha256(
                " ".join(row["text"].lower().split()).encode()
            ).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append(row)
        rows_by_domain.append(rows)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        (
            row["text"]
            for rows in rows_by_domain
            for row in rows
            if row["role"] == "train"
        ),
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<eos>", "<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer.save(str(root / "tokenizer.json"))
    tensors = {role: [] for role in ("train", "validation", "test")}
    counts = {}
    hashes = {}
    for domain, rows in zip(DOMAINS, rows_by_domain):
        counts[domain] = {}
        for role, destination in tensors.items():
            selected = [row["text"] for row in rows if row["role"] == role]
            sequences = tokenizer.encode_batch(selected)
            flat = [token for sequence in sequences for token in (*sequence.ids, 1)]
            usable = len(flat) // (length + 1) * (length + 1)
            blocks = torch.tensor(flat[:usable], dtype=torch.int32).reshape(
                -1, length + 1
            )
            if len(blocks) < 64:
                raise ValueError(f"too few blocks: {domain}/{role}: {len(blocks)}")
            destination.append(blocks)
            counts[domain][role] = {
                "documents": len(selected),
                "blocks": len(blocks),
                "tokens": usable,
            }
        hashes[domain] = hashlib.sha256(
            (root / "raw" / f"{domain}.jsonl").read_bytes()
        ).hexdigest()
    target = root / "corpus.pt"
    torch.save(
        {
            "domains": DOMAINS,
            "length": length,
            "vocab_size": tokenizer.get_vocab_size(),
            **tensors,
        },
        target,
    )
    save_json(
        root / "manifest.json",
        {
            "source": "McAuley-Lab/Amazon-Reviews-2023",
            "source_selection": "bounded source-file prefix",
            "split_unit": "parent_asin hashed globally; no product crosses roles",
            "global_duplicate_texts_removed": duplicates,
            "counts": counts,
            "source_sha256": hashes,
            "tokenizer_training_role": "train only",
            "sequence_length": length,
            "vocab_size": tokenizer.get_vocab_size(),
        },
    )
    return target


def make_features(root: Path, dimension: int = 6) -> Path:
    from transformers import AutoModel, AutoTokenizer

    name = "Qwen/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModel.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    texts = ["Customer reviews of " + name.replace("_", " ") + "." for name in DOMAINS]
    tokens = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        full = model(**tokens).last_hidden_state[:, -1].float().cpu()
    full = torch.nn.functional.normalize(full, dim=1)
    centered = full - full.mean(0)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    features = u[:, :dimension] * s[:dimension]
    features /= features.square().mean(0).sqrt().clamp_min(1e-8)
    target = root / "task_features.pt"
    torch.save(
        {
            "features": features,
            "full_embeddings": full,
            "descriptions": texts,
            "model": name,
            "revision": model.config._commit_hash,
            "feature_fit_uses": "category descriptions only, no review text or outcomes",
        },
        target,
    )
    return target


def make_token_features(root: Path, dimension=6, samples=256):
    """Gradient-free Hellinger/PCA task features, using training inputs only."""
    corpus = torch.load(root / "corpus.pt", weights_only=False)
    histograms = []
    for task, blocks in enumerate(corpus["train"]):
        indices = torch.randperm(
            len(blocks),
            generator=torch.Generator().manual_seed(
                31415 + (0 if corpus.get("parallel_examples") else task)
            ),
        )[:samples]
        tokens = blocks[indices].long().flatten()
        counts = torch.bincount(
            tokens[tokens >= 0], minlength=corpus["vocab_size"]
        ).float()
        histograms.append((counts / counts.sum()).sqrt())
    full = torch.stack(histograms)
    centered = full - full.mean(0)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    features = u[:, :dimension] * s[:dimension]
    features /= features.square().mean(0).sqrt().clamp_min(1e-8)
    target = root / "token_features.pt"
    torch.save(
        {
            "features": features,
            "full_histograms": full,
            "samples_per_task": samples,
            "feature_fit_uses": "training-input token frequencies only; no gradients or model outputs",
            "definition": "square-root token frequency, centered PCA, unit coordinate RMS",
            "input_sampling_seed": 31415,
        },
        target,
    )
    return target


def prepare_multilingual(root: Path, model_path: str, length=128):
    """WMT24++ targets as parallel-content multilingual LM, not MT scoring."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    languages = (
        "ar_SA",
        "bg_BG",
        "bn_IN",
        "cs_CZ",
        "de_DE",
        "el_GR",
        "es_MX",
        "fa_IR",
        "fr_FR",
        "hi_IN",
        "id_ID",
        "it_IT",
        "ja_JP",
        "ko_KR",
        "nl_NL",
        "pl_PL",
        "pt_BR",
        "ru_RU",
        "tr_TR",
        "zh_CN",
    )
    revision = "fd7405c06494bc66a57b25f55d217a72f96e60dc"
    root.mkdir(parents=True, exist_ok=True)

    def fetch(language):
        path = hf_hub_download(
            "google/wmt24pp",
            f"en-{language}.jsonl",
            repo_type="dataset",
            revision=revision,
        )
        return {
            r["segment_id"]: r
            for line in Path(path).read_text().split("\n")
            if line
            for r in [json.loads(line)]
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        tables = list(pool.map(fetch, languages))
    keys = set.intersection(*(set(table) for table in tables))
    keys = {k for k in keys if not any(t[k]["is_bad_source"] for t in tables)}
    seen_source = set()
    seen_target = [set() for _ in tables]
    selected = []
    for key in sorted(keys):
        source = " ".join(tables[0][key]["source"].split())
        targets = [" ".join(t[key]["target"].split()) for t in tables]
        if source in seen_source or any(
            text in seen for text, seen in zip(targets, seen_target)
        ):
            continue
        if any(t[key]["source"] != tables[0][key]["source"] for t in tables):
            raise ValueError("unaligned source")
        seen_source.add(source)
        for text, seen in zip(targets, seen_target):
            seen.add(text)
        selected.append(key)
    roles = {role: [] for role in ("train", "validation", "test")}
    for key in selected:
        document = tables[0][key]["document_id"]
        bucket = (
            int(hashlib.sha256(("wmt-tbe-v1:" + document).encode()).hexdigest()[:8], 16)
            % 10
        )
        role = "train" if bucket < 6 else "validation" if bucket < 8 else "test"
        roles[role].append(key)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = AutoConfig.from_pretrained(
            model_path, local_files_only=True
        ).bos_token_id
    tensors = {role: [] for role in roles}
    truncated = {}
    for language, table in zip(languages, tables):
        truncated[language] = {}
        for role, ids in roles.items():
            sequences = tokenizer(
                [table[k]["target"] for k in ids], add_special_tokens=False
            )["input_ids"]
            blocks = torch.full((len(ids), length + 1), -1, dtype=torch.int32)
            cut = 0
            for i, sequence in enumerate(sequences):
                tokens = [bos_id, *sequence, tokenizer.eos_token_id]
                cut += len(tokens) > length + 1
                used = tokens[: length + 1]
                blocks[i, : len(used)] = torch.tensor(used)
            tensors[role].append(blocks)
            truncated[language][role] = cut
    torch.save(
        {
            "domains": languages,
            "length": length,
            "vocab_size": len(tokenizer),
            "parallel_examples": True,
            **tensors,
        },
        root / "corpus.pt",
    )
    save_json(
        root / "manifest.json",
        {
            "source": "google/wmt24pp",
            "revision": revision,
            "license": "apache-2.0",
            "task": "unconditional target-language modeling, not translation quality",
            "languages": languages,
            "split_unit": "globally hashed source document; aligned content IDs across all languages",
            "segment_ids": roles,
            "document_ids": {
                role: [tables[0][k]["document_id"] for k in ids]
                for role, ids in roles.items()
            },
            "examples_per_role": {k: len(v) for k, v in roles.items()},
            "filter": "bad sources and duplicate source/within-language target strings removed jointly",
            "truncated_examples": truncated,
            "tokenizer": model_path,
            "max_tokens": length,
        },
    )
    path = make_token_features(root, samples=min(256, len(roles["train"])))
    torch.save(torch.load(path, weights_only=False), root / "task_features.pt")
    print({k: len(v) for k, v in roles.items()}, flush=True)


def prepare_native(source: Path, root: Path, model_path: str, length=128):
    """Same product/text split and documents, native pretrained-model tokenizer."""
    from transformers import AutoTokenizer

    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    seen = set()
    tensors = {role: [] for role in ("train", "validation", "test")}
    counts = {}
    for domain in DOMAINS:
        selected = {role: [] for role in tensors}
        for line in (source / "raw" / f"{domain}.jsonl").read_text().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            key = hashlib.sha256(
                " ".join(row["text"].lower().split()).encode()
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            selected[row["role"]].append(row["text"])
        counts[domain] = {}
        for role, texts in selected.items():
            ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
            flat = [t for row in ids for t in (*row, tokenizer.eos_token_id)]
            usable = len(flat) // (length + 1) * (length + 1)
            blocks = torch.tensor(flat[:usable], dtype=torch.int32).reshape(
                -1, length + 1
            )
            tensors[role].append(blocks)
            counts[domain][role] = {"documents": len(texts), "blocks": len(blocks)}
        print(f"NATIVE DATA {domain}", flush=True)
    torch.save(
        {"domains": DOMAINS, "length": length, "vocab_size": len(tokenizer), **tensors},
        root / "corpus.pt",
    )
    torch.save(
        torch.load(source / "task_features.pt", weights_only=False),
        root / "task_features.pt",
    )
    save_json(
        root / "manifest.json",
        {
            "source_manifest": str(source / "manifest.json"),
            "tokenizer": model_path,
            "counts": counts,
            "split_unit": "same product-disjoint and deduplicated documents",
        },
    )
    make_token_features(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=12000)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--token-features-only", action="store_true")
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--native-model")
    parser.add_argument("--multilingual", action="store_true")
    args = parser.parse_args()
    if args.multilingual:
        prepare_multilingual(args.root, args.native_model)
        return
    if args.native_source:
        prepare_native(args.native_source, args.root, args.native_model)
        return
    if args.token_features_only:
        print(make_token_features(args.root), flush=True)
        return
    if not args.features_only:
        print(prepare(args.root, args.documents), flush=True)
    print(make_features(args.root), flush=True)


if __name__ == "__main__":
    main()
