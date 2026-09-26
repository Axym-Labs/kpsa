"""Continuous language-conditioned feature scaling: a bounded premise test.

This is not pruning, training, or damage recovery. The intact model is the
reference. The profile coordinate is either parameter-group OPG or OPG of the
actual continuous multipliers. Importance does not supply a signed improvement
direction, so both signs are tested. This is not a natural-gradient step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .common import save_json, seed_everything
from .domain_applications import application_scores, load_checkpoint
from .domain_gates import FeatureGates, intervention_scales
from .domain_optimizer import ParameterPartition
from .domain_train import batch_loss, evaluate_details, logit_residual_norm


@torch.enable_grad()
def gate_profiles(model, domains, samples=64, parallel=False, include_fisher=False):
    """Squared derivatives w.r.t. actual feature multipliers, not weight traces."""
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    for p in parameters:
        p.requires_grad_(False)
    try:
        with FeatureGates(model) as gates:
            count = sum(g.numel() for g in gates.values)
            estimators = (
                ("raw", "normalized", "fisher")
                if include_fisher
                else ("raw", "normalized")
            )
            result = {e: torch.zeros(count, len(domains)) for e in estimators}
            # Diagnostic only: it is never used to fit the importance index or
            # choose intervention strength. Squaring alone discards this sign.
            result["mean_gradient"] = torch.zeros(count, len(domains))
            for task, data in enumerate(domains):
                if len(data) < samples:
                    raise ValueError("insufficient gate calibration examples")
                order = torch.randperm(
                    len(data),
                    generator=torch.Generator().manual_seed(
                        31415 + (0 if parallel else task)
                    ),
                )[:samples]
                for block in data[order]:
                    gates.zero_grad()
                    batch = block[None].to(parameters[0].device).long()
                    loss, logits = batch_loss(model, batch)
                    norm = logit_residual_norm(logits.detach(), batch[:, 1:]).clamp_min(
                        1e-12
                    )
                    loss.backward(retain_graph=include_fisher)
                    gradient = gates.gradients().flatten().cpu()
                    result["mean_gradient"][:, task] += gradient / samples
                    scores = gradient.square() / samples
                    result["raw"][:, task] += scores
                    result["normalized"][:, task] += scores / float(norm)
                    if include_fisher:
                        gates.zero_grad()
                        probabilities = logits.detach().float().softmax(-1)
                        labels = torch.multinomial(
                            probabilities.flatten(0, 1), 1
                        ).reshape(batch[:, 1:].shape)
                        labels[batch[:, 1:] < 0] = -1
                        torch.nn.functional.cross_entropy(
                            logits.float().flatten(0, 1),
                            labels.flatten(),
                            ignore_index=-1,
                        ).backward()
                        result["fisher"][:, task] += (
                            gates.gradients().flatten().square().cpu()
                            * int((labels >= 0).sum())
                            / samples
                        )
                print(f"GATE PROFILE {task + 1}/{len(domains)}", flush=True)
            return result
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
        model.train(training)


def translation_examples(manifest, tables, count):
    """Use the existing document-disjoint split and the same sources per language."""
    if set(manifest["document_ids"]["train"]) & set(
        manifest["document_ids"]["validation"]
    ):
        raise ValueError("calibration/evaluation document leakage")
    ids = manifest["segment_ids"]["validation"]
    if len(ids) < count:
        raise ValueError("insufficient validation examples")
    order = torch.randperm(
        len(ids), generator=torch.Generator().manual_seed(90210)
    ).tolist()
    by_document = {}
    for index in order:
        by_document.setdefault(
            manifest["document_ids"]["validation"][index], []
        ).append(ids[index])
    selected = []
    while len(selected) < count:
        for available in by_document.values():
            if available and len(selected) < count:
                selected.append(available.pop(0))
    rows = [[table[i] for i in selected] for table in tables]
    for column in zip(*rows):
        if len({r["source"] for r in column}) != 1 or any(
            r["is_bad_source"] for r in column
        ):
            raise ValueError("unaligned or invalid parallel source")
        if any(r["document_id"] in manifest["document_ids"]["train"] for r in column):
            raise ValueError("calibration/evaluation document leakage")
    return rows


def conditional_tokens(prefix, target, eos):
    if not prefix or not target:
        raise ValueError("nonempty prompt and target required")
    sequence = prefix + target + [eos]
    inputs = torch.tensor([sequence[:-1]])
    labels = torch.tensor([sequence[1:]])
    labels[:, : len(prefix) - 1] = -1
    return inputs, labels


def translation_prompt(tokenizer, source, language, tokenize=False):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": f"Translate the following English text into {language}. Output only the translation.\n\n{source}",
            }
        ],
        tokenize=tokenize,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def translation_generation_options():
    """Qwen3's published non-thinking recipe, fixed across interventions."""
    return {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": 1024,
        "use_cache": True,
    }


@torch.enable_grad()
def translation_profiles(
    model, tokenizer, part, indices, manifest, tables, names, samples=64
):
    ids = manifest["segment_ids"]["train"]
    if len(ids) < samples:
        raise ValueError("insufficient translation calibration data")
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(31415))[
        :samples
    ]
    chosen = [ids[i] for i in order]
    profiles = {
        e: torch.zeros(len(indices), len(tables)) for e in ("raw", "normalized")
    }
    for task, table in enumerate(tables):
        for index in chosen:
            row = table[index]
            prefix = translation_prompt(
                tokenizer, row["source"], names[task], tokenize=True
            )
            target = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
            inputs, labels = conditional_tokens(prefix, target, tokenizer.eos_token_id)
            inputs, labels = inputs.cuda(), labels.cuda()
            model.zero_grad(set_to_none=True)
            logits = model(input_ids=inputs, use_cache=False).logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), ignore_index=-1
            )
            norm = logit_residual_norm(logits.detach(), labels).clamp_min(1e-12)
            loss.backward()
            score = part.gradient_scores().cpu()[indices] / samples
            profiles["raw"][:, task] += score
            profiles["normalized"][:, task] += score / float(norm)
        print(f"TRANSLATION PROFILE {task + 1}/{len(tables)}", flush=True)
    model.zero_grad(set_to_none=True)
    return profiles


@torch.no_grad()
def translation(checkpoint, data, profile_cache, output, count=32):
    """Actual translation generation, not an LM-loss proxy or a steering claim."""
    from huggingface_hub import hf_hub_download
    from sacrebleu.metrics import CHRF
    from transformers import AutoTokenizer

    seed_everything(11)
    torch.set_num_threads(4)
    manifest = json.loads((data / "manifest.json").read_text())
    names = (
        "Arabic",
        "Bulgarian",
        "Bengali",
        "Czech",
        "German",
        "Greek",
        "Spanish",
        "Persian",
        "French",
        "Hindi",
        "Indonesian",
        "Italian",
        "Japanese",
        "Korean",
        "Dutch",
        "Polish",
        "Portuguese",
        "Russian",
        "Turkish",
        "Chinese",
    )
    tables = []
    for language in manifest["languages"]:
        path = hf_hub_download(
            "google/wmt24pp",
            f"en-{language}.jsonl",
            repo_type="dataset",
            revision=manifest["revision"],
            local_files_only=True,
        )
        tables.append(
            {
                r["segment_id"]: r
                for line in Path(path).read_text().splitlines()
                for r in [json.loads(line)]
            }
        )
    examples = translation_examples(manifest, tables, count)
    payload = torch.load(profile_cache, weights_only=False)
    if (
        payload["signature"]["checkpoint"] != str(checkpoint.resolve())
        or payload["signature"]["corpus"] != str((data / "corpus.pt").resolve())
        or payload["signature"]["checkpoint_mtime_ns"] != checkpoint.stat().st_mtime_ns
    ):
        raise ValueError("mismatched profile provenance")
    features = torch.load(data / "token_features.pt", weights_only=False)["features"]
    model, _ = load_checkpoint(checkpoint)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    part = ParameterPartition(model, "swiglu")
    indices = torch.cat(
        [
            torch.arange(s.offset, s.offset + s.count)
            for s in part.slices
            if ".mlp.down_proj.weight" in s.name
        ]
    )
    calibrated_cache = output.with_suffix(".pt")
    calibration_signature = {
        **payload["signature"],
        "samples": 64,
        "loss": "source-conditioned, target-only translation cross entropy",
        "coordinate": "mean parameter OPG per coupled SwiGLU feature",
    }
    if calibrated_cache.exists():
        saved = torch.load(calibrated_cache, weights_only=False)
        if saved["signature"] != calibration_signature:
            raise ValueError("incompatible translation calibration cache")
        profiles = saved["profiles"]
    else:
        profiles = translation_profiles(
            model, tokenizer, part, indices, manifest, tables, names
        )
        calibrated_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"signature": calibration_signature, "profiles": profiles}, calibrated_cache
        )
    configs = [
        ("none", "identity", 0.0),
        ("none", "uniform", -0.1),
        ("none", "uniform", 0.1),
    ] + [
        (e, m, 0.1)
        for e in ("raw", "normalized")
        for m in ("full", "tbe", "jl", "pooled")
    ]
    metric = CHRF(word_order=2)
    result = {
        "protocol": 2,
        "data_role": "validation",
        "model": str(checkpoint.resolve()),
        "source_profile_signature": calibration_signature,
        "examples_per_language": count,
        "sampling": "document-balanced validation sources; shared across languages",
        "generation": {
            **translation_generation_options(),
            "seed": "11000 + task_index * 1000 + batch_start; reset for each method",
            "recipe_source": "https://huggingface.co/Qwen/Qwen3-1.7B#best-practices",
            "batch_size": 8,
            "batch_order": "ascending source character length, restored to original order",
            "enable_thinking": False,
            "prompt": "Translate the following English text into {language}. Output only the translation.\\n\\n{source}",
        },
        "configs": configs,
        "records": [],
        "complete": False,
        "claim_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with FeatureGates(model) as gates:
        shape = (len(gates.values), gates.values[0].numel())
        for estimator, method, strength in configs:
            if method not in {"identity", "uniform"}:
                atlas = profiles[estimator]
                scores = application_scores(
                    atlas,
                    features,
                    "mean" if method == "pooled" else method,
                    811,
                    score_link="linear",
                )
                if method != "pooled":
                    scores = scores / atlas.mean(1, keepdim=True).clamp_min(1e-30)
            for task, rows in enumerate(examples):
                scales = intervention_scales(
                    None if method in {"identity", "uniform"} else scores[:, task],
                    shape,
                    method,
                    strength,
                )
                gates.set(scales)
                prompts = [
                    translation_prompt(tokenizer, r["source"], names[task])
                    for r in rows
                ]
                predictions, capped = [None] * len(rows), [None] * len(rows)
                order = sorted(range(len(rows)), key=lambda i: len(rows[i]["source"]))
                for start in range(0, len(rows), 8):
                    selected = order[start : start + 8]
                    encoded = tokenizer(
                        [prompts[i] for i in selected],
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                    seed_everything(11000 + task * 1000 + start)
                    generated = model.generate(
                        **encoded,
                        **translation_generation_options(),
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    suffix = generated[:, encoded.input_ids.shape[1] :]
                    texts = tokenizer.batch_decode(suffix, skip_special_tokens=True)
                    stop_ids = model.generation_config.eos_token_id
                    stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids
                    for index, text, tokens in zip(selected, texts, suffix.cpu()):
                        predictions[index] = text
                        capped[index] = not any(int(x) in stop_ids for x in tokens)
                references = [r["target"] for r in rows]
                score = metric.corpus_score(predictions, [references]).score
                result["records"].append(
                    {
                        "estimator": estimator,
                        "method": method,
                        "strength": strength,
                        "language": manifest["languages"][task],
                        "chrfpp": score,
                        "capped": capped,
                        "segment_ids": [r["segment_id"] for r in rows],
                        "document_ids": [r["document_id"] for r in rows],
                        "predictions": predictions,
                        "references": references,
                    }
                )
                result["wall_seconds"] = time.perf_counter() - started
                result["metric_signature"] = str(metric.get_signature())
                save_json(output, result)
                print(
                    f"TRANSLATION {estimator}/{method}/{manifest['languages'][task]} chrF++={score:.3f} capped={sum(capped)}",
                    flush=True,
                )
        gates.set(torch.ones(shape))
    result["complete"] = True
    save_json(output, result)
    return result


def run(
    checkpoint,
    data,
    profile_cache,
    output,
    eval_blocks=64,
    strengths=(-0.3, -0.1, 0.1, 0.3),
    estimators=("raw", "normalized", "fisher"),
    coordinate="parameter",
):
    if coordinate not in {"parameter", "gate"}:
        raise ValueError(coordinate)
    seed_everything(11)
    torch.set_num_threads(4)
    payload = torch.load(profile_cache, weights_only=False)
    signature = payload["signature"]
    if (
        signature["checkpoint"] != str(checkpoint.resolve())
        or signature["corpus"] != str((data / "corpus.pt").resolve())
        or signature["checkpoint_mtime_ns"] != checkpoint.stat().st_mtime_ns
    ):
        raise ValueError("profile cache is not from this checkpoint and corpus")
    corpus = torch.load(data / "corpus.pt", weights_only=False)
    features = torch.load(data / "token_features.pt", weights_only=False)["features"]
    model, _ = load_checkpoint(checkpoint)
    model.eval()
    part = ParameterPartition(model, "swiglu")
    indices = torch.cat(
        [
            torch.arange(s.offset, s.offset + s.count)
            for s in part.slices
            if ".mlp.down_proj.weight" in s.name
        ]
    )
    if coordinate == "parameter":
        source = {e: payload["source"]["swiglu"][e][indices] for e in estimators}
    else:
        signature = {
            **signature,
            "coordinate": "continuous feature multiplier",
            "diagnostics": ["calibration_mean_gradient"],
            "include_fisher": "fisher" in estimators,
        }
        cache = output.with_suffix(".pt")
        if cache.exists():
            saved = torch.load(cache, weights_only=False)
            if saved["signature"] != signature:
                raise ValueError("gate-profile cache provenance mismatch")
            source = saved["source"]
        else:
            source = gate_profiles(
                model,
                corpus["train"],
                samples=signature["samples"],
                parallel=corpus.get("parallel_examples", False),
                include_fisher="fisher" in estimators,
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"signature": signature, "source": source}, cache)
    baseline, baseline_details = evaluate_details(
        model, corpus["validation"], eval_blocks
    )
    records = []
    started = time.perf_counter()
    methods = ("full", "tbe", "jl", "pooled", "permuted", "wrong_task", "random")
    result = {
        "protocol": 1,
        "model": str(checkpoint.resolve()),
        "data": str(data.resolve()),
        "source_profile_signature": signature,
        "importance_coordinate": coordinate,
        "evaluation_blocks": eval_blocks,
        "data_role": "validation",
        "tasks": corpus["domains"],
        "strengths": list(strengths),
        "methods": methods,
        "estimators": estimators,
        "baseline_nll": baseline.tolist(),
        "baseline_details": baseline_details,
        "intervention": "positive continuous multiplier on SwiGLU features; no masking",
        "direction": "within-layer ranks of queried importance / task-mean importance",
        "scope": "seen-task compact queries, unchanged pretrained model, no adaptation",
        "claim_ready": False,
        "records": records,
        "complete": False,
    }
    save_json(output, result)
    with FeatureGates(model) as gates:
        shape = (len(gates.values), gates.values[0].numel())
        identity, _ = evaluate_details(model, corpus["validation"], eval_blocks)
        result["identity_max_abs_error"] = float((identity - baseline).abs().max())
        if result["identity_max_abs_error"] > 1e-6:
            raise RuntimeError("identity scaling changed the model")
        for estimator in (*estimators, "none"):
            if estimator != "none":
                atlas = source[estimator]
                pooled = atlas.mean(1, keepdim=True).clamp_min(1e-30)
            for method in methods if estimator != "none" else ("uniform",):
                if method != "uniform":
                    scores = application_scores(
                        atlas,
                        features,
                        "mean" if method == "pooled" else method,
                        811,
                        score_link="linear",
                    )
                    direction = scores if method == "pooled" else scores / pooled
                for strength in strengths:
                    losses, details, diagnostics = [], [], []
                    first_order = []
                    for task in range(len(features)):
                        scales = intervention_scales(
                            None if method == "uniform" else direction[:, task],
                            shape,
                            method,
                            strength,
                        )
                        if not bool((scales > 0).all()):
                            raise RuntimeError(
                                "continuous scaling must never remove a feature"
                            )
                        gates.set(scales)
                        if coordinate == "gate":
                            first_order.append(
                                float(
                                    source["mean_gradient"][:, task]
                                    @ scales.log().flatten()
                                )
                            )
                        value, detail = evaluate_details(
                            model, [corpus["validation"][task]], eval_blocks
                        )
                        losses.append(float(value[0]))
                        details.append(detail[0])
                        diagnostics.append(
                            {
                                "minimum": float(scales.min()),
                                "maximum": float(scales.max()),
                                "rms_log_scale": float(
                                    scales.log().square().mean().sqrt()
                                ),
                                "zero_count": int((scales == 0).sum()),
                            }
                        )
                    row = {
                        "estimator": estimator,
                        "method": method,
                        "strength": strength,
                        "nll": losses,
                        "delta_nll": (torch.tensor(losses) - baseline).tolist(),
                        "details": details,
                        "multiplier_diagnostics": diagnostics,
                        "calibration_first_order_delta_nll": first_order or None,
                    }
                    records.append(row)
                    result["wall_seconds"] = time.perf_counter() - started
                    save_json(output, result)
                    print(
                        f"SCALE {estimator}/{method}/{strength:+g}: "
                        f"delta NLL={float((torch.tensor(losses) - baseline).mean()):+.6f}",
                        flush=True,
                    )
        gates.set(torch.ones(shape))
        identity, _ = evaluate_details(model, corpus["validation"], eval_blocks)
        result["restoration_max_abs_error"] = float((identity - baseline).abs().max())
    result.update(
        complete=True,
        full_mlp_atlas_floats=int(indices.numel() * len(features)),
        tbe_mlp_index_floats=int(
            indices.numel() * (features.shape[1] + 1)
            + features.numel()
            + features.shape[1]
        ),
        wall_seconds=time.perf_counter() - started,
    )
    save_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "data", "profile-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--eval-blocks", type=int, default=64)
    parser.add_argument("--translation", action="store_true")
    parser.add_argument(
        "--coordinate", choices=("parameter", "gate"), default="parameter"
    )
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[-0.3, -0.1, 0.1, 0.3]
    )
    parser.add_argument(
        "--estimators", nargs="+", default=["raw", "normalized", "fisher"]
    )
    args = parser.parse_args()
    if args.translation:
        if args.coordinate != "parameter":
            parser.error("translation currently uses parameter-coordinate calibration")
        translation(
            args.checkpoint,
            args.data,
            args.profile_cache,
            args.output,
            args.eval_blocks,
        )
    else:
        run(
            args.checkpoint,
            args.data,
            args.profile_cache,
            args.output,
            args.eval_blocks,
            tuple(args.strengths),
            tuple(args.estimators),
            args.coordinate,
        )
