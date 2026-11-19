"""Write the camera-ready figure/table CSVs into kpsa-paper/data/empirical.

Single source for every number changed in the revision:
  bandwidth .1 everywhere (Fig 4 vision, forecasting in Fig 2, ranking figures),
  the v3 cost benchmark (isolated, matched precision, stacked multi-atlas query),
  category log-probability functionals for the multi-functional accuracy,
  alternative estimators without the kNN average.

Files written (existing files are overwritten only for the rows they own):
  paper_application1_primary.csv   vision rows from the Fig 4 rerun at .1 (protein rows kept)
  paper_application2_primary.csv   ETTh1/Weather rows at .1 with moving-block intervals (others kept)
  app1_budget_precision_recall.csv, app1_budget_recall_multiples.csv   reruns at .1
  retrieval_comparison_vision.csv  accuracy by method and N, with v3 cost
  multi_functional_cost.csv        throughput/memory vs number of functionals (v3, stacked KPSA)
  multi_functional_accuracy.csv    category functionals, per functional and pooled
  retrieval_baselines_all.csv      all settings x estimators at .1
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..paper_figures import absolute_series, normalized_series

RESULTS = Path.home() / "main/data/kpsa-rebuttal/results"
CACHE = Path.home() / "main/data/kpsa-rebuttal"
PAPER = Path.home() / "main/workspace/kpsa-paper/data/empirical"
FR = ("0.0002", "0.0005", "0.001")


def read_csv(name):
    with (PAPER / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def write(name, rows):
    with (PAPER / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", name, len(rows))


# ---------------------------------------------------------------- Fig 4 (vision rows)
def application1_vision():
    payload = json.loads((RESULTS / "fig4_bandwidth_rerun.json").read_text())
    fractions = tuple(payload["protocol"]["fractions"])
    common = dict(budget_field="fraction", budgets=fractions,
                  budget_percent={f: 100 * f for f in fractions}, query_field="query_column",
                  valid_field="target_correct",
                  method_map={"prototype_800_scale_0.1": "Semantic KPSA", "class_onehot": "Categorical KPSA",
                              "nearest_encoder": "KPSA-1NN", "scalar_mass": "Constant sensitivity",
                              "direct_coordinate_taylor": "Activation attribution"})
    absolute = absolute_series(payload, metric="degradation", metric_label="target_margin_decrease", **common)
    gap = {(r["budget"], r["method"]): r for r in normalized_series(
        payload, metric="degradation", reference="Activation attribution", **common)}
    old = read_csv("paper_application1_primary.csv")
    rows = [{"modality": "Vision", **r, "gap_recovery_pct": gap[(r["budget"], r["method"])]["mean"],
             "gap_ci95_low_pct": gap[(r["budget"], r["method"])]["ci95_low"],
             "gap_ci95_high_pct": gap[(r["budget"], r["method"])]["ci95_high"]} for r in absolute]
    write("paper_application1_primary.csv", rows + [r for r in old if r["modality"] != "Vision"])


# ---------------------------------------------------------------- Fig 2 (forecasting rows)
def _block_ratio(num, low, high, block, subtract_low=True, draws=5000, seed=0):
    num, low, high = (np.asarray(x, dtype=np.float64) for x in (num, low, high))
    n = len(num)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(draws, -1)[:, :n]
    den = high[idx].mean(1) - (low[idx].mean(1) if subtract_low else 0)
    ratio = (num[idx].mean(1) - (low[idx].mean(1) if subtract_low else 0)) / den
    point = (num.mean() - (low.mean() if subtract_low else 0)) / (high.mean() - (low.mean() if subtract_low else 0))
    lo, hi = np.quantile(ratio[np.isfinite(ratio)], (0.025, 0.975))
    return 100 * point, 100 * lo, 100 * hi


def application2_forecasting():
    from .forecast_baselines import predict_all

    labels = {"kpsa@0.1": "Semantic KPSA", "temporal_categorical": "Categorical KPSA", "knn@1": "Semantic nearest",
              "scalar": "Constant sensitivity", "direct_gradient": "Direct gradient"}
    new = []
    for ds in ("ETTh1", "Weather"):
        res = json.loads((RESULTS / f"forecast_{ds}.json").read_text())
        block = int(res["block_length_queries"])
        rec = res["records"]
        cache = torch.load(CACHE / f"forecast_{ds}_cache.pt", weights_only=False)
        upl = cache["units_per_layer"]
        scope = torch.arange(11 * upl, 12 * upl)
        preds = predict_all(cache, "query")
        truth = cache["query_profiles"]
        n = len(truth)
        for f, count in zip(FR, res["feature_counts"]):
            budget = 100 * float(f)
            for m, label in labels.items():
                if m in ("scalar", "direct_gradient"):
                    mean, lo, hi = (0.0, 0.0, 0.0) if m == "scalar" else (100.0, 100.0, 100.0)
                else:
                    mean, lo, hi = _block_ratio(rec[m][f], rec["scalar"][f], rec["direct_gradient"][f], block)
                new.append({"modality": ds, "measurement": "causal", "budget": budget, "method": label,
                            "mean": mean, "ci95_low": lo, "ci95_high": hi, "queries": n})
            cover = {}
            for m in labels:
                p, t = preds[m][:, scope], truth[:, scope]
                top = torch.topk(p, count, dim=1).indices
                cover[m] = (t.gather(1, top).sum(1) / t.sum(1)).numpy()
            for m, label in labels.items():
                mean, lo, hi = _block_ratio(cover[m], np.zeros(n), cover["direct_gradient"], block, subtract_low=False)
                new.append({"modality": ds, "measurement": "energy", "budget": budget, "method": label,
                            "mean": mean, "ci95_low": lo, "ci95_high": hi, "queries": n})
    old = read_csv("paper_application2_primary.csv")
    order = ["Vision", "ETTh1", "Weather", "Protein"]
    keep = [r for r in old if r["modality"] not in ("ETTh1", "Weather")]
    rows = sorted(keep + new, key=lambda r: (order.index(r["modality"]),
                                              ("causal", "energy").index(r["measurement"]), float(r["budget"])))
    write("paper_application2_primary.csv", rows)


# ---------------------------------------------------------------- ranking figures
def ranking():
    for name, src in (("app1_budget_precision_recall.csv", "app1_budget_pr_fixed_s0.1.json"),
                      ("app1_budget_recall_multiples.csv", "app1_budget_recall_multiples_s0.1.json")):
        payload = json.loads((RESULTS / src).read_text())
        fixed = payload["protocol"].get("fixed_oracle_fraction")
        rows = []
        for method, summaries in payload["summaries"].items():
            label = "KPSA-1NN" if method == "Semantic nearest" else method
            for s in summaries:
                for metric in ("precision", "recall", "f1"):
                    v = s[metric]
                    rows.append({"method": label, "oracle_fraction": s.get("oracle_fraction", fixed),
                                 "predicted_fraction": s["predicted_fraction"],
                                 "retrieval_multiplier": s.get("retrieval_multiplier", ""), "metric": metric,
                                 "mean": v["mean"], "ci95_low": v["ci95_low"], "ci95_high": v["ci95_high"],
                                 "queries": v["n"]})
        write(name, rows)


# ---------------------------------------------------------------- cost benchmark (v3)
def bench(path):
    rows = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get("batch", 32) != 32:
            continue
        rows[(r["method"], r["precision"], r.get("n"), r.get("k"))].append(r)
    return {key: {"ips": st.median(r["images_per_second_median"] for r in rs),
                  "peak": max(r["peak_allocated_bytes"] for r in rs),
                  "reserved": max(r.get("peak_reserved_bytes", 0) for r in rs),
                  "stored": rs[0].get("stored_bytes", 0)}
            for key, rs in rows.items() if "images_per_second_median" in rs[0]}


def comparison_vision():
    r = json.loads((RESULTS / "vision_retrieval_eval.json").read_text())
    b = bench(RESULTS / "cost_bench_v3.jsonl")
    rows = []
    for n, block in r["by_n"].items():
        n = int(n)
        ch = block["dev_chosen"]
        methods = {
            "Semantic KPSA": ("kpsa_proto@0.1", ("kpsa_proto", 6400, 1)),
            "Semantic KPSA (exact kernel)": (ch["kpsa_exact"], ("kpsa_exact", n, 1)),
            "Categorical KPSA": ("class_average", ("class_average", 6400, 1)),
            "KPSA-1NN": ("knn@1", ("knn", n, 1)),
            "Linear (ridge)": (ch["ridge_z"], ("ridge_z", 6400, 1)),
            "Constant sensitivity": ("constant", None),
            "Activation attribution": ("activation_attribution", ("attribution", 6400, 1)),
        }
        storage = block["storage_bytes"]
        for label, (m, key) in methods.items():
            fam = m.split("@")[0]
            for prec in ("fp32", "bf16"):
                cost = b.get((key[0], prec, key[1], key[2])) if key else None
                for f in FR:
                    s = block["summary"]["test"][m][f]
                    rows.append({
                        "method": label, "config": m, "source_examples": n, "budget": f, "precision": prec,
                        "gap_recovered": s["gap_recovered"], "ci95_low": s["gap_ci95"][0],
                        "ci95_high": s["gap_ci95"][1], "queries": s["n"],
                        "persistent_bytes": storage.get(fam, 0) if fam != "activation_attribution" else 0,
                        "images_per_second": cost["ips"] if cost else "",
                        "peak_allocated_bytes": cost["peak"] if cost else "",
                        "peak_reserved_bytes": cost["reserved"] if cost else "",
                    })
    write("retrieval_comparison_vision.csv", rows)


def multi():
    b = bench(RESULTS / "cost_multi_v3.jsonl")
    labels = {"kpsa_multi_stacked": "Semantic KPSA", "attribution_multi": "Activation attribution (sequential)",
              "attribution_multi_vjp": "Activation attribution (batched VJP)",
              "direct_multi": "Direct gradient profile (sequential)", "knn_multi": "KPSA-1NN"}
    rows = [{"method": labels[m], "functionals": k, "precision": p, "images_per_second": v["ips"],
             "peak_allocated_bytes": v["peak"], "stored_bytes": v["stored"]}
            for (m, p, n, k), v in sorted(b.items()) if m in labels]
    # End-to-end KPSA throughput, timed in one run per functional count (e2e_multi_bench):
    # building M atlases from 800 sources, then answering 50,000 queries.
    for line in (RESULTS / "e2e_multi.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["side"] == "kpsa":
            rows.append({"method": "Semantic KPSA (end-to-end, 800 sources, 50k queries)", "functionals": r["functionals"],
                         "precision": "fp32", "images_per_second": r["images_per_second"],
                         "peak_allocated_bytes": "", "stored_bytes": ""})
    write("multi_functional_cost.csv", rows)
    rows = []
    for n_src, name in ((6400, "vision_multi_functional_category.json"),
                        (800, "vision_multi_functional_category_n800.json")):
        acc = json.loads((RESULTS / name).read_text())
        for c, by in acc["summary_by_functional"].items():
            for m, cs in by.items():
                for cnt, v in cs.items():
                    rows.append({"sources": n_src, "functional": c, "method": m, "selected_groups": cnt,
                                 "gap_recovered": v[0], "ci95_low": v[1], "ci95_high": v[2], "queries": acc["queries"]})
        for m, cs in acc["mean_over_functionals"].items():
            for cnt, v in cs.items():
                lo, hi = acc["mean_over_functionals_ci95"][m][cnt]
                rows.append({"sources": n_src, "functional": "mean", "method": m, "selected_groups": cnt,
                             "gap_recovered": v, "ci95_low": lo, "ci95_high": hi, "queries": acc["queries"]})
    write("multi_functional_accuracy.csv", rows)


def baselines_all():
    rows = []

    def add(setting, application, label, s, n):
        for f, v in s.items():
            rows.append({"setting": setting, "application": application, "method": label, "budget": f,
                         "gap_recovered": v[0], "ci95_low": v[1], "ci95_high": v[2], "queries": n})

    v = json.loads((RESULTS / "vision_param_baselines.json").read_text())
    ch = v["dev_chosen"]
    for label, m in (("KPSA", "kpsa_proto@0.1"), ("KPSA, exact kernel", ch["kpsa_exact"]),
                     ("Categorical KPSA", "class_average"), ("KPSA-1NN", "knn@1"), ("Linear (ridge)", ch["ridge_z"])):
        add("ImageNet-1k / ViT-B/16", "parameter perturbation", label,
            {fb: v["summary"][m][g] for fb, g in zip(FR, ("8", "19", "37"))}, v["n"])
    for ds in ("ETTh1", "Weather"):
        r = json.loads((RESULTS / f"forecast_{ds}.json").read_text())
        ch = r.get("dev_chosen", r.get("dev_chosen_from_ETTh1"))
        for label, m in (("KPSA", "kpsa@0.1"), ("Categorical KPSA", "temporal_categorical"), ("KPSA-1NN", "knn@1"),
                         ("Linear (ridge)", ch["ridge"])):
            add(f"{ds} / Chronos-Bolt", "parameter perturbation", label,
                {f: r["summary"][m][f]["gap_block"] for f in FR}, r["queries"])
    r = json.loads((RESULTS / "protein_paper_split.json").read_text())
    for app, appname in (("app2", "parameter perturbation"), ("app1", "neuron localization")):
        ch = r[app]["dev_chosen"]
        for label, m in (("KPSA", "kpsa@0.1"), ("Categorical KPSA", "categorical"), ("KPSA-1NN", "knn@1"),
                         ("Linear (ridge)", ch["ridge"])):
            add("Swiss-Prot / ESM2", appname, label, r[app]["summary"][m], r["n"][app])
    r = json.loads((RESULTS / "vision_retrieval_eval.json").read_text())["by_n"]["6400"]
    ch = r["dev_chosen"]
    for label, m in (("KPSA", "kpsa_proto@0.1"), ("KPSA, exact kernel", ch["kpsa_exact"]),
                     ("Categorical KPSA", "class_average"), ("KPSA-1NN", "knn@1"), ("Linear (ridge)", ch["ridge_z"])):
        s = r["summary"]["test"][m]
        add("ImageNet-200 / ViT-B/16", "neuron localization", label,
            {f: (s[f]["gap_recovered"], *s[f]["gap_ci95"]) for f in FR}, s["0.001"]["n"])
    write("retrieval_baselines_all.csv", rows)


def main():
    application1_vision()
    application2_forecasting()
    ranking()
    comparison_vision()
    multi()
    baselines_all()


if __name__ == "__main__":
    main()
