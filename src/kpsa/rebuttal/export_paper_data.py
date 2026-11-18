"""Export rebuttal results as the paper's frozen figure/table CSVs.

Writes into kpsa-paper/data/empirical/:
  retrieval_comparison_vision.csv  accuracy (gap at each budget) by method and N,
                                   with persistent storage, peak memory, throughput
  multi_functional_cost.csv        throughput and peak memory vs number of functionals,
                                   with mean gap over functionals
  retrieval_baselines_all.csv      every setting (vision/forecasting/protein, both
                                   applications) x method x budget, gap with 95% CI
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path.home() / "main/data/kpsa-rebuttal/results"
PAPER = Path.home() / "main/workspace/kpsa-paper/data/empirical"
FR = ("0.0002", "0.0005", "0.001")


def bench(path):
    rows = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        rows[(r["method"], r["precision"], r.get("n"), r.get("k"))].append(r)
    out = {}
    for key, rs in rows.items():
        out[key] = {
            "ips": st.median(r["images_per_second_median"] for r in rs) if "images_per_second_median" in rs[0] else None,
            "peak": max(r.get("peak_allocated_bytes", 0) for r in rs),
            "stored": rs[0].get("stored_bytes", 0),
        }
    return out


def write(name, rows):
    with (PAPER / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", PAPER / name, len(rows))


def vision(kpsa_config):
    r = json.loads((RESULTS / "vision_retrieval_eval.json").read_text())
    b = {}
    for f in ("cost_bench_v2.jsonl",):
        if (RESULTS / f).exists():
            b.update(bench(RESULTS / f))
    rows = []
    for n, block in r["by_n"].items():
        n = int(n)
        ch = block["dev_chosen"]
        methods = {
            "Semantic KPSA": (ch["kpsa_proto"] if kpsa_config == "dev" else "kpsa_proto@0.025", ("kpsa_proto", 6400, 1)),
            "Semantic KPSA (exact kernel)": (ch["kpsa_exact"], ("kpsa_exact", n, 1)),
            "Categorical KPSA": ("class_average", ("class_average", 6400, 1)),
            "KPSA-1NN": ("knn@1", ("knn", n, 1)),
            "kNN": (ch["knn"], ("knn", n, 16)),  # cost measured at k = 16 (upper bound for the dev-chosen k)
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
                        "gap_recovered": s["gap_recovered"], "ci95_low": s["gap_ci95"][0], "ci95_high": s["gap_ci95"][1],
                        "energy_captured": s["energy_captured"], "queries": s["n"],
                        "persistent_bytes": storage.get(fam, 0) if fam not in ("activation_attribution",) else 0,
                        "images_per_second": cost["ips"] if cost else "",
                        "peak_allocated_bytes": cost["peak"] if cost else "",
                    })
    write("retrieval_comparison_vision.csv", rows)


def multi():
    b = bench(RESULTS / "cost_multi_v2.jsonl")
    acc = json.loads((RESULTS / "vision_multi_functional_app2.json").read_text())
    labels = {"kpsa_multi": "Semantic KPSA", "attribution_multi": "Activation attribution (sequential)",
              "attribution_multi_vjp": "Activation attribution (batched VJP)",
              "direct_multi": "Direct gradient profile (sequential)"}
    rows = []
    for (method, prec, n, k), v in sorted(b.items()):
        gaps = ""
        if method == "kpsa_multi":
            gaps = sum(acc["summary_by_functional"][str(j)]["kpsa@0.1"]["37"][0] for j in range(1, k + 1)) / k
        rows.append({"method": labels[method], "functionals": k, "precision": prec, "images_per_second": v["ips"],
                     "peak_allocated_bytes": v["peak"], "stored_bytes": v["stored"],
                     "mean_gap_recovered_0.1pct_param_perturbation": gaps})
    write("multi_functional_cost.csv", rows)
    rows = []
    for j, block in acc["summary_by_functional"].items():
        for m, by in block.items():
            for c, v in by.items():
                rows.append({"functional_rank": j, "method": m, "selected_groups": c, "gap_recovered": v[0],
                             "ci95_low": v[1], "ci95_high": v[2], "queries": acc["queries"]})
    write("multi_functional_accuracy.csv", rows)


def everything():
    rows = []

    def add(setting, application, reference, label, s, n):
        for f, v in s.items():
            rows.append({"setting": setting, "application": application, "reference": reference, "method": label,
                         "budget": f, "gap_recovered": v[0], "ci95_low": v[1], "ci95_high": v[2], "queries": n})

    v = json.loads((RESULTS / "vision_param_baselines.json").read_text())
    ch = v["dev_chosen"]
    for label, m in (("Semantic KPSA (submission setting)", "kpsa_proto@0.1"), ("Semantic KPSA (exact kernel)", ch["kpsa_exact"]),
                     ("Categorical KPSA", "class_average"), ("KPSA-1NN", "knn@1"), ("kNN", ch["knn"]),
                     ("Linear (ridge)", ch["ridge_z"])):
        add("ImageNet-1k / ViT-B/16", "parameter perturbation", "direct gradient", label,
            {fb: v["summary"][m][g] for fb, g in zip(FR, ("8", "19", "37"))}, v["n"])
    for ds in ("ETTh1", "Weather"):
        r = json.loads((RESULTS / f"forecast_{ds}.json").read_text())
        ch = r.get("dev_chosen", r.get("dev_chosen_from_ETTh1"))
        for label, m in (("Semantic KPSA (submission setting)", "kpsa@0.25"), ("Semantic KPSA (dev-selected scale)", ch["kpsa"]),
                         ("Categorical KPSA", "temporal_categorical"), ("KPSA-1NN", "knn@1"), ("kNN", ch["knn"]),
                         ("Linear (ridge)", ch["ridge"])):
            add(f"{ds} / Chronos-Bolt", "parameter perturbation", "direct gradient", label,
                {f: r["summary"][m][f]["gap_block"] for f in FR}, r["queries"])
    p = RESULTS / "protein_paper_split.json"
    if p.exists():
        r = json.loads(p.read_text())
        for app, appname, ref in (("app2", "parameter perturbation", "direct gradient"),
                                  ("app1", "neuron localization", "activation attribution")):
            ch = r[app]["dev_chosen"]
            for label, m in (("Semantic KPSA (submission setting)", "kpsa@0.1"), ("Semantic KPSA (dev-selected scale)", ch["kpsa"]),
                             ("Categorical KPSA", "categorical"), ("KPSA-1NN", "knn@1"), ("kNN", ch["knn"]),
                             ("Linear (ridge)", ch["ridge"])):
                add("Swiss-Prot / ESM2", appname, ref, label, r[app]["summary"][m], r["n"][app])
    r = json.loads((RESULTS / "vision_retrieval_eval.json").read_text())["by_n"]["6400"]
    ch = r["dev_chosen"]
    for label, m in (("Semantic KPSA (submission setting)", "kpsa_proto@0.025"), ("Semantic KPSA (dev-selected scale)", ch["kpsa_proto"]),
                     ("Semantic KPSA (exact kernel)", ch["kpsa_exact"]), ("Categorical KPSA", "class_average"),
                     ("KPSA-1NN", "knn@1"), ("kNN", ch["knn"]), ("Linear (ridge)", ch["ridge_z"])):
        s = r["summary"]["test"][m]
        add("ImageNet-200 / ViT-B/16 (N=6,400)", "neuron localization", "activation attribution", label,
            {f: (s[f]["gap_recovered"], *s[f]["gap_ci95"]) for f in FR}, s["0.001"]["n"])
    write("retrieval_baselines_all.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpsa-config", choices=("dev", "paper"), default="dev")
    a = parser.parse_args()
    vision(a.kpsa_config)
    if (RESULTS / "cost_multi_v2.jsonl").exists() and (RESULTS / "vision_multi_functional_app2.json").exists():
        multi()
    everything()


if __name__ == "__main__":
    main()
