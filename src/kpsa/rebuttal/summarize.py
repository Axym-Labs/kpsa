"""Markdown tables from the rebuttal result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(v):
    return f"{100 * v:.1f}"


def vision_table(path):
    r = json.loads(Path(path).read_text())
    lines = []
    fr = ["0.0002", "0.0005", "0.001"]
    for n, block in r["by_n"].items():
        ch = block["dev_chosen"]
        test = block["summary"]["test"]
        lines.append(f"\n### N = {n} (test n = {test['constant']['0.001']['n']})\n")
        lines.append("| Method | config | 0.02% | 0.05% | 0.1% | energy@0.1% | storage MiB |")
        lines.append("|---|---|---|---|---|---|---|")
        rows = [("KPSA prototype (paper)", "kpsa_proto@0.025"), ("KPSA exact RBF (paper)", "kpsa_exact@0.1")]
        rows += [(f"{fam} (dev-chosen)", m) for fam, m in ch.items() if fam not in ("constant",)]
        for label, m in rows:
            s = test[m]
            fam = m.split("@")[0]
            mib = block["storage_bytes"].get(fam, 0) / 2**20
            cells = [f"{pct(s[f]['gap_recovered'])} [{pct(s[f]['gap_ci95'][0])}, {pct(s[f]['gap_ci95'][1])}]" for f in fr]
            lines.append(f"| {label} | {m} | " + " | ".join(cells) + f" | {pct(s['0.001']['energy_captured'])} | {mib:.1f} |")
        d = block["paired_test_differences"]["kpsa_proto_paper"]
        lines.append("\nPaper KPSA (prototype) minus baseline, gap points at 0.02/0.05/0.1%:")
        for bm, v in d.items():
            lines.append(f"- {bm}: " + ", ".join(f"{pct(v[f][0])} [{pct(v[f][1])}, {pct(v[f][2])}]" for f in fr))
        if "paper_subset" in block:
            ps = block["paper_subset"]
            lines.append(f"\nPaper 100-query subset (n={ps['n']}): " + "; ".join(
                f"{m}: " + ", ".join(pct(ps[m][f][0]) for f in fr) for m in ps if m != "n"))
    # Sweep summary.
    lines.append("\n### Gap recovered at 0.1% vs N (test)\n")
    fams = ["kpsa_proto@0.025", "kpsa_exact@0.1", "knn@1"]
    ns = list(r["by_n"])
    extra = sorted({f for b in r["by_n"].values() for f in b["dev_chosen"]} - {"constant", "activation_attribution", "direct_profile"})
    lines.append("| N | " + " | ".join(fams + [f"{e}*" for e in extra]) + " |")
    lines.append("|---" * (1 + len(fams) + len(extra)) + "|")
    for n in ns:
        b = r["by_n"][n]
        t = b["summary"]["test"]
        cells = [pct(t[m]["0.001"]["gap_recovered"]) for m in fams]
        cells += [f"{pct(t[b['dev_chosen'][e]]['0.001']['gap_recovered'])} ({b['dev_chosen'][e].split('@')[-1]})" for e in extra]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("\n(* dev-chosen hyperparameter in parentheses)")
    # Bandwidth/k curves at largest N.
    n = ns[-1]
    t = r["by_n"][n]["summary"]["test"]
    lines.append(f"\n### Smoothing curves at N={n} (test, mean gap over budgets)\n")
    for prefix in ("knn@", "kpsa_exact@", "kpsa_proto@", "ridge_z@", "ridge_phi@"):
        ms = [m for m in t if m.startswith(prefix)]
        lines.append(f"- {prefix}: " + ", ".join(
            f"{m.split('@')[1]}: {pct(sum(t[m][f]['gap_recovered'] for f in fr) / 3)}" for m in ms))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vision", type=Path)
    a = p.parse_args()
    if a.vision:
        print(vision_table(a.vision))


if __name__ == "__main__":
    main()
