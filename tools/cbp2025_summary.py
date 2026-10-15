#!/usr/bin/env python3
"""Summarise CBP2025 pipeline results into one markdown report.

    python tools/cbp2025_summary.py                 # all categories
    python tools/cbp2025_summary.py --category media

Reads results/cbp2025/<category>/*.hybrid.json and the per-run summaries under
experiments/cbp2025/. The headline is hybrid MPKI with the validation-gated
attach policy -- the only one that never looks at test.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load(category_dir: Path):
    hyb = [json.loads(f.read_text()) for f in sorted(category_dir.glob("*.hybrid.json"))]
    exp = REPO / "experiments" / "cbp2025" / category_dir.name
    runs = [json.loads(f.read_text()) for f in exp.glob("*/*/*/summary.json")]
    return hyb, [r for r in runs if r.get("status") == "ok"]


def category_report(name: str, hyb: list, runs: list) -> list[str]:
    out = [f"## {name}", ""]
    if not hyb:
        return out + ["(no results yet)", ""]
    variants = sorted({v for h in hyb for v in h["variants"]})
    base_miss = sum(h["tage_test_mispredictions"] for h in hyb)
    instr = sum(h["test_instructions"] for h in hyb)
    out.append(f"{len(hyb)} traces, test ranges total {instr/1e6:.0f} M "
               f"instructions. Pooled TAGE-SC-L test MPKI "
               f"**{1000*base_miss/instr:.3f}**.")
    out.append("")
    out.append("| variant | branches | gated: attached | pooled hybrid MPKI | "
               "reduction | traces improved | attach-all reduction |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        hs = [h for h in hyb if v in h["variants"]]
        miss_g = sum(h["variants"][v]["attach_gated"]["mispredictions"] for h in hs)
        miss_a = sum(h["variants"][v]["attach_all"]["mispredictions"] for h in hs)
        base = sum(h["tage_test_mispredictions"] for h in hs)
        ins = sum(h["test_instructions"] for h in hs)
        nb = sum(h["variants"][v]["branches_modelled"] for h in hs)
        att = sum(h["variants"][v]["attach_gated"]["attached"] for h in hs)
        improved = sum(h["variants"][v]["attach_gated"]["mispredictions"]
                       < h["tage_test_mispredictions"] for h in hs)
        out.append(f"| `{v}` | {nb} | {att} | {1000*miss_g/ins:.3f} | "
                   f"{(base-miss_g)/base:+.2%} | {improved}/{len(hs)} | "
                   f"{(base-miss_a)/base:+.2%} |")
    orc = [h for h in hyb if "oracle_perfect_on_modelled" in h]
    if orc:
        saved = sum(h["oracle_perfect_on_modelled"]["mpki_reduction"]
                    * h["tage_test_mispredictions"] for h in orc)
        base = sum(h["tage_test_mispredictions"] for h in orc)
        out.append("")
        out.append(f"Ceiling (perfect prediction on every modelled branch): "
                   f"{saved/base:.2%} MPKI reduction. Coverage, not just model "
                   f"quality, bounds the hybrid.")

    # Per-branch: how often does each variant beat TAGE on the same instances?
    by_v = defaultdict(list)
    for r in runs:
        by_v[r["variant"]].append(r)
    out += ["", "| variant | branch runs | beat TAGE on test | median reduction "
            "| pooled branch reduction |", "|---|---:|---:|---:|---:|"]
    for v in variants:
        rs = by_v.get(v, [])
        if not rs:
            continue
        red = np.array([r["reduction_vs_tage"] for r in rs], dtype=float)
        tm = sum(r["test_tage_misses"] for r in rs)
        mm = sum(r["test_model_misses"] for r in rs)
        wins = sum(r["test_model_misses"] < r["test_tage_misses"] for r in rs)
        out.append(f"| `{v}` | {len(rs)} | {wins} | {np.nanmedian(red):+.1%} | "
                   f"{(tm-mm)/tm if tm else 0:+.1%} |")

    # Paired comparison: features vs tokens, and features vs shuffled control.
    key = lambda r: (r["trace"], r["pc"])  # noqa: E731
    idx = {v: {key(r): r for r in by_v.get(v, [])} for v in variants}
    for a, b, label in (("tage_raw", "tokens", "features vs plain BranchNet"),
                        ("tage_raw", "tage_raw_shuffled",
                         "features vs SHUFFLED features (control)")):
        if a in idx and b in idx:
            common = set(idx[a]) & set(idx[b])
            if not common:
                continue
            ma = sum(idx[a][k]["test_model_misses"] for k in common)
            mb = sum(idx[b][k]["test_model_misses"] for k in common)
            better = sum(idx[a][k]["test_model_misses"] < idx[b][k]["test_model_misses"]
                         for k in common)
            out.append("")
            out.append(f"- **{label}**: on {len(common)} paired branches, "
                       f"`{a}` has {ma:,} test misses vs {mb:,} "
                       f"({(mb-ma)/mb if mb else 0:+.1%}); better on "
                       f"{better}/{len(common)}.")
    out.append("")
    out.append("| trace | TAGE MPKI | " + " | ".join(f"{v} (gated)" for v in variants)
               + " | ceiling |")
    out.append("|---|---:|" + "---:|" * (len(variants) + 1))
    for h in hyb:
        cells = []
        for v in variants:
            r = h["variants"].get(v)
            cells.append("—" if r is None else
                         f"{r['attach_gated']['mpki']:.3f} "
                         f"({r['attach_gated']['mpki_reduction']:+.1%})")
        ceil = h.get("oracle_perfect_on_modelled", {}).get("mpki_reduction")
        out.append(f"| {h['trace']} | {h['tage_test_mpki']:.3f} | "
                   + " | ".join(cells)
                   + f" | {'—' if ceil is None else f'{ceil:.1%}'} |")
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", nargs="*", default=None)
    ap.add_argument("--out", type=Path,
                    default=REPO / "results" / "cbp2025" / "SUMMARY.md")
    args = ap.parse_args()
    root = REPO / "results" / "cbp2025"
    cats = args.category or sorted(d.name for d in root.iterdir() if d.is_dir())
    lines = ["# CBP2025 — BranchNet-big + TAGE-SC-L state + fetch-safe raw features",
             "",
             "Generated by `tools/cbp2025_summary.py`. Split: chronological "
             "within each trace (train 0-60%, validation 60-70%, test 70-100% "
             "of branches, H-branch gaps) — **cross-phase, one run**, not "
             "cross-input. Hybrid MPKI is exact, over the test range; "
             "\"gated\" attaches a model only where it beat TAGE on validation significantly (one-sided exact McNemar, p < 0.05).",
             ""]
    for c in cats:
        hyb, runs = load(root / c)
        lines += category_report(c, hyb, runs)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
