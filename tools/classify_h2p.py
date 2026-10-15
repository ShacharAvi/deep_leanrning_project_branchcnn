#!/usr/bin/env python3
"""Classify *why* each hard-to-predict branch is hard, before training anything.

Project spec §2.3 asks for branches to be classified by apparent source of
difficulty, and warns against assuming a CNN will fix all of them. TAGE-SC-L's
own internal state answers this directly, because TAGE already tried every
history length from 6 to 3000 and recorded what happened:

    hit_bank       which tagged component supplied the prediction. Bank 0 means
                   nothing matched and it fell back to bimodal -- no history
                   pattern of any length helped. High banks mean the correlation
                   is real but deep.
    confidence     TAGE's own 3-level estimate.
    alt_disagree   provider and alternate disagree -> genuinely uncertain.

All of it is computed at prediction time, so none of it is label leakage.

Categories, and what each implies for the project:

    random              ~50% accurate, low confidence. Nothing will predict it.
                        A model that "succeeds" here is leaking, not learning.
    data-dependent      mostly bimodal fallback: no control-flow history of any
                        length correlates. Likely needs operand values, which
                        our traces do not carry (spec §20.3).
    deep-correlation    high hit_bank and TAGE does reasonably well -> the
                        correlation exists but is far back. THE CNN TARGET.
    confidently-wrong   high confidence yet frequently wrong -> systematically
                        mispredicted; the case BranchNet was built for.
    uncertain           low/medium confidence with provider-alt disagreement ->
                        TAGE is guessing; a richer model may separate the cases.
    well-predicted      TAGE already handles it; little headroom.

    python tools/classify_h2p.py --dir results/mcf --min-executions 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED = "tage_hit_bank_mean"

#: A branch whose TAGE accuracy exceeds the always-taken/not-taken floor by less
#: than this is "bias-dominated": history is buying nothing, and the remaining
#: mispredictions are rare exceptions under severe class imbalance.
BIAS_MARGIN = 0.01


def load_per_pc(path: Path) -> dict[int, dict]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if REQUIRED not in (reader.fieldnames or []):
            raise SystemExit(
                f"{path.name} has no TAGE state columns. Re-run "
                f"run_predictors.py with tage_sc_l_64KB selected."
            )
        return {int(r["pc"], 16): r for r in reader}


def classify(row: dict) -> tuple[str, str]:
    """Return (category, one-line reason) for a single branch in one trace."""
    execs = int(row["executions"])
    misses = int(row["tage_sc_l_64KB_mispredictions"])
    accuracy = 1 - misses / execs
    taken_rate = float(row["taken_rate"])
    bias = max(taken_rate, 1 - taken_rate)

    bimodal = float(row["tage_bimodal_frac"])
    hit_bank = float(row["tage_hit_bank_mean"])
    high = float(row["tage_conf_high_frac"])
    low = float(row["tage_conf_low_frac"])
    disagree = float(row["tage_alt_disagree_frac"])

    if accuracy < 0.60 and bias < 0.60:
        return "random", f"{accuracy:.0%} accurate at {taken_rate:.0%} taken -- chance"
    if bimodal > 0.70:
        return ("data-dependent",
                f"{bimodal:.0%} bimodal fallback -- no history length correlates")
    # Check this BEFORE accuracy-based categories. A 98%-taken branch gives TAGE
    # a 98% score for free; a high hit_bank there means TAGE found a pattern that
    # buys nothing. Judged on accuracy alone these look like strong
    # deep-correlation targets, and they are not.
    if accuracy - bias < BIAS_MARGIN:
        return ("bias-dominated",
                f"{bias:.1%} bias floor vs {accuracy:.1%} TAGE -- "
                f"history gains {accuracy-bias:+.2%}")
    if accuracy > 0.98:
        return "well-predicted", f"TAGE already {accuracy:.1%} accurate"
    if high > 0.70 and accuracy < 0.90:
        return ("confidently-wrong",
                f"{high:.0%} high-confidence yet {1-accuracy:.0%} wrong")
    if hit_bank >= 10 and accuracy >= 0.70:
        return ("deep-correlation",
                f"mean hit_bank {hit_bank:.1f} -- correlation is real but deep")
    if low > 0.20 or disagree > 0.20:
        return ("uncertain",
                f"{low:.0%} low-confidence, {disagree:.0%} provider/alt disagree")
    return "deep-correlation", f"mean hit_bank {hit_bank:.1f}, {accuracy:.1%} accurate"


ACTION = {
    "random": "SKIP -- unpredictable; use as a leakage canary",
    "bias-dominated": "SKIP -- history gains ~nothing over always-taken",
    "data-dependent": "SKIP for PC+direction models; needs operand values",
    "deep-correlation": "MODEL -- the CNN/Transformer target",
    "confidently-wrong": "MODEL -- the BranchNet case",
    "uncertain": "MODEL -- may benefit from richer context",
    "well-predicted": "SKIP -- no headroom",
}


def summarise(category: str, misrate: float, mean) -> str:
    """One-line reason, built from cross-trace averages so it matches the table."""
    accuracy = 1 - misrate
    if category == "random":
        return f"{accuracy:.0%} accurate -- chance"
    if category == "data-dependent":
        return (f"{mean('tage_bimodal_frac'):.0%} bimodal fallback -- "
                f"no history length correlates")
    if category == "bias-dominated":
        bias = max(mean("taken_rate"), 1 - mean("taken_rate"))
        return (f"{bias:.1%} bias floor vs {accuracy:.1%} TAGE -- "
                f"history gains {accuracy-bias:+.2%}")
    if category == "well-predicted":
        return f"TAGE already {accuracy:.1%} accurate"
    if category == "confidently-wrong":
        return (f"{mean('tage_conf_high_frac'):.0%} high-confidence yet "
                f"{1-accuracy:.0%} wrong")
    if category == "uncertain":
        return (f"{mean('tage_conf_low_frac'):.0%} low-confidence, "
                f"{mean('tage_alt_disagree_frac'):.0%} provider/alt disagree")
    return (f"mean hit_bank {mean('tage_hit_bank_mean'):.1f} -- "
            f"correlation is real but deep")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "results" / "mcf")
    parser.add_argument("--min-executions", type=int, default=10_000)
    parser.add_argument("--min-misses", type=int, default=1_000)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--select-on", nargs="+", default=None,
                        help="restrict selection to these traces. Spec §28: "
                             "'No test-set branch statistics were used to choose "
                             "branches.' Pass the TRAIN traces only.")
    args = parser.parse_args()

    csvs = sorted(args.dir.glob("*_per_pc.csv"))
    if not csvs:
        print(f"no per-PC CSVs in {args.dir}", file=sys.stderr)
        return 1
    traces = {p.name[: -len("_per_pc.csv")]: load_per_pc(p) for p in csvs}
    names = sorted(traces)

    if args.select_on:
        unknown = [n for n in args.select_on if n not in traces]
        if unknown:
            print(f"--select-on names not found: {unknown}", file=sys.stderr)
            return 1
        names = sorted(args.select_on)
        print(f"SELECTING ON {len(names)} trace(s) only; "
              f"{len(traces) - len(names)} held out of selection entirely.")
    else:
        print("WARNING: selecting on ALL traces, including any test trace. "
              "Spec §28 forbids using test-set branch statistics for selection; "
              "pass --select-on with the training traces.")

    hard = {
        n: {pc for pc, r in rows.items()
            if int(r["executions"]) >= args.min_executions
            and int(r["tage_sc_l_64KB_mispredictions"]) >= args.min_misses}
        for n, rows in traces.items()
    }
    shared = set.intersection(*hard.values()) if hard else set()
    print(f"{len(names)} traces; {len(shared)} H2P branches hard in ALL of them\n")

    # Classify per trace, then take the majority verdict across traces. A branch
    # whose category is not stable across members is itself worth flagging.
    per_branch = {}
    for pc in shared:
        votes, reasons = [], {}
        for n in names:
            cat, why = classify(traces[n][pc])
            votes.append(cat)
            reasons[n] = why
        counts = Counter(votes)
        category, agreement = counts.most_common(1)[0]
        mean = lambda col: sum(  # noqa: E731
            float(traces[n][pc][col]) for n in names) / len(names)
        total_misses = sum(
            int(traces[n][pc]["tage_sc_l_64KB_mispredictions"]) for n in names)
        total_execs = sum(int(traces[n][pc]["executions"]) for n in names)
        # Reason is rebuilt from the cross-trace aggregates so it agrees with
        # the numbers printed alongside it.
        per_branch[pc] = {
            "category": category,
            "agreement": agreement / len(names),
            "votes": dict(counts),
            "reason": summarise(category, total_misses / total_execs, mean),
            "total_misses": total_misses,
            "total_executions": total_execs,
            "mean_hit_bank": mean("tage_hit_bank_mean"),
            "bimodal_frac": mean("tage_bimodal_frac"),
        }

    by_cat = defaultdict(list)
    for pc, info in per_branch.items():
        by_cat[info["category"]].append((info["total_misses"], pc, info))

    total_misses = sum(i["total_misses"] for i in per_branch.values())
    print(f"{'category':20s} {'branches':>9s} {'misses':>14s} {'share':>7s}  action")
    print("-" * 96)
    order = ["deep-correlation", "confidently-wrong", "uncertain",
             "data-dependent", "random", "bias-dominated", "well-predicted"]
    for cat in order:
        items = by_cat.get(cat, [])
        if not items:
            continue
        m = sum(i[0] for i in items)
        print(f"{cat:20s} {len(items):>9d} {m:>14,} {m/total_misses:>6.1%}  {ACTION[cat]}")

    print(f"\n{'pc':>10s} {'category':19s} {'misses':>12s} {'TAGE acc':>9s} "
          f"{'hit_bank':>9s} {'bimodal':>8s} {'stable':>7s}  reason")
    print("-" * 118)
    for cat in order:
        for misses, pc, info in sorted(by_cat.get(cat, []), reverse=True):
            acc = 1 - info["total_misses"] / info["total_executions"]
            print(f"{pc:>10x} {cat:19s} {misses:>12,} {acc:>8.2%} "
                  f"{info['mean_hit_bank']:>9.1f} {info['bimodal_frac']:>7.1%} "
                  f"{info['agreement']:>6.0%}  {info['reason']}")

    unstable = [pc for pc, i in per_branch.items() if i["agreement"] < 0.75]
    if unstable:
        print(f"\n{len(unstable)} branch(es) whose category disagrees across traces "
              f"-- behaviour is phase-dependent: "
              + ", ".join(f"{pc:#x}" for pc in sorted(unstable)))

    modelled = [pc for pc, i in per_branch.items()
                if i["category"] in ("deep-correlation", "confidently-wrong", "uncertain")]
    m = sum(per_branch[pc]["total_misses"] for pc in modelled)
    print(f"\nWorth modelling: {len(modelled)}/{len(per_branch)} branches, "
          f"{m:,} mispredictions ({m/total_misses:.1%} of the shared H2P set)")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {f"{pc:#x}": info for pc, info in sorted(per_branch.items())}, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
