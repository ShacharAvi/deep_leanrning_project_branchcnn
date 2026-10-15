#!/usr/bin/env python3
"""Where does the hybrid's remaining headroom go?

    python tools/cbp2025_gap_analysis.py --group web_g0 --variant tokens

Per modelled branch, on the TEST traces, every TAGE misprediction falls into
one of two buckets:

  both wrong     neither predictor is right -> no arbiter can fix it; only a
                 better MODEL (more information, more coverage) can
  model-only     the model is right, TAGE wrong -> an arbiter COULD recover it

and the oracle's headroom is exactly the model-only bucket. The learned
arbiter (refitted on validation, as in evaluation) recovers part of it and
loses some TAGE-only executions. The AUC of the arbiter on test disagreements
says whether the missing part is an ARBITER problem (AUC high: rankable, just
mis-thresholded) or an INFORMATION problem (AUC near 0.5: the available
signals cannot tell the two cases apart).

Coverage check: per test trace, the model's gain over TAGE is related to how
similar that trace is to its nearest training trace (cosine of per-branch
execution counts). If gains concentrate on test traces that resemble training
traces, more inputs (coverage) is the lever.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.arbiter import Executions  # noqa: E402
from src.evaluation.learned_arbiter import (  # noqa: E402
    arbiter_inputs, fit_learned_arbiter, predictions_differ)


def trace_vectors(category: str, traces: list[str]) -> dict[str, dict]:
    out = {}
    for t in traces:
        d = {}
        path = REPO / "data" / "cbp2025" / category / f"{t}.full_per_pc.csv"
        for r in csv.DictReader(open(path, newline="")):
            d[r["pc"]] = int(r["executions"])
        out[t] = d
    return out


def cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], float)
    vb = np.array([b.get(k, 0) for k in keys], float)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def main() -> int:
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="web_g0")
    ap.add_argument("--variant", default="tokens")
    args = ap.parse_args()

    root = REPO / "data" / "cbp2025" / "xt" / args.group
    exp = REPO / "experiments" / "cbp2025_xt" / args.group
    split = json.loads((root / "split.json").read_text())
    groups = json.loads((REPO / "results/cbp2025/groups.json").read_text())
    cat = next(g["category"] for g in groups["groups"]
               if g["name"] == args.group)

    tot = dict(tage_miss=0, both_wrong=0, model_only=0, tage_only=0,
               recovered=0, lost=0)
    aucs, per_trace = [], {}
    keys = ("logits", "labels", "tage_correct", "tage_conf")
    for f in sorted(exp.glob(f"*/{args.variant}/summary.json")):
        s = json.loads(f.read_text())
        if s.get("status") != "ok":
            continue
        vz, tz = np.load(f.parent / "val.npz"), np.load(f.parent / "test.npz")
        val = Executions(**{k: vz[k] for k in keys})
        tst = Executions(**{k: tz[k] for k in keys})
        pc = int(s["pc"], 16)
        Xv = arbiter_inputs(val, root / "logs", pc, vz["trace"], vz["indices"])
        Xt = arbiter_inputs(tst, root / "logs", pc, tz["trace"], tz["indices"])
        arb = fit_learned_arbiter(val, Xv, vz["trace"], "lr")
        use = arb.use_model(tst, Xt)
        m, t = tst.model_correct, tst.tage_correct
        tot["tage_miss"] += int((~t).sum())
        tot["both_wrong"] += int((~t & ~m).sum())
        tot["model_only"] += int((~t & m).sum())
        tot["tage_only"] += int((t & ~m).sum())
        tot["recovered"] += int((use & ~t & m).sum())
        tot["lost"] += int((use & t & ~m).sum())
        differ = predictions_differ(tst)
        if arb.model is not None and differ.sum() > 50:
            prob = arb.model.predict_proba(Xt[differ])[:, 1]
            y = m[differ]
            if 0 < y.mean() < 1:
                aucs.append((s["pc"], roc_auc_score(y, prob), int(differ.sum())))
        for tr in np.unique(tz["trace"]):
            sel = tz["trace"] == tr
            d = per_trace.setdefault(str(tr), [0, 0])
            d[0] += int((~t[sel]).sum())
            d[1] += int((~m[sel]).sum())

    tm = tot["tage_miss"]
    print(f"== {args.group} / {args.variant}: TAGE misses on modelled branches "
          f"(test) = {tm:,}")
    print(f"  both wrong (no arbiter can fix)     {tot['both_wrong']:8,}  "
          f"{tot['both_wrong']/tm:6.1%}")
    print(f"  model right, TAGE wrong (oracle)    {tot['model_only']:8,}  "
          f"{tot['model_only']/tm:6.1%}")
    print(f"    learned arbiter recovers          {tot['recovered']:8,}  "
          f"{tot['recovered']/tot['model_only']:6.1%} of that")
    print(f"    ... while losing (TAGE was right) {tot['lost']:8,}")
    print(f"  TAGE right, model wrong (risk pool) {tot['tage_only']:8,}")
    if aucs:
        w = np.array([a[2] for a in aucs], float)
        print(f"  arbiter AUC on test disagreements: weighted mean "
              f"{np.average([a[1] for a in aucs], weights=w):.3f} "
              f"(0.5 = signals cannot separate; 1.0 = perfect)")

    vecs = trace_vectors(cat, split["train"] + split["test"])
    print("\n  per test trace: model vs TAGE misses on modelled branches, "
          "and similarity to nearest TRAIN trace")
    rows = []
    for tr, (tmiss, mmiss) in sorted(per_trace.items()):
        sim = max(cosine(vecs[tr], vecs[x]) for x in split["train"])
        rows.append((sim, (tmiss - mmiss) / tmiss if tmiss else 0.0, tr))
    for sim, gain, tr in sorted(rows, reverse=True):
        print(f"    {tr:8s} nearest-train similarity {sim:.3f}   model gain "
              f"vs TAGE (attach-all) {gain:+.1%}")
    sims, gains = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    if len(rows) > 2:
        print(f"  correlation(similarity, gain) = "
              f"{np.corrcoef(sims, gains)[0, 1]:+.2f} over {len(rows)} traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
