#!/usr/bin/env python3
"""Per-branch vs shared learned arbiter, on saved cross-trace results.

    python tools/cbp2025_shared_arbiter.py --group web_g0
    python tools/cbp2025_shared_arbiter.py --group web_g0_pool

CPU only: reads the saved val/test logits and the TAGE logs; trains nothing
on the GPU. Writes results/cbp2025/crosstrace/<group>_shared.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.arbiter import Executions  # noqa: E402
from src.evaluation.learned_arbiter import (  # noqa: E402
    arbiter_inputs, extra_columns, fit_corrected_shared_arbiter,
    fit_learned_arbiter, fit_shared_arbiter, score_mask)

KEYS = ("logits", "labels", "tage_correct", "tage_conf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="web_g0")
    ap.add_argument("--variants", default="tokens,tage_raw,tage_raw_shuffled,"
                                          "tokens_labelshuffle")
    ap.add_argument("--arbiter-extra", default=None,
                    help="feature dir under the group (e.g. prodfeat_L1024) "
                         "whose producers/inst_hist columns are added to the "
                         "ARBITER's inputs")
    ap.add_argument("--shuffle-extra", action="store_true",
                    help="CONTROL: permute the extra arbiter columns")
    ap.add_argument("--tag", default="_shared")
    args = ap.parse_args()
    root = REPO / "data" / "cbp2025" / "xt" / args.group
    exp = REPO / "experiments" / "cbp2025_xt" / args.group
    out_dir = REPO / "results" / "cbp2025" / "crosstrace"

    base_miss = instr = 0
    split = json.loads((root / "split.json").read_text())
    for t in split["test"]:
        s = json.loads((root / "logs" / f"{t}.summary.json").read_text())
        tage = next(p for p in s["predictors"] if p["name"] == "tage_sc_l_64KB")
        base_miss += int(tage["mispredictions"])
        instr += int(s["metadata"]["metric_instructions"])

    report = {"group": args.group, "tage_mpki": 1000 * base_miss / instr,
              "variants": {}}
    for variant in args.variants.split(","):
        branches_val, tests = [], {}
        for f in sorted(exp.glob(f"*/{variant}/summary.json")):
            s = json.loads(f.read_text())
            if s.get("status") != "ok":
                continue
            vz, tz = np.load(f.parent / "val.npz"), np.load(f.parent / "test.npz")
            val = Executions(**{k: vz[k] for k in KEYS})
            tst = Executions(**{k: tz[k] for k in KEYS})
            pc = int(s["pc"], 16)
            Xv = arbiter_inputs(val, root / "logs", pc, vz["trace"], vz["indices"])
            Xt = arbiter_inputs(tst, root / "logs", pc, tz["trace"], tz["indices"])
            if args.arbiter_extra:
                d = root / args.arbiter_extra
                Ev = extra_columns(d, pc, vz["trace"], vz["indices"])
                Et = extra_columns(d, pc, tz["trace"], tz["indices"])
                if args.shuffle_extra:            # CONTROL
                    r = np.random.default_rng(0)
                    Ev, Et = Ev[r.permutation(len(Ev))], Et[r.permutation(len(Et))]
                Xv, Xt = np.hstack([Xv, Ev]), np.hstack([Xt, Et])
            branches_val.append((s["pc"], val, Xv, vz["trace"]))
            tests[s["pc"]] = (tst, Xt, val, Xv, vz["trace"])
        if not tests:
            continue
        res = {"branches": len(tests)}
        # Per-branch learned arbiter (the current method).
        saved = 0
        for name, (tst, Xt, val, Xv, tv) in tests.items():
            arb = fit_learned_arbiter(val, Xv, tv, "lr")
            sc = score_mask(tst, arb.use_model(tst, Xt))
            saved += sc["tage_misses"] - sc["hybrid_misses"]
        res["per_branch_lr"] = {"mpki_reduction": saved / base_miss}
        # Shared arbiters.
        for kind in ("lr", "gbdt"):
            sh = fit_shared_arbiter(branches_val, kind)     # VALIDATION only
            saved, used = 0, 0
            for name, (tst, Xt, *_rest) in tests.items():
                sc = score_mask(tst, sh.use_model(name, tst, Xt))
                saved += sc["tage_misses"] - sc["hybrid_misses"]
                used += bool(sh.enabled.get(name))
            res[f"shared_{kind}"] = {"mpki_reduction": saved / base_miss,
                                     "branches_enabled": used,
                                     "per_branch_oof": sh.oof}
        co = fit_corrected_shared_arbiter(branches_val)       # VALIDATION only
        saved, used = 0, 0
        for name, (tst, Xt, *_rest) in tests.items():
            sc = score_mask(tst, co.use_model(name, tst, Xt))
            saved += sc["tage_misses"] - sc["hybrid_misses"]
            used += bool(co.enabled.get(name))
        res["shared_lr_corrected"] = {"mpki_reduction": saved / base_miss,
                                      "branches_enabled": used,
                                      "lambdas": co.lambdas,
                                      "per_branch_oof": co.oof}
        report["variants"][variant] = res
        print(f"{args.group} {variant:22s} per-branch LR "
              f"{res['per_branch_lr']['mpki_reduction']:+.2%} | shared LR "
              f"{res['shared_lr']['mpki_reduction']:+.2%} "
              f"({res['shared_lr']['branches_enabled']}/{len(tests)} branches) | "
              f"shared GBDT {res['shared_gbdt']['mpki_reduction']:+.2%} "
              f"({res['shared_gbdt']['branches_enabled']}/{len(tests)}) | "
              f"shared LR + correction "
              f"{res['shared_lr_corrected']['mpki_reduction']:+.2%} "
              f"({res['shared_lr_corrected']['branches_enabled']}/{len(tests)})",
              flush=True)
    (out_dir / f"{args.group}{args.tag}.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
