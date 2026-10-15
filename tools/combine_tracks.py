#!/usr/bin/env python3
"""Combine the generalist (BranchCNN) and the specialists (per-branch BranchNet).

    python tools/combine_tracks.py --fold 1

Everything is scored on BranchCNN's sample set for the fold's test traces
(every conditional branch with a full history window), joined per execution on
(trace, position among all branches) -- verified identical in both pipelines.
MPKI denominator: the test traces' instructions (BranchCNN's own definition).

Systems compared (arbiters fitted on VALIDATION only):

  tage            CBP2016 TAGE-SC-L alone (BranchCNN's own baseline)
  branchcnn       the generalist alone (TAGE state is already one of its inputs)
  A: tage|cnn     our learned arbiter added to THEIR architecture: per execution,
                  TAGE or BranchCNN
  B: cnn|spec     generalist + specialists: on the modelled hard branches, per
                  execution, BranchCNN or our per-branch BranchNet; BranchCNN
                  everywhere else
  C: full         A everywhere, then B's specialist arbitration on top of A's
                  choice on the hard branches
  oracle-*        perfect arbitration ceilings (upper bounds only)

Validation is one trace per fold, so the arbiters' out-of-fold safety check
uses contiguous TIME BLOCKS of the validation trace as folds (instead of whole
traces). A branch keeps its specialist only if the out-of-fold decisions beat
the incumbent significantly (exact McNemar), exactly as before.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.arbiter import HIGH, LOW, MEDIUM, Executions, score  # noqa: E402
from src.evaluation.learned_arbiter import (  # noqa: E402
    fit_corrected_shared_arbiter, fit_learned_arbiter, fit_shared_arbiter,
    model_columns, predictions_differ, score_mask)

BCNN = REPO / "branchcnn" / "results"
XT = REPO / "experiments" / "cbp2025_xt"
OUT = REPO / "results" / "cbp2025" / "combined"


def tage_state_columns(pred, hit, conf):
    """CBP2016 TAGE state as arbiter inputs (BranchCNN's own encoding)."""
    return np.stack([np.where(pred > 0, 1.0, -1.0), hit / 36.0,
                     (hit == 0).astype(float), (conf & 1).astype(float),
                     ((conf >> 1) & 1).astype(float), ((conf >> 2) & 1).astype(float)],
                    axis=1)


def conf_level(conf):
    lv = np.full(conf.shape, MEDIUM, dtype=np.int8)
    lv[(conf & 1) == 1] = HIGH
    lv[((conf >> 1) & 1) == 1] = LOW
    return lv


def blocks(trace, pos, n_blocks=5):
    """Group id = trace + contiguous time block, for out-of-fold folds."""
    out = np.empty(trace.shape, dtype=object)
    for t in np.unique(trace):
        sel = trace == t
        p = pos[sel]
        edges = np.quantile(p, np.linspace(0, 1, n_blocks + 1)[1:-1])
        out[sel] = [f"{t}#{b}" for b in np.searchsorted(edges, p, side="right")]
    return out.astype(str)


def load_cnn(run, role):
    d = np.load(run / f"predictions_{role}.npz")
    return {k: d[k] for k in d.files}


def specialist(fold, variant, role, pc_dirs):
    """Per-branch specialist predictions: {pc: dict(trace, pos, logit, label)}."""
    out = {}
    for d in pc_dirs:
        s = json.loads((d / variant / "summary.json").read_text()) \
            if (d / variant / "summary.json").exists() else {}
        if s.get("status") != "ok":
            continue
        z = np.load(d / variant / ("val.npz" if role == "validation" else "test.npz"))
        out[d.name] = {"trace": z["trace"].astype(str), "pos": z["indices"],
                       "logit": z["logits"], "label": z["labels"]}
    return out


def join(cnn, spec):
    """Index into the BranchCNN arrays for each specialist execution, and keep
    only executions present in both."""
    key_c = {}
    for t in np.unique(cnn["trace"]):
        sel = np.where(cnn["trace"] == t)[0]
        key_c[t] = (cnn["pos"][sel], sel)
    idx, keep = np.full(spec["pos"].shape, -1), np.zeros(spec["pos"].shape, bool)
    for t in np.unique(spec["trace"]):
        if t not in key_c:
            continue
        cp, ci = key_c[t]
        order = np.argsort(cp)
        sp = spec["pos"][spec["trace"] == t]
        slot = np.searchsorted(cp[order], sp)
        slot = np.clip(slot, 0, cp.size - 1)
        hit = cp[order][slot] == sp
        where = np.where(spec["trace"] == t)[0]
        idx[where[hit]] = ci[order][slot[hit]]
        keep[where[hit]] = True
    return idx, keep


def run_fold(fold, variant, run_name=None, xt_group=None):
    run = BCNN / (run_name or f"repro_fold{fold}")
    meta = json.loads((run / "predictions_meta.json").read_text())
    pc_dirs = sorted(p for p in (XT / (xt_group or f"bcnn_web_f{fold}")).iterdir()
                     if p.is_dir())
    res = {"fold": fold, "specialist_variant": variant,
           "test_instructions": meta["test"]["instructions"]}
    instr = meta["test"]["instructions"]
    mpki = lambda misses: 1000.0 * misses / instr  # noqa: E731

    cnn = {r: load_cnn(run, r) for r in ("validation", "test")}
    for r in cnn.values():
        r["cnn_ok"] = (r["logit"] > 0).astype(np.int8) == r["label"]
        r["tage_ok"] = r["tage_pred"] == r["label"]
        r["X_tage"] = tage_state_columns(r["tage_pred"], r["tage_hit"], r["tage_conf"])
        r["lvl"] = conf_level(r["tage_conf"])
        r["groups"] = blocks(r["trace"], r["pos"])
    tst = cnn["test"]
    base_tage = int((~tst["tage_ok"]).sum())
    base_cnn = int((~tst["cnn_ok"]).sum())
    res["tage"] = {"misses": base_tage, "mpki": mpki(base_tage)}
    res["branchcnn"] = {"misses": base_cnn, "mpki": mpki(base_cnn)}

    # ---- A: arbiter added to BranchCNN's architecture (TAGE vs BranchCNN)
    def ex_tc(r):    # "model" = BranchCNN, incumbent = TAGE
        return Executions(r["logit"].astype(np.float32), r["label"], r["tage_ok"],
                          r["lvl"])
    v, t = cnn["validation"], tst
    Xv = np.hstack([model_columns(v["logit"]), v["X_tage"]])
    Xt = np.hstack([model_columns(t["logit"]), t["X_tage"]])
    arbA = fit_learned_arbiter(ex_tc(v), Xv, v["groups"], "lr")
    useA = arbA.use_model(ex_tc(t), Xt)            # True = use BranchCNN
    choiceA_ok = np.where(useA, t["cnn_ok"], t["tage_ok"])   # TAGE if never kept
    choiceA_ok_val = np.where(arbA.use_model(ex_tc(v), Xv), v["cnn_ok"], v["tage_ok"])
    res["A_tage|cnn"] = {"misses": int((~choiceA_ok).sum()),
                         "mpki": mpki(int((~choiceA_ok).sum())),
                         "kept": arbA.model is not None, "oof_p": arbA.oof_p,
                         "share_using_branchcnn": float(useA.mean())}
    oA = int((~t["tage_ok"] & ~t["cnn_ok"]).sum())
    res["oracle_tage|cnn"] = {"misses": oA, "mpki": mpki(oA)}

    # ---- B / C: specialists on the hard branches
    specs = {r: specialist(fold, variant, r, pc_dirs) for r in ("validation", "test")}
    for inc_name, inc_ok_key in (("B_cnn|spec", "cnn_ok"), ("C_full", "choiceA")):
        branches_val, tests, covered = [], {}, 0
        for pc in specs["test"]:
            if pc not in specs["validation"]:
                continue
            parts = {}
            for role in ("validation", "test"):
                s = specs[role][pc]
                r = cnn[role]
                idx, keep = join(r, s)
                ii = idx[keep]
                if not np.array_equal(s["label"][keep].astype(np.int8), r["label"][ii]):
                    raise RuntimeError(f"label mismatch joining {pc} ({role})")
                if inc_ok_key == "choiceA":
                    inc_ok = (choiceA_ok_val if role == "validation" else choiceA_ok)[ii]
                else:
                    inc_ok = r[inc_ok_key][ii]
                ex = Executions(s["logit"][keep].astype(np.float32),
                                s["label"][keep], inc_ok.astype(bool), r["lvl"][ii])
                X = np.hstack([model_columns(s["logit"][keep]),
                               model_columns(r["logit"][ii]), r["X_tage"][ii]])
                parts[role] = (ex, X, r["groups"][ii], ii)
            ev, Xv_, gv, _ = parts["validation"]
            et, Xt_, gt, it = parts["test"]
            branches_val.append((pc, ev, Xv_, gv))
            tests[pc] = (et, Xt_, it)
        if not tests:
            continue
        inc_total = base_cnn if inc_ok_key == "cnn_ok" else int((~choiceA_ok).sum())
        sh = fit_corrected_shared_arbiter(branches_val)
        saved = oracle_saved = used = 0
        per_branch = {}
        for pc, (et, Xt_, it) in tests.items():
            sc = score_mask(et, sh.use_model(pc, et, Xt_))
            saved += sc["tage_misses"] - sc["hybrid_misses"]
            o = score(et, "oracle")
            oracle_saved += o["tage_misses"] - o["hybrid_misses"]
            used += bool(sh.enabled.get(pc))
            per_branch[pc] = {"incumbent_misses": sc["tage_misses"],
                              "combined_misses": sc["hybrid_misses"],
                              "specialist_alone_misses": int((~et.model_correct).sum()),
                              "executions": len(et), "enabled": bool(sh.enabled.get(pc))}
        res[inc_name] = {"misses": inc_total - saved, "mpki": mpki(inc_total - saved),
                         "branches": len(tests), "branches_enabled": used,
                         "per_branch": per_branch}
        res[f"oracle_{inc_name}"] = {"misses": inc_total - oracle_saved,
                                     "mpki": mpki(inc_total - oracle_saved)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--variant", default="tage_raw")
    ap.add_argument("--run", default=None, help="BranchCNN run name (default repro_fold<k>)")
    ap.add_argument("--xt-group", default=None, help="specialists' experiment group")
    ap.add_argument("--tag", default=None, help="output name (default fold<k>)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k in a.fold:
        r = run_fold(k, a.variant, a.run, a.xt_group)
        (OUT / f"{a.tag or f'fold{k}'}_{a.variant}.json").write_text(json.dumps(r, indent=2))
        tm = r["tage"]["misses"]
        print(f"\nfold {k} (specialists: {a.variant}); TAGE-SC-L {r['tage']['mpki']:.3f} MPKI")
        for key in ("branchcnn", "A_tage|cnn", "B_cnn|spec", "C_full",
                    "oracle_tage|cnn", "oracle_B_cnn|spec", "oracle_C_full"):
            if key in r:
                m = r[key]["misses"]
                print(f"  {key:22s} MPKI {r[key]['mpki']:.3f}  vs TAGE {(m - tm) / tm * 100:+6.2f}%"
                      + (f"  (specialists kept on {r[key]['branches_enabled']}/{r[key]['branches']} branches)"
                         if "branches" in r[key] else ""))


if __name__ == "__main__":
    main()
