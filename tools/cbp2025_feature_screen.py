#!/usr/bin/env python3
"""Screen candidate feature groups cheaply before any BranchNet run.

    python tools/cbp2025_feature_screen.py --group web_g0

Stages (each resumable):

  1. profile    build/cbp2025_producers profile on the TRAIN traces only
  2. choose     per branch, up to 4 dataflow producers that appear in the
                chain on >= 50% of executions (pooled over train traces),
                nearest first -> data/cbp2025/xt/<group>/producers.json
  3. extract    producer + inst_hist features for train/val/test traces at
                commit lags 1024 (guaranteed safe), 256, 64 (assumptions)
  4. screen     per branch, a small gradient-boosted tree model predicting the
                branch outcome, trained on train-trace executions, scored on
                VALIDATION traces (never test):
                  base       TAGE-SC-L per-execution state (14 columns)
                  + group    base + one candidate group
                  + shuffled base + the same group, rows permuted (control)
                A group is informative if it beats base AND its shuffled copy.

This is a proxy: the tree model does not see the global branch history that
BranchNet sees, so a group can look useful here and still be redundant with
history. It filters out dead weight before spending GPU hours, nothing more.
Writes results/cbp2025/screen/<group>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PRODUCERS = REPO / "build" / "cbp2025_producers"
LAGS = (1024, 256, 64)


def _run(cmd):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"{cmd}: {res.stderr[-500:]}")


def trace_gz(category: str, tid: str) -> Path:
    meta = json.loads((REPO / "data" / "cbp2025" / category /
                       f"{tid}.meta.json").read_text())
    return REPO / meta["trace"]


def profile_one(args):
    category, tid, pcs, out = args
    if not Path(out).exists():
        _run([PRODUCERS, "profile", trace_gz(category, tid), "--pcs", pcs,
              "--out", out])
    return tid


def extract_one(args):
    category, tid, producers, lag, out = args
    if not Path(out + ".json").exists():
        _run([PRODUCERS, "features", trace_gz(category, tid), "--producers",
              producers, "--commit-lag", lag, "--out", out])
    return tid


def choose_producers(profiles: list[Path], min_frac: float) -> dict:
    inst = defaultdict(int)
    cnt = defaultdict(lambda: defaultdict(int))
    depth = {}
    for f in profiles:
        for bpc, v in json.loads(f.read_text()).items():
            inst[bpc] += v["instances"]
            for c in v["chain"]:
                cnt[bpc][c["pc"]] += c["count"]
                depth[(bpc, c["pc"])] = min(depth.get((bpc, c["pc"]), 9),
                                            c["depth"])
    out = {}
    for bpc, counts in cnt.items():
        cands = [(depth[(bpc, p)], -c / inst[bpc], p) for p, c in counts.items()
                 if c / inst[bpc] >= min_frac]
        out[bpc] = [p for _, _, p in sorted(cands)[:4]]
    return out


def screen(args, root: Path, split: dict, branches: list[dict]) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from src.dataset import load_raw_features
    from src.evaluation import load_prediction_log

    def tage_rows(tid, pc, idx):
        sub = load_prediction_log(root / "logs" / tid).for_pc(pc)
        slot = np.searchsorted(sub.branch_index, idx)
        return (sub.tage_features()[slot], sub.taken[slot],
                ~sub.mispredicted("tage_sc_l_64KB")[slot])

    cache = {}

    def feats(lag, tid):
        key = (lag, tid)
        if key not in cache:
            cache[key] = load_raw_features(root / f"prodfeat_L{lag}" /
                                           f"{tid}.feat")
        return cache[key]

    def table(traces, pc, lag, max_rows):
        Xb, y, tc, groups = [], [], [], defaultdict(list)
        for tid in traces:
            bi, pcs, f, side = feats(lag, tid)
            sel = pcs == np.uint64(pc)
            idx = bi[sel]
            if idx.size == 0:
                continue
            t_feat, taken, t_ok = tage_rows(tid, pc, idx)
            Xb.append(t_feat); y.append(taken); tc.append(t_ok)
            g = np.array(side["feature_groups"])
            for name in ("producers", "inst_hist"):
                groups[name].append(f[sel][:, g == name])
        if not Xb:
            return None
        out = {"base": np.concatenate(Xb), "y": np.concatenate(y),
               "tage_ok": np.concatenate(tc)}
        for k, v in groups.items():
            out[k] = np.concatenate(v)
        if max_rows and out["y"].size > max_rows:
            keep = np.sort(np.random.default_rng(0).choice(
                out["y"].size, max_rows, replace=False))
            out = {k: v[keep] for k, v in out.items()}
        return out

    def fit_score(Xtr, ytr, Xva, yva):
        m = HistGradientBoostingClassifier(max_depth=6, max_iter=200,
                                           learning_rate=0.1, random_state=0)
        m.fit(Xtr, ytr)
        return int((m.predict(Xva) != yva).sum())

    results = []
    for br in branches:
        pc = int(br["pc"], 16)
        row = {"pc": br["pc"], "category": br["category"]}
        tr0 = table(split["train"], pc, LAGS[0], args.max_train_rows)
        va0 = table(split["validation"], pc, LAGS[0], None)
        if tr0 is None or va0 is None or va0["y"].size < 200:
            continue
        row["val_rows"] = int(va0["y"].size)
        row["tage_misses"] = int((~va0["tage_ok"]).sum())
        row["base_misses"] = fit_score(tr0["base"], tr0["y"], va0["base"],
                                       va0["y"])
        rng = np.random.default_rng(1)
        candidates = [("inst_hist", LAGS[0])] + \
            [("producers", lag) for lag in LAGS]
        for name, lag in candidates:
            tr = tr0 if lag == LAGS[0] else table(split["train"], pc, lag,
                                                  args.max_train_rows)
            va = va0 if lag == LAGS[0] else table(split["validation"], pc,
                                                  lag, None)
            key = name if name == "inst_hist" else f"producers_L{lag}"
            Xtr = np.hstack([tr["base"], tr[name]])
            Xva = np.hstack([va["base"], va[name]])
            row[key] = fit_score(Xtr, tr["y"], Xva, va["y"])
            Xtr_s = np.hstack([tr["base"], tr[name][rng.permutation(len(tr["y"]))]])
            Xva_s = np.hstack([va["base"], va[name][rng.permutation(len(va["y"]))]])
            row[key + "_shuffled"] = fit_score(Xtr_s, tr["y"], Xva_s, va["y"])
        results.append(row)
        print(f"  {br['pc']} {br['category']:17s} val {row['val_rows']:7,} | "
              f"TAGE {row['tage_misses']:6,} base {row['base_misses']:6,} | " +
              " | ".join(f"{k.replace('producers_', 'prod')} "
                         f"{row[k]:6,} (shuf {row[k + '_shuffled']:6,})"
                         for k in ["inst_hist"] +
                         [f"producers_L{l}" for l in LAGS]), flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="web_g0")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--max-train-rows", type=int, default=300_000)
    args = ap.parse_args()

    root = REPO / "data" / "cbp2025" / "xt" / args.group
    split = json.loads((root / "split.json").read_text())
    groups = json.loads((REPO / "results/cbp2025/groups.json").read_text())
    base = args.group.replace("_pool", "")
    category = next(g["category"] for g in groups["groups"] if g["name"] == base)
    branches = json.loads((root / "h2p.json").read_text())["branches"]

    # 1-2. profile TRAIN traces only, choose producers
    prof_dir = root / "profiles"
    prof_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(profile_one, [(category, t, str(root / "h2p.txt"),
                                     str(prof_dir / f"{t}.json"))
                                    for t in split["train"]]))
    producers = choose_producers([prof_dir / f"{t}.json" for t in split["train"]],
                                 args.min_frac)
    (root / "producers.json").write_text(json.dumps(producers, indent=2))
    n_with = sum(bool(v) for v in producers.values())
    print(f"[{args.group}] producers chosen for {n_with}/{len(producers)} "
          f"branches (train traces only)", flush=True)

    # 3. extract for every trace at every lag
    jobs = []
    for lag in LAGS:
        d = root / f"prodfeat_L{lag}"
        d.mkdir(exist_ok=True)
        for t in split["train"] + split["validation"] + split["test"]:
            jobs.append((category, t, str(root / "producers.json"), str(lag),
                         str(d / f"{t}.feat")))
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(extract_one, jobs))
    print(f"[{args.group}] features extracted at lags {LAGS}", flush=True)

    # 4. screen on validation
    results = screen(args, root, split, branches)
    tot = defaultdict(int)
    for r in results:
        for k, v in r.items():
            if isinstance(v, int) and k != "val_rows":
                tot[k] += v
    print(f"\n[{args.group}] pooled validation mispredictions over "
          f"{len(results)} branches:")
    print(f"  TAGE-SC-L          {tot['tage_misses']:8,}")
    print(f"  base (TAGE state)  {tot['base_misses']:8,}")
    for k in ["inst_hist"] + [f"producers_L{l}" for l in LAGS]:
        print(f"  + {k:16s} {tot[k]:8,}  ({(tot['base_misses'] - tot[k]) / tot['base_misses']:+.1%} vs base) | "
              f"shuffled {tot[k + '_shuffled']:8,}  "
              f"({(tot['base_misses'] - tot[k + '_shuffled']) / tot['base_misses']:+.1%})")
    out = REPO / "results" / "cbp2025" / "screen"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.group}.json").write_text(json.dumps(
        {"group": args.group, "lags": LAGS, "producers": producers,
         "per_branch": results, "totals": dict(tot)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
