#!/usr/bin/env python3
"""End-to-end BranchNet-big + features pipeline over CBP2025 traces.

    python scripts/cbp2025_pipeline.py --category media \\
        --traces new_data/media/*_trace.gz

Per trace, every stage resumable (outputs are skipped if present):

  1. convert   build/cbp2025_convert -> SBBT + split boundaries
  2. select    TAGE-SC-L per-PC stats over the TRAIN range only (spec §28: no
               test statistics choose branches) -> top-K H2P branches
  3. log       TAGE-SC-L per-instance log (internal state + correctness) for
               those branches, whole trace
  4. testbase  TAGE-SC-L totals over the TEST range -> hybrid MPKI baseline
  5. features  fetch-safe raw features for those branches
  6. dataset   BranchNet-compatible HDF5, split_role=chronological
  7. train     BranchNet-big per branch, per feature variant
  8. hybrid    exact hybrid MPKI on the test range, per variant

THE SPLIT. A CBP2025 trace is one run, so train / validation / test are
disjoint time ranges of its branch stream: [H, b0) / [b0+H, b1) / [b1+H, end),
b0 = 60%, b1 = 70% of the branches. The H-branch gaps keep every window inside
its own range. This is a cross-PHASE split (later execution of the same run),
not cross-input; say so wherever a number from it is quoted.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONVERT = REPO / "build" / "cbp2025_convert"
RUN_PREDICTORS = REPO / "build" / "run_predictors"
TAGE = "tage_sc_l_64KB"
HISTORY = 582

#: Feature variants. ``tage`` columns are TAGE-SC-L's per-instance internal
#: state (all groups, so including TAGE's own prediction -- the model may learn
#: when to override TAGE); ``raw`` are the converter's fetch-safe features.
VARIANTS = {
    "tokens": dict(tage=False, raw=False, shuffle=False),
    "tage": dict(tage=True, raw=False, shuffle=False),
    "raw": dict(tage=False, raw=True, shuffle=False),
    "tage_raw": dict(tage=True, raw=True, shuffle=False),
    # CONTROL: same columns, rows permuted across instances. A gain that
    # survives this did not come from the features.
    "tage_raw_shuffled": dict(tage=True, raw=True, shuffle=True),
}


def trace_id_of(path: Path) -> str:
    name = path.name
    for suffix in ("_trace.gz", ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def run(cmd: list, log) -> None:
    log(f"$ {' '.join(str(c) for c in cmd)}")
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {cmd}\n"
                           f"{res.stderr[-2000:]}")


class Paths:
    def __init__(self, work: Path, tid: str):
        self.tid = tid
        self.sbbt = work / f"{tid}.sbbt.zst"
        self.meta = work / f"{tid}.meta.json"
        self.train_pc = work / f"{tid}.train_per_pc.csv"
        self.h2p = work / f"{tid}.h2p.txt"
        self.h2p_json = work / f"{tid}.h2p.json"
        self.log_prefix = work / "logs" / tid
        self.full_json = work / f"{tid}.full.json"
        self.test_json = work / f"{tid}.test.json"
        self.feat_prefix = work / "features" / f"{tid}.feat"
        self.dataset = work / "datasets" / f"{tid}.hdf5"


# --------------------------------------------------------------------------
# CPU stages
# --------------------------------------------------------------------------

def select_h2p(train_csv: Path, top_k: int, min_exec: int, min_miss: int):
    rows = []
    with open(train_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            ex = int(row["executions"])
            miss = int(row[f"{TAGE}_mispredictions"])
            if ex >= min_exec and miss >= min_miss:
                rows.append({"pc": row["pc"], "train_executions": ex,
                             "train_tage_misses": miss,
                             "train_tage_accuracy": 1 - miss / ex,
                             "train_taken_rate": float(row["taken_rate"])})
    rows.sort(key=lambda r: -r["train_tage_misses"])
    return rows[:top_k]


def prepare_trace(trace: str, work: str, top_k: int, min_exec: int,
                  min_miss: int) -> dict:
    """Stages 1-6 for one trace. Runs in a worker process."""
    from src.dataset import build_trace_dataset

    trace, work = Path(trace), Path(work)
    tid = trace_id_of(trace)
    p = Paths(work, tid)
    for d in (work, work / "logs", work / "features", work / "datasets"):
        d.mkdir(parents=True, exist_ok=True)
    logfile = open(work / f"{tid}.prep.log", "a")

    def log(msg):
        logfile.write(msg + "\n")
        logfile.flush()

    started = time.time()
    if not (p.sbbt.exists() and p.meta.exists()):
        run([CONVERT, trace, "--sbbt", p.sbbt, "--meta", p.meta], log)
    meta = json.loads(p.meta.read_text())
    b_train, b_val = meta["boundaries"]

    if not p.train_pc.exists():
        run([RUN_PREDICTORS, p.sbbt, "--predictors", f"always_taken,{TAGE}",
             "--limit", b_train["conditional_before"],
             "--json", work / f"{tid}.train.json", "--per-pc", p.train_pc], log)

    if not p.h2p_json.exists():
        chosen = select_h2p(p.train_pc, top_k, min_exec, min_miss)
        p.h2p_json.write_text(json.dumps({
            "rule": f"TAGE-SC-L misses on the TRAIN range only; executions >= "
                    f"{min_exec}, misses >= {min_miss}; top {top_k} by misses",
            "branches": chosen}, indent=2))
        p.h2p.write_text("".join(f"{r['pc']}\n" for r in chosen))
    chosen = json.loads(p.h2p_json.read_text())["branches"]
    if not chosen:
        log("no branch passes the H2P thresholds; trace skipped")
        return {"trace": tid, "status": "no_h2p", "meta": meta}

    if not p.log_prefix.with_suffix(".json").exists():
        run([RUN_PREDICTORS, p.sbbt, "--predictors", f"always_taken,{TAGE}",
             "--json", p.full_json, "--log", p.log_prefix, "--log-pcs", p.h2p],
            log)
    if not p.test_json.exists():
        run([RUN_PREDICTORS, p.sbbt, "--predictors", TAGE,
             "--warmup", b_val["instr_num"], "--json", p.test_json], log)
    if not p.feat_prefix.with_name(p.feat_prefix.name + ".json").exists():
        run([CONVERT, trace, "--feature-pcs", p.h2p, "--features",
             p.feat_prefix], log)
    if not p.dataset.exists():
        build_trace_dataset(
            p.sbbt, p.dataset, [int(r["pc"], 16) for r in chosen],
            split_role="chronological",
            extra_attrs={"split_boundaries": json.dumps(meta["boundaries"]),
                         "source_trace": str(trace)})
    log(f"prepared in {time.time() - started:.0f}s")
    return {"trace": tid, "status": "ok"}


# --------------------------------------------------------------------------
# GPU stages
# --------------------------------------------------------------------------

def ranges(meta: dict) -> dict:
    b0 = meta["boundaries"][0]["branch_index"]
    b1 = meta["boundaries"][1]["branch_index"]
    n = meta["num_branches"]
    return {"train": (HISTORY, b0), "validation": (b0 + HISTORY, b1),
            "test": (b1 + HISTORY, n)}


def tage_correct_on(log, pc: int, indices: np.ndarray) -> np.ndarray:
    """TAGE-SC-L correctness on exactly these instances of ``pc``."""
    sub = log.for_pc(pc)
    slot = np.searchsorted(sub.branch_index, indices)
    if not np.array_equal(sub.branch_index[slot], indices):
        raise RuntimeError(f"TAGE log does not cover every instance of {pc:#x}")
    return ~sub.mispredicted(TAGE)[slot]


def mcnemar_one_sided(model_only: int, tage_only: int) -> float:
    """P(model is no better than TAGE), exact McNemar on discordant pairs.

    Only instances where exactly one predictor is right carry information
    about which is better; under "equally good" each is a fair coin.
    """
    from scipy.stats import binomtest
    n = model_only + tage_only
    if n == 0:
        return 1.0
    return float(binomtest(model_only, n, 0.5, alternative="greater").pvalue)


def dataset_indices(ds) -> np.ndarray:
    """Instance order of ``iter_batches(shuffle=False)``: sources in order."""
    return np.concatenate([s.indices for s in ds.sources]) if ds.sources \
        else np.empty(0, np.int64)


def train_trace(work: Path, exp_root: Path, tid: str, variants: list[str],
                args, log) -> dict:
    import torch  # noqa: F401  (imported late so prep workers stay light)
    from src.dataset import (AuxFeatureSpec, BranchDataset,
                             CombinedFeatureSpec, RawFeatureSpec,
                             as_signed_pc, vocabulary_size)
    from src.evaluation import load_prediction_log
    from src.models import build_model, dataset_config
    from src.training import (TrainConfig, evaluate, resolve_device,
                              save_run, seed_everything,
                              summarise_against_baselines, train_branch_model)

    p = Paths(work, tid)
    meta = json.loads(p.meta.read_text())
    rng_ = ranges(meta)
    chosen = json.loads(p.h2p_json.read_text())["branches"]
    tage_log = load_prediction_log(p.log_prefix)
    encoding = dataset_config("branchnet_big")
    vocab = vocabulary_size(**encoding)
    device = resolve_device(args.device)

    results = []
    for br in chosen:
        # Signed, as the dataset stores it; the log joins use the bit pattern.
        pc = as_signed_pc(int(br["pc"], 16))
        for variant in variants:
            spec_v = VARIANTS[variant]
            outdir = exp_root / tid / br["pc"] / variant
            if (outdir / "metrics.json").exists() and not args.retrain:
                results.append(json.loads((outdir / "summary.json").read_text()))
                continue
            log(f"[{tid}] {br['pc']} {variant}")
            parts = []
            if spec_v["tage"]:
                parts.append(AuxFeatureSpec(
                    log_dir=p.log_prefix.parent,
                    groups=("structural", "confidence", "directional")))
            if spec_v["raw"]:
                parts.append(RawFeatureSpec(feature_dir=p.feat_prefix.parent))
            aux = None
            if len(parts) == 1:
                aux = parts[0]
            elif parts:
                aux = CombinedFeatureSpec(parts)

            common = dict(br_pc=pc, history_length=HISTORY,
                          pc_bits=encoding["pc_bits"],
                          pc_hash_bits=encoding["pc_hash_bits"],
                          hash_dir_with_pc=encoding["hash_dir_with_pc"],
                          in_memory=True, aux_features=aux)
            seed_everything(args.seed)
            try:
                train_set = BranchDataset(
                    [p.dataset], index_range=rng_["train"],
                    max_samples=args.max_train_samples, seed=args.seed, **common)
                val_set = BranchDataset(
                    [p.dataset], index_range=rng_["validation"],
                    max_samples=args.max_val_samples, seed=args.seed, **common)
                test_set = BranchDataset(
                    [p.dataset], index_range=rng_["test"], **common)
            except Exception as exc:  # e.g. branch absent from a range
                log(f"  skip: {exc}")
                continue
            validation_source = "trace 60-70%"
            if len(val_set) < args.min_eval:
                # Programs have phases: a branch can be absent from the whole
                # validation window (measured on media_11: zero executions in
                # 40-80% of the trace). Fall back to the chronological tail of
                # the TRAINING range, with an H gap -- in-distribution, and
                # still never test.
                train_set = BranchDataset(
                    [p.dataset], index_range=rng_["train"],
                    time_slice=("head", args.val_fallback_fraction),
                    max_samples=args.max_train_samples, seed=args.seed, **common)
                val_set = BranchDataset(
                    [p.dataset], index_range=rng_["train"],
                    time_slice=("tail", args.val_fallback_fraction),
                    max_samples=args.max_val_samples, seed=args.seed, **common)
                validation_source = (f"train-range tail "
                                     f"{args.val_fallback_fraction:.0%}")
                log(f"  validation window empty for this branch; using the "
                    f"{validation_source}")
            sizes = (len(train_set), len(val_set), len(test_set))
            if sizes[0] < args.min_train or sizes[1] < args.min_eval or \
                    sizes[2] < args.min_eval:
                log(f"  skip: too few samples train/val/test = {sizes}")
                summary = {"trace": tid, "pc": br["pc"], "variant": variant,
                           "status": "too_few_samples", "sizes": sizes}
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / "summary.json").write_text(json.dumps(summary))
                continue

            model_kwargs = {}
            model_name = "branchnet_big"
            feature_names, kept = [], []
            if aux is not None:
                model_name = "branchnet_features"
                mask = train_set.aux_feature_mask()   # train split only
                feature_names = train_set.aux_feature_names
                kept = [n for n, m in zip(feature_names, mask) if m]
                if not mask.any():
                    log("  skip: every feature constant on train")
                    continue
                model_kwargs.update(num_features=train_set.num_aux_features,
                                    feature_mask=mask)
                if spec_v["shuffle"]:
                    for ds in (train_set, val_set, test_set):
                        r = np.random.default_rng(args.seed)
                        for s in ds.sources:
                            s.features = s.features[
                                r.permutation(s.features.shape[0])]

            config = TrainConfig(
                branch_pc=pc, model=model_name, history_length=HISTORY,
                pc_bits=encoding["pc_bits"],
                pc_hash_bits=encoding["pc_hash_bits"],
                hash_dir_with_pc=encoding["hash_dir_with_pc"],
                batch_size=args.batch_size, learning_rate=args.lr,
                max_epochs=args.max_epochs, patience=args.patience,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples, seed=args.seed,
                device=args.device)
            model = build_model(model_name, vocabulary_size=vocab,
                                history_length=HISTORY, **model_kwargs)
            training = train_branch_model(model, train_set, val_set, config,
                                          log=log)
            val = evaluate(model, val_set, device, args.batch_size)
            test = evaluate(model, test_set, device, args.batch_size)

            val_idx, test_idx = dataset_indices(val_set), dataset_indices(test_set)
            tage_val = tage_correct_on(tage_log, pc, val_idx)
            tage_test = tage_correct_on(tage_log, pc, test_idx)
            # Discordant pairs on VALIDATION: the attach decision's evidence.
            model_val = val["_correct"].astype(bool)
            val_model_only = int((model_val & ~tage_val).sum())
            val_tage_only = int((~model_val & tage_val).sum())
            comparison = summarise_against_baselines(
                test, taken_rate=test_set.taken_rate,
                tage_accuracy=float(tage_test.mean()))
            summary = {
                "trace": tid, "pc": br["pc"], "variant": variant, "status": "ok",
                "sizes": sizes, "model": model_name,
                "validation_source": validation_source,
                "features_kept": len(kept), "features_total": len(feature_names),
                "val_model_accuracy": val["accuracy"],
                "val_tage_accuracy": float(tage_val.mean()),
                "val_model_only_correct": val_model_only,
                "val_tage_only_correct": val_tage_only,
                "val_mcnemar_p": mcnemar_one_sided(val_model_only,
                                                   val_tage_only),
                "test_model_accuracy": test["accuracy"],
                "test_tage_accuracy": float(tage_test.mean()),
                "test_bias_accuracy": comparison["bias_accuracy"],
                "test_model_misses": int(test["mispredictions"]),
                "test_tage_misses": int((~tage_test).sum()),
                "test_instances": int(test["samples"]),
                "reduction_vs_tage": comparison["misprediction_reduction_vs_tage"],
                "best_epoch": training["best_epoch"],
                "epoch_seconds": float(np.mean([h["seconds"] for h in
                                                training["history"]])),
            }
            save_run(outdir, config=config, model=model, training=training,
                     test=test, comparison=comparison,
                     datasets={"sizes": sizes, "ranges": rng_,
                               "split_kind": "chronological (cross-phase, "
                                             "one run)",
                               "variant": variant,
                               "features": None if aux is None else {
                                   **aux.describe(), "columns": feature_names,
                                   "kept": kept,
                                   "shuffled_control": spec_v["shuffle"]}},
                     extra={"summary": summary})
            np.save(outdir / "test_indices.npy", test_idx)
            np.save(outdir / "val_correct.npy", val["_correct"])
            np.save(outdir / "val_indices.npy", val_idx)
            (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
            results.append(summary)
            log(f"  test model {summary['test_model_accuracy']:.4%}  TAGE "
                f"{summary['test_tage_accuracy']:.4%}  reduction "
                f"{summary['reduction_vs_tage']:+.1%}")
    return {"results": results}


def hybrid_report(work: Path, exp_root: Path, tid: str, variants: list[str],
                  alpha: float = 0.05):
    """Exact hybrid MPKI over the test range, per variant and attach policy.

    ``attach_all``: every trained model overrides TAGE on its branch.
    ``attach_gated``: only where the model beat TAGE on VALIDATION -- the
    BranchNet protocol (spec §23 step 6) -- *significantly*: one-sided exact
    McNemar p < ``alpha`` on the validation discordant pairs. A bare "more
    correct on validation" rule attaches on noise when validation holds a few
    hundred instances, which on small branches made the hybrid worse than
    TAGE. Neither policy looks at test.
    """
    p = Paths(work, tid)
    test_base = json.loads(p.test_json.read_text())
    tage = next(x for x in test_base["predictors"] if x["name"] == TAGE)
    base_miss = int(tage["mispredictions"])
    instr = int(test_base["metadata"]["metric_instructions"])
    out = {"trace": tid, "test_instructions": instr, "gate_alpha": alpha,
           "tage_test_mispredictions": base_miss,
           "tage_test_mpki": 1000 * base_miss / instr, "variants": {}}
    oracle_saved = None
    for variant in variants:
        rows = []
        for f in sorted(exp_root.glob(f"{tid}/*/{variant}/summary.json")):
            s = json.loads(f.read_text())
            if s.get("status") == "ok":
                rows.append(s)
        res = {"branches_modelled": len(rows)}
        if rows and oracle_saved is None:
            oracle_saved = sum(r["test_tage_misses"] for r in rows)
        for policy in ("attach_all", "attach_gated"):
            miss = base_miss
            attached = 0
            for r in rows:
                if policy == "attach_gated" and \
                        r.get("val_mcnemar_p", 1.0) >= alpha:
                    continue
                miss += r["test_model_misses"] - r["test_tage_misses"]
                attached += 1
            res[policy] = {"attached": attached,
                           "mispredictions": miss,
                           "mpki": 1000 * miss / instr,
                           "mpki_reduction": (base_miss - miss) / base_miss
                           if base_miss else 0.0}
        out["variants"][variant] = res
    if oracle_saved is not None:
        out["oracle_perfect_on_modelled"] = {
            "mpki": 1000 * (base_miss - oracle_saved) / instr,
            "mpki_reduction": oracle_saved / base_miss if base_miss else 0.0}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True)
    ap.add_argument("--traces", nargs="+", required=True,
                    help="CBP2025 .gz traces (globs allowed)")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--variants", default="tokens,tage_raw,tage_raw_shuffled")
    ap.add_argument("--stages", default="prep,train,hybrid")
    ap.add_argument("--prep-jobs", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-exec", type=int, default=3000,
                    help="min executions of a branch in the TRAIN range")
    ap.add_argument("--min-miss", type=int, default=100,
                    help="min TAGE misses of a branch in the TRAIN range")
    ap.add_argument("--min-train", type=int, default=2000)
    ap.add_argument("--min-eval", type=int, default=300)
    ap.add_argument("--val-fallback-fraction", type=float, default=0.15,
                    help="share of the train range used as validation when a "
                         "branch is absent from the validation window")
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-samples", type=int, default=1_000_000)
    ap.add_argument("--max-val-samples", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    traces = sorted({Path(t) for pat in args.traces for t in glob.glob(pat)},
                    key=lambda t: (len(trace_id_of(t)), trace_id_of(t)))
    if not traces:
        ap.error("no traces matched")
    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v}; choose from {list(VARIANTS)}")
    work = args.work or REPO / "data" / "cbp2025" / args.category
    exp_root = REPO / "experiments" / "cbp2025" / args.category
    res_root = REPO / "results" / "cbp2025" / args.category
    res_root.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    runlog = open(res_root / "pipeline.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        runlog.write(line + "\n")
        runlog.flush()

    log(f"=== {args.category}: {len(traces)} traces, variants {variants}")
    if "prep" in stages:
        with ProcessPoolExecutor(max_workers=args.prep_jobs) as pool:
            futs = {pool.submit(prepare_trace, str(t), str(work), args.top_k,
                                args.min_exec, args.min_miss): t
                    for t in traces}
            for fut, t in futs.items():
                try:
                    r = fut.result()
                    log(f"prep {r['trace']}: {r['status']}")
                except Exception:
                    log(f"prep {trace_id_of(t)} FAILED\n{traceback.format_exc()}")

    for t in traces:
        tid = trace_id_of(t)
        p = Paths(work, tid)
        if not p.dataset.exists():
            continue
        if "train" in stages:
            try:
                train_trace(work, exp_root, tid, variants, args, log)
            except Exception:
                log(f"train {tid} FAILED\n{traceback.format_exc()}")
                continue
        if "hybrid" in stages:
            try:
                rep = hybrid_report(work, exp_root, tid, variants)
                (res_root / f"{tid}.hybrid.json").write_text(
                    json.dumps(rep, indent=2))
                parts = [f"TAGE {rep['tage_test_mpki']:.3f} MPKI"]
                for v, r in rep["variants"].items():
                    parts.append(f"{v} {r['attach_gated']['mpki']:.3f} "
                                 f"({r['attach_gated']['mpki_reduction']:+.2%})")
                log(f"hybrid {tid}: " + " | ".join(parts))
            except Exception:
                log(f"hybrid {tid} FAILED\n{traceback.format_exc()}")
    log("=== done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
