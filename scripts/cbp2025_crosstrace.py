#!/usr/bin/env python3
"""Cross-trace (cross-input) BranchNet-big + confidence-based hybrid.

    python scripts/cbp2025_crosstrace.py --group web_g0
    python scripts/cbp2025_crosstrace.py --group web_g0 --max-train-traces 4

The BranchNet / Tarsa protocol on CBP2025: a group (from
tools/cbp2025_groups.py) holds many traces of ONE program on different inputs.
Its traces are split, by trace, into train / validation / test:

  setup  split; select H2P branches on TRAIN traces only (spec §28; Tarsa's
         rule: hot and hard in >= 3 train traces); per-trace TAGE logs, raw
         features and HDF5 datasets for that branch list
  train  one BranchNet-big per branch per variant, pooled over train traces;
         saves per-execution logits on validation and test
  eval   hybrid MPKI on the WHOLE test traces under three policies
         (src/evaluation/arbiter.py): BranchNet per-branch gate, our
         per-execution confidence arbiter, and the oracle ceiling

Variant names: ``<features>[@a<alpha>][#t<n>]`` -- e.g. ``tage_raw@a3`` weights
the training loss by 1 + 3*[TAGE not high-confidence]; ``#t4`` trains on the
first 4 train traces only (data-volume curve).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.cbp2025_pipeline import (  # noqa: E402
    CONVERT, HISTORY, RUN_PREDICTORS, TAGE, run)

FEATURES = {
    "tokens": dict(tage=False, raw=None, shuffle=False),
    "tage": dict(tage=True, raw=None, shuffle=False),
    "raw_path": dict(tage=False, raw=("path",), shuffle=False),
    "raw_committed": dict(tage=False, raw=("committed",), shuffle=False),
    "tage_raw": dict(tage=True, raw=("path", "inst_mix", "committed"),
                     shuffle=False),
    "tage_raw_shuffled": dict(tage=True, raw=("path", "inst_mix", "committed"),
                              shuffle=True),
    # CONTROL: training labels permuted; test must land on the bias floor.
    "tokens_labelshuffle": dict(tage=False, raw=None, shuffle=False,
                                label_shuffle=True),
}

# Dataflow-producer + instruction-history features (tools/cbp2025_producers
# .cpp, chosen by tools/cbp2025_feature_screen.py), at a given commit lag.
# 1024 is the guaranteed-safe lag; smaller lags are an assumption.
for _lag in (1024, 256):
    _extra = [(f"prodfeat_L{_lag}", ("producers", "inst_hist"))]
    FEATURES[f"tage_raw_prod{_lag}"] = dict(
        tage=True, raw=("path", "inst_mix", "committed"), shuffle=False,
        extra_raw=_extra)
    FEATURES[f"tage_raw_prod{_lag}_shuffled"] = dict(
        tage=True, raw=("path", "inst_mix", "committed"), shuffle=True,
        extra_raw=_extra)
    FEATURES[f"tage_raw_prod{_lag}_labelshuffle"] = dict(
        tage=True, raw=("path", "inst_mix", "committed"), shuffle=False,
        extra_raw=_extra, label_shuffle=True)


def parse_variant(name: str) -> dict:
    base, alpha, ntr = name, 0.0, None
    if "#t" in base:
        base, n = base.split("#t")
        ntr = int(n)
    if "@a" in base:
        base, a = base.split("@a")
        alpha = float(a)
    if base not in FEATURES:
        raise ValueError(f"unknown feature set {base!r}; have {list(FEATURES)}")
    return {"features": base, "alpha": alpha, "train_traces": ntr,
            **FEATURES[base]}


class GroupPaths:
    def __init__(self, group: str, fold: int, suffix: str = ""):
        tag = (group if fold == 0 else f"{group}_f{fold}") + suffix
        self.tag = tag
        self.root = REPO / "data" / "cbp2025" / "xt" / tag
        self.split = self.root / "split.json"
        self.h2p = self.root / "h2p.txt"
        self.h2p_json = self.root / "h2p.json"
        self.logs = self.root / "logs"
        self.features = self.root / "features"
        self.datasets = self.root / "datasets"
        self.exp = REPO / "experiments" / "cbp2025_xt" / tag
        self.results = REPO / "results" / "cbp2025" / "crosstrace"


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def make_split(members: list[str], group: str, fold: int, val_frac: float,
               test_frac: float) -> dict:
    """Seeded permutation by trace. Folds rotate which slice is test."""
    seed = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)
    order = list(np.random.default_rng(seed).permutation(sorted(members)))
    n = len(order)
    n_test = max(1, round(test_frac * n))
    n_val = max(1, round(val_frac * n))
    # Fold k rotates the permutation by k*n_test so test sets are disjoint.
    shift = (fold * n_test) % n
    order = order[shift:] + order[:shift]
    split = {"test": order[:n_test], "validation": order[n_test:n_test + n_val],
             "train": order[n_test + n_val:]}
    roles = [t for r in split.values() for t in r]
    assert len(roles) == len(set(roles)) == n, "a trace landed in two roles"
    return split


def pooled_rows(csv_paths: list[Path]) -> dict[str, dict]:
    """Per-PC rows summed over traces; fraction columns execution-weighted."""
    frac = ["tage_hit_bank_mean", "tage_bimodal_frac", "tage_conf_high_frac",
            "tage_conf_med_frac", "tage_conf_low_frac", "tage_alt_disagree_frac"]
    acc: dict[str, dict] = {}
    for path in csv_paths:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                ex = int(row["executions"])
                a = acc.setdefault(row["pc"], {"executions": 0, "taken": 0,
                                               "misses": 0,
                                               **{f: 0.0 for f in frac}})
                miss = int(row[f"{TAGE}_mispredictions"])
                a["executions"] += ex
                a["taken"] += int(row["taken"])
                a["misses"] += miss
                for f in frac:
                    a[f] += float(row[f]) * ex
                a.setdefault("_rows", []).append((ex, miss))
    return acc


def select_branches(train: list[str], work: Path, top_k: int, min_exec: int,
                    min_miss: int, min_traces: int) -> list[dict]:
    from tools.classify_h2p import classify

    csvs = [work / f"{t}.full_per_pc.csv" for t in train]
    acc = pooled_rows(csvs)
    need = min(min_traces, len(train))
    chosen = []
    for pc, a in acc.items():
        eligible = sum(ex >= min_exec and miss >= min_miss
                       for ex, miss in a["_rows"])
        if eligible < need:
            continue
        ex = a["executions"]
        row = {"executions": ex, f"{TAGE}_mispredictions": a["misses"],
               "taken_rate": a["taken"] / ex,
               "tage_hit_bank_mean": a["tage_hit_bank_mean"] / ex,
               "tage_bimodal_frac": a["tage_bimodal_frac"] / ex,
               "tage_conf_high_frac": a["tage_conf_high_frac"] / ex,
               "tage_conf_low_frac": a["tage_conf_low_frac"] / ex,
               "tage_alt_disagree_frac": a["tage_alt_disagree_frac"] / ex}
        category, reason = classify(row)
        chosen.append({"pc": pc, "train_executions": ex,
                       "train_tage_misses": a["misses"],
                       "train_tage_accuracy": 1 - a["misses"] / ex,
                       "eligible_in_train_traces": eligible,
                       "category": category, "category_reason": reason})
    chosen.sort(key=lambda r: -r["train_tage_misses"])
    return chosen[:top_k]


def prepare_member(tid: str, role: str, work: str, root: str) -> str:
    """TAGE log, raw features and HDF5 for one trace (worker process)."""
    from src.dataset import build_trace_dataset

    work, root = Path(work), Path(root)
    gp_logs, gp_feat, gp_ds = root / "logs", root / "features", root / "datasets"
    sbbt = work / f"{tid}.sbbt.zst"
    meta = json.loads((work / f"{tid}.meta.json").read_text())
    trace_gz = REPO / meta["trace"]
    pcs = root / "h2p.txt"
    logf = open(root / "setup.log", "a")

    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    if not (gp_logs / f"{tid}.json").exists():
        run([RUN_PREDICTORS, sbbt, "--predictors", f"always_taken,{TAGE}",
             "--log", gp_logs / tid, "--log-pcs", pcs,
             "--json", gp_logs / f"{tid}.summary.json"], log)
    if not (gp_feat / f"{tid}.feat.json").exists():
        run([CONVERT, trace_gz, "--feature-pcs", pcs,
             "--features", gp_feat / f"{tid}.feat"], log)
    ds = gp_ds / f"{tid}.hdf5"
    if not ds.exists():
        stats = build_trace_dataset(
            sbbt, ds, [int(l, 16) for l in pcs.read_text().split()],
            split_role=role, extra_attrs={"group": root.name})
        if stats.missing_pcs:
            log(f"{tid}: {len(stats.missing_pcs)} selected PCs absent")
    return tid


def _pc_counts(path: Path) -> dict[str, int]:
    with open(path, newline="") as fh:
        return {r["pc"]: int(r["executions"]) for r in csv.DictReader(fh)}


def _cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], float)
    vb = np.array([b.get(k, 0) for k in keys], float)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def pool_candidates(work: Path, split: dict, h2p: set[str], *,
                    min_shared: int, max_sim: float, hot: int = 1000):
    """Extra TRAINING traces for a group: other recordings of the program.

    A trace outside the group qualifies if at least ``min_shared`` of the
    group's selected branches are hot in it (same program, other input), and
    it is NOT a near-copy of any validation or test trace (cosine of per-branch
    execution counts >= ``max_sim``) -- a near-copy would leak the held-out
    behaviour into training. Validation/test membership never changes.
    """
    members = {t for r in ("train", "validation", "test") for t in split[r]}
    held = {t: _pc_counts(work / f"{t}.full_per_pc.csv")
            for t in split["validation"] + split["test"]}
    out, rejected = [], []
    for f in sorted(work.glob("*.full_per_pc.csv")):
        t = f.name.split(".")[0]
        if t in members:
            continue
        counts = _pc_counts(f)
        shared = sum(counts.get(pc, 0) >= hot for pc in h2p)
        if shared < min_shared:
            continue
        sim = max(_cosine(counts, v) for v in held.values())
        (rejected if sim >= max_sim else out).append(
            {"trace": t, "shared_hard_branches": shared,
             "max_similarity_to_heldout": round(sim, 4)})
    return out, rejected


def setup_pooled(args, gp: GroupPaths, base: GroupPaths, group: dict,
                 log) -> dict:
    """Same split, branches and held-out traces as ``base``; more train traces."""
    import shutil
    work = REPO / "data" / "cbp2025" / group["category"]
    if not base.split.exists():
        raise RuntimeError(f"run the base group {base.tag} first")
    for d in (gp.root, gp.logs, gp.features, gp.datasets):
        d.mkdir(parents=True, exist_ok=True)
    base_split = json.loads(base.split.read_text())
    if not gp.split.exists():
        h2p = {b["pc"] for b in json.loads(base.h2p_json.read_text())["branches"]}
        extra, rejected = pool_candidates(
            work, base_split, h2p, min_shared=args.pool_min_shared,
            max_sim=args.pool_max_sim)
        split = {**base_split,
                 "train": base_split["train"] + [e["trace"] for e in extra],
                 "pooled_from": base.tag, "pool_extra": extra,
                 "pool_rejected_near_heldout": rejected}
        gp.split.write_text(json.dumps(split, indent=2))
        # Branch list is the base's: selected on the ORIGINAL train traces.
        shutil.copy(base.h2p_json, gp.h2p_json)
        shutil.copy(base.h2p, gp.h2p)
    split = json.loads(gp.split.read_text())
    # Reuse the base's per-trace artefacts; build only the new ones.
    for t in base_split["train"] + base_split["validation"] + base_split["test"]:
        for src, dst in ((base.logs / f"{t}.bin", gp.logs / f"{t}.bin"),
                         (base.logs / f"{t}.json", gp.logs / f"{t}.json"),
                         (base.logs / f"{t}.summary.json",
                          gp.logs / f"{t}.summary.json"),
                         (base.features / f"{t}.feat.bin",
                          gp.features / f"{t}.feat.bin"),
                         (base.features / f"{t}.feat.json",
                          gp.features / f"{t}.feat.json"),
                         (base.datasets / f"{t}.hdf5", gp.datasets / f"{t}.hdf5")):
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
    extra = [e["trace"] for e in split["pool_extra"]]
    log(f"[{gp.tag}] pooled: +{len(extra)} train traces "
        f"({len(split['pool_rejected_near_heldout'])} rejected as near-copies "
        f"of held-out traces); train/val/test = {len(split['train'])}/"
        f"{len(split['validation'])}/{len(split['test'])}")
    with ProcessPoolExecutor(max_workers=args.prep_jobs) as pool:
        for f in [pool.submit(prepare_member, t, "train", str(work),
                              str(gp.root)) for t in extra]:
            f.result()
    log(f"[{gp.tag}] setup done")
    return split


def setup(args, gp: GroupPaths, group: dict, log) -> dict:
    work = REPO / "data" / "cbp2025" / group["category"]
    for d in (gp.root, gp.logs, gp.features, gp.datasets):
        d.mkdir(parents=True, exist_ok=True)
    if not gp.split.exists() and args.split_json:
        # Explicit split (e.g. another study's exact train/val/test traces).
        given = json.loads(Path(args.split_json).read_text())
        split = {r: list(given[r]) for r in ("train", "validation", "test")}
        roles = [t for r in split.values() for t in r]
        assert len(roles) == len(set(roles)), "a trace is in two roles"
        gp.split.write_text(json.dumps({"group": group["name"], "fold": args.fold,
                                        "split_kind": "cross-trace (explicit)",
                                        "source": str(args.split_json), **split},
                                       indent=2))
    if not gp.split.exists():
        split = make_split(group["members"], group["name"], args.fold,
                           args.val_frac, args.test_frac)
        gp.split.write_text(json.dumps({"group": group["name"],
                                        "fold": args.fold,
                                        "split_kind": "cross-trace", **split},
                                       indent=2))
    split = json.loads(gp.split.read_text())
    if not gp.h2p_json.exists():
        chosen = select_branches(split["train"], work, args.top_k,
                                 args.min_exec, args.min_miss, args.min_traces)
        gp.h2p_json.write_text(json.dumps({
            "rule": f"train traces only; hot (>={args.min_exec}) and hard "
                    f"(>={args.min_miss} TAGE misses) in >= {args.min_traces} "
                    f"train traces; top {args.top_k} by pooled TAGE misses",
            "branches": chosen}, indent=2))
        gp.h2p.write_text("".join(f"{b['pc']}\n" for b in chosen))
    chosen = json.loads(gp.h2p_json.read_text())["branches"]
    log(f"[{gp.tag}] split train/val/test = {len(split['train'])}/"
        f"{len(split['validation'])}/{len(split['test'])}; "
        f"{len(chosen)} branches")
    if not chosen:
        return split
    jobs = [(t, r) for r in ("train", "validation", "test") for t in split[r]]
    with ProcessPoolExecutor(max_workers=args.prep_jobs) as pool:
        futs = [pool.submit(prepare_member, t, r, str(work), str(gp.root))
                for t, r in jobs]
        for f in futs:
            f.result()
    log(f"[{gp.tag}] setup done")
    return split


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

class TageConfidenceWeights:
    """Loss weight ``1 + alpha * [TAGE not high-confidence]`` per execution.

    TAGE's confidence is computed at prediction time, so this is an input-side
    signal: the weighting depends on what TAGE knew, never on the label.
    """

    def __init__(self, log_dir: Path, alpha: float):
        self.log_dir, self.alpha = Path(log_dir), float(alpha)

    def load(self, dataset_path, br_pc, indices):
        conf = tage_info(self.log_dir, dataset_path, br_pc, indices)["conf"]
        return 1.0 + self.alpha * (conf != 0).astype(np.float32)


def tage_info(log_dir: Path, dataset_path: Path, br_pc: int,
              indices: np.ndarray) -> dict:
    """TAGE correctness, confidence level and the actual direction, exactly
    on these executions (exact branch_index join; raises otherwise)."""
    import h5py
    from src.evaluation import load_prediction_log
    from src.evaluation.arbiter import HIGH, LOW, MEDIUM

    with h5py.File(dataset_path, "r") as fh:
        tid = str(fh.attrs["trace_id"])
    sub = load_prediction_log(log_dir / tid).for_pc(br_pc)
    slot = np.searchsorted(sub.branch_index, indices)
    if slot.size and (slot.max() >= len(sub) or
                      not np.array_equal(sub.branch_index[slot], indices)):
        raise RuntimeError(f"TAGE log for {tid} misses executions of "
                           f"{br_pc:#x}")
    conf = np.full(indices.size, MEDIUM, dtype=np.int8)
    conf[sub.flag("high_confidence")[slot]] = HIGH
    conf[sub.flag("low_confidence")[slot]] = LOW
    return {"correct": ~sub.mispredicted(TAGE)[slot], "conf": conf,
            "taken": sub.taken[slot], "trace": tid}


def split_arrays(ds, test_or_val: dict, log_dir: Path, br_pc: int,
                 *, check: bool = True) -> dict:
    """Per-execution arrays in dataset order (sources concatenated)."""
    parts = {"logits": test_or_val["_logits"], "correct": test_or_val["_correct"]}
    tc, conf, taken, idx, trace = [], [], [], [], []
    for s in ds.sources:
        info = tage_info(log_dir, s.path, br_pc, s.indices)
        tc.append(info["correct"]); conf.append(info["conf"])
        taken.append(info["taken"]); idx.append(s.indices)
        trace.append(np.full(s.indices.size, info["trace"]))
    out = {"logits": parts["logits"].astype(np.float32),
           "tage_correct": np.concatenate(tc), "tage_conf": np.concatenate(conf),
           "labels": np.concatenate(taken).astype(np.uint8),
           "indices": np.concatenate(idx), "trace": np.concatenate(trace)}
    # Cross-check: the model's correctness implied by logits+labels must equal
    # what evaluate() measured. Catches any order mismatch between the two.
    # (Skipped only for a label-shuffle control's validation set, whose
    # labels are permuted on purpose; ``labels`` here are always the true ones.)
    implied = (out["logits"] > 0) == (out["labels"] > 0)
    if check and not np.array_equal(implied.astype(np.uint8), parts["correct"]):
        raise RuntimeError("logits/labels order disagrees with evaluate()")
    return out


def release_device_memory() -> None:
    """Free GPU memory between runs.

    PyTorch's MPS allocator keeps freed buffers cached. Over dozens of runs in
    one process that cache grew to ~30 GB of wired memory on a 36 GB machine,
    pushing everything else into swap (epochs went from ~70 s to ~2,000 s).
    Dropping the run's objects and emptying the cache after each run keeps the
    footprint flat.
    """
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_group(args, gp: GroupPaths, split: dict, variants: list[str], log):
    from src.dataset import (AuxFeatureSpec, BranchDataset,
                             CombinedFeatureSpec, RawFeatureSpec,
                             as_signed_pc, vocabulary_size)
    from src.models import build_model, dataset_config
    from src.training import (TrainConfig, evaluate, resolve_device,
                              save_run, seed_everything, train_branch_model)

    chosen = json.loads(gp.h2p_json.read_text())["branches"]
    if args.branches:
        chosen = chosen[:args.branches]
    files = {r: [gp.datasets / f"{t}.hdf5" for t in split[r]]
             for r in ("train", "validation", "test")}
    enc = dataset_config("branchnet_big")
    vocab = vocabulary_size(**enc)
    device = resolve_device(args.device)

    for bi, br in enumerate(chosen):
        pc = as_signed_pc(int(br["pc"], 16))
        for name in variants:
            v = parse_variant(name)
            if v.get("label_shuffle") and bi >= args.control_branches:
                continue
            outdir = gp.exp / br["pc"] / name
            if (outdir / "summary.json").exists() and not args.retrain:
                continue
            log(f"[{gp.tag}] {br['pc']} ({br['category']}) {name}")
            parts = []
            if v["tage"]:
                parts.append(AuxFeatureSpec(
                    log_dir=gp.logs,
                    groups=("structural", "confidence", "directional")))
            if v["raw"]:
                parts.append(RawFeatureSpec(feature_dir=gp.features,
                                            groups=v["raw"]))
            for sub, groups_ in v.get("extra_raw", ()):
                parts.append(RawFeatureSpec(feature_dir=gp.root / sub,
                                            groups=groups_))
            aux = (None if not parts else parts[0] if len(parts) == 1
                   else CombinedFeatureSpec(parts))
            weights = (TageConfidenceWeights(gp.logs, v["alpha"])
                       if v["alpha"] > 0 else None)
            train_files = files["train"][:v["train_traces"]] \
                if v["train_traces"] else files["train"]
            common = dict(br_pc=pc, history_length=HISTORY,
                          pc_bits=enc["pc_bits"],
                          pc_hash_bits=enc["pc_hash_bits"],
                          hash_dir_with_pc=enc["hash_dir_with_pc"],
                          in_memory=True, aux_features=aux,
                          allow_missing=True)
            seed_everything(args.seed)
            outdir.mkdir(parents=True, exist_ok=True)
            try:
                train_set = BranchDataset(
                    train_files, expected_split_role="train",
                    max_samples=args.max_train_samples, seed=args.seed,
                    sample_weights=weights, **common)
                val_set = BranchDataset(
                    files["validation"], expected_split_role="validation",
                    max_samples=args.max_val_samples, seed=args.seed, **common)
                test_set = BranchDataset(files["test"],
                                         expected_split_role="test", **common)
            except KeyError as exc:
                log(f"  skip: {exc}")
                (outdir / "summary.json").write_text(json.dumps(
                    {"status": "missing", "pc": br["pc"], "variant": name}))
                continue
            validation_source = "validation traces"
            if len(val_set) < args.min_samples and len(train_set) > 0:
                # The branch is (nearly) absent from every validation trace --
                # on infra_g0, 5 of 16 branches by the luck of the split. Hold
                # out a share of the TRAINING traces that contain it as this
                # branch's validation: still whole-trace separation from the
                # traces the model trains on, and test is untouched.
                have = [s_.path for s_ in train_set.sources]
                k = max(1, round(args.val_fallback_frac * len(have)))
                order = np.random.default_rng(args.seed).permutation(len(have))
                held = [have[i] for i in sorted(order[:k])]
                keep = [f for f in have if f not in held]
                if keep:
                    train_set = BranchDataset(
                        keep, expected_split_role="train",
                        max_samples=args.max_train_samples, seed=args.seed,
                        sample_weights=weights, **common)
                    val_set = BranchDataset(
                        held, expected_split_role="train",
                        max_samples=args.max_val_samples, seed=args.seed,
                        **common)
                    validation_source = ("held-out train traces: " +
                                         ",".join(f.stem for f in held))
                    log(f"  validation traces lack this branch; holding out "
                        f"{k} of {len(have)} train traces that contain it")
            sizes = (len(train_set), len(val_set), len(test_set))
            if min(sizes) < args.min_samples:
                log(f"  skip: too few samples {sizes}")
                (outdir / "summary.json").write_text(json.dumps(
                    {"status": "too_few_samples", "sizes": sizes,
                     "pc": br["pc"], "variant": name}))
                continue

            model_name, model_kwargs, kept = "branchnet_big", {}, []
            if aux is not None:
                model_name = "branchnet_features"
                mask = train_set.aux_feature_mask()          # train split only
                kept = [n for n, m in zip(train_set.aux_feature_names, mask) if m]
                if not mask.any():
                    log("  skip: all features constant on train")
                    continue
                model_kwargs = dict(num_features=train_set.num_aux_features,
                                    feature_mask=mask)
                if v["shuffle"]:
                    r = np.random.default_rng(args.seed)
                    for ds in (train_set, val_set, test_set):
                        for s in ds.sources:
                            s.features = s.features[
                                r.permutation(s.features.shape[0])]
            if v.get("label_shuffle"):
                train_set.shuffle_labels(seed=args.seed)
                val_set.shuffle_labels(seed=args.seed + 1)

            config = TrainConfig(
                branch_pc=pc, model=model_name, history_length=HISTORY,
                pc_bits=enc["pc_bits"], pc_hash_bits=enc["pc_hash_bits"],
                hash_dir_with_pc=enc["hash_dir_with_pc"],
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
            val_arr = split_arrays(val_set, val, gp.logs, pc,
                                   check=not v.get("label_shuffle"))
            test_arr = split_arrays(test_set, test, gp.logs, pc)
            np.savez_compressed(outdir / "val.npz", **val_arr)
            np.savez_compressed(outdir / "test.npz", **test_arr)
            summary = {
                "status": "ok", "pc": br["pc"], "category": br["category"],
                "variant": name, **{k: v[k] for k in ("features", "alpha",
                                                      "train_traces")},
                "sizes": sizes, "features_kept": len(kept),
                "validation_source": validation_source,
                "train_samples_available": train_set.total_available,
                "val_model_accuracy": val["accuracy"],
                "val_tage_accuracy": float(val_arr["tage_correct"].mean()),
                "test_model_accuracy": test["accuracy"],
                "test_tage_accuracy": float(test_arr["tage_correct"].mean()),
                "test_bias_accuracy": max(test_set.taken_rate,
                                          1 - test_set.taken_rate),
                "best_epoch": training["best_epoch"],
                "epoch_seconds": float(np.mean(
                    [h["seconds"] for h in training["history"]])),
            }
            save_run(outdir, config=config, model=model, training=training,
                     test=test, comparison={}, datasets={
                         "sizes": sizes, "split": split, "variant": name,
                         "features": None if aux is None else
                         {**aux.describe(), "kept": kept}},
                     extra={"summary": summary})
            (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
            log(f"  test model {summary['test_model_accuracy']:.4%}  TAGE "
                f"{summary['test_tage_accuracy']:.4%}  bias "
                f"{summary['test_bias_accuracy']:.4%}")
            del model, train_set, val_set, test_set, val, test, training
            release_device_memory()


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

def evaluate_group(args, gp: GroupPaths, split: dict, variants: list[str],
                   log) -> dict:
    from src.evaluation.arbiter import (Executions, branch_gate,
                                        fit_confidence_rule, score)
    from src.evaluation.learned_arbiter import (arbiter_inputs,
                                                fit_learned_arbiter,
                                                score_mask)

    base_miss = instr = 0
    for t in split["test"]:
        s = json.loads((gp.logs / f"{t}.summary.json").read_text())
        tage = next(p for p in s["predictors"] if p["name"] == TAGE)
        base_miss += int(tage["mispredictions"])
        instr += int(s["metadata"]["metric_instructions"])
    report = {"group": gp.tag, "test_traces": split["test"],
              "test_instructions": instr, "tage_misses": base_miss,
              "tage_mpki": 1000 * base_miss / instr, "variants": {}}

    for name in variants:
        per_branch, totals = [], {}
        for f in sorted(gp.exp.glob(f"*/{name}/summary.json")):
            s = json.loads(f.read_text())
            if s.get("status") != "ok":
                continue
            d = f.parent
            vz, tz = np.load(d / "val.npz"), np.load(d / "test.npz")
            keys = ("logits", "labels", "tage_correct", "tage_conf")
            val = Executions(**{k: vz[k] for k in keys})
            tst = Executions(**{k: tz[k] for k in keys})
            rules = {"branch": branch_gate(val),
                     "confidence": fit_confidence_rule(val)}
            row = {"pc": s["pc"], "category": s["category"]}
            for pol, rule in rules.items():
                sc = score(tst, rule)
                row[pol] = {**sc, "rule": [rule.conf_set, rule.tau],
                            "val_p": rule.val_p}
            if args.learned:
                pc_u = int(s["pc"], 16)
                Xv = arbiter_inputs(val, gp.logs, pc_u, vz["trace"],
                                    vz["indices"])
                Xt = arbiter_inputs(tst, gp.logs, pc_u, tz["trace"],
                                    tz["indices"])
                for kind in ("lr", "gbdt"):
                    arb = fit_learned_arbiter(val, Xv, vz["trace"], kind,
                                              seed=args.seed)  # VAL only
                    row[f"learned_{kind}"] = {
                        **score_mask(tst, arb.use_model(tst, Xt)),
                        "kept": arb.model is not None, "oof_p": arb.oof_p,
                        "oof_gain": arb.oof_gain, "rows": arb.trained_on}
                    if kind == "lr":
                        # The learned arbiter validates itself out-of-fold BY
                        # TRACE, so it needs >= 2 validation traces. Where a
                        # branch has only one (fallback validation), use the
                        # threshold rule, which needs no folds.
                        one_trace = np.unique(vz["trace"]).size < 2
                        row["learned_lr_fb"] = (
                            dict(score(tst, rules["confidence"]),
                                 source="threshold (1 validation trace)")
                            if one_trace else
                            dict(row["learned_lr"], source="learned"))
            row["oracle"] = score(tst, "oracle")
            row["attach_all"] = score(tst, _always(tst))
            per_branch.append(row)
        res = {"branches": len(per_branch)}
        pols = ["branch", "confidence", "oracle", "attach_all"]
        if args.learned:
            pols += ["learned_lr", "learned_gbdt", "learned_lr_fb"]
        for pol in pols:
            saved = sum(r[pol]["tage_misses"] - r[pol]["hybrid_misses"]
                        for r in per_branch)
            res[pol] = {"hybrid_mpki": 1000 * (base_miss - saved) / instr,
                        "mpki_reduction": saved / base_miss if base_miss else 0,
                        "branches_used": sum(r[pol]["overridden"] > 0
                                             for r in per_branch)}
        cats = {}
        for r in per_branch:
            c = cats.setdefault(r["category"], {"branches": 0, "tage_misses": 0,
                                                "confidence_saved": 0,
                                                "branch_saved": 0})
            c["branches"] += 1
            c["tage_misses"] += r["confidence"]["tage_misses"]
            c["confidence_saved"] += (r["confidence"]["tage_misses"]
                                      - r["confidence"]["hybrid_misses"])
            c["branch_saved"] += (r["branch"]["tage_misses"]
                                  - r["branch"]["hybrid_misses"])
        res["by_category"] = cats
        res["per_branch"] = per_branch
        report["variants"][name] = res
        log(f"[{gp.tag}] {name:24s} TAGE {report['tage_mpki']:.3f} | "
            f"branch-gate {res['branch']['mpki_reduction']:+.2%} | "
            f"confidence {res['confidence']['mpki_reduction']:+.2%} | "
            + (f"learned-lr {res['learned_lr']['mpki_reduction']:+.2%} | "
               f"learned-lr+fallback {res['learned_lr_fb']['mpki_reduction']:+.2%} | "
               f"learned-gbdt {res['learned_gbdt']['mpki_reduction']:+.2%} | "
               if args.learned else "") +
            f"oracle {res['oracle']['mpki_reduction']:+.2%}")
    gp.results.mkdir(parents=True, exist_ok=True)
    (gp.results / f"{gp.tag}{args.results_tag}.json").write_text(
        json.dumps(report, indent=2))
    return report


def _always(ex):
    from src.evaluation.arbiter import Rule
    return Rule("all", 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--stages", default="setup,train,eval")
    ap.add_argument("--variants", default="tokens,tage_raw,tage_raw_shuffled")
    ap.add_argument("--max-train-traces", type=int, nargs="*", default=None,
                    help="data-volume curve: also run each variant with only "
                         "the first N train traces")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--min-exec", type=int, default=3000)
    ap.add_argument("--min-miss", type=int, default=100)
    ap.add_argument("--min-traces", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=500)
    ap.add_argument("--val-fallback-frac", type=float, default=0.25,
                    help="share of a branch's train traces held out as its "
                         "validation when the validation traces lack it")
    ap.add_argument("--branches", type=int, default=None,
                    help="train only the top N selected branches")
    ap.add_argument("--control-branches", type=int, default=2,
                    help="label-shuffle control runs on this many top branches")
    ap.add_argument("--prep-jobs", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-samples", type=int, default=1_000_000)
    ap.add_argument("--max-val-samples", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--split-json", default=None,
                    help="explicit split {category, train, validation, test}; "
                         "--group then only names the run")
    ap.add_argument("--pool", action="store_true",
                    help="pooled coverage: same held-out traces and branches "
                         "as the base group, plus extra TRAIN traces of the "
                         "same program from outside the group")
    ap.add_argument("--pool-min-shared", type=int, default=12,
                    help="an extra trace must have this many of the group's "
                         "selected branches hot")
    ap.add_argument("--pool-max-sim", type=float, default=0.98,
                    help="reject extra traces at least this similar to any "
                         "validation/test trace")
    ap.add_argument("--no-learned", dest="learned", action="store_false",
                    help="skip the learned arbiters in the eval stage")
    ap.add_argument("--results-tag", default="",
                    help="suffix for the eval results file")
    args = ap.parse_args()

    groups = json.loads((REPO / "results" / "cbp2025" / "groups.json").read_text())
    group = next((g for g in groups["groups"] if g["name"] == args.group), None)
    if group is None and args.split_json:
        given = json.loads(Path(args.split_json).read_text())
        group = {"name": args.group, "category": given["category"],
                 "members": given["train"] + given["validation"] + given["test"]}
    if group is None:
        ap.error(f"no group {args.group} in results/cbp2025/groups.json")
    variants = args.variants.split(",")
    for n in args.max_train_traces or []:
        variants += [f"{v}#t{n}" for v in args.variants.split(",")
                     if "labelshuffle" not in v]
    for v in variants:
        parse_variant(v)

    gp = GroupPaths(args.group, args.fold, "_pool" if args.pool else "")
    gp.results.mkdir(parents=True, exist_ok=True)
    runlog = open(gp.results / f"{gp.tag}.log", "a")

    def log(msg):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        runlog.write(line + "\n")
        runlog.flush()

    stages = set(args.stages.split(","))
    try:
        if "setup" in stages:
            split = (setup_pooled(args, gp, GroupPaths(args.group, args.fold),
                                  group, log) if args.pool
                     else setup(args, gp, group, log))
        else:
            split = json.loads(gp.split.read_text())
        if "train" in stages:
            train_group(args, gp, split, variants, log)
        if "eval" in stages:
            evaluate_group(args, gp, split, variants, log)
    except Exception:
        log(f"FAILED\n{traceback.format_exc()}")
        return 1
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
