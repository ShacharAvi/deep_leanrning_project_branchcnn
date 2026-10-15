"""CBP2025 conversion, fetch-safe features, and the chronological split.

The converter is the one place where a leak could enter the new pipeline
silently, so the central tests here build a synthetic trace with a known
answer and check that (a) the SBBT stream is exactly what the CBP2025 records
imply and (b) no feature can see a value the predictor could not have had at
fetch.
"""

from __future__ import annotations

import gzip
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.dataset import (
    AuxFeatureError,
    BranchDataset,
    RawFeatureSpec,
    as_signed_pc,
    build_trace_dataset,
    load_raw_features,
)
from src.evaluation.prediction_log import as_unsigned_pc
from src.traces import SbbtReader

REPO = Path(__file__).resolve().parent.parent
CONVERT = REPO / "build" / "cbp2025_convert"

pytestmark = pytest.mark.skipif(not CONVERT.exists(),
                                reason="build/cbp2025_convert not built")

ALU, LOAD, STORE, COND, JMP, IND, FP, SLOW, CALL, CALLI, RET = \
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11
FLAGS = 64


def rec(pc, cls, *, taken=False, target=None, ins=(), outs=()):
    """One CBP2025 trace record. ``outs`` is [(reg, value)], int regs only."""
    b = struct.pack("<QB", pc, cls)
    if cls in (LOAD, STORE):
        b += struct.pack("<QBB", 0x1000, 8, 0)
        if cls == STORE:
            b += struct.pack("<B", 0)
    if cls in (COND, JMP, IND, CALL, CALLI, RET):
        b += struct.pack("<B", int(taken))
        if taken:
            b += struct.pack("<Q", target)
    b += struct.pack("<B", len(ins)) + bytes(ins)
    b += struct.pack("<B", len(outs)) + bytes(r for r, _ in outs)
    for _, v in outs:
        b += struct.pack("<Q", v & 0xFFFF_FFFF_FFFF_FFFF)
    return b


def write_trace(path: Path, records: list[bytes]) -> None:
    with gzip.open(path, "wb") as fh:
        fh.write(b"".join(records))


def convert(tmp_path, records, *, pcs=None, lag=8, extra=()):
    trace = tmp_path / "t_trace.gz"
    write_trace(trace, records)
    sbbt = tmp_path / "t.sbbt.zst"
    meta = tmp_path / "t.meta.json"
    cmd = [str(CONVERT), str(trace), "--sbbt", str(sbbt), "--meta", str(meta),
           "--commit-lag", str(lag), "--decode-lag", "2", "--mix-window", "4",
           *extra]
    if pcs is not None:
        (tmp_path / "pcs.txt").write_text("".join(f"{p:#x}\n" for p in pcs))
        cmd += ["--feature-pcs", str(tmp_path / "pcs.txt"),
                "--features", str(tmp_path / "t.feat")]
    subprocess.run(cmd, check=True, capture_output=True)
    return sbbt, json.loads(meta.read_text())


# ---------------------------------------------------------------- SBBT stream

def test_sbbt_stream_matches_records(tmp_path):
    B = 0x4000
    records = [
        rec(0x1000, ALU, outs=[(1, 5)]),
        rec(B, COND, taken=True, target=0x5000, ins=[FLAGS]),     # taken
        rec(0x5000, ALU),
        rec(0x5004, LOAD, ins=[2], outs=[(3, 9)]),
        rec(B, COND, taken=False, ins=[FLAGS]),                   # not taken
        rec(0x4004, CALL, taken=True, target=0x8000),
        rec(0x8000, RET, taken=True, target=0x4008),
        rec(0x4008, IND, taken=True, target=0x9000, ins=[4]),
    ]
    sbbt, meta = convert(tmp_path, records)
    r = SbbtReader(sbbt)
    assert r.header.num_instructions == 8
    assert r.header.num_branches == 5
    a = r.read_all()
    assert list(a.ip) == [B, B, 0x4004, 0x8000, 0x4008]
    assert list(a.opcode) == [0b0001, 0b0001, 0b1000, 0b0110, 0b0010]
    assert list(a.taken) == [1, 0, 1, 1, 1]
    # The not-taken instance gets the target its taken instance revealed.
    assert list(a.target) == [0x5000, 0x5000, 0x8000, 0x4008, 0x9000]
    # ninstr counts from the previous branch, inclusive of this one.
    assert list(a.ninstr) == [2, 3, 1, 1, 1]
    assert list(a.index) == [0, 1, 2, 3, 4]
    assert meta["num_conditional"] == 2


def test_split_boundaries_are_consistent(tmp_path):
    records = []
    for i in range(50):
        records.append(rec(0x100, ALU))
        records.append(rec(0x200, COND, taken=bool(i % 2), target=0x100,
                           ins=[FLAGS]))
    _, meta = convert(tmp_path, records)
    b0, b1 = meta["boundaries"]
    assert (b0["branch_index"], b1["branch_index"]) == (30, 35)
    # Branch k is preceded by 2k+1 instructions and is itself instruction 2k+2.
    assert b0["instr_num"] == 2 * 30 + 2
    assert b0["conditional_before"] == 30


# ---------------------------------------------------------- fetch-safe features

def _features(tmp_path):
    bi, pc, f, side = load_raw_features(tmp_path / "t.feat")
    return bi, pc, f, side["feature_names"]


def test_branch_own_source_value_never_leaks(tmp_path):
    """The compare that decides a B.cond writes NZCV right before it. That
    value must not be visible -- only flags committed >= lag instructions ago.
    """
    B, lag = 0x4000, 8
    records = [rec(0x10, ALU, outs=[(FLAGS, 0xA << 28)])]          # old flags
    records += [rec(0x20 + 4 * i, ALU) for i in range(lag + 2)]    # retire it
    records += [rec(0x90, ALU, outs=[(FLAGS, 0x5 << 28)]),         # the compare
                rec(B, COND, taken=True, target=0x10, ins=[FLAGS])]
    convert(tmp_path, records, pcs=[B], lag=lag)
    _, _, f, names = _features(tmp_path)
    col = {n: i for i, n in enumerate(names)}
    assert f[0, col["src0_valid"]] == 1 and f[0, col["src0_is_flags"]] == 1
    nzcv = [int(f[0, col[f"src0_bit{b}"]]) for b in range(4)]
    assert nzcv == [0, 1, 0, 1], "saw the stale committed 0xA, not the live 0x5"


def test_commit_lag_boundary_is_exact(tmp_path):
    """A write ``lag - 1`` instructions back may still be in flight and must be
    invisible; at exactly ``lag`` back it has retired (the converter's rule:
    writes by instructions at least ``commit-lag`` older are committed)."""
    B, lag = 0x4000, 8
    for gap, visible in ((lag - 1, False), (lag, True)):
        sub = tmp_path / f"g{gap}"
        sub.mkdir()
        records = [rec(0x10, ALU, outs=[(5, 7)])]
        records += [rec(0x20 + 4 * i, ALU) for i in range(gap - 1)]
        records += [rec(B, COND, taken=True, target=0x10, ins=[5])]
        convert(sub, records, pcs=[B], lag=lag)
        _, _, f, names = _features(sub)
        col = {n: i for i, n in enumerate(names)}
        assert bool(f[0, col["src0_valid"]]) is visible, gap
        if visible:
            assert [int(f[0, col[f"src0_bit{b}"]]) for b in range(4)] == \
                [1, 1, 1, 0]   # 7


def test_local_history_excludes_current_outcome(tmp_path):
    B = 0x4000
    outcomes = [1, 1, 0, 1, 0, 0]
    records = []
    for t in outcomes:
        records += [rec(0x10, ALU),
                    rec(B, COND, taken=bool(t), target=0x10, ins=[FLAGS])]
    convert(tmp_path, records, pcs=[B])
    bi, _, f, names = _features(tmp_path)
    col = {n: i for i, n in enumerate(names)}
    assert list(bi) == list(range(6))
    for k in range(len(outcomes)):
        expect = [outcomes[k - 1 - j] if k - 1 - j >= 0 else 0
                  for j in range(4)]
        got = [int(f[k, col[f"local_hist_{j}"]]) for j in range(4)]
        assert got == expect, (k, got, expect)
    assert f[0, col["first_occurrence"]] == 1
    assert (f[1:, col["first_occurrence"]] == 0).all()


def test_features_only_for_requested_pcs_with_branch_index(tmp_path):
    A, B = 0x4000, 0x5000
    records = []
    for i in range(5):
        records += [rec(A, COND, taken=True, target=B, ins=[FLAGS]),
                    rec(B, COND, taken=bool(i % 2), target=A, ins=[FLAGS])]
    sbbt, _ = convert(tmp_path, records, pcs=[B])
    bi, pc, _, _ = _features(tmp_path)
    a = SbbtReader(sbbt).read_all()
    assert set(pc.tolist()) == {B}
    assert list(bi) == list(a.index[a.ip == B])


# ------------------------------------------------- dataset split and join

def test_chronological_dataset_requires_index_range(tmp_path):
    B = 0x4000
    records = []
    for i in range(400):
        records += [rec(0x10, ALU),
                    rec(B, COND, taken=bool(i % 3 == 0), target=0x10,
                        ins=[FLAGS])]
    sbbt, meta = convert(tmp_path, records, pcs=[B])
    h5 = tmp_path / "t.hdf5"
    build_trace_dataset(sbbt, h5, [B], split_role="chronological")
    with pytest.raises(ValueError, match="index_range"):
        BranchDataset([h5], br_pc=B, history_length=8)

    b0 = meta["boundaries"][0]["branch_index"]
    train = BranchDataset([h5], br_pc=B, history_length=8,
                          index_range=(8, b0))
    test = BranchDataset([h5], br_pc=B, history_length=8,
                         index_range=(b0 + 8, meta["num_branches"]))
    tr = np.concatenate([s.indices for s in train.sources])
    te = np.concatenate([s.indices for s in test.sources])
    assert tr.max() < b0 <= te.min() - 8

    spec = RawFeatureSpec(feature_dir=tmp_path, groups=("path",))
    # RawFeatureSpec looks for <trace_id>.feat; the trace id here is "t".
    feats, names = spec.load(h5, B, te)
    assert feats.shape == (te.size, len(names))
    with pytest.raises(AuxFeatureError, match="partial"):
        spec.load(h5, B, te + 1)


def test_pc_sign_conventions_round_trip():
    kernel = 0xffff800010177f98
    signed = as_signed_pc(kernel)
    assert signed < 0 and signed == np.int64(np.uint64(kernel).view(np.int64))
    assert as_unsigned_pc(signed) == np.uint64(kernel)
    assert as_signed_pc(0x4000) == 0x4000


# ------------------------------------------------- dataflow producer features

PRODUCERS = REPO / "build" / "cbp2025_producers"


def _producer_trace(tmp_path, values, gap, lag):
    """A loop: ldr x1 <- [mem]; cmp x1 -> NZCV; b.cond on NZCV; `gap` filler
    ALU instructions per iteration. Returns the trace and branch PC."""
    LD, CMP, B = 0x100, 0x104, 0x108
    records = []
    for v in values:
        records.append(rec(LD, LOAD, ins=[2], outs=[(1, v)]))
        records.append(rec(CMP, ALU, ins=[1], outs=[(FLAGS, (v & 0xF) << 28)]))
        records.append(rec(B, COND, taken=bool(v & 1), target=LD, ins=[FLAGS]))
        records += [rec(0x200 + 4 * i, ALU) for i in range(gap)]
    trace = tmp_path / "p_trace.gz"
    write_trace(trace, records)
    return trace, LD, CMP, B


@pytest.mark.skipif(not PRODUCERS.exists(), reason="producers tool not built")
def test_profile_finds_the_dataflow_chain(tmp_path):
    trace, LD, CMP, B = _producer_trace(tmp_path, list(range(50)), 5, 64)
    (tmp_path / "pcs.txt").write_text(f"{B:#x}\n")
    out = tmp_path / "prof.json"
    subprocess.run([str(PRODUCERS), "profile", str(trace), "--pcs",
                    str(tmp_path / "pcs.txt"), "--out", str(out)], check=True,
                   capture_output=True)
    prof = json.loads(out.read_text())[f"{B:#x}"]
    chain = {(c["depth"], int(c["pc"], 16)) for c in prof["chain"]}
    assert (1, CMP) in chain and (2, LD) in chain
    assert prof["instances"] == 50


@pytest.mark.skipif(not PRODUCERS.exists(), reason="producers tool not built")
def test_producer_values_are_committed_only(tmp_path):
    """Iteration = 3 + 5 = 8 instructions; lag 20 => the newest committed load
    is >= 3 iterations back, and the 2-3 newer ones are in flight."""
    lag, values = 20, [7 * i + 3 for i in range(30)]
    trace, LD, CMP, B = _producer_trace(tmp_path, values, 5, lag)
    (tmp_path / "prod.json").write_text(json.dumps({f"{B:#x}": [f"{LD:#x}"]}))
    subprocess.run([str(PRODUCERS), "features", str(trace), "--producers",
                    str(tmp_path / "prod.json"), "--commit-lag", str(lag),
                    "--out", str(tmp_path / "pf")], check=True,
                   capture_output=True)
    bi, pc, f, side = load_raw_features(tmp_path / "pf")
    col = {n: i for i, n in enumerate(side["feature_names"])}
    for k in range(len(values)):
        n_branch = 8 * k + 2                 # instruction number of branch k
        committed = [j for j in range(k + 1) if n_branch - 8 * j >= lag]
        if not committed:
            assert f[k, col["prod0_valid"]] == 0
            continue
        j = committed[-1]                    # newest committed load
        v = values[j]
        assert f[k, col["prod0_valid"]] == 1
        assert [int(f[k, col[f"prod0_v_bit{b}"]]) for b in range(4)] == \
            [(v >> b) & 1 for b in range(4)], k
        # Never this iteration's own load: it is 2 instructions old, < lag.
        assert j < k
        inflight = sum(1 for jj in range(k + 1) if n_branch - 8 * jj < lag)
        assert np.isclose(f[k, col["prod0_inflight"]],
                          np.log2(1 + inflight) / 6), (k, inflight)


@pytest.mark.skipif(not PRODUCERS.exists(), reason="producers tool not built")
def test_inst_hist_excludes_the_branch_itself(tmp_path):
    trace, LD, CMP, B = _producer_trace(tmp_path, [1, 2, 3], 5, 64)
    (tmp_path / "prod.json").write_text(json.dumps({f"{B:#x}": []}))
    subprocess.run([str(PRODUCERS), "features", str(trace), "--producers",
                    str(tmp_path / "prod.json"), "--commit-lag", "64",
                    "--out", str(tmp_path / "pf")], check=True,
                   capture_output=True)
    _, _, f, side = load_raw_features(tmp_path / "pf")
    col = {n: i for i, n in enumerate(side["feature_names"])}
    # Instruction right before branch 0 is the compare (ALU), then the load.
    assert f[0, col["prev0_alu"]] == 1 and f[0, col["prev0_branch"]] == 0
    assert f[0, col["prev1_load"]] == 1
