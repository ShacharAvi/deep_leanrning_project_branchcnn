"""Stage 3: extract one row per dynamic conditional branch into Parquet.

Reuses the verified record reader from ``trace_inspection`` (a port of
``TraceReader::readInstr``). One row per conditional-branch *record*; simulator
pieces are never iterated, so multi-output instructions cannot duplicate a row.

    python -m src.branch_extraction
    python -m src.branch_extraction --workload int --limit 200000

``outcome`` is the training label. ``next_pc`` and ``taken_target`` are
resolution-time fields that reveal the label exactly and must not be used as
per-occurrence model inputs.
"""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

from .trace_inspection import (
    COND_BR,
    OFFICIAL_SAMPLE_STATS,
    SAMPLE_TRACES,
    iter_trace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "extracted"

# Marks "no taken target" in the packed array before conversion to pd.NA.
_NO_TARGET = 0xFFFFFFFFFFFFFFFF

COLUMN_ORDER = [
    "workload_id",
    "trace_id",
    "instruction_index",
    "branch_index",
    "branch_pc",
    "next_pc",
    "taken_target",
    "pc_occurrence",
    "outcome",
]


def output_stem(trace_id: str) -> str:
    base = trace_id[: -len("_trace")] if trace_id.endswith("_trace") else trace_id
    return f"{base}_branches"


def extract_branches(
    trace_path: Path,
    workload_id: str,
    trace_id: str,
    limit: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Return (branch table, number of trace records decoded).

    ``limit`` caps decoded *records*, not extracted branches.
    """
    instruction_index = array("Q")
    branch_pc = array("Q")
    next_pc = array("Q")
    taken_target = array("Q")
    pc_occurrence = array("Q")
    outcome = array("B")

    seen_per_pc: dict[int, int] = {}
    records = 0

    for record in iter_trace(trace_path, limit=limit):
        records += 1
        if record.inst_class != COND_BR:
            continue

        occurrence = seen_per_pc.get(record.pc, 0) + 1
        seen_per_pc[record.pc] = occurrence

        instruction_index.append(record.index)
        branch_pc.append(record.pc)
        next_pc.append(record.next_pc)
        taken_target.append(record.next_pc if record.taken else _NO_TARGET)
        pc_occurrence.append(occurrence)
        outcome.append(1 if record.taken else 0)

    row_count = len(branch_pc)

    target_values = np.array(taken_target, dtype=np.uint64)
    target_column = pd.array(target_values, dtype="UInt64")
    target_column[target_values == np.uint64(_NO_TARGET)] = pd.NA

    frame = pd.DataFrame(
        {
            "workload_id": pd.Categorical.from_codes(
                np.zeros(row_count, dtype=np.int8), categories=[workload_id]
            ),
            "trace_id": pd.Categorical.from_codes(
                np.zeros(row_count, dtype=np.int8), categories=[trace_id]
            ),
            "instruction_index": np.array(instruction_index, dtype=np.uint64),
            "branch_index": np.arange(row_count, dtype=np.uint64),
            "branch_pc": np.array(branch_pc, dtype=np.uint64),
            "next_pc": np.array(next_pc, dtype=np.uint64),
            "taken_target": target_column,
            "pc_occurrence": np.array(pc_occurrence, dtype=np.uint64),
            "outcome": np.array(outcome, dtype=np.uint8),
        }
    )
    return frame[COLUMN_ORDER], records


def validate(frame: pd.DataFrame) -> bool:
    """Run the Stage 3 acceptance checks, printing one line per check."""
    checks: list[tuple[str, bool, str]] = []

    outcomes = set(frame["outcome"].unique().tolist())
    checks.append(("outcome values are only 0/1", outcomes <= {0, 1}, str(sorted(outcomes))))

    checks.append(("row count is non-zero", len(frame) > 0, f"{len(frame):,} rows"))

    instr = frame["instruction_index"].to_numpy()
    strictly_increasing = bool(len(instr) < 2 or np.all(np.diff(instr) > 0))
    checks.append(
        ("execution order preserved", strictly_increasing, "instruction_index strictly increasing")
    )

    duplicates = int(frame["instruction_index"].duplicated().sum())
    checks.append(
        ("no duplicate branch rows", duplicates == 0, f"{duplicates} duplicated instruction_index")
    )

    branch_index = frame["branch_index"].to_numpy()
    contiguous = bool(np.array_equal(branch_index, np.arange(len(frame), dtype=np.uint64)))
    checks.append(("branch_index is contiguous", contiguous, "0..n-1"))

    taken = frame["outcome"].to_numpy(dtype=bool)
    target_present = frame["taken_target"].notna().to_numpy()
    checks.append(
        (
            "taken_target present iff taken",
            bool(np.array_equal(taken, target_present)),
            "consistent with the trace format",
        )
    )

    fallthrough = frame.loc[~taken]
    bad_fallthrough = int(
        (fallthrough["next_pc"].to_numpy() != fallthrough["branch_pc"].to_numpy() + 4).sum()
    )
    checks.append(
        ("not-taken next_pc == pc+4", bad_fallthrough == 0, f"{bad_fallthrough} mismatches")
    )

    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:<32} {detail}")
    return all(bool(passed) for _, passed, _ in checks)


def report(frame: pd.DataFrame, records: int, workload_id: str, partial: bool) -> None:
    total = len(frame)
    taken = int(frame["outcome"].sum())
    not_taken = total - taken
    taken_pct = 100.0 * taken / total if total else 0.0

    print(f"  total decoded instructions : {records:,}")
    print(f"  conditional branches       : {total:,}")
    print(f"  unique branch PCs          : {frame['branch_pc'].nunique():,}")
    print(f"  taken                      : {taken:,} ({taken_pct:.2f}%)")
    print(f"  not taken                  : {not_taken:,} ({100 - taken_pct:.2f}%)")

    reference = OFFICIAL_SAMPLE_STATS.get(workload_id)
    if reference and not partial:
        print(
            f"  vs official README         : Instr {records - reference['instr']:+,} / "
            f"CondDirect {total - reference['cond_direct']:+,} (0 = exact match)"
        )
    print()


def preview(frame: pd.DataFrame, rows: int = 10) -> str:
    shown = frame.head(rows).copy()
    for column in ("branch_pc", "next_pc", "taken_target"):
        shown[column] = shown[column].map(
            lambda value: "-" if pd.isna(value) else f"0x{int(value):x}"
        )
    return shown.to_string(index=False)


def write_outputs(frame: pd.DataFrame, outdir: Path, stem: str, preview_rows: int) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    parquet_path = outdir / f"{stem}.parquet"
    csv_path = outdir / f"{stem}_preview.csv"

    frame.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    frame.head(preview_rows).to_csv(csv_path, index=False)

    reloaded = pd.read_parquet(parquet_path)
    if reloaded.shape != frame.shape:
        raise RuntimeError(
            f"Parquet round-trip changed shape: wrote {frame.shape}, read {reloaded.shape}"
        )

    size_mb = parquet_path.stat().st_size / 1e6
    print(f"  wrote {parquet_path.name:<28} {size_mb:6.2f} MB  ({len(frame):,} rows)")
    print(f"  wrote {csv_path.name:<28} (first {preview_rows} rows, for eyeballing)")
    print(f"  read back with pandas.read_parquet -> {reloaded.shape[0]:,} rows OK")
    return parquet_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=sorted(SAMPLE_TRACES),
        action="append",
        help="Sample workload to extract (default: both).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap decoded records.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--preview-rows", type=int, default=25)
    args = parser.parse_args()

    workloads = args.workload or sorted(SAMPLE_TRACES)
    missing = [w for w in workloads if not SAMPLE_TRACES[w].is_file()]
    if missing:
        print(f"BLOCKED: sample traces missing for {', '.join(missing)}.")
        print("Run: python scripts/fetch_cbp2025.py")
        return 1

    frames: list[pd.DataFrame] = []
    ok = True

    for workload_id in workloads:
        trace_path = SAMPLE_TRACES[workload_id]
        trace_id = trace_path.name[: -len(".gz")]

        print("=" * 78)
        print(f"EXTRACTING {workload_id}: {trace_path.name}")
        print("=" * 78)

        frame, records = extract_branches(
            trace_path, workload_id=workload_id, trace_id=trace_id, limit=args.limit
        )
        report(frame, records, workload_id, partial=args.limit is not None)

        print("  validation")
        ok &= validate(frame)
        print()

        write_outputs(frame, args.outdir, output_stem(trace_id), args.preview_rows)
        print()
        print(preview(frame))
        print()
        frames.append(frame)

    if len(frames) > 1:
        print("=" * 78)
        print("COMBINED DATASET")
        print("=" * 78)
        combined = pd.concat(frames, ignore_index=True)
        # branch_index is per-trace; group by trace_id before using it downstream.
        for column in ("workload_id", "trace_id"):
            combined[column] = combined[column].astype("category")
        print(f"  rows                : {len(combined):,}")
        print(f"  unique branch PCs   : {combined['branch_pc'].nunique():,}")
        print(f"  taken share         : {100.0 * combined['outcome'].mean():.2f}%")
        write_outputs(combined, args.outdir, "sample_all_branches", args.preview_rows)
        print()

    print("=" * 78)
    print("ALL VALIDATION CHECKS PASSED" if ok else "SOME VALIDATION CHECKS FAILED")
    print(f"Output directory: {args.outdir}")
    print("=" * 78)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
