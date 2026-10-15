"""Stage 2: verify the CBP2025 trace format against a real sample trace.

Read-only. This module decodes trace records but writes no dataset; Parquet
extraction is Stage 3 and will import the reader from here.

The record layout is a direct port of ``TraceReader::readInstr`` in the official
``lib/trace_reader.h``.

Run:
    python -m src.trace_inspection            # both samples, full scan
    python -m src.trace_inspection --limit 50 # decode only the first 50 records
"""

from __future__ import annotations

import argparse
import gzip
import platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_CBP = PROJECT_ROOT / "external" / "cbp2025"

SAMPLE_TRACES = {
    "int": EXTERNAL_CBP / "sample_traces" / "int" / "sample_int_trace.gz",
    "fp": EXTERNAL_CBP / "sample_traces" / "fp" / "sample_fp_trace.gz",
}

# Official README reference for the bundled sample traces.
# "Instr" is per trace record; CondDirect NumBr may count simulator pieces.
OFFICIAL_SAMPLE_STATS = {
    "int": {"instr": 997301, "cond_direct": 128874},
    "fp": {"instr": 997741, "cond_direct": 111265},
}

# InstClass, lib/sim_common_structs.h
ALU, LOAD, STORE, COND_BR = 0, 1, 2, 3
UNCOND_DIRECT, UNCOND_INDIRECT, FP_OP, SLOW_ALU, UNDEF = 4, 5, 6, 7, 8
CALL_DIRECT, CALL_INDIRECT, RETURN = 9, 10, 11

CLASS_NAMES = {
    ALU: "aluOp",
    LOAD: "loadOp",
    STORE: "stOp",
    COND_BR: "condBrOp",
    UNCOND_DIRECT: "uncondDirBrOp",
    UNCOND_INDIRECT: "uncondIndBrOp",
    FP_OP: "fpOp",
    SLOW_ALU: "slowAluOp",
    UNDEF: "undefOp",
    CALL_DIRECT: "callDirBrOp",
    CALL_INDIRECT: "callIndBrOp",
    RETURN: "retBrOp",
}

BRANCH_CLASSES = frozenset(
    {COND_BR, UNCOND_DIRECT, UNCOND_INDIRECT, CALL_DIRECT, CALL_INDIRECT, RETURN}
)

_U64 = struct.Struct("<Q")

# Int regs: 0-30 GPR, 31 SP, 64 flags, 65 zero. SIMD/FP: 32-63.
VEC_OFFSET, CC_OFFSET, ZERO_OFFSET = 32, 64, 65


def reg_is_int(reg: int) -> bool:
    return reg < VEC_OFFSET or reg in (CC_OFFSET, ZERO_OFFSET)


class TraceFormatError(RuntimeError):
    """The byte stream did not match the official record layout."""


@dataclass
class TraceRecord:
    """One ``readInstr()`` macro-op, i.e. one record in the gzip stream."""

    index: int  # 0-based, chronological
    pc: int
    inst_class: int
    taken: bool
    next_pc: int
    num_in_regs: int
    num_out_regs: int
    num_pieces: int  # simulator pieces this record would expand into

    @property
    def is_cond_branch(self) -> bool:
        return self.inst_class == COND_BR


def _read_exact(stream: BinaryIO, size: int, what: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise TraceFormatError(f"Truncated stream while reading {what}")
    return data


def read_records(stream: BinaryIO, limit: int | None = None) -> Iterator[TraceRecord]:
    """Decode records sequentially. Order is execution order."""
    index = 0
    while limit is None or index < limit:
        head = stream.read(8)
        if not head:
            return
        if len(head) != 8:
            raise TraceFormatError(f"Truncated PC at record {index}")
        pc = _U64.unpack(head)[0]

        inst_class = _read_exact(stream, 1, "inst class")[0]
        if inst_class == UNDEF or inst_class not in CLASS_NAMES:
            raise TraceFormatError(
                f"Invalid InstClass {inst_class} at record {index} (pc=0x{pc:x}); "
                "byte layout or endianness assumption is wrong"
            )

        if inst_class in (LOAD, STORE):
            _read_exact(stream, 8, "effective address")
            _read_exact(stream, 1, "access size")
            _read_exact(stream, 1, "base update flag")
            if inst_class == STORE:
                _read_exact(stream, 1, "reg offset flag")

        taken = False
        next_pc = pc + 4
        if inst_class in BRANCH_CLASSES:
            raw_taken = _read_exact(stream, 1, "taken")[0]
            if raw_taken not in (0, 1):
                raise TraceFormatError(
                    f"Taken byte is {raw_taken}, expected 0/1, at record {index}"
                )
            taken = bool(raw_taken)
            if taken:
                next_pc = _U64.unpack(_read_exact(stream, 8, "taken target"))[0]

        num_in_regs = _read_exact(stream, 1, "num input regs")[0]
        _read_exact(stream, num_in_regs, "input reg names")

        num_out_regs = _read_exact(stream, 1, "num output regs")[0]
        out_regs = _read_exact(stream, num_out_regs, "output reg names")

        # One 64-bit value per output register; SIMD registers carry a second
        # 64-bit lane. A non-zero upper lane adds a simulator piece.
        extra_pieces = 0
        for reg in out_regs:
            _read_exact(stream, 8, "output value")
            if not reg_is_int(reg):
                upper = _U64.unpack(_read_exact(stream, 8, "output value hi"))[0]
                if upper != 0:
                    extra_pieces += 1

        yield TraceRecord(
            index=index,
            pc=pc,
            inst_class=inst_class,
            taken=taken,
            next_pc=next_pc,
            num_in_regs=num_in_regs,
            num_out_regs=num_out_regs,
            num_pieces=max(num_out_regs, 1) + extra_pieces,
        )
        index += 1


def iter_trace(path: Path, limit: int | None = None) -> Iterator[TraceRecord]:
    with gzip.open(path, "rb") as stream:
        yield from read_records(stream, limit=limit)


def require_sample_trace(workload_id: str) -> Path:
    path = SAMPLE_TRACES[workload_id]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing official sample trace: {path}\n"
            "Clone https://github.com/ramisheikh/cbp2025 into external/cbp2025."
        )
    return path


def _report_environment() -> None:
    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    print(f"python        : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"executable    : {sys.executable}")
    print(f"platform      : {platform.platform()}")
    print(f"byteorder     : {sys.byteorder}")
    print(f"project root  : {PROJECT_ROOT}")
    print(f"cbp2025 clone : {EXTERNAL_CBP} (exists={EXTERNAL_CBP.is_dir()})")
    for name, path in SAMPLE_TRACES.items():
        size = f"{path.stat().st_size:,} bytes" if path.is_file() else "MISSING"
        print(f"  sample {name:<3} : {path.name} -> {size}")
    print()


def _report_encoding(path: Path) -> None:
    print("-- raw container ------------------------------------------------")
    with path.open("rb") as raw:
        magic = raw.read(4)
    print(f"first raw bytes : {magic.hex(' ')} (gzip magic = 1f 8b)")
    with gzip.open(path, "rb") as stream:
        head = stream.read(48)
    print(f"decompressed[48]: {head.hex(' ')}")
    printable = sum(1 for b in head if 32 <= b < 127)
    print(f"printable ratio : {printable}/{len(head)} -> binary, not text/CSV")
    print()


def _report_first_records(path: Path, count: int = 8) -> None:
    print(f"-- first {count} decoded records ----------------------------------")
    header = f"{'idx':>5} {'pc':>18} {'class':<14} {'taken':>5} {'next_pc':>18} {'pieces':>6}"
    print(header)
    for record in iter_trace(path, limit=count):
        print(
            f"{record.index:>5} 0x{record.pc:016x} "
            f"{CLASS_NAMES[record.inst_class]:<14} "
            f"{int(record.taken):>5} 0x{record.next_pc:016x} {record.num_pieces:>6}"
        )
    print()


def _report_full_scan(workload_id: str, path: Path, limit: int | None) -> None:
    scope = "FULL SCAN" if limit is None else f"PARTIAL SCAN (limit={limit:,})"
    print(f"-- {scope} ------------------------------------------------")

    records = 0
    pieces = 0
    class_counts: dict[int, int] = {}
    cond_total = 0
    cond_taken = 0
    cond_multi_piece = 0
    cond_pcs: set[int] = set()
    fallthrough_mismatch = 0

    for record in iter_trace(path, limit=limit):
        records += 1
        pieces += record.num_pieces
        class_counts[record.inst_class] = class_counts.get(record.inst_class, 0) + 1
        if record.is_cond_branch:
            cond_total += 1
            cond_taken += record.taken
            cond_pcs.add(record.pc)
            if record.num_pieces > 1:
                cond_multi_piece += 1
            if not record.taken and record.next_pc != record.pc + 4:
                fallthrough_mismatch += 1

    print(f"trace records             : {records:,}")
    print(f"simulator pieces (derived): {pieces:,}")
    for inst_class in sorted(class_counts):
        name = CLASS_NAMES[inst_class]
        print(f"  {name:<14} {class_counts[inst_class]:>12,}")
    print()

    not_taken = cond_total - cond_taken
    pct = (100.0 * cond_taken / cond_total) if cond_total else 0.0
    print(f"conditional branches      : {cond_total:,}")
    print(f"unique branch PCs         : {len(cond_pcs):,}")
    print(f"taken                     : {cond_taken:,} ({pct:.2f}%)")
    print(f"not taken                 : {not_taken:,} ({100 - pct:.2f}%)")
    print(f"cond branches >1 piece    : {cond_multi_piece:,}  <- duplicate-row risk")
    print(f"not-taken next_pc != pc+4 : {fallthrough_mismatch:,}  <- expect 0")

    reference = OFFICIAL_SAMPLE_STATS.get(workload_id)
    if reference and limit is None:
        print()
        print("vs official README sample run:")
        delta_instr = records - reference["instr"]
        delta_cond = cond_total - reference["cond_direct"]
        print(f"  Instr      README={reference['instr']:,} ours={records:,} delta={delta_instr:+,}")
        print(
            f"  CondDirect README={reference['cond_direct']:,} "
            f"ours={cond_total:,} delta={delta_cond:+,}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=sorted(SAMPLE_TRACES),
        action="append",
        help="Sample workload to inspect (default: both).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Decode at most N records instead of the whole trace.",
    )
    args = parser.parse_args()

    _report_environment()

    workloads = args.workload or sorted(SAMPLE_TRACES)
    missing = [w for w in workloads if not SAMPLE_TRACES[w].is_file()]
    if missing:
        print(f"BLOCKED: sample traces missing for {', '.join(missing)}.")
        print("Clone the CBP2025 framework into external/cbp2025 first (see the repository README).")
        return 1

    for workload_id in workloads:
        path = SAMPLE_TRACES[workload_id]
        print("=" * 78)
        print(f"TRACE {workload_id}: {path}")
        print("=" * 78)
        _report_encoding(path)
        try:
            _report_first_records(path)
            _report_full_scan(workload_id, path, args.limit)
        except TraceFormatError as error:
            print(f"FORMAT MISMATCH: {error}")
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
