"""Focused tests for the Stage 3 extraction logic.

Most tests build a synthetic trace in the official byte layout, so the
extraction rules are checked without depending on the sample traces. The final
test runs against a real official trace and is skipped when it is absent.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import pandas as pd
import pytest

from src.branch_extraction import extract_branches, output_stem, validate
from src.trace_inspection import (
    ALU,
    CALL_DIRECT,
    COND_BR,
    FP_OP,
    LOAD,
    OFFICIAL_SAMPLE_STATS,
    SAMPLE_TRACES,
    STORE,
    UNCOND_DIRECT,
    UNDEF,
    TraceFormatError,
    reg_is_int,
)

FLAG_REG = 64
GPR0 = 0
SIMD0 = 32
BRANCH_CLASSES_NEEDING_TAKEN = (COND_BR, UNCOND_DIRECT, CALL_DIRECT)


def encode_record(
    pc: int,
    inst_class: int,
    *,
    taken: bool | None = None,
    target: int | None = None,
    in_regs: tuple[int, ...] = (),
    out_regs: tuple[int, ...] = (),
    out_values: tuple[int, ...] | None = None,
) -> bytes:
    """Encode one record exactly as TraceReader::readInstr expects to read it."""
    blob = bytearray()
    blob += struct.pack("<Q", pc)
    blob.append(inst_class)

    if inst_class in (LOAD, STORE):
        blob += struct.pack("<Q", 0x4000)  # effective address
        blob.append(8)  # access size
        blob.append(0)  # no base update
        if inst_class == STORE:
            blob.append(0)  # no register offset

    if inst_class in BRANCH_CLASSES_NEEDING_TAKEN:
        assert taken is not None, "branch records need an explicit taken flag"
        blob.append(1 if taken else 0)
        if taken:
            assert target is not None, "taken branches need a target"
            blob += struct.pack("<Q", target)

    blob.append(len(in_regs))
    blob += bytes(in_regs)
    blob.append(len(out_regs))
    blob += bytes(out_regs)

    # One 64-bit value per output register, two for SIMD registers.
    slots = sum(1 if reg_is_int(reg) else 2 for reg in out_regs)
    values = list(out_values) if out_values is not None else [0] * slots
    assert len(values) == slots, f"expected {slots} output values, got {len(values)}"
    for value in values:
        blob += struct.pack("<Q", value)

    return bytes(blob)


def write_trace(path: Path, records: list[bytes]) -> Path:
    with gzip.open(path, "wb") as stream:
        stream.write(b"".join(records))
    return path


def extract(path: Path) -> pd.DataFrame:
    frame, _ = extract_branches(path, workload_id="test", trace_id="synthetic_trace")
    return frame


def test_extracts_only_conditional_branches(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, out_regs=(GPR0,)),
            encode_record(0x1004, COND_BR, taken=True, target=0x2000, out_regs=(FLAG_REG,)),
            encode_record(0x2000, LOAD, out_regs=(GPR0,)),
            encode_record(0x2004, UNCOND_DIRECT, taken=True, target=0x3000),
            encode_record(0x3000, COND_BR, taken=False, out_regs=(FLAG_REG,)),
            encode_record(0x3004, FP_OP, out_regs=(SIMD0,), out_values=(1, 0)),
        ],
    )
    frame = extract(trace)

    assert len(frame) == 2
    assert frame["branch_pc"].tolist() == [0x1004, 0x3000]
    assert frame["outcome"].tolist() == [1, 0]
    assert frame["instruction_index"].tolist() == [1, 4]


def test_outcome_is_binary_and_taken_target_matches(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x40, COND_BR, taken=True, target=0x900, out_regs=(FLAG_REG,)),
            encode_record(0x44, COND_BR, taken=False, out_regs=(FLAG_REG,)),
        ],
    )
    frame = extract(trace)

    assert set(frame["outcome"].unique().tolist()) <= {0, 1}
    assert frame.loc[0, "taken_target"] == 0x900
    assert frame.loc[0, "next_pc"] == 0x900
    assert pd.isna(frame.loc[1, "taken_target"])
    assert frame.loc[1, "next_pc"] == 0x44 + 4


def test_multi_piece_branch_yields_exactly_one_row(tmp_path: Path) -> None:
    """A conditional branch with two output registers is 2 simulator pieces.

    Extraction is per record, so it must still produce a single row.
    """
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(
                0x500,
                COND_BR,
                taken=True,
                target=0x600,
                out_regs=(GPR0, FLAG_REG),
                out_values=(7, 0),
            ),
            encode_record(
                0x600, COND_BR, taken=False, out_regs=(SIMD0,), out_values=(1, 1)
            ),
        ],
    )
    frame = extract(trace)

    assert len(frame) == 2
    assert frame["branch_pc"].tolist() == [0x500, 0x600]
    assert frame["instruction_index"].duplicated().sum() == 0


def test_pc_occurrence_counts_per_address(tmp_path: Path) -> None:
    records = []
    for index in range(3):
        records.append(
            encode_record(0xA00, COND_BR, taken=index % 2 == 0, target=0xB00, out_regs=(FLAG_REG,))
        )
        records.append(encode_record(0xC00, COND_BR, taken=False, out_regs=(FLAG_REG,)))
    frame = extract(write_trace(tmp_path / "t.gz", records))

    assert frame.loc[frame["branch_pc"] == 0xA00, "pc_occurrence"].tolist() == [1, 2, 3]
    assert frame.loc[frame["branch_pc"] == 0xC00, "pc_occurrence"].tolist() == [1, 2, 3]


def test_execution_order_and_validation(tmp_path: Path) -> None:
    records = [
        encode_record(0x100 + 8 * i, COND_BR, taken=i % 3 == 0, target=0x900, out_regs=(FLAG_REG,))
        for i in range(25)
    ]
    frame = extract(write_trace(tmp_path / "t.gz", records))

    instr = frame["instruction_index"].to_numpy()
    assert (instr[1:] > instr[:-1]).all()
    assert frame["branch_index"].tolist() == list(range(25))
    assert validate(frame) is True


def test_parquet_roundtrip(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x10, COND_BR, taken=True, target=0x80, out_regs=(FLAG_REG,)),
            encode_record(0x14, COND_BR, taken=False, out_regs=(FLAG_REG,)),
        ],
    )
    frame = extract(trace)

    target = tmp_path / "out.parquet"
    frame.to_parquet(target, index=False, engine="pyarrow", compression="snappy")
    reloaded = pd.read_parquet(target)

    assert reloaded.shape == frame.shape
    assert reloaded["branch_pc"].tolist() == frame["branch_pc"].tolist()
    assert reloaded["outcome"].tolist() == frame["outcome"].tolist()
    assert pd.isna(reloaded.loc[1, "taken_target"])


def test_invalid_inst_class_is_rejected(tmp_path: Path) -> None:
    trace = write_trace(tmp_path / "t.gz", [encode_record(0x10, UNDEF)])
    with pytest.raises(TraceFormatError):
        extract(trace)


def test_truncated_record_is_rejected(tmp_path: Path) -> None:
    good = encode_record(0x10, COND_BR, taken=True, target=0x80, out_regs=(FLAG_REG,))
    trace = write_trace(tmp_path / "t.gz", [good, good[:-3]])
    with pytest.raises(TraceFormatError):
        extract(trace)


def test_output_stem() -> None:
    assert output_stem("sample_int_trace") == "sample_int_branches"
    assert output_stem("sample_fp_trace") == "sample_fp_branches"
    assert output_stem("custom") == "custom_branches"


@pytest.mark.parametrize("workload_id", sorted(SAMPLE_TRACES))
def test_matches_official_counts(workload_id: str) -> None:
    trace_path = SAMPLE_TRACES[workload_id]
    if not trace_path.is_file():
        pytest.skip(f"official sample trace missing: run scripts/fetch_cbp2025.py")

    frame, records = extract_branches(
        trace_path, workload_id=workload_id, trace_id=trace_path.name[: -len(".gz")]
    )
    reference = OFFICIAL_SAMPLE_STATS[workload_id]

    assert records == reference["instr"]
    assert len(frame) == reference["cond_direct"]
    assert validate(frame) is True
