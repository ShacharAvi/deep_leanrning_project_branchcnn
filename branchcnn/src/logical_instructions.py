"""Full-detail reader for CBP2025 logical trace instructions.

One ``LogicalInstruction`` per ``TraceReader::readInstr()`` call, i.e. per record
in the gzip stream, *before* the simulator splits instructions into pieces.

This is a deliberately separate implementation from ``trace_inspection.read_records``,
which stays untouched so the validated Stage 3 output cannot change. The two
readers are cross-checked against each other by
``scripts/probe_logical_instructions.py`` and by the test suite.

Verified against external/cbp2025/lib/trace_reader.h. Field order in
``readInstr`` (lines 697-800):

    PC                  8 bytes   always
    InstClass           1 byte    always
    effective address   8 bytes   load/store only
    access size         1 byte    load/store only
    base update flag    1 byte    load/store only
    reg offset flag     1 byte    store only
    taken               1 byte    branch classes only
    taken target        8 bytes   branch and taken only
    num input regs      1 byte    always
    input reg ids       1 byte each
    num output regs     1 byte    always
    output reg ids      1 byte each
    output values       8 bytes per INT reg, 16 per SIMD reg (lo then hi)

Register values are available only for *destinations*. The official reader fills
source operand values with 0xdeadbeef, so source values must be reconstructed
from shadow state and are never read from the trace.
"""

from __future__ import annotations

import gzip
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Sequence, Tuple

from .trace_inspection import (
    BRANCH_CLASSES,
    CC_OFFSET,
    CLASS_NAMES,
    COND_BR,
    LOAD,
    STORE,
    UNDEF,
    VEC_OFFSET,
    ZERO_OFFSET,
    TraceFormatError,
    reg_is_int,
)

_U64 = struct.Struct("<Q")

__all__ = [
    "LogicalInstruction",
    "iter_logical_instructions",
    "read_logical_instructions",
    "register_kind",
]


def register_kind(reg: int) -> str:
    """Official register id groups (trace_reader.h ``enum Offset``)."""
    if reg == CC_OFFSET:
        return "flags"
    if reg == ZERO_OFFSET:
        return "zero"
    if reg == 31:
        return "stack_pointer"
    if reg < VEC_OFFSET:
        return "gpr"
    if reg < CC_OFFSET:
        return "simd"
    return "unknown"


@dataclass
class LogicalInstruction:
    """One logical trace instruction, exactly as stored in the file."""

    index: int
    pc: int
    inst_class: int

    taken: bool = False
    next_pc: int = 0
    taken_target: Optional[int] = None

    eff_addr: Optional[int] = None
    mem_size: Optional[int] = None
    base_upd_flag: bool = False
    has_reg_offset: bool = False

    in_regs: Tuple[int, ...] = ()
    out_regs: Tuple[int, ...] = ()
    # Aligned 1:1 with out_regs. ``out_values_hi`` is None for INT registers and
    # holds the upper 64 bits for SIMD registers.
    out_values_lo: Tuple[int, ...] = ()
    out_values_hi: Tuple[Optional[int], ...] = ()

    base_update_reg: Optional[int] = None
    num_pieces: int = 1
    mem_pieces: int = 0

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.inst_class]

    @property
    def is_branch(self) -> bool:
        return self.inst_class in BRANCH_CLASSES

    @property
    def is_conditional_branch(self) -> bool:
        return self.inst_class == COND_BR

    @property
    def is_load(self) -> bool:
        return self.inst_class == LOAD

    @property
    def is_store(self) -> bool:
        return self.inst_class == STORE

    @property
    def reads_flags(self) -> bool:
        return CC_OFFSET in self.in_regs

    @property
    def writes_flags(self) -> bool:
        return CC_OFFSET in self.out_regs

    def destination_value(self, reg: int) -> Optional[int]:
        """Low 64 bits written to ``reg`` by this instruction, if any."""
        for index, out_reg in enumerate(self.out_regs):
            if out_reg == reg:
                return self.out_values_lo[index]
        return None


def _detect_base_update(
    inst_class: int, base_upd_flag: bool, in_regs: Sequence[int], out_regs: Sequence[int]
) -> Optional[int]:
    """Port of ``Instr::capture_base_update_log_reg`` (trace_reader.h:222-320).

    Returns the base-update register id, or None.

    Note: the official overlap test keeps only registers below ``vecOffset``,
    so flags (64) and zero (65) are excluded even though ``reg_is_int`` treats
    them as integer registers.
    """
    if inst_class == STORE:
        # Asserted in the official reader: a store has at most one output, and
        # having one is equivalent to the base-update flag being set.
        return out_regs[0] if len(out_regs) == 1 else None

    if inst_class != LOAD:
        return None

    if len(out_regs) <= 1:
        return None

    overlap = {r for r in in_regs if r < VEC_OFFSET} & {r for r in out_regs if r < VEC_OFFSET}
    if not base_upd_flag or not overlap:
        return None
    return sorted(overlap)[0]


def _read_exact(stream: BinaryIO, size: int, what: str, index: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise TraceFormatError(f"Truncated stream reading {what} at record {index}")
    return data


def read_logical_instructions(
    stream: BinaryIO, limit: Optional[int] = None
) -> Iterator[LogicalInstruction]:
    """Decode logical instructions sequentially in execution order."""
    index = 0
    while limit is None or index < limit:
        head = stream.read(8)
        if not head:
            return
        if len(head) != 8:
            raise TraceFormatError(f"Truncated PC at record {index}")
        pc = _U64.unpack(head)[0]

        inst_class = _read_exact(stream, 1, "inst class", index)[0]
        if inst_class == UNDEF or inst_class not in CLASS_NAMES:
            raise TraceFormatError(
                f"Invalid InstClass {inst_class} at record {index} (pc=0x{pc:x})"
            )

        eff_addr: Optional[int] = None
        mem_size: Optional[int] = None
        base_upd_flag = False
        has_reg_offset = False
        if inst_class in (LOAD, STORE):
            eff_addr = _U64.unpack(_read_exact(stream, 8, "effective address", index))[0]
            mem_size = _read_exact(stream, 1, "access size", index)[0]
            base_upd_flag = bool(_read_exact(stream, 1, "base update flag", index)[0])
            if inst_class == STORE:
                has_reg_offset = bool(_read_exact(stream, 1, "reg offset flag", index)[0])

        taken = False
        next_pc = pc + 4
        taken_target: Optional[int] = None
        if inst_class in BRANCH_CLASSES:
            raw_taken = _read_exact(stream, 1, "taken", index)[0]
            if raw_taken not in (0, 1):
                raise TraceFormatError(
                    f"Taken byte is {raw_taken}, expected 0/1, at record {index}"
                )
            taken = bool(raw_taken)
            if taken:
                taken_target = _U64.unpack(_read_exact(stream, 8, "taken target", index))[0]
                next_pc = taken_target

        num_in_regs = _read_exact(stream, 1, "num input regs", index)[0]
        in_regs = tuple(_read_exact(stream, num_in_regs, "input reg ids", index))

        num_out_regs = _read_exact(stream, 1, "num output regs", index)[0]
        out_regs = tuple(_read_exact(stream, num_out_regs, "output reg ids", index))

        values_lo: list[int] = []
        values_hi: list[Optional[int]] = []
        extra_pieces = 0
        for reg in out_regs:
            values_lo.append(_U64.unpack(_read_exact(stream, 8, "output value", index))[0])
            if reg_is_int(reg):
                values_hi.append(None)
            else:
                upper = _U64.unpack(_read_exact(stream, 8, "output value hi", index))[0]
                values_hi.append(upper)
                if upper != 0:
                    extra_pieces += 1

        # Piece accounting, mirroring readInstr (trace_reader.h:767, 823-854).
        total_pieces = (num_out_regs if num_out_regs > 0 else 1) + extra_pieces
        base_update_reg = _detect_base_update(inst_class, base_upd_flag, in_regs, out_regs)
        has_base_update = base_update_reg is not None

        if inst_class == STORE:
            store_value_regs = num_in_regs - (1 + int(has_reg_offset))
            mem_pieces = store_value_regs if store_value_regs != 0 else 1
            total_pieces = mem_pieces + int(has_base_update)
        elif inst_class == LOAD:
            mem_pieces = total_pieces - int(has_base_update)
        else:
            mem_pieces = 0

        yield LogicalInstruction(
            index=index,
            pc=pc,
            inst_class=inst_class,
            taken=taken,
            next_pc=next_pc,
            taken_target=taken_target,
            eff_addr=eff_addr,
            mem_size=mem_size,
            base_upd_flag=base_upd_flag,
            has_reg_offset=has_reg_offset,
            in_regs=in_regs,
            out_regs=out_regs,
            out_values_lo=tuple(values_lo),
            out_values_hi=tuple(values_hi),
            base_update_reg=base_update_reg,
            num_pieces=total_pieces,
            mem_pieces=mem_pieces,
        )
        index += 1


def iter_logical_instructions(
    path: Path, limit: Optional[int] = None, buffer_size: int = 1 << 20
) -> Iterator[LogicalInstruction]:
    """Stream a trace from disk.

    Each record needs several small reads, so the decompressed stream is wrapped
    in a large buffer; this matters a lot on multi-GB traces.
    """
    with gzip.open(path, "rb") as raw:
        stream = io.BufferedReader(raw, buffer_size=buffer_size)
        yield from read_logical_instructions(stream, limit=limit)
