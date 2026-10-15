"""SBBT trace format: layout constants, codecs, and opcode semantics.

SBBT ("Simple Binary Branch Trace") v1.0.0 is the format used by MBPlib and the
CBP-5 trace mirror. The authoritative reader is
``third_party/mbplib/src/sim/sbbt_reader.cpp``; this module is a verified
reimplementation of that layout. Everything here is little-endian.

Layout
------
24-byte header, then 16 bytes per branch::

    header:  uint64 sbbt_mark       == 0x0000010A54424253   ("SBBT\\n" + version 1)
             uint64 num_instructions
             uint64 num_branches

    record:  word0 (uint64 LE): bits  0..3   opcode
                                bits  4..10  padding (always 0)
                                bit   11     outcome (1 = taken)
                                bits 12..63  ip      (52-bit, sign-extended from bit 51)
             word1 (uint64 LE): bits  0..11  ninstr  (instructions since previous branch)
                                bits 12..63  target  (52-bit, sign-extended from bit 51)

``encode_records`` is the exact inverse of ``decode_records``; the round-trip is
covered by tests, which keeps this docstring honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

SBBT_MARK = 0x0000010A54424253
SBBT_MARK_NO_VERSION = 0x0000000A54424253
SBBT_MARK_VERSION_MASK = 0x000000FFFFFFFFFF

HEADER_BYTES = 24
RECORD_BYTES = 16

_IP_BITS = 52
_IP_SIGN_BIT = np.uint64(1) << np.uint64(_IP_BITS - 1)
_IP_MASK = (np.uint64(1) << np.uint64(_IP_BITS)) - np.uint64(1)


class SbbtFormatError(ValueError):
    """Raised when a file is not a well-formed SBBT v1 trace."""


class OpCode(IntEnum):
    """Branch opcode bits, mirroring ``mbp::Branch::OpCode``.

    The low 4 bits encode a base type (bits 2..3) plus two independent flags:
    ``CND`` (conditional) and ``IND`` (indirect).
    """

    JUMP = 0b0000
    RET = 0b0100
    CALL = 0b1000

    CND = 0b0001
    IND = 0b0010

    TYPE_MASK = 0b1100
    NUM_OPCODES = 0b10000


_TYPE_NAMES = {OpCode.JUMP: "JUMP", OpCode.RET: "RET", OpCode.CALL: "CALL"}


def opcode_name(opcode: int) -> str:
    """Human-readable opcode, e.g. 1 -> 'JUMP|CND'."""
    base = opcode & OpCode.TYPE_MASK
    name = _TYPE_NAMES.get(base, f"TYPE({base:#06b})")
    if opcode & OpCode.CND:
        name += "|CND"
    if opcode & OpCode.IND:
        name += "|IND"
    return name


def sign_extend_52(value: np.ndarray) -> np.ndarray:
    """Sign-extend 52-bit addresses to full 64-bit.

    Not optional: the CBP-5 MOBILE traces run in high kernel address space, where
    raw values like 0xfffc00025fe20 must become 0xffffffc00025fe20. Low bits are
    unaffected, so ``pc & ((1 << pc_bits) - 1)`` is the same either way -- but PC
    identity, hex display and cross-trace PC matching all depend on this.
    """
    value = np.asarray(value, dtype=np.uint64)
    return (value ^ _IP_SIGN_BIT) - _IP_SIGN_BIT


@dataclass(frozen=True)
class TraceHeader:
    """Header fields of an SBBT trace."""

    num_instructions: int
    num_branches: int

    @property
    def branches_per_kilo_instruction(self) -> float:
        return 1000.0 * self.num_branches / self.num_instructions


def decode_header(raw: bytes) -> TraceHeader:
    """Parse and validate a 24-byte SBBT header."""
    if len(raw) < HEADER_BYTES:
        raise SbbtFormatError(
            f"file is empty or too small: got {len(raw)} bytes, need {HEADER_BYTES}"
        )

    mark, num_instructions, num_branches = (
        int(v) for v in np.frombuffer(raw, dtype=np.uint64, count=3)
    )

    if (mark & SBBT_MARK_VERSION_MASK) != SBBT_MARK_NO_VERSION:
        raise SbbtFormatError(
            f"not an SBBT trace: magic {mark:#018x} "
            f"(expected {SBBT_MARK_NO_VERSION:#018x} in the low 40 bits)"
        )
    if mark != SBBT_MARK:
        version = mark & ~SBBT_MARK_VERSION_MASK
        raise SbbtFormatError(
            f"unsupported SBBT version {version:#x} (this reader handles v1 only)"
        )

    return TraceHeader(num_instructions=num_instructions, num_branches=num_branches)


def encode_header(header: TraceHeader) -> bytes:
    return np.array(
        [SBBT_MARK, header.num_instructions, header.num_branches], dtype=np.uint64
    ).tobytes()


@dataclass(frozen=True)
class BranchArrays:
    """Decoded branches as a struct-of-arrays.

    Every element carries its **absolute index among all branches in the trace**.
    For a contiguous run that is just ``start_index + arange(n)``, so it is left
    implicit; once elements are filtered out (e.g. conditional-only) the run is
    no longer contiguous and ``index_array`` holds the positions explicitly.

    This matters because the prediction log, the H2P statistics and the per-branch
    datasets all join on these indices. A position that silently shifts under
    filtering would misalign histories against labels.
    """

    ip: np.ndarray  # int64, sign-extended
    target: np.ndarray  # int64, sign-extended
    opcode: np.ndarray  # uint8
    taken: np.ndarray  # uint8, 0/1
    ninstr: np.ndarray  # uint16, instructions since previous branch
    start_index: int = 0
    index_array: np.ndarray | None = None  # set only when non-contiguous

    def __len__(self) -> int:
        return int(self.ip.shape[0])

    @property
    def is_contiguous(self) -> bool:
        """True when these branches are an unbroken run of the trace."""
        return self.index_array is None

    @property
    def is_conditional(self) -> np.ndarray:
        return (self.opcode & np.uint8(OpCode.CND)).astype(bool)

    @property
    def is_indirect(self) -> np.ndarray:
        return (self.opcode & np.uint8(OpCode.IND)).astype(bool)

    @property
    def branch_type(self) -> np.ndarray:
        """Base type (JUMP / RET / CALL) with the CND and IND flags stripped."""
        return self.opcode & np.uint8(OpCode.TYPE_MASK)

    @property
    def index(self) -> np.ndarray:
        """Absolute branch index of each element."""
        if self.index_array is not None:
            return self.index_array
        return np.arange(self.start_index, self.start_index + len(self), dtype=np.int64)

    def __getitem__(self, selector) -> "BranchArrays":
        """Select a subset, preserving absolute indices."""
        selected = self.index[selector]
        contiguous = isinstance(selector, slice) and selector.step in (None, 1)
        return BranchArrays(
            ip=self.ip[selector],
            target=self.target[selector],
            opcode=self.opcode[selector],
            taken=self.taken[selector],
            ninstr=self.ninstr[selector],
            start_index=int(selected[0]) if len(selected) else self.start_index,
            index_array=None if contiguous else selected,
        )


def decode_records(buf: bytes, start_index: int = 0) -> BranchArrays:
    """Decode a whole number of 16-byte SBBT records."""
    if len(buf) % RECORD_BYTES:
        raise SbbtFormatError(
            f"buffer of {len(buf)} bytes is not a multiple of {RECORD_BYTES}"
        )

    count = len(buf) // RECORD_BYTES
    words = np.frombuffer(buf, dtype=np.uint64, count=2 * count).reshape(count, 2)
    w0, w1 = words[:, 0], words[:, 1]

    return BranchArrays(
        ip=sign_extend_52(w0 >> np.uint64(12)).astype(np.int64),
        target=sign_extend_52(w1 >> np.uint64(12)).astype(np.int64),
        opcode=(w0 & np.uint64(0xF)).astype(np.uint8),
        taken=((w0 >> np.uint64(11)) & np.uint64(1)).astype(np.uint8),
        ninstr=(w1 & np.uint64(0xFFF)).astype(np.uint16),
        start_index=start_index,
    )


def encode_records(
    ip, target, opcode, taken, ninstr
) -> bytes:
    """Inverse of :func:`decode_records`. Used to build synthetic test traces."""
    ip = np.asarray(ip, dtype=np.int64).astype(np.uint64) & _IP_MASK
    target = np.asarray(target, dtype=np.int64).astype(np.uint64) & _IP_MASK
    opcode = np.asarray(opcode, dtype=np.uint64) & np.uint64(0xF)
    taken = np.asarray(taken, dtype=np.uint64) & np.uint64(1)
    ninstr = np.asarray(ninstr, dtype=np.uint64) & np.uint64(0xFFF)

    count = ip.shape[0]
    words = np.empty((count, 2), dtype=np.uint64)
    words[:, 0] = opcode | (taken << np.uint64(11)) | (ip << np.uint64(12))
    words[:, 1] = ninstr | (target << np.uint64(12))
    return words.tobytes()


def build_trace_bytes(
    ip, target, opcode, taken, ninstr, num_instructions: int | None = None
) -> bytes:
    """Build a complete, valid, uncompressed SBBT trace. For tests and fixtures."""
    body = encode_records(ip, target, opcode, taken, ninstr)
    count = len(body) // RECORD_BYTES
    if num_instructions is None:
        num_instructions = int(np.asarray(ninstr, dtype=np.uint64).sum())
    header = encode_header(
        TraceHeader(num_instructions=num_instructions, num_branches=count)
    )
    return header + body
