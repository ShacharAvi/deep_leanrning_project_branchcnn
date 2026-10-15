"""Tests for the SBBT codec (spec section 27, "Trace tests").

The codec is tested by round-tripping synthetic traces built with
``build_trace_bytes``, so the documented bit layout is executable rather than
merely asserted in a docstring.
"""

import numpy as np
import pytest

from src.traces import (
    HEADER_BYTES,
    RECORD_BYTES,
    SBBT_MARK,
    OpCode,
    SbbtFormatError,
    TraceHeader,
    build_trace_bytes,
    decode_header,
    decode_records,
    encode_header,
    encode_records,
    opcode_name,
    sign_extend_52,
)


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #

def test_header_roundtrip():
    header = TraceHeader(num_instructions=110_000_000, num_branches=7_392_094)
    assert decode_header(encode_header(header)) == header


def test_header_magic_is_the_documented_constant():
    assert SBBT_MARK == 0x0000010A54424253
    # "SBBT\n" in the low 5 bytes, version 1 above it.
    assert SBBT_MARK.to_bytes(8, "little")[:5] == b"SBBT\n"


def test_header_rejects_empty_file():
    with pytest.raises(SbbtFormatError, match="empty or too small"):
        decode_header(b"")


def test_header_rejects_short_file():
    with pytest.raises(SbbtFormatError, match="empty or too small"):
        decode_header(b"\x00" * (HEADER_BYTES - 1))


def test_header_rejects_bad_magic():
    raw = bytearray(encode_header(TraceHeader(10, 1)))
    raw[0] ^= 0xFF
    with pytest.raises(SbbtFormatError, match="not an SBBT trace"):
        decode_header(bytes(raw))


def test_header_rejects_unsupported_version():
    bad = np.array([0x0000020A54424253, 10, 1], dtype=np.uint64).tobytes()
    with pytest.raises(SbbtFormatError, match="unsupported SBBT version"):
        decode_header(bad)


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #

def test_record_roundtrip_preserves_every_field():
    ip = [0x45A975, 0x7FFFFFFFFFFF, 0x1]
    target = [0x45A910, 0x2, 0x3]
    opcode = [OpCode.JUMP | OpCode.CND, OpCode.CALL, OpCode.RET | OpCode.IND]
    taken = [1, 1, 0]
    ninstr = [11, 4095, 0]

    decoded = decode_records(encode_records(ip, target, opcode, taken, ninstr))

    assert decoded.ip.tolist() == ip
    assert decoded.target.tolist() == target
    assert decoded.opcode.tolist() == [int(o) for o in opcode]
    assert decoded.taken.tolist() == taken
    assert decoded.ninstr.tolist() == ninstr


def test_record_is_sixteen_bytes():
    assert RECORD_BYTES == 16
    assert len(encode_records([1], [2], [0], [1], [3])) == RECORD_BYTES


def test_decode_rejects_partial_record():
    buf = encode_records([1], [2], [0], [1], [3])[:-1]
    with pytest.raises(SbbtFormatError, match="not a multiple of"):
        decode_records(buf)


def test_decode_empty_buffer_yields_no_branches():
    assert len(decode_records(b"")) == 0


def test_ninstr_is_twelve_bits():
    """ninstr saturates at 4095; the header is the authority for MPKI."""
    decoded = decode_records(encode_records([0], [0], [0], [0], [4095]))
    assert decoded.ninstr.tolist() == [4095]


def test_start_index_offsets_absolute_positions():
    buf = encode_records([1, 2, 3], [0, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1])
    decoded = decode_records(buf, start_index=1000)
    assert decoded.index.tolist() == [1000, 1001, 1002]


# --------------------------------------------------------------------------- #
# sign extension -- the MOBILE traces depend on this
# --------------------------------------------------------------------------- #

def test_sign_extend_leaves_low_addresses_untouched():
    assert sign_extend_52(np.array([0x45A975], np.uint64)).tolist() == [0x45A975]


def test_sign_extend_restores_kernel_addresses():
    """Raw 0xfffc00025fe20 is a real PC from SHORT_MOBILE-6."""
    raw = np.array([0xFFFC00025FE20], dtype=np.uint64)
    assert sign_extend_52(raw).astype(np.int64).tolist() == [
        np.int64(np.uint64(0xFFFFFFC00025FE20))
    ]


def test_sign_extension_survives_the_codec():
    kernel_pc = np.int64(np.uint64(0xFFFFFFC00025FE20))
    decoded = decode_records(encode_records([kernel_pc], [0], [1], [1], [1]))
    assert decoded.ip.tolist() == [kernel_pc]


def test_sign_extension_does_not_disturb_low_pc_bits():
    """Token encoding uses pc & ((1 << pc_bits) - 1); it must not care."""
    kernel_pc = np.int64(np.uint64(0xFFFFFFC00025FE20))
    raw = np.uint64(0xFFFC00025FE20)
    assert (np.uint64(kernel_pc) & np.uint64(0xFFF)) == (raw & np.uint64(0xFFF))


# --------------------------------------------------------------------------- #
# opcodes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "opcode,expected",
    [
        (0b0000, "JUMP"),
        (0b0001, "JUMP|CND"),
        (0b0010, "JUMP|IND"),
        (0b0011, "JUMP|CND|IND"),
        (0b0100, "RET"),
        (0b1000, "CALL"),
        (0b1010, "CALL|IND"),
    ],
)
def test_opcode_names(opcode, expected):
    assert opcode_name(opcode) == expected


def test_conditional_and_indirect_flags_are_independent():
    decoded = decode_records(
        encode_records(
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0b0000, 0b0001, 0b0010, 0b0011],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        )
    )
    assert decoded.is_conditional.tolist() == [False, True, False, True]
    assert decoded.is_indirect.tolist() == [False, False, True, True]


def test_branch_type_strips_flags():
    decoded = decode_records(
        encode_records([0, 0], [0, 0], [OpCode.CALL | OpCode.IND, OpCode.RET], [1, 1], [1, 1])
    )
    assert decoded.branch_type.tolist() == [int(OpCode.CALL), int(OpCode.RET)]


# --------------------------------------------------------------------------- #
# whole-trace construction
# --------------------------------------------------------------------------- #

def test_build_trace_bytes_is_self_consistent():
    raw = build_trace_bytes([1, 2], [3, 4], [1, 1], [1, 0], [5, 7])
    header = decode_header(raw)
    assert header.num_branches == 2
    assert header.num_instructions == 12
    assert len(raw) == HEADER_BYTES + 2 * RECORD_BYTES


def test_selection_preserves_start_index_for_slices():
    buf = encode_records([1, 2, 3, 4], [0] * 4, [1] * 4, [1, 0, 1, 0], [1] * 4)
    decoded = decode_records(buf, start_index=100)
    assert decoded[1:3].index.tolist() == [101, 102]
