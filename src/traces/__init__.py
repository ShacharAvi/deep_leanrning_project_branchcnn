"""Trace parsing for SBBT branch traces."""

from .formats import (
    HEADER_BYTES,
    RECORD_BYTES,
    SBBT_MARK,
    BranchArrays,
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
from .parser import DEFAULT_CHUNK_BRANCHES, SbbtReader, read_header

__all__ = [
    "BranchArrays",
    "DEFAULT_CHUNK_BRANCHES",
    "HEADER_BYTES",
    "OpCode",
    "RECORD_BYTES",
    "SBBT_MARK",
    "SbbtFormatError",
    "SbbtReader",
    "TraceHeader",
    "build_trace_bytes",
    "decode_header",
    "decode_records",
    "encode_header",
    "encode_records",
    "opcode_name",
    "read_header",
    "sign_extend_52",
]
