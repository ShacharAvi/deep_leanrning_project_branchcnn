"""Reader for SBBT branch traces (``.sbbt`` and ``.sbbt.zst``).

Two access modes:

``SbbtReader.chunks()``
    Streaming. Constant memory, and the only safe way to touch the large traces
    -- ``LONG_MOBILE-12`` is 7 GB decompressed from a 2.7 MB file.

``SbbtReader.read_all()``
    Everything in memory as a :class:`~src.traces.formats.BranchArrays`.
    Convenient for the short traces and tests; refuses oversized traces unless
    explicitly allowed.

The history stream used for modelling is **all branches**, not only conditional
ones. That matches BranchNet, whose tracer records every branch
(``br_type != NOT_BR``) and whose ``create_ml_datasets.py`` builds the history
array over all of them. Unconditional branches always carry ``taken == 1``, so
they contribute PC context with a constant direction bit. Use
``conditional_only=True`` on the helpers below for the ablation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from .formats import (
    HEADER_BYTES,
    RECORD_BYTES,
    BranchArrays,
    SbbtFormatError,
    TraceHeader,
    decode_header,
    decode_records,
)

DEFAULT_CHUNK_BRANCHES = 1 << 22  # 4 Mi branches == 64 MiB per chunk
_READ_ALL_LIMIT_BYTES = 8 << 30  # refuse >8 GiB in read_all() without opt-in


def _read_exactly(stream: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes, or fewer only at end of stream.

    Decompression streams routinely return short reads; a plain ``read(n)`` is
    not enough.
    """
    parts, remaining = [], size
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            break
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


class SbbtReader:
    """Sequential reader for one SBBT trace file.

    Args:
        path: ``.sbbt`` or ``.sbbt.zst``.
        trace_id: Label carried into emitted records. Defaults to the filename
            with extensions stripped.
    """

    def __init__(self, path: str | os.PathLike, trace_id: str | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        name = self.path.name
        for suffix in (".sbbt.zst", ".sbbt"):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                break
        else:
            raise SbbtFormatError(
                f"unrecognised trace extension: {name!r} (expected .sbbt or .sbbt.zst)"
            )

        self.trace_id = trace_id if trace_id is not None else stem
        self._compressed = name.endswith(".zst")
        self.header = self._read_header()

    def __repr__(self) -> str:
        return (
            f"SbbtReader({self.trace_id!r}, "
            f"instructions={self.header.num_instructions:,}, "
            f"branches={self.header.num_branches:,})"
        )

    # -- internals ---------------------------------------------------------

    def _open_stream(self, fh: BinaryIO) -> BinaryIO:
        if not self._compressed:
            return fh
        import zstandard as zstd

        return zstd.ZstdDecompressor().stream_reader(fh)

    def _read_header(self) -> TraceHeader:
        with open(self.path, "rb") as fh:
            return decode_header(_read_exactly(self._open_stream(fh), HEADER_BYTES))

    # -- public API --------------------------------------------------------

    @property
    def decompressed_bytes(self) -> int:
        """Size the trace expands to, from the header. Useful before reading."""
        return HEADER_BYTES + RECORD_BYTES * self.header.num_branches

    def chunks(
        self,
        branches_per_chunk: int = DEFAULT_CHUNK_BRANCHES,
        *,
        conditional_only: bool = False,
        verify_count: bool = True,
    ) -> Iterator[BranchArrays]:
        """Stream the trace in chunks of decoded branches.

        ``start_index`` on each chunk is the absolute index among *all* branches,
        regardless of ``conditional_only`` -- so positions stay comparable across
        filtered and unfiltered passes.

        Args:
            branches_per_chunk: Branches per yielded chunk.
            conditional_only: Drop non-conditional branches from each chunk.
            verify_count: Check the total against the header. Catches truncation,
                which is otherwise silent and corrupts every downstream result.

        Raises:
            SbbtFormatError: On trailing partial records or a count mismatch.
        """
        if branches_per_chunk < 1:
            raise ValueError("branches_per_chunk must be >= 1")

        want = branches_per_chunk * RECORD_BYTES
        seen = 0

        with open(self.path, "rb") as fh:
            stream = self._open_stream(fh)
            decode_header(_read_exactly(stream, HEADER_BYTES))

            tail = b""
            while True:
                block = stream.read(want)
                if not block:
                    break
                if tail:
                    block = tail + block
                usable = len(block) - (len(block) % RECORD_BYTES)
                tail = block[usable:]
                if not usable:
                    continue

                chunk = decode_records(block[:usable], start_index=seen)
                seen += len(chunk)
                if conditional_only:
                    chunk = chunk[chunk.is_conditional]
                if len(chunk):
                    yield chunk

            if tail:
                raise SbbtFormatError(
                    f"{self.trace_id}: trailing {len(tail)} bytes after "
                    f"{seen:,} complete records -- file is truncated"
                )

        if verify_count and seen != self.header.num_branches:
            raise SbbtFormatError(
                f"{self.trace_id}: read {seen:,} branches but the header "
                f"declares {self.header.num_branches:,} -- file is truncated or corrupt"
            )

    def read_all(
        self, *, conditional_only: bool = False, allow_large: bool = False
    ) -> BranchArrays:
        """Read the whole trace into memory.

        Raises:
            MemoryError: If the trace exceeds 8 GiB and ``allow_large`` is False.
                Use :meth:`chunks` instead; nothing in this project needs the
                whole of a large trace resident.
        """
        size = self.decompressed_bytes
        if size > _READ_ALL_LIMIT_BYTES and not allow_large:
            raise MemoryError(
                f"{self.trace_id} decompresses to {size / 2**30:.1f} GiB. "
                f"Use chunks(), or pass allow_large=True if you really mean it."
            )

        parts = list(self.chunks(conditional_only=conditional_only))
        if not parts:
            return BranchArrays(
                ip=np.empty(0, np.int64),
                target=np.empty(0, np.int64),
                opcode=np.empty(0, np.uint8),
                taken=np.empty(0, np.uint8),
                ninstr=np.empty(0, np.uint16),
                start_index=0,
            )
        if len(parts) == 1:
            return parts[0]

        indices = np.concatenate([p.index for p in parts])
        contiguous = all(p.is_contiguous for p in parts) and not conditional_only
        return BranchArrays(
            ip=np.concatenate([p.ip for p in parts]),
            target=np.concatenate([p.target for p in parts]),
            opcode=np.concatenate([p.opcode for p in parts]),
            taken=np.concatenate([p.taken for p in parts]),
            ninstr=np.concatenate([p.ninstr for p in parts]),
            start_index=int(indices[0]),
            index_array=None if contiguous else indices,
        )

    def iter_records(self, *, conditional_only: bool = False) -> Iterator[dict]:
        """Yield the normalised per-branch record of spec Phase 1.

        ``{"pc": int, "taken": int, "branch_index": int, "trace_id": str}``

        Python-level iteration, so this is for tests, debugging and tiny traces.
        Real passes use :meth:`chunks` and stay in numpy.
        """
        for chunk in self.chunks(conditional_only=conditional_only):
            ips = chunk.ip.tolist()
            takens = chunk.taken.tolist()
            indices = chunk.index.tolist()
            for pc, taken, branch_index in zip(ips, takens, indices):
                yield {
                    "pc": pc,
                    "taken": int(taken),
                    "branch_index": branch_index,
                    "trace_id": self.trace_id,
                }


def read_header(path: str | os.PathLike) -> TraceHeader:
    """Read just the header. Only the first compressed block is touched."""
    return SbbtReader(path).header
