"""Build per-branch training datasets from SBBT traces.

Output format matches BranchNet's (BranchNet's ``bin/create_ml_datasets.py``),
which was read and verified rather than guessed at. One HDF5 file per trace:

``history``
    One token per branch, over **all** branches in the trace -- conditional and
    not. ``token = ((pc & pc_mask) << 1) | direction``. BranchNet's tracer
    records every branch and its history spans all of them, so unconditional
    branches contribute PC context with a constant direction bit.

``br_indices_<hex pc>``
    Positions in ``history`` where that static branch occurs. A training sample
    for occurrence ``i`` is ``history[i-H : i]`` with label ``history[i] & 1``:
    the window stops immediately before the target, so **the target's own
    outcome is never an input**.

``num_taken_<pc>`` / ``num_not_taken_<pc>`` attrs
    Class balance, for ``pos_weight``.

Windows are deliberately **not** materialised. At H=582 that is ~600x more
bytes for no information, so they are sliced on the fly at load time.

``PC_BITS`` is 30 at storage and masked down to the model's ``pc_bits`` when
loading, so one build serves every ``pc_bits`` ablation.
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from ..traces import SbbtReader

#: PC bits kept in storage. The loader masks further; see pytorch_dataset.
STORAGE_PC_BITS = 30

BUILDER_VERSION = 1


def as_signed_pc(pc: int) -> int:
    """A PC as the parser stores it: sign-extended int64.

    Hex lists from the C++ driver print kernel PCs as their unsigned 64-bit
    pattern (``0xffff8000...``), which does not fit int64 as a Python int.
    """
    pc = int(pc) & 0xFFFF_FFFF_FFFF_FFFF
    return pc - (1 << 64) if pc >= (1 << 63) else pc


def token_dtype(pc_bits: int) -> np.dtype:
    """Smallest unsigned type holding ``pc_bits + 1`` bits (BranchNet's rule)."""
    width = pc_bits + 1
    if width <= 8:
        return np.dtype(np.uint8)
    if width <= 16:
        return np.dtype(np.uint16)
    if width <= 32:
        return np.dtype(np.uint32)
    if width <= 64:
        return np.dtype(np.uint64)
    raise ValueError(f"pc_bits={pc_bits} needs more than 64 bits per token")


def encode_tokens(pc: np.ndarray, taken: np.ndarray, pc_bits: int) -> np.ndarray:
    """``((pc & mask) << 1) | direction``, the shared token of spec §6.2."""
    dtype = token_dtype(pc_bits)
    mask = (np.uint64(1) << np.uint64(pc_bits)) - np.uint64(1)
    low = pc.astype(np.uint64) & mask
    return ((low << np.uint64(1)) | taken.astype(np.uint64)).astype(dtype)


def decode_token(token: int, pc_bits: int = STORAGE_PC_BITS) -> tuple[int, int]:
    """Inverse of :func:`encode_tokens` for one token. Used by tests."""
    return (token >> 1) & ((1 << pc_bits) - 1), token & 1


@dataclass
class BuildStats:
    """What a build produced, for logging and verification."""

    trace_id: str
    total_branches: int
    history_length: int
    branch_counts: dict[int, int] = field(default_factory=dict)
    taken_counts: dict[int, int] = field(default_factory=dict)
    missing_pcs: list[int] = field(default_factory=list)


def build_trace_dataset(
    trace_path: str | Path,
    output_path: str | Path,
    target_pcs,
    *,
    split_role: str,
    pc_bits: int = STORAGE_PC_BITS,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
    chunk_branches: int = 1 << 22,
    overwrite: bool = False,
    extra_attrs: dict | None = None,
) -> BuildStats:
    """Convert one trace into a per-branch training dataset.

    Streams the trace, so memory is bounded by the chunk size plus the
    occurrence indices of the selected branches -- never the whole trace.

    Args:
        trace_path: ``.sbbt`` or ``.sbbt.zst``.
        output_path: HDF5 file to create.
        target_pcs: Static branch PCs to index (ints).
        split_role: ``train`` / ``validation`` / ``test``. Recorded in the file
            so a later stage cannot silently use a test trace for training.
            ``chronological`` means the file serves all three as disjoint time
            ranges of one trace; ``BranchDataset`` then demands an
            ``index_range`` (CBP2025, where each trace is one run).
        pc_bits: PC bits kept per token.
        compression: HDF5 filter. gzip level 4 roughly halves these files at a
            fraction of level 9's cost; branch streams compress well.
        overwrite: Replace an existing output instead of failing.

    Returns:
        :class:`BuildStats`, including any requested PC that never appeared.
    """
    trace_path, output_path = Path(trace_path), Path(output_path)
    targets = np.array(sorted(set(as_signed_pc(pc) for pc in target_pcs)),
                       dtype=np.int64)
    if targets.size == 0:
        raise ValueError("no target PCs given")
    if split_role not in ("train", "validation", "test", "chronological"):
        raise ValueError(f"split_role must be train/validation/test/chronological, "
                         f"got {split_role!r}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists (pass overwrite=True)")

    reader = SbbtReader(trace_path)
    dtype = token_dtype(pc_bits)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Buffers of occurrence indices, concatenated once at the end.
    occurrences: dict[int, list[np.ndarray]] = {int(pc): [] for pc in targets}
    taken_counts: dict[int, int] = {int(pc): 0 for pc in targets}
    total = 0

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    with h5py.File(tmp_path, "w") as fh:
        history = fh.create_dataset(
            "history",
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(min(1 << 20, max(1 << 12, chunk_branches)),),
            compression=compression,
            compression_opts=compression_opts if compression == "gzip" else None,
        )

        for chunk in reader.chunks(branches_per_chunk=chunk_branches):
            tokens = encode_tokens(chunk.ip, chunk.taken, pc_bits)
            start = total
            total += len(chunk)
            history.resize((total,))
            history[start:total] = tokens

            # Which of this chunk's branches are targets?
            hit = np.isin(chunk.ip, targets)
            if hit.any():
                hit_ip = chunk.ip[hit]
                hit_idx = chunk.index[hit]
                hit_taken = chunk.taken[hit]
                for pc in np.unique(hit_ip):
                    sel = hit_ip == pc
                    key = int(pc)
                    occurrences[key].append(hit_idx[sel])
                    taken_counts[key] += int(hit_taken[sel].sum())

        counts: dict[int, int] = {}
        missing: list[int] = []
        for pc in (int(p) for p in targets):
            parts = occurrences[pc]
            indices = (np.concatenate(parts) if parts
                       else np.empty(0, dtype=np.int64))
            counts[pc] = int(indices.size)
            if indices.size == 0:
                missing.append(pc)
                continue
            # Strictly increasing: a violation means the stream was reordered.
            if indices.size > 1 and not np.all(np.diff(indices) > 0):
                raise RuntimeError(
                    f"{trace_path.name}: occurrence indices for {pc:#x} are not "
                    f"strictly increasing -- the trace stream was reordered"
                )
            fh.create_dataset(
                f"br_indices_{hex(pc)}",
                data=indices,
                compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None,
            )
            fh.attrs[f"num_taken_{hex(pc)}"] = taken_counts[pc]
            fh.attrs[f"num_not_taken_{hex(pc)}"] = counts[pc] - taken_counts[pc]

        fh.attrs["builder_version"] = BUILDER_VERSION
        fh.attrs["pc_bits"] = pc_bits
        fh.attrs["token_encoding"] = "((pc & ((1 << pc_bits) - 1)) << 1) | direction"
        fh.attrs["history_covers"] = "all branches (conditional and unconditional)"
        fh.attrs["trace_path"] = str(trace_path)
        fh.attrs["trace_id"] = reader.trace_id
        fh.attrs["split_role"] = split_role
        fh.attrs["num_branches"] = total
        fh.attrs["trace_header_instructions"] = reader.header.num_instructions
        fh.attrs["trace_header_branches"] = reader.header.num_branches
        fh.attrs["target_pcs"] = np.array(targets, dtype=np.int64)
        fh.attrs["created_utc"] = _datetime.datetime.now(
            _datetime.timezone.utc).isoformat()
        for key, value in (extra_attrs or {}).items():
            fh.attrs[key] = value

    tmp_path.replace(output_path)
    return BuildStats(
        trace_id=reader.trace_id,
        total_branches=total,
        history_length=total,
        branch_counts=counts,
        taken_counts=taken_counts,
        missing_pcs=missing,
    )


def read_hard_branches(path: str | Path) -> list[int]:
    """Read a hard-branch list: one hex PC per line, ``#`` comments allowed.

    Same format as BranchNet's hard_brs files and ``--log-pcs``.
    """
    pcs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            pcs.append(int(line, 16))
    if not pcs:
        raise ValueError(f"{path} contains no PCs")
    return pcs
