"""Tests for :class:`src.traces.SbbtReader` (spec section 27, "Trace tests").

Most tests use synthetic traces written to tmp_path. Tests marked ``real_trace``
run against ``data/SHORT_MOBILE-73.sbbt.zst`` and are skipped when it is absent,
so the suite still passes on a checkout without the data.
"""

import numpy as np
import pytest
import zstandard as zstd

from src.traces import (
    HEADER_BYTES,
    OpCode,
    SbbtFormatError,
    SbbtReader,
    build_trace_bytes,
    read_header,
)

from .conftest import SMOKE_TRACE


@pytest.fixture
def synthetic(tmp_path):
    """A 6-branch trace mixing conditional and unconditional branches.

    Index:      0         1         2       3         4         5
    Kind:       cond      cond      JUMP    cond      CALL      cond
    """
    raw = build_trace_bytes(
        ip=[0x1000, 0x1004, 0x1008, 0x100C, 0x1010, 0x1014],
        target=[0x2000, 0x2004, 0x2008, 0x200C, 0x2010, 0x2014],
        opcode=[
            OpCode.JUMP | OpCode.CND,
            OpCode.JUMP | OpCode.CND,
            OpCode.JUMP,
            OpCode.JUMP | OpCode.CND,
            OpCode.CALL,
            OpCode.JUMP | OpCode.CND,
        ],
        taken=[1, 0, 1, 1, 1, 0],
        ninstr=[10, 20, 30, 40, 50, 60],
        num_instructions=250,
    )
    path = tmp_path / "synthetic.sbbt"
    path.write_bytes(raw)
    return path


def _compress(path, raw):
    path.write_bytes(zstd.ZstdCompressor().compress(raw))
    return path


# --------------------------------------------------------------------------- #
# header and construction
# --------------------------------------------------------------------------- #

def test_reads_header(synthetic):
    header = read_header(synthetic)
    assert header.num_branches == 6
    assert header.num_instructions == 250


def test_trace_id_defaults_to_stem(tmp_path, synthetic):
    assert SbbtReader(synthetic).trace_id == "synthetic"
    raw = synthetic.read_bytes()
    assert SbbtReader(_compress(tmp_path / "LONG_MOBILE-11.sbbt.zst", raw)).trace_id == (
        "LONG_MOBILE-11"
    )


def test_explicit_trace_id_wins(synthetic):
    assert SbbtReader(synthetic, trace_id="run-a").trace_id == "run-a"


def test_rejects_unknown_extension(tmp_path, synthetic):
    path = tmp_path / "trace.bin"
    path.write_bytes(synthetic.read_bytes())
    with pytest.raises(SbbtFormatError, match="unrecognised trace extension"):
        SbbtReader(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SbbtReader(tmp_path / "nope.sbbt")


def test_decompressed_bytes_matches_layout(synthetic):
    assert SbbtReader(synthetic).decompressed_bytes == HEADER_BYTES + 6 * 16


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def test_read_all_returns_every_branch(synthetic):
    branches = SbbtReader(synthetic).read_all()
    assert len(branches) == 6
    assert branches.ip.tolist() == [0x1000, 0x1004, 0x1008, 0x100C, 0x1010, 0x1014]
    assert branches.taken.tolist() == [1, 0, 1, 1, 1, 0]


def test_first_and_last_record(synthetic):
    """Spec section 27: 'first record', 'last record'."""
    branches = SbbtReader(synthetic).read_all()
    assert (branches.ip[0], branches.taken[0], branches.ninstr[0]) == (0x1000, 1, 10)
    assert (branches.ip[-1], branches.taken[-1], branches.ninstr[-1]) == (0x1014, 0, 60)


def test_conditional_only_filters_and_keeps_absolute_indices(synthetic):
    branches = SbbtReader(synthetic).read_all(conditional_only=True)
    assert branches.ip.tolist() == [0x1000, 0x1004, 0x100C, 0x1014]
    # Indices still refer to positions in the unfiltered stream.
    assert branches.index.tolist() == [0, 1, 3, 5]
    assert not branches.is_contiguous


def test_conditional_only_indices_survive_chunk_boundaries(synthetic):
    """Filtering inside chunked reads must not renumber branches."""
    for chunk_size in (1, 2, 3, 4, 5, 6, 100):
        chunks = list(
            SbbtReader(synthetic).chunks(
                branches_per_chunk=chunk_size, conditional_only=True
            )
        )
        assert np.concatenate([c.index for c in chunks]).tolist() == [0, 1, 3, 5]


def test_unfiltered_read_stays_contiguous(synthetic):
    branches = SbbtReader(synthetic).read_all()
    assert branches.is_contiguous
    assert branches.index.tolist() == list(range(6))


def test_boolean_selection_preserves_absolute_indices(synthetic):
    branches = SbbtReader(synthetic).read_all()
    taken_only = branches[branches.taken.astype(bool)]
    assert taken_only.index.tolist() == [0, 2, 3, 4]
    assert taken_only.ip.tolist() == [0x1000, 0x1008, 0x100C, 0x1010]


def test_selection_is_composable(synthetic):
    """Filter twice: indices must still point at the original stream."""
    branches = SbbtReader(synthetic).read_all()
    conditional = branches[branches.is_conditional]
    assert conditional[conditional.taken.astype(bool)].index.tolist() == [0, 3]


def test_unconditional_branches_are_always_taken(synthetic):
    """Holds in the real traces too, and is why they are safe history context."""
    branches = SbbtReader(synthetic).read_all()
    assert branches.taken[~branches.is_conditional].tolist() == [1, 1]


def test_zstd_and_plain_agree(tmp_path, synthetic):
    raw = synthetic.read_bytes()
    plain = SbbtReader(synthetic).read_all()
    packed = SbbtReader(_compress(tmp_path / "t.sbbt.zst", raw)).read_all()
    assert plain.ip.tolist() == packed.ip.tolist()
    assert plain.taken.tolist() == packed.taken.tolist()


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 6, 7, 1000])
def test_chunking_is_independent_of_chunk_size(synthetic, chunk_size):
    reader = SbbtReader(synthetic)
    chunks = list(reader.chunks(branches_per_chunk=chunk_size))
    assert sum(len(c) for c in chunks) == 6
    assert np.concatenate([c.ip for c in chunks]).tolist() == reader.read_all().ip.tolist()


def test_chunk_start_indices_are_contiguous_and_absolute(synthetic):
    chunks = list(SbbtReader(synthetic).chunks(branches_per_chunk=2))
    assert [c.start_index for c in chunks] == [0, 2, 4]
    assert np.concatenate([c.index for c in chunks]).tolist() == list(range(6))


def test_chunk_size_must_be_positive(synthetic):
    with pytest.raises(ValueError, match="must be >= 1"):
        list(SbbtReader(synthetic).chunks(branches_per_chunk=0))


# --------------------------------------------------------------------------- #
# truncation and corruption -- silent failures here would poison everything
# --------------------------------------------------------------------------- #

def test_trailing_partial_record_is_rejected(tmp_path, synthetic):
    path = tmp_path / "truncated.sbbt"
    path.write_bytes(synthetic.read_bytes()[:-5])
    with pytest.raises(SbbtFormatError, match="truncated"):
        SbbtReader(path).read_all()


def test_missing_whole_records_are_rejected(tmp_path, synthetic):
    """Byte-aligned truncation: only the header count reveals it."""
    path = tmp_path / "short.sbbt"
    path.write_bytes(synthetic.read_bytes()[: HEADER_BYTES + 3 * 16])
    with pytest.raises(SbbtFormatError, match="header declares"):
        SbbtReader(path).read_all()


def test_count_verification_can_be_disabled(tmp_path, synthetic):
    path = tmp_path / "short.sbbt"
    path.write_bytes(synthetic.read_bytes()[: HEADER_BYTES + 3 * 16])
    branches = list(SbbtReader(path).chunks(verify_count=False))
    assert sum(len(c) for c in branches) == 3


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.sbbt"
    path.write_bytes(b"")
    with pytest.raises(SbbtFormatError, match="empty or too small"):
        SbbtReader(path)


def test_header_only_trace_reads_as_zero_branches(tmp_path):
    raw = build_trace_bytes([], [], [], [], [], num_instructions=0)
    path = tmp_path / "headeronly.sbbt"
    path.write_bytes(raw)
    assert len(SbbtReader(path).read_all()) == 0


def test_read_all_refuses_oversized_traces(tmp_path, monkeypatch):
    """Guard against accidentally materialising LONG_MOBILE-12 (7 GB)."""
    raw = build_trace_bytes([1], [2], [1], [1], [1])
    path = tmp_path / "big.sbbt"
    path.write_bytes(raw)
    reader = SbbtReader(path)
    monkeypatch.setattr(
        type(reader), "decompressed_bytes", property(lambda self: 9 << 30)
    )
    with pytest.raises(MemoryError, match="Use chunks"):
        reader.read_all()


# --------------------------------------------------------------------------- #
# normalised records (spec Phase 1)
# --------------------------------------------------------------------------- #

def test_iter_records_shape(synthetic):
    records = list(SbbtReader(synthetic, trace_id="t0").iter_records())
    assert len(records) == 6
    assert records[0] == {
        "pc": 0x1000,
        "taken": 1,
        "branch_index": 0,
        "trace_id": "t0",
    }
    assert [r["branch_index"] for r in records] == list(range(6))


def test_iter_records_conditional_only_keeps_absolute_indices(synthetic):
    records = list(SbbtReader(synthetic).iter_records(conditional_only=True))
    assert [r["branch_index"] for r in records] == [0, 1, 3, 5]


# --------------------------------------------------------------------------- #
# real trace
# --------------------------------------------------------------------------- #

@pytest.mark.real_trace
def test_real_trace_header_matches_known_values():
    header = read_header(SMOKE_TRACE)
    assert header.num_instructions == 110_000_000
    assert header.num_branches == 7_392_094


@pytest.mark.real_trace
def test_real_trace_streams_exactly_the_declared_branch_count():
    reader = SbbtReader(SMOKE_TRACE)
    total = sum(len(chunk) for chunk in reader.chunks())
    assert total == reader.header.num_branches


@pytest.mark.real_trace
def test_real_trace_padding_bits_are_always_zero():
    """Confirms the documented field layout against production data."""
    with open(SMOKE_TRACE, "rb") as fh:
        raw = zstd.ZstdDecompressor().stream_reader(fh).read()
    words = np.frombuffer(raw, dtype=np.uint64, offset=HEADER_BYTES)
    assert not np.any((words[0::2] >> np.uint64(4)) & np.uint64(0x7F))


@pytest.mark.real_trace
def test_real_trace_conditional_stats_are_stable():
    """Regression guard on decode correctness, from a verified earlier run."""
    reader = SbbtReader(SMOKE_TRACE)
    conditional = taken = 0
    for chunk in reader.chunks(conditional_only=True):
        conditional += len(chunk)
        taken += int(chunk.taken.sum())
    assert conditional == 7_039_641
    assert taken / conditional == pytest.approx(0.5175, abs=1e-4)
