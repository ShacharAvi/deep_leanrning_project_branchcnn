"""Dataset tests -- spec §28 leakage checklist, as assertions rather than prose.

The checklist is the most consequential part of this project: a leak does not
crash, it produces an excellent-looking number. So each item gets a test, and
several are written to fail loudly if someone later "simplifies" the window or
split logic.

Most tests run on a small synthetic trace built with ``build_trace_bytes``, so
the expected windows can be computed by hand. 
"""

import numpy as np
import pytest
import torch

from src.dataset import (
    BranchDataset,
    build_trace_dataset,
    decode_token,
    encode_tokens,
    preprocess_history,
    read_hard_branches,
    remove_incomplete_histories,
    token_dtype,
    vocabulary_size,
)
from src.traces import OpCode, SbbtReader, build_trace_bytes


TARGET = 0x1000
OTHER = 0x2000


@pytest.fixture
def tiny_trace(tmp_path):
    """400 branches: TARGET every 5th, an unconditional every 7th.

    Small enough that every expected window can be derived by hand.
    """
    ips, takens, opcodes = [], [], []
    for i in range(400):
        if i % 5 == 0:
            ips.append(TARGET)
            takens.append(i % 2)
            opcodes.append(OpCode.JUMP | OpCode.CND)
        elif i % 7 == 0:
            ips.append(0x3000)
            takens.append(1)
            opcodes.append(OpCode.JUMP)        # unconditional: tracked, not predicted
        else:
            ips.append(OTHER)
            takens.append((i // 3) % 2)
            opcodes.append(OpCode.JUMP | OpCode.CND)

    raw = build_trace_bytes(ip=ips, target=[p + 8 for p in ips], opcode=opcodes,
                            taken=takens, ninstr=[4] * len(ips))
    path = tmp_path / "tiny.sbbt"
    path.write_bytes(raw)
    return path, ips, takens


@pytest.fixture
def tiny_dataset(tmp_path, tiny_trace):
    path, ips, takens = tiny_trace
    out = tmp_path / "tiny.hdf5"
    stats = build_trace_dataset(path, out, [TARGET, OTHER], split_role="train")
    return out, ips, takens, stats


# --------------------------------------------------------------------------- #
# token encoding
# --------------------------------------------------------------------------- #

def test_token_roundtrip():
    pcs = np.array([0x1000, 0x401234, 0x3FFFFFFF], dtype=np.int64)
    takens = np.array([1, 0, 1], dtype=np.uint8)
    tokens = encode_tokens(pcs, takens, 30)
    for token, pc, taken in zip(tokens.tolist(), pcs.tolist(), takens.tolist()):
        assert decode_token(token, 30) == (pc & ((1 << 30) - 1), taken)


def test_token_is_pc_shifted_plus_direction():
    """Spec §6.2: token_id = (pc_low << 1) | direction."""
    token = int(encode_tokens(np.array([0xABC], np.int64),
                              np.array([1], np.uint8), 12)[0])
    assert token == (0xABC << 1) | 1


def test_token_dtype_follows_width():
    assert token_dtype(6) == np.uint8      # 7 bits
    assert token_dtype(12) == np.uint16    # 13 bits
    assert token_dtype(30) == np.uint32    # 31 bits


def test_vocabulary_size_matches_the_spec_example():
    """Spec §6.2: p=12 -> 2^13 = 8192 token ids."""
    assert vocabulary_size(12, 12, False) == 8192


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #

def test_history_covers_all_branches_not_just_conditional(tiny_dataset):
    """Matches BranchNet: its tracer records every branch."""
    import h5py
    out, ips, _, stats = tiny_dataset
    with h5py.File(out, "r") as fh:
        assert fh["history"].shape[0] == len(ips)
        assert fh.attrs["history_covers"].startswith("all branches")
    assert stats.total_branches == len(ips)


def test_occurrence_indices_are_positions_in_history(tiny_dataset):
    import h5py
    out, ips, _, _ = tiny_dataset
    with h5py.File(out, "r") as fh:
        indices = fh[f"br_indices_{hex(TARGET)}"][:]
        history = fh["history"][:]
    expected = [i for i, pc in enumerate(ips) if pc == TARGET]
    assert indices.tolist() == expected
    for i in indices:
        assert decode_token(int(history[i]))[0] == TARGET


def test_class_balance_attrs_match_a_recount(tiny_dataset):
    import h5py
    out, ips, takens, _ = tiny_dataset
    with h5py.File(out, "r") as fh:
        taken = int(fh.attrs[f"num_taken_{hex(TARGET)}"])
        not_taken = int(fh.attrs[f"num_not_taken_{hex(TARGET)}"])
    actual = [t for pc, t in zip(ips, takens) if pc == TARGET]
    assert taken == sum(actual)
    assert not_taken == len(actual) - sum(actual)


def test_split_role_is_recorded(tiny_dataset):
    import h5py
    out, *_ = tiny_dataset
    with h5py.File(out, "r") as fh:
        assert fh.attrs["split_role"] == "train"


def test_invalid_split_role_rejected(tmp_path, tiny_trace):
    path, *_ = tiny_trace
    with pytest.raises(ValueError, match="split_role"):
        build_trace_dataset(path, tmp_path / "x.hdf5", [TARGET], split_role="holdout")


def test_refuses_to_overwrite_silently(tmp_path, tiny_dataset):
    out, *_ = tiny_dataset
    path = tmp_path / "tiny.sbbt"
    with pytest.raises(FileExistsError):
        build_trace_dataset(path, out, [TARGET], split_role="train")


def test_absent_pc_is_reported_not_silently_dropped(tmp_path, tiny_trace):
    path, *_ = tiny_trace
    stats = build_trace_dataset(path, tmp_path / "m.hdf5", [TARGET, 0xDEAD],
                                split_role="train")
    assert stats.missing_pcs == [0xDEAD]


def test_build_is_deterministic(tmp_path, tiny_trace):
    """Same trace in, byte-identical datasets out (spec §27)."""
    import h5py
    path, *_ = tiny_trace
    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    build_trace_dataset(path, a, [TARGET], split_role="train")
    build_trace_dataset(path, b, [TARGET], split_role="train")
    with h5py.File(a) as fa, h5py.File(b) as fb:
        assert np.array_equal(fa["history"][:], fb["history"][:])
        assert np.array_equal(fa[f"br_indices_{hex(TARGET)}"][:],
                              fb[f"br_indices_{hex(TARGET)}"][:])


def test_hard_branch_file_parsing(tmp_path):
    path = tmp_path / "brs.txt"
    path.write_text("# comment\n\n  40638e  # trailing\n406fae\n")
    assert read_hard_branches(path) == [0x40638E, 0x406FAE]


# --------------------------------------------------------------------------- #
# spec §28 -- the leakage checklist
# --------------------------------------------------------------------------- #

def test_target_outcome_is_not_in_its_own_window(tiny_dataset):
    """Checklist item 1. The single most important property here."""
    import h5py
    out, ips, takens, _ = tiny_dataset
    H = 20
    ds = BranchDataset([out], br_pc=TARGET, history_length=H, pc_bits=30)
    with h5py.File(out, "r") as fh:
        indices = remove_incomplete_histories(
            fh[f"br_indices_{hex(TARGET)}"][:], H)

    for sample_idx in range(min(10, len(ds))):
        inputs, label = ds[sample_idx]
        position = int(indices[sample_idx])
        assert len(inputs) == H
        # The window is exactly the H branches BEFORE the target.
        expected = [((ips[j] & ((1 << 30) - 1)) << 1) | takens[j]
                    for j in range(position - H, position)]
        assert inputs.tolist() == expected
        assert float(label) == float(takens[position])


def test_no_future_branch_appears_in_any_window(tiny_dataset):
    """Checklist item 2."""
    import h5py
    out, ips, takens, _ = tiny_dataset
    H = 15
    ds = BranchDataset([out], br_pc=TARGET, history_length=H, pc_bits=30)
    with h5py.File(out, "r") as fh:
        indices = remove_incomplete_histories(
            fh[f"br_indices_{hex(TARGET)}"][:], H)
    full = [((ips[j] & ((1 << 30) - 1)) << 1) | takens[j] for j in range(len(ips))]

    for sample_idx in range(len(ds)):
        inputs, _ = ds[sample_idx]
        position = int(indices[sample_idx])
        # Every token in the window must come from strictly before the target.
        assert inputs.tolist() == full[position - H:position]


def test_incomplete_histories_are_dropped(tiny_dataset):
    """Checklist item 4: every window is exactly H long, never padded."""
    import h5py
    out, *_ = tiny_dataset
    H = 50
    with h5py.File(out, "r") as fh:
        raw = fh[f"br_indices_{hex(TARGET)}"][:]
    kept = remove_incomplete_histories(raw, H)
    assert (kept >= H).all()
    assert len(kept) == int((raw >= H).sum())

    ds = BranchDataset([out], br_pc=TARGET, history_length=H, pc_bits=30)
    assert len(ds) == len(kept)
    for i in range(len(ds)):
        assert len(ds[i][0]) == H


def test_dataset_refuses_to_mix_split_roles(tmp_path, tiny_trace):
    """Checklist item 3: a test trace must not be usable as training data."""
    path, *_ = tiny_trace
    train = tmp_path / "tr.hdf5"
    test = tmp_path / "te.hdf5"
    build_trace_dataset(path, train, [TARGET], split_role="train")
    build_trace_dataset(path, test, [TARGET], split_role="test")

    BranchDataset([train], br_pc=TARGET, history_length=10,
                  pc_bits=30, expected_split_role="train")
    with pytest.raises(ValueError, match="refusing to mix splits"):
        BranchDataset([train, test], br_pc=TARGET, history_length=10,
                      pc_bits=30, expected_split_role="train")


def test_dataset_is_deterministic(tiny_dataset):
    """Checklist item 6."""
    out, *_ = tiny_dataset
    a = BranchDataset([out], br_pc=TARGET, history_length=20, pc_bits=30)
    b = BranchDataset([out], br_pc=TARGET, history_length=20, pc_bits=30)
    assert len(a) == len(b)
    for i in (0, len(a) // 2, len(a) - 1):
        assert torch.equal(a[i][0], b[i][0])
        assert a[i][1] == b[i][1]


def test_pos_weight_matches_a_recount(tiny_dataset):
    out, ips, takens, _ = tiny_dataset
    ds = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30)
    actual = [t for pc, t in zip(ips, takens) if pc == TARGET]
    taken, not_taken = sum(actual), len(actual) - sum(actual)
    assert ds.pos_weight == pytest.approx(not_taken / taken)
    assert ds.taken_rate == pytest.approx(taken / len(actual))


def test_unknown_branch_is_rejected(tiny_dataset):
    out, *_ = tiny_dataset
    with pytest.raises(KeyError, match="no index for branch"):
        BranchDataset([out], br_pc=0xBEEF, history_length=10, pc_bits=30)


def test_cannot_request_more_pc_bits_than_stored(tmp_path, tiny_trace):
    path, *_ = tiny_trace
    out = tmp_path / "narrow.hdf5"
    build_trace_dataset(path, out, [TARGET], split_role="train", pc_bits=12)
    with pytest.raises(ValueError, match="cannot"):
        BranchDataset([out], br_pc=TARGET, history_length=10, pc_bits=30)


# --------------------------------------------------------------------------- #
# history preprocessing
# --------------------------------------------------------------------------- #

def test_preprocess_masks_to_pc_bits():
    tokens = np.array([(0xABCDEF << 1) | 1], dtype=np.int64)
    out = preprocess_history(tokens, pc_bits=12, pc_hash_bits=12,
                             hash_dir_with_pc=False, dtype=np.int64)
    assert int(out[0]) == ((0xABCDEF << 1) | 1) & ((1 << 13) - 1)


def test_preprocess_does_not_mutate_its_input():
    """It operates in-place internally; a shared array must survive."""
    tokens = np.array([(0xABCDEF << 1) | 1], dtype=np.int64)
    before = tokens.copy()
    preprocess_history(tokens, 12, 12, False, np.int64)
    assert np.array_equal(tokens, before)


def test_preprocess_preserves_the_direction_bit():
    """Direction must survive folding -- it is the label's source."""
    for pc in (0x1234, 0xABCDE, 0x3FFFF):
        for direction in (0, 1):
            token = np.array([(pc << 1) | direction], dtype=np.int64)
            out = preprocess_history(token, 20, 8, False, np.int64)
            assert int(out[0]) & 1 == direction


def test_folded_tokens_stay_within_the_vocabulary():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1 << 31, size=5000, dtype=np.int64)
    for pc_bits, hash_bits, hash_dir in [(30, 12, False), (12, 12, False),
                                         (30, 8, False), (12, 7, True)]:
        out = preprocess_history(tokens, pc_bits, hash_bits, hash_dir, np.int64)
        assert out.min() >= 0
        assert out.max() < vocabulary_size(pc_bits, hash_bits, hash_dir)


def test_hash_bits_wider_than_pc_bits_rejected():
    with pytest.raises(ValueError, match="pc_hash_bits"):
        preprocess_history(np.array([1]), pc_bits=8, pc_hash_bits=12,
                           hash_dir_with_pc=False, dtype=np.int64)


# --------------------------------------------------------------------------- #
# training-set subsampling
# --------------------------------------------------------------------------- #

def test_subsampling_reduces_length_deterministically(tiny_dataset):
    out, *_ = tiny_dataset
    full = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30)
    a = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30,
                      max_samples=20, seed=7)
    b = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30,
                      max_samples=20, seed=7)
    assert len(a) == 20 < len(full)
    assert a.total_available == len(full)
    for i in range(len(a)):
        assert torch.equal(a[i][0], b[i][0]) and a[i][1] == b[i][1]


def test_different_seeds_select_different_samples(tiny_dataset):
    out, *_ = tiny_dataset
    a = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30,
                      max_samples=20, seed=1)
    b = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30,
                      max_samples=20, seed=2)
    assert any(not torch.equal(a[i][0], b[i][0]) for i in range(len(a)))


def test_subsampled_windows_are_still_true_histories(tiny_dataset):
    """Subsampling must thin the sample set, never alter a window."""
    out, ips, takens, _ = tiny_dataset
    H = 10
    sub = BranchDataset([out], br_pc=TARGET, history_length=H, pc_bits=30,
                        max_samples=15, seed=3)
    full_tokens = [((ips[j] & ((1 << 30) - 1)) << 1) | takens[j]
                   for j in range(len(ips))]
    for i in range(len(sub)):
        inputs, label = sub[i]
        # The window must appear verbatim somewhere in the real token stream,
        # immediately followed by a TARGET whose direction is the label.
        tokens = inputs.tolist()
        matched = False
        for pos in range(H, len(full_tokens)):
            if full_tokens[pos - H:pos] == tokens:
                if ips[pos] == TARGET and float(takens[pos]) == float(label):
                    matched = True
                    break
        assert matched, f"sample {i} is not a genuine window"


def test_cap_above_available_is_a_noop(tiny_dataset):
    out, *_ = tiny_dataset
    full = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30)
    capped = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30,
                           max_samples=10_000_000)
    assert len(capped) == len(full)


def test_refuses_to_subsample_a_test_set(tiny_dataset):
    """Hybrid MPKI needs every instance; a thinned test set makes it fiction."""
    out, *_ = tiny_dataset
    with pytest.raises(ValueError, match="refusing to subsample a test set"):
        BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30,
                      expected_split_role="test", max_samples=10)


def test_subsampled_indices_stay_sorted(tiny_dataset):
    """Sequential HDF5 reads; also guards against accidental shuffling."""
    out, *_ = tiny_dataset
    ds = BranchDataset([out], br_pc=TARGET, history_length=5, pc_bits=30,
                       max_samples=25, seed=0)
    for source in ds.sources:
        assert np.all(np.diff(source.indices) > 0)
