"""Tests for the causal feature extractor, the splits, and the CNN.

Synthetic traces are built in the official byte layout using the helpers from
``test_branch_extraction``, so the rules are checked without depending on the
real traces being present.

The tests that matter most are the causality ones: that a sample's window stops
before its own branch slot, that per-PC local history contains only earlier
outcomes, and that an instruction executing after a branch can never be one of
its producers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import (
    DEFAULT_SPLIT,
    NUM_SCALARS,
    NUM_SEQ_NUMERIC,
    TRACE_FAMILIES,
    BranchDataset,
    DatasetConfig,
    SplitSpec,
    check_split,
    fold_pc,
    scalar_feature_names,
)
from src.feature_extraction import (
    LOCAL_HISTORY_BITS,
    MIX_CLASSES,
    SLICE_CLASSES,
    ExtractionConfig,
    FeatureExtractor,
    extract_trace,
    load_features,
    save_features,
)
from src.logical_instructions import read_logical_instructions
from src.model import BranchCNN, ModelConfig, count_parameters
from src.trace_inspection import (
    ALU,
    COND_BR,
    LOAD,
    STORE,
    UNCOND_DIRECT,
)
from tests.test_branch_extraction import encode_record, write_trace

FLAG_REG = 64
ZERO_REG = 65
GPR0, GPR1, GPR2 = 0, 1, 2


def run(path: Path, config: ExtractionConfig = ExtractionConfig()):
    return extract_trace(path, "test", "synthetic_trace", config)


# --- causality -------------------------------------------------------------


def test_sample_points_at_its_own_branch_slot(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, out_regs=(GPR0,)),
            encode_record(0x1004, COND_BR, taken=True, target=0x2000, in_regs=(GPR0,)),
            encode_record(0x2000, UNCOND_DIRECT, taken=True, target=0x3000),
            encode_record(0x3000, COND_BR, taken=False, in_regs=(GPR0,)),
        ],
    )
    features = run(trace)

    assert features.num_samples == 2
    # The slot a sample points at is its own branch: same class, same direction.
    assert np.all(features.h_class[features.s_pos] == COND_BR)
    assert np.all(features.h_dir[features.s_pos] == features.s_out)
    assert list(features.s_out) == [1, 0]


def test_window_stops_before_the_predicted_branch(tmp_path: Path, monkeypatch) -> None:
    """The batch window must be ``[pos - H, pos)``, never including ``pos``."""
    records = [encode_record(0x1000, ALU, out_regs=(GPR0,))]
    for step in range(40):
        records.append(
            encode_record(
                0x2000 + 4 * (step % 3),
                COND_BR,
                taken=bool(step % 2),
                target=0x5000,
                in_regs=(GPR0,),
            )
        )
    trace = write_trace(tmp_path / "t.gz", records)
    features = run(trace)
    path = tmp_path / "synthetic_trace.npz"
    save_features(features, path)

    monkeypatch.setitem(TRACE_FAMILIES, "synthetic_trace", "A")
    dataset = BranchDataset(
        tmp_path,
        DatasetConfig(history_length=4, pc_bits=8),
        SplitSpec(train=("A",), validation=("B",), test=("C",)),
    )
    dataset.load(roles=("train",))
    batch = next(dataset.iter_batches("train", batch_size=8))

    # Reconstruct the expected window for the first sample directly.
    rows = dataset.splits["train"].traces[0].rows
    position = int(features.s_pos[rows[0]])
    expected = features.h_dir[position - 4 : position]
    assert list(batch["tokens"][0] & 1) == list(expected)
    # The branch's own direction is the label and must not be the last input.
    assert int(features.h_dir[position]) == int(batch["label"][0])


def test_local_history_holds_only_earlier_outcomes(tmp_path: Path) -> None:
    outcomes = [True, True, False, True]
    records = [encode_record(0x1000, ALU, out_regs=(GPR0,))]
    for taken in outcomes:
        records.append(
            encode_record(
                0x2000, COND_BR, taken=taken, target=0x4000, in_regs=(GPR0,)
            )
        )
    features = run(write_trace(tmp_path / "t.gz", records))

    assert list(features.s_nseen) == [0, 1, 2, 3]
    # bit 0 is the most recent earlier outcome.
    assert list(features.s_local) == [0b0, 0b1, 0b11, 0b110]


def test_writer_after_the_branch_is_not_a_producer(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, out_regs=(GPR0,)),  # index 0
            encode_record(0x1004, COND_BR, taken=False, in_regs=(GPR0,)),  # index 1
            encode_record(0x1008, ALU, out_regs=(GPR0,)),  # index 2
        ],
    )
    features = run(trace)
    # Distance back to the producer is 1 (index 1 - index 0), not -1.
    assert int(features.s_near_writer[0]) == 1
    assert int(features.s_dep_ninstr[0]) == 1


def test_never_written_source_reports_no_producer(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [encode_record(0x1004, COND_BR, taken=False, in_regs=(GPR1,))],
    )
    features = run(trace)
    assert int(features.s_near_writer[0]) == np.iinfo(np.uint32).max
    assert int(features.s_dep_ninstr[0]) == 0


def test_zero_register_source_has_no_producer(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [encode_record(0x1004, COND_BR, taken=False, in_regs=(ZERO_REG,))],
    )
    features = run(trace)
    assert int(features.s_dep_ninstr[0]) == 0


# --- history contents ------------------------------------------------------


def test_history_covers_unconditional_branches_too(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, COND_BR, taken=False, in_regs=(GPR0,)),
            encode_record(0x1004, UNCOND_DIRECT, taken=True, target=0x2000),
            encode_record(0x2000, COND_BR, taken=False, in_regs=(GPR0,)),
        ],
    )
    features = run(trace)
    assert features.num_branches == 3
    assert features.num_samples == 2
    assert list(features.h_class) == [COND_BR, UNCOND_DIRECT, COND_BR]


def test_backward_branch_flag(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x2000, COND_BR, taken=True, target=0x1000, in_regs=(GPR0,)),
            encode_record(0x2004, COND_BR, taken=True, target=0x9000, in_regs=(GPR0,)),
            encode_record(0x2008, COND_BR, taken=False, in_regs=(GPR0,)),
        ],
    )
    features = run(trace)
    assert list(features.h_back) == [1, 0, 0]


def test_block_mix_counts_instruction_classes(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, out_regs=(GPR0,)),
            encode_record(0x1004, ALU, out_regs=(GPR1,)),
            encode_record(0x1008, LOAD, out_regs=(GPR2,)),
            encode_record(0x100C, STORE, in_regs=(GPR0,)),
            encode_record(0x1010, COND_BR, taken=False, in_regs=(GPR0,)),
        ],
    )
    features = run(trace)
    alu, load, store = (MIX_CLASSES.index(c) for c in (ALU, LOAD, STORE))
    assert int(features.h_mix[0, alu]) == 2
    assert int(features.h_mix[0, load]) == 1
    assert int(features.h_mix[0, store]) == 1
    assert int(features.h_ninstr[0]) == 4


def test_slice_mix_counts_classes_in_the_dependency_slice(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, out_regs=(GPR1,)),
            encode_record(0x1004, LOAD, in_regs=(GPR1,), out_regs=(GPR0,)),
            encode_record(0x1008, COND_BR, taken=False, in_regs=(GPR0,)),
        ],
    )
    features = run(trace)
    alu, load = (SLICE_CLASSES.index(c) for c in (ALU, LOAD))
    assert int(features.s_slice_mix[0, load]) == 1
    assert int(features.s_slice_mix[0, alu]) == 1
    assert int(features.s_nload[0]) == 1
    assert int(features.s_dep_ninstr[0]) == 2
    assert int(features.s_dep_npcs[0]) == 2


def test_flag_writer_class_is_recorded(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, in_regs=(GPR0,), out_regs=(FLAG_REG,)),
            encode_record(0x1004, COND_BR, taken=False, in_regs=(FLAG_REG,)),
        ],
    )
    features = run(trace)
    assert int(features.s_flags[0]) == 1
    assert int(features.s_fwclass[0]) == ALU


def test_no_flag_dependency_reports_sentinel_class(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, out_regs=(GPR0,)),
            encode_record(0x1004, COND_BR, taken=False, in_regs=(GPR0,)),
        ],
    )
    features = run(trace)
    assert int(features.s_flags[0]) == 0
    assert int(features.s_fwclass[0]) == 255


# --- bounds and configuration ---------------------------------------------


def test_depth_bound_is_reported_as_incomplete(tmp_path: Path) -> None:
    records = [encode_record(0x1000, ALU, out_regs=(GPR0,))]
    for step in range(10):
        records.append(
            encode_record(0x1004 + 4 * step, ALU, in_regs=(GPR0,), out_regs=(GPR0,))
        )
    records.append(encode_record(0x2000, COND_BR, taken=False, in_regs=(GPR0,)))
    features = run(write_trace(tmp_path / "t.gz", records), ExtractionConfig(max_depth=3))
    assert int(features.s_dep_depth[0]) == 3
    assert int(features.s_incomplete[0]) == 1


def test_sample_stride_thins_samples_but_not_history(tmp_path: Path) -> None:
    records = [encode_record(0x1000, ALU, out_regs=(GPR0,))]
    for step in range(10):
        records.append(
            encode_record(0x2000, COND_BR, taken=bool(step % 2), target=0x4000, in_regs=(GPR0,))
        )
    trace = write_trace(tmp_path / "t.gz", records)

    full = run(trace)
    thinned = run(trace, ExtractionConfig(sample_stride=3))
    assert thinned.num_samples == 4
    # History is untouched, so windows are identical between the two runs.
    assert full.num_branches == thinned.num_branches
    assert np.array_equal(full.h_dir, thinned.h_dir)
    assert np.array_equal(full.h_pc, thinned.h_pc)


def test_save_load_round_trip(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, out_regs=(GPR0,)),
            encode_record(0x1004, COND_BR, taken=True, target=0x2000, in_regs=(GPR0,)),
        ],
    )
    features = run(trace)
    path = tmp_path / "out.npz"
    save_features(features, path)
    restored = load_features(path)

    assert restored.trace_id == features.trace_id
    assert restored.instructions_seen == features.instructions_seen
    assert restored.config.max_depth == features.config.max_depth
    for name in ("h_pc", "h_dir", "h_mix", "s_pos", "s_out", "s_slice_mix", "s_local"):
        assert np.array_equal(getattr(restored, name), getattr(features, name))


# --- splits ----------------------------------------------------------------


def test_default_split_is_family_disjoint() -> None:
    check_split(DEFAULT_SPLIT)
    train = set(DEFAULT_SPLIT.traces("train"))
    validation = set(DEFAULT_SPLIT.traces("validation"))
    test = set(DEFAULT_SPLIT.traces("test"))
    assert not train & validation
    assert not train & test
    assert not validation & test
    # The default split is the media study: it must use every media trace exactly once.
    # (The registry also holds web traces, which have their own explicit splits.)
    media = {t for t in TRACE_FAMILIES if t.startswith("media_")}
    assert train | validation | test == media


def test_overlapping_families_are_rejected() -> None:
    with pytest.raises(ValueError, match="family|families"):
        check_split(SplitSpec(train=("A", "B"), validation=("B",), test=("C",)))


def test_empty_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="no families"):
        check_split(SplitSpec(train=("A",), validation=(), test=("C",)))


def test_family_members_never_straddle_a_split() -> None:
    for role in ("train", "validation", "test"):
        families = {TRACE_FAMILIES[t] for t in DEFAULT_SPLIT.traces(role)}
        for family in families:
            members = {t for t, f in TRACE_FAMILIES.items() if f == family}
            assert members <= set(DEFAULT_SPLIT.traces(role))


def test_fold_pc_mixes_high_bits() -> None:
    # Two addresses differing only above the mask must not collide.
    low = np.array([0x1234], dtype=np.uint64)
    high = np.array([0x1234 | (1 << 20)], dtype=np.uint64)
    assert fold_pc(low, 8)[0] != fold_pc(high, 8)[0]
    assert 0 <= int(fold_pc(high, 8)[0]) < (1 << 8)


def test_fold_pc_is_in_range() -> None:
    values = np.array([0, 1, 0xFFFFFFFF, 0xDEADBEEF12], dtype=np.uint64)
    folded = fold_pc(values, 10)
    assert folded.min() >= 0
    assert folded.max() < (1 << 10)


# --- feature assembly ------------------------------------------------------


def _toy_dataset(tmp_path: Path, monkeypatch, history_length: int = 4) -> BranchDataset:
    records = [encode_record(0x1000, ALU, out_regs=(GPR0,))]
    for step in range(60):
        records.append(encode_record(0x1100, LOAD, in_regs=(GPR0,), out_regs=(GPR1,)))
        records.append(
            encode_record(
                0x2000 + 4 * (step % 5),
                COND_BR,
                taken=bool(step % 3),
                target=0x9000,
                in_regs=(GPR1,),
            )
        )
    features = run(write_trace(tmp_path / "t.gz", records))
    save_features(features, tmp_path / "synthetic_trace.npz")

    monkeypatch.setitem(TRACE_FAMILIES, "synthetic_trace", "A")
    dataset = BranchDataset(
        tmp_path,
        DatasetConfig(history_length=history_length, pc_bits=8),
        SplitSpec(train=("A",), validation=("B",), test=("C",)),
    )
    dataset.load(roles=("train",))
    return dataset


def test_scalar_names_match_width() -> None:
    assert len(scalar_feature_names()) == NUM_SCALARS
    assert len(set(scalar_feature_names())) == NUM_SCALARS


def test_batch_shapes_and_finiteness(tmp_path: Path, monkeypatch) -> None:
    dataset = _toy_dataset(tmp_path, monkeypatch)
    batch = next(dataset.iter_batches("train", batch_size=16))

    assert batch["tokens"].shape == (16, 4)
    assert batch["classes"].shape == (16, 4)
    assert batch["seq_numeric"].shape == (16, 4, NUM_SEQ_NUMERIC)
    assert batch["scalars"].shape == (16, NUM_SCALARS)
    assert batch["pc"].shape == (16,)
    assert np.isfinite(batch["scalars"]).all()
    assert np.isfinite(batch["seq_numeric"]).all()
    assert set(np.unique(batch["label"])) <= {0.0, 1.0}
    assert batch["tokens"].max() < dataset.vocabulary_size
    assert batch["pc"].max() < dataset.num_pc_ids


def test_samples_without_a_full_window_are_dropped(tmp_path: Path, monkeypatch) -> None:
    dataset = _toy_dataset(tmp_path, monkeypatch, history_length=8)
    entry = dataset.splits["train"].traces[0]
    assert np.all(entry.features.s_pos[entry.rows] >= 8)


def test_normaliser_is_fitted_on_train_only(tmp_path: Path, monkeypatch) -> None:
    dataset = _toy_dataset(tmp_path, monkeypatch)
    mean, std = dataset.normaliser()
    assert mean.shape == (NUM_SCALARS,)
    assert std.shape == (NUM_SCALARS,)
    assert np.all(std > 0)


def test_shuffle_labels_preserves_inputs(tmp_path: Path, monkeypatch) -> None:
    dataset = _toy_dataset(tmp_path, monkeypatch)
    plain = next(dataset.iter_batches("train", batch_size=32, seed=1))
    shuffled = next(
        dataset.iter_batches("train", batch_size=32, seed=1, shuffle_labels=True)
    )
    assert np.array_equal(plain["tokens"], shuffled["tokens"])
    assert np.array_equal(plain["scalars"], shuffled["scalars"])
    assert plain["label"].sum() == shuffled["label"].sum()


# --- model -----------------------------------------------------------------


def _model_config(**overrides) -> ModelConfig:
    base = dict(
        vocabulary_size=512,
        num_pc_ids=256,
        num_scalars=NUM_SCALARS,
        num_seq_numeric=NUM_SEQ_NUMERIC,
        history_length=16,
        conv_channels=(8, 8),
        head_dims=(16,),
    )
    base.update(overrides)
    return ModelConfig(**base)


def _forward(config: ModelConfig, batch: int = 4):
    model = BranchCNN(config)
    tokens = torch.randint(0, config.vocabulary_size, (batch, config.history_length))
    classes = torch.randint(0, 12, (batch, config.history_length))
    numeric = torch.randn(batch, config.history_length, config.num_seq_numeric)
    scalars = torch.randn(batch, config.num_scalars)
    pc = torch.randint(0, config.num_pc_ids, (batch,))
    return model(tokens, classes, numeric, scalars, pc)


def test_model_forward_shape() -> None:
    logit, aux = _forward(_model_config())
    assert logit.shape == (4,)
    assert aux is None


def test_model_ablations_change_head_width() -> None:
    full = BranchCNN(_model_config())
    no_scalars = BranchCNN(_model_config(use_scalars=False))
    no_pc = BranchCNN(_model_config(use_pc=False))
    assert no_scalars.head_inputs == full.head_inputs - NUM_SCALARS
    assert no_pc.head_inputs == full.head_inputs - full.config.pc_embed_dim
    for config in (
        _model_config(use_scalars=False),
        _model_config(use_pc=False),
        _model_config(use_sequence=False),
    ):
        logit, _ = _forward(config)
        assert logit.shape == (4,)


def test_model_requires_some_input() -> None:
    with pytest.raises(ValueError, match="at least one input group"):
        BranchCNN(
            _model_config(use_sequence=False, use_scalars=False, use_pc=False)
        )


def test_auxiliary_head_emits_targets() -> None:
    logit, aux = _forward(_model_config(aux_targets=4))
    assert logit.shape == (4,)
    assert aux is not None and aux.shape == (4, 4)


def test_model_is_trainable() -> None:
    config = _model_config()
    model = BranchCNN(config)
    before = count_parameters(model)
    assert before > 0
    logit, _ = _forward(config)
    logit.sum().backward()


def test_gradient_reaches_every_enabled_input_group() -> None:
    config = _model_config()
    model = BranchCNN(config)
    tokens = torch.randint(0, config.vocabulary_size, (4, config.history_length))
    classes = torch.randint(0, 12, (4, config.history_length))
    numeric = torch.randn(4, config.history_length, config.num_seq_numeric)
    scalars = torch.randn(4, config.num_scalars, requires_grad=True)
    pc = torch.randint(0, config.num_pc_ids, (4,))
    logit, _ = model(tokens, classes, numeric, scalars, pc)
    logit.sum().backward()
    assert scalars.grad is not None and torch.any(scalars.grad != 0)
    assert model.token_embed.weight.grad is not None
    assert model.pc_embed.weight.grad is not None


# --- the existing pipeline must keep working -------------------------------


def test_stage3_extraction_is_unaffected(tmp_path: Path) -> None:
    """The new module must not change the validated Stage 3 output."""
    from src.branch_extraction import extract_branches

    trace = write_trace(
        tmp_path / "t.gz",
        [
            encode_record(0x1000, ALU, out_regs=(GPR0,)),
            encode_record(0x1004, COND_BR, taken=True, target=0x2000, in_regs=(GPR0,)),
            encode_record(0x2000, COND_BR, taken=False, in_regs=(GPR0,)),
        ],
    )
    frame, _ = extract_branches(trace, workload_id="test", trace_id="synthetic_trace")
    assert len(frame) == 2
    assert list(frame["outcome"]) == [1, 0]

    # And an extraction pass over the same trace agrees on the outcomes.
    features = run(trace)
    assert list(features.s_out) == list(frame["outcome"])


def test_reader_and_extractor_agree_on_branch_count(tmp_path: Path) -> None:
    records = [
        encode_record(0x1000, ALU, out_regs=(GPR0,)),
        encode_record(0x1004, COND_BR, taken=False, in_regs=(GPR0,)),
        encode_record(0x1008, UNCOND_DIRECT, taken=True, target=0x2000),
        encode_record(0x2000, LOAD, out_regs=(GPR1,)),
        encode_record(0x2004, COND_BR, taken=True, target=0x3000, in_regs=(GPR1,)),
    ]
    trace = write_trace(tmp_path / "t.gz", records)
    features = run(trace)

    import gzip
    import io

    with gzip.open(trace, "rb") as raw:
        decoded = list(read_logical_instructions(io.BufferedReader(raw)))
    branch_classes = sum(1 for d in decoded if d.is_branch)
    cond = sum(1 for d in decoded if d.is_conditional_branch)
    assert features.num_branches == branch_classes
    assert features.num_samples == cond


def test_extractor_handles_empty_trace(tmp_path: Path) -> None:
    features = run(write_trace(tmp_path / "t.gz", []))
    assert features.num_branches == 0
    assert features.num_samples == 0
    assert features.h_mix.shape == (0, len(MIX_CLASSES))
    assert features.s_slice_mix.shape == (0, len(SLICE_CLASSES))


def test_local_history_width_is_respected(tmp_path: Path) -> None:
    records = [encode_record(0x1000, ALU, out_regs=(GPR0,))]
    for _ in range(LOCAL_HISTORY_BITS + 10):
        records.append(
            encode_record(0x2000, COND_BR, taken=True, target=0x4000, in_regs=(GPR0,))
        )
    features = run(write_trace(tmp_path / "t.gz", records))
    assert features.s_local.max() < (1 << LOCAL_HISTORY_BITS)
    # All-taken history saturates to every bit set.
    assert int(features.s_local[-1]) == (1 << LOCAL_HISTORY_BITS) - 1
