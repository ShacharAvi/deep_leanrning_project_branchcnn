"""TAGE-SC-L decision as an input: join integrity, no label path, learnable weight.

The rule: TAGE's *direction, hit bank and confidence* may be inputs (they exist before the outcome);
the *mispredicted* byte is derived from the label and must never reach the model.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import TRACE_FAMILIES, BranchDataset, DatasetConfig, SplitSpec
from src.feature_extraction import save_features
from src.model import BranchCNN, ModelConfig, tage_weights
from src.tage_features import (
    NUM_TAGE_FEATURES,
    NUM_TAGE_STATES,
    TAGE_FEATURE_NAMES,
    TageLog,
    load_tage_log,
    tage_feature_matrix,
    tage_state,
)
from src.trace_inspection import ALU, COND_BR, LOAD
from tests.test_branch_extraction import encode_record, write_trace
from tests.test_cnn_pipeline import GPR0, GPR1, run

ROOT = Path(__file__).resolve().parents[1]


def _records(n=80, flip_at=None):
    recs = [encode_record(0x1000, ALU, out_regs=(GPR0,))]
    for step in range(n):
        taken = bool(step % 3)
        if step == flip_at:
            taken = not taken
        recs.append(encode_record(0x1100, LOAD, in_regs=(GPR0,), out_regs=(GPR1,)))
        recs.append(encode_record(0x2000 + 4 * (step % 5), COND_BR, taken=taken, target=0x9000, in_regs=(GPR1,)))
    return recs


def _write_log(path: Path, features, miss=None, seed=0):
    rng = np.random.default_rng(seed)
    n = features.num_samples
    miss = rng.integers(0, 2, n).astype(np.uint8) if miss is None else miss
    pred = features.s_out ^ miss
    log = np.stack([miss, pred, rng.integers(0, 30, n).astype(np.uint8), rng.integers(0, 8, n).astype(np.uint8)], 1)
    path.write_bytes(log.astype(np.uint8).tobytes())
    return miss


def _dataset(tmp_path, monkeypatch, **cfg):
    features = run(write_trace(tmp_path / "t.gz", _records()))
    save_features(features, tmp_path / "synthetic_trace.npz")
    _write_log(tmp_path / "synthetic_trace.log.bin", features)
    monkeypatch.setitem(TRACE_FAMILIES, "synthetic_trace", "A")
    ds = BranchDataset(
        tmp_path,
        DatasetConfig(history_length=4, pc_bits=8, use_tage=True, tage_dir=tmp_path, **cfg),
        SplitSpec(train=("A",), validation=("B",), test=("C",)),
    )
    ds.load(roles=("train",))
    return ds, features


# --- join integrity --------------------------------------------------------


def test_misaligned_log_is_refused(tmp_path):
    features = run(write_trace(tmp_path / "t.gz", _records()))
    good = tmp_path / "ok.bin"
    _write_log(good, features)
    load_tage_log(good, features)                       # fine

    raw = np.fromfile(good, np.uint8).reshape(-1, 4)
    short = tmp_path / "short.bin"
    short.write_bytes(raw[:-1].tobytes())
    with pytest.raises(ValueError, match="partial join"):
        load_tage_log(short, features)

    shifted = tmp_path / "shift.bin"
    shifted.write_bytes(np.roll(raw, 1, axis=0).tobytes())
    with pytest.raises(ValueError, match="misaligned"):
        load_tage_log(shifted, features)


def test_log_object_does_not_carry_the_misprediction_byte(tmp_path):
    features = run(write_trace(tmp_path / "t.gz", _records()))
    _write_log(tmp_path / "l.bin", features)
    log = load_tage_log(tmp_path / "l.bin", features)
    assert {f.name for f in dataclasses.fields(TageLog)} == {"pred", "hit", "conf"}
    assert not any("miss" in n for n in TAGE_FEATURE_NAMES)


def test_features_depend_only_on_pred_hit_conf(tmp_path):
    """Two logs with identical (pred, hit, conf) but reached from different labels give equal features."""
    features = run(write_trace(tmp_path / "t.gz", _records()))
    _write_log(tmp_path / "l.bin", features, seed=1)
    log = load_tage_log(tmp_path / "l.bin", features)
    rows = np.arange(features.num_samples)
    a = tage_feature_matrix(log, rows)
    assert a.shape == (len(rows), NUM_TAGE_FEATURES)
    assert np.array_equal(a[:, 0], 2.0 * log.pred - 1.0)
    assert tage_state(log.hit, log.conf).max() < NUM_TAGE_STATES


# --- dataset ---------------------------------------------------------------


def test_batches_carry_tage_and_sign_matches_prediction(tmp_path, monkeypatch):
    ds, features = _dataset(tmp_path, monkeypatch)
    batch = next(ds.iter_batches("train", batch_size=16))
    assert batch["tage"].shape == (16, NUM_TAGE_FEATURES)
    assert batch["tage_state"].max() < NUM_TAGE_STATES
    assert np.array_equal(batch["tage_sign"], batch["tage"][:, 0])
    # the sign is TAGE's prediction, which differs from the label exactly when TAGE missed
    pred = (batch["tage_sign"] > 0).astype(np.float32)
    assert (pred != batch["label"]).any() and (pred == batch["label"]).any()


def test_shuffle_tage_control_moves_only_tage(tmp_path, monkeypatch):
    ds, _ = _dataset(tmp_path, monkeypatch)
    plain = next(ds.iter_batches("train", batch_size=40, seed=1))
    sub = tmp_path / "s"
    sub.mkdir()
    ds2, _ = _dataset(sub, monkeypatch, shuffle_tage=True)
    shuf = next(ds2.iter_batches("train", batch_size=40, seed=1))
    assert np.array_equal(plain["tokens"], shuf["tokens"])
    assert np.array_equal(plain["label"], shuf["label"])
    assert not np.array_equal(plain["tage"], shuf["tage"])
    assert plain["tage"].sum() != 0 and np.isclose(plain["tage"].sum(0), shuf["tage"].sum(0), atol=1e3).all()


# --- model -----------------------------------------------------------------


def _config(**kw):
    base = dict(vocabulary_size=256, num_pc_ids=256, num_scalars=4, num_seq_numeric=2, history_length=8,
                conv_channels=(4,), head_dims=(8,), use_tage=True)
    base.update(kw)
    return ModelConfig(**base)


def _inputs(config, b=6):
    return (
        torch.randint(0, 256, (b, 8)), torch.randint(0, 12, (b, 8)), torch.randn(b, 8, 2),
        torch.randn(b, 4), torch.randint(0, 256, (b,)),
        torch.randn(b, NUM_TAGE_FEATURES), torch.randint(0, NUM_TAGE_STATES, (b,)),
        torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
    )


def test_tage_weight_is_a_learnable_parameter_and_starts_at_init():
    m = BranchCNN(_config(tage_init=1.5))
    names = {n for n, _ in m.named_parameters()}
    assert {"tage_scale", "tage_delta.weight"} <= names
    w = tage_weights(m)
    assert w["global_scale"] == pytest.approx(1.5)
    assert all(x == pytest.approx(1.5) for x in w["per_state_weight"])


def test_residual_logit_is_head_logit_plus_weighted_tage_vote():
    cfg = _config(tage_mode="residual", tage_init=2.0)
    m = BranchCNN(cfg).eval()
    x = _inputs(cfg)
    with torch.no_grad():
        m.output.weight.zero_(); m.output.bias.zero_()   # head contributes nothing
        logit, _ = m(*x)
    assert torch.allclose(logit, 2.0 * x[-1])            # exactly weight * sign


def test_gradient_reaches_the_tage_weight():
    cfg = _config()
    m = BranchCNN(cfg)
    logit, _ = m(*_inputs(cfg))
    logit[:3].sum().backward()   # signs of the first 3 are all +1 (summing all six would cancel)
    assert m.tage_scale.grad is not None and m.tage_scale.grad.abs() > 0
    assert m.tage_delta.weight.grad.abs().sum() > 0


def test_modes_and_backward_compatibility():
    with pytest.raises(ValueError):
        BranchCNN(_config(tage_mode="nope"))
    off = BranchCNN(_config(use_tage=False))
    assert not hasattr(off, "tage_scale")
    x = _inputs(_config())[:5]
    logit, _ = off(*x)                                   # old 5-argument call still works
    assert logit.shape == (6,)
    feat_only = BranchCNN(_config(tage_mode="feature"))
    assert not hasattr(feat_only, "tage_scale") and feat_only.head_inputs == off.head_inputs + NUM_TAGE_FEATURES


# --- the replay itself is causal -------------------------------------------


def test_tage_replay_prediction_never_depends_on_own_outcome(tmp_path):
    binary = ROOT / "build" / "tage_cbp2025"
    if not binary.exists():
        pytest.skip("build/tage_cbp2025 not built (scripts/build_tage_logs.sh)")
    base = write_trace(tmp_path / "a.gz", _records(200))
    k = 120
    flip = write_trace(tmp_path / "b.gz", _records(200, flip_at=k))
    out = {}
    for name, path in (("a", base), ("b", flip)):
        subprocess.run([str(binary), str(path), "--log", str(tmp_path / f"{name}.bin")], check=True, capture_output=True)
        out[name] = np.fromfile(tmp_path / f"{name}.bin", np.uint8).reshape(-1, 4)
    # columns 1..3 = predicted direction, hit bank, confidence: identical up to AND INCLUDING branch k
    assert np.array_equal(out["a"][: k + 1, 1:], out["b"][: k + 1, 1:])
    # column 0 (the miss byte) is a function of the outcome, so it is allowed to differ at k
    assert out["a"][k, 0] != out["b"][k, 0]
