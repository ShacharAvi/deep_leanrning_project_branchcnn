"""Arbitration policies, the cross-trace split, and per-sample weights.

The arbiter is where the project's own contribution lives, so the tests pin
the properties that make its numbers trustworthy: it is fitted on validation
arrays only, a rule chosen on noise is rejected, and per-execution
arbitration recovers the case per-branch attachment cannot.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.arbiter import (
    HIGH, LOW, MEDIUM, Executions, Rule, branch_gate, fit_confidence_rule,
    score)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def synthetic(n=20000, seed=0, model_good_on_low=True):
    """TAGE: right on high-confidence executions, 60% on low-confidence ones.
    Model: right on low-confidence executions, 80% on high-confidence ones."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).astype(np.uint8)
    conf = rng.choice([HIGH, MEDIUM, LOW], n, p=[0.7, 0.1, 0.2]).astype(np.int8)
    tage_correct = np.where(conf == HIGH, True, rng.random(n) < 0.6)
    if model_good_on_low:
        model_correct = np.where(conf == HIGH, rng.random(n) < 0.8, True)
    else:
        model_correct = rng.random(n) < 0.5
    sign = np.where(labels == 1, 1.0, -1.0)
    logits = np.where(model_correct, sign, -sign) * rng.uniform(0.5, 3, n)
    return Executions(logits.astype(np.float32), labels, tage_correct, conf)


def test_confidence_arbiter_beats_branch_gate_when_model_is_mixed():
    val, test = synthetic(seed=1), synthetic(seed=2)
    rule = fit_confidence_rule(val)
    assert rule.conf_set in ("low", "low+medium")
    conf_score = score(test, rule)
    gate_score = score(test, branch_gate(val))
    assert conf_score["hybrid_misses"] < conf_score["tage_misses"]
    assert conf_score["hybrid_misses"] < gate_score["hybrid_misses"]


def test_rule_fitted_on_noise_is_rejected():
    val = synthetic(seed=3, model_good_on_low=False)
    rule = fit_confidence_rule(val)
    assert rule.conf_set is None
    assert score(val, rule)["hybrid_misses"] == score(val, rule)["tage_misses"]


def test_ordering_on_validation():
    """On the arrays it was fitted on, oracle <= confidence rule <= TAGE."""
    val = synthetic(seed=4)
    rule = fit_confidence_rule(val)
    s, o = score(val, rule), score(val, "oracle")
    assert o["hybrid_misses"] <= s["hybrid_misses"] <= s["tage_misses"]


def test_misaligned_arrays_raise():
    with pytest.raises(ValueError, match="misaligned"):
        Executions(np.zeros(5, np.float32), np.zeros(4), np.zeros(5, bool),
                   np.zeros(5, np.int8))


def test_split_is_a_partition_and_folds_rotate_test():
    from scripts.cbp2025_crosstrace import make_split
    members = [f"web_{i}" for i in range(23)]
    seen_test = set()
    for fold in range(4):
        s = make_split(members, "web_g0", fold, 0.2, 0.2)
        roles = s["train"] + s["validation"] + s["test"]
        assert sorted(roles) == sorted(members)
        assert not (set(s["test"]) & seen_test), "test sets overlap across folds"
        seen_test |= set(s["test"])
    # Deterministic for a given group and fold.
    assert make_split(members, "web_g0", 0, .2, .2) == \
        make_split(members, "web_g0", 0, .2, .2)


def _write_pc_csv(path, rows):
    cols = ["pc", "executions", "taken", "taken_rate",
            "tage_sc_l_64KB_mispredictions", "tage_hit_bank_mean",
            "tage_bimodal_frac", "tage_conf_high_frac", "tage_conf_med_frac",
            "tage_conf_low_frac", "tage_alt_disagree_frac"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for pc, ex, miss in rows:
            w.writerow([pc, ex, ex // 2, 0.5, miss, 12, 0.0, 0.5, 0.3, 0.2, 0.1])


def test_branch_selection_reads_train_traces_only(tmp_path):
    """Only train CSVs exist; selection must succeed without opening others,
    and must require the branch to be hard in >= 3 train traces."""
    from scripts.cbp2025_crosstrace import select_branches
    for i in range(3):
        _write_pc_csv(tmp_path / f"t{i}.full_per_pc.csv",
                      [("0xa", 10000, 2000), ("0xb", 10000, 3000 if i == 0 else 10)])
    chosen = select_branches(["t0", "t1", "t2"], tmp_path, top_k=8,
                             min_exec=3000, min_miss=100, min_traces=3)
    assert [c["pc"] for c in chosen] == ["0xa"]   # 0xb hard in only one trace


def test_iter_batches_weights_follow_their_instances(tmp_path):
    from tests.test_cbp2025 import CONVERT, FLAGS, COND, ALU, convert, rec
    if not CONVERT.exists():
        pytest.skip("converter not built")
    from src.dataset import BranchDataset, build_trace_dataset
    B = 0x4000
    records = []
    for i in range(300):
        records += [rec(0x10, ALU),
                    rec(B, COND, taken=bool(i % 3 == 0), target=0x10, ins=[FLAGS])]
    sbbt, meta = convert(tmp_path, records)
    h5 = tmp_path / "t.hdf5"
    build_trace_dataset(sbbt, h5, [B], split_role="train")

    class IndexWeights:              # weight == branch index: easy to check
        def load(self, path, br_pc, indices):
            return indices.astype(np.float32)

    ds = BranchDataset([h5], br_pc=B, history_length=8, max_samples=100,
                       sample_weights=IndexWeights())
    assert ds.has_sample_weights
    for tokens, labels, w in ds.iter_batches(32, shuffle=True, seed=3,
                                             with_weights=True):
        # Rebuild each sample's position from its weight and check the label.
        pos = w.numpy().astype(np.int64)
        assert set(pos) <= set(ds.sources[0].indices)
        hist = ds.sources[0].history
        assert np.array_equal((hist[pos] & 1).astype(np.float32), labels.numpy())


# ------------------------------------------------------------ learned arbiter

def _learned_case(seed, signal=True, n=30000):
    """Disagreements where a TAGE-state column says who is right."""
    from src.evaluation.learned_arbiter import model_columns
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).astype(np.uint8)
    hint = rng.random(n) < 0.3                 # e.g. "TAGE low confidence"
    tage_correct = np.where(hint, rng.random(n) < 0.3, rng.random(n) < 0.95)
    p_model = np.where(hint, 0.85, 0.6) if signal else np.full(n, 0.6)
    model_correct = rng.random(n) < p_model
    sign = np.where(labels == 1, 1.0, -1.0)
    logits = (np.where(model_correct, sign, -sign)
              * rng.uniform(0.2, 3, n)).astype(np.float32)
    ex = Executions(logits, labels, tage_correct,
                    np.where(hint, LOW, HIGH).astype(np.int8))
    tage_state = np.stack([hint.astype(float),
                           rng.random(n)], axis=1)  # one real, one noise column
    X = np.concatenate([model_columns(logits), tage_state], axis=1)
    traces = np.array([f"t{i % 6}" for i in range(n)])
    return ex, X, traces


@pytest.mark.parametrize("kind", ["lr", "gbdt"])
def test_learned_arbiter_finds_the_pattern_and_generalises(kind):
    from src.evaluation.learned_arbiter import fit_learned_arbiter, score_mask
    val, Xv, tv = _learned_case(1)
    test, Xt, _ = _learned_case(2)
    arb = fit_learned_arbiter(val, Xv, tv, kind)
    assert arb.model is not None and arb.oof_p < 0.05
    s = score_mask(test, arb.use_model(test, Xt))
    assert s["hybrid_misses"] < s["tage_misses"]
    # It only ever overrides where the two actually disagree.
    from src.evaluation.learned_arbiter import predictions_differ
    use = arb.use_model(test, Xt)
    assert not (use & ~predictions_differ(test)).any()


def test_learned_arbiter_rejects_noise():
    """Model worse than TAGE everywhere and no informative column: the
    out-of-fold check must refuse to attach it."""
    from src.evaluation.learned_arbiter import fit_learned_arbiter, model_columns
    rng = np.random.default_rng(5)
    n = 20000
    labels = rng.integers(0, 2, n).astype(np.uint8)
    tage_correct = rng.random(n) < 0.9
    model_correct = rng.random(n) < 0.6
    sign = np.where(labels == 1, 1.0, -1.0)
    logits = (np.where(model_correct, sign, -sign)).astype(np.float32)
    ex = Executions(logits, labels, tage_correct, np.zeros(n, np.int8))
    X = np.concatenate([model_columns(logits), rng.random((n, 3))], axis=1)
    traces = np.array([f"t{i % 6}" for i in range(n)])
    arb = fit_learned_arbiter(ex, X, traces, "lr")
    assert arb.model is None


def test_shared_arbiter_pools_branches_and_gates_each():
    """Two branches share the pattern; a third is pure noise and must stay off."""
    from src.evaluation.learned_arbiter import fit_shared_arbiter, score_mask
    a = _learned_case(11, n=8000)
    b = _learned_case(12, n=8000)
    c = _learned_case(13, n=8000)
    # Branch c: the model is wrong on EVERY execution where it disagrees with
    # TAGE (right only where TAGE is right), so arbitration can never help.
    rng = np.random.default_rng(13)
    ex = c[0]
    model_right = ex.tage_correct & (rng.random(len(ex)) < 0.8)
    sign = np.where(ex.labels == 1, 1.0, -1.0)
    logits = (np.where(model_right, sign, -sign) * 2.0).astype(np.float32)
    c_ex = Executions(logits, ex.labels, ex.tage_correct, ex.tage_conf)
    sh = fit_shared_arbiter([("a", *a), ("b", *b), ("c", c_ex, c[1], c[2])])
    assert sh.enabled["a"] and sh.enabled["b"] and not sh.enabled["c"]
    t, Xt, _ = _learned_case(14, n=8000)
    s = score_mask(t, sh.use_model("a", t, Xt))
    assert s["hybrid_misses"] < s["tage_misses"]


def test_corrected_shared_arbiter_learns_branch_specific_signal():
    """Branches a, b share a pattern; branch c's TAGE weakness is flagged by a
    DIFFERENT column. The pure shared arbiter cannot know that; the per-branch
    correction should pick it up and beat it on c."""
    from src.evaluation.learned_arbiter import (
        fit_corrected_shared_arbiter, fit_shared_arbiter, score_mask)
    a, b = _learned_case(21, n=8000), _learned_case(22, n=8000)

    def swapped(seed, n=8000):
        ex, X, tr = _learned_case(seed, n=n)
        X = X.copy()
        X[:, [-2, -1]] = X[:, [-1, -2]]   # the informative column moves
        return ex, X, tr

    c_val, c_test = swapped(23), swapped(24)
    branches = [("a", *a), ("b", *b), ("c", *c_val)]
    shared = fit_shared_arbiter(branches)
    corrected = fit_corrected_shared_arbiter(branches)
    t, Xt, _ = c_test
    s_sh = score_mask(t, shared.use_model("c", t, Xt))
    s_co = score_mask(t, corrected.use_model("c", t, Xt))
    assert corrected.enabled["c"]
    assert s_co["hybrid_misses"] < s_sh["hybrid_misses"]
    assert s_co["hybrid_misses"] < s_co["tage_misses"]
