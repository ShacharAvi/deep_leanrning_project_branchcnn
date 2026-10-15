"""Prediction log and hybrid MPKI tests.

The load-bearing property here is **alignment**: a prediction log is only
useful if its ``branch_index`` means the same thing as ``chunk.index`` from
:mod:`src.traces`. If those drift, a hybrid evaluation still produces a
plausible-looking number — it is just wrong. So the central test joins the log
against an independently-parsed trace and requires exact agreement on branch
index, PC and direction.
"""

import json
import subprocess

import numpy as np
import pytest

from src.evaluation import (
    LOG_DTYPE,
    PredictionLogError,
    hybrid_mispredictions,
    load_prediction_log,
    oracle_ceiling,
)
from src.predictors import Branch
from src.traces import SbbtReader

from .conftest import REPO_ROOT
from .test_predictor_crossvalidation import make_trace, pseudorandom_branches

DRIVER = REPO_ROOT / "build" / "run_predictors"

pytestmark = pytest.mark.skipif(
    not DRIVER.exists(), reason="build/run_predictors not built"
)

PREDICTORS = "always_taken,bimodal_64KB,gshare_64KB,tage_sc_l_64KB"


@pytest.fixture
def logged_run(tmp_path):
    """Run the driver over a synthetic trace, logging two selected PCs."""
    branches = pseudorandom_branches(20_000)
    trace = make_trace(tmp_path, branches)

    conditional_pcs = sorted({b.pc for b in branches if b.conditional})
    selected = conditional_pcs[:2]
    pc_file = tmp_path / "hard_brs.txt"
    pc_file.write_text(
        "# selected branches\n" + "\n".join(f"{pc:x}" for pc in selected) + "\n"
    )

    prefix = tmp_path / "log"
    summary = tmp_path / "summary.json"
    subprocess.run(
        [str(DRIVER), str(trace), "--predictors", PREDICTORS,
         "--json", str(summary), "--log", str(prefix),
         "--log-pcs", str(pc_file)],
        capture_output=True, check=True,
    )
    return {
        "trace": trace,
        "branches": branches,
        "selected": selected,
        "log": load_prediction_log(prefix),
        "summary": json.loads(summary.read_text()),
        "prefix": prefix,
    }


# --------------------------------------------------------------------------- #
# format
# --------------------------------------------------------------------------- #

def test_record_layout_is_packed_25_bytes():
    """Must match LogRecord in tools/run_predictors.cpp (v2, with TAGE state)."""
    assert LOG_DTYPE.itemsize == 25
    assert LOG_DTYPE.names == (
        "branch_index", "pc", "taken", "mispredicted",
        "hit_bank", "alt_bank", "tage_flags", "sc_sum", "sc_threshold")


def test_sidecar_describes_the_binary(logged_run):
    meta = logged_run["log"].meta
    assert meta["format"] == "branch-prediction-log"
    assert meta["record_bytes"] == LOG_DTYPE.itemsize
    assert meta["numpy_dtype"]["formats"] == [
        "<i8", "<u8", "u1", "u1", "i1", "i1", "u1", "<i2", "<i2"]
    assert meta["has_tage_state"] is True
    assert meta["filtered_to_pcs"] is True
    assert meta["num_filter_pcs"] == 2


def test_log_holds_only_the_selected_pcs(logged_run):
    assert sorted(int(p) for p in logged_run["log"].pcs()) == logged_run["selected"]


def test_log_is_much_smaller_than_logging_everything(logged_run):
    """PC filtering is the whole reason this is affordable on real traces."""
    logged = len(logged_run["log"])
    total_conditional = logged_run["summary"]["metadata"][
        "conditional_branches_counted"]
    assert logged < total_conditional / 5


# --------------------------------------------------------------------------- #
# alignment -- the property everything downstream depends on
# --------------------------------------------------------------------------- #

def test_branch_index_matches_the_independently_parsed_trace(logged_run):
    """The join key must mean the same thing on both sides of the pipeline."""
    log = logged_run["log"]
    selected = set(logged_run["selected"])

    reader = SbbtReader(logged_run["trace"])
    branches = reader.read_all()
    want = branches[branches.is_conditional]
    want = want[np.isin(want.ip, list(selected))]

    assert log.branch_index.tolist() == want.index.tolist()
    assert log.pc.tolist() == [int(x) for x in want.ip.tolist()]
    assert log.taken.tolist() == want.taken.astype(bool).tolist()


def test_branch_index_counts_unconditional_branches_too(logged_run):
    """branch_index is a position among ALL branches, not among conditionals."""
    log = logged_run["log"]
    branches = logged_run["branches"]
    # Index of the first logged instance, counted over every branch.
    first = log.branch_index[0]
    assert branches[first].pc == int(log.pc[0])
    assert branches[first].conditional
    # Position among conditionals only would be smaller, since the workload
    # contains unconditional branches.
    conditional_position = sum(
        1 for b in branches[:first] if b.conditional
    )
    assert conditional_position < first


def test_branch_index_is_strictly_increasing(logged_run):
    assert np.all(np.diff(logged_run["log"].branch_index) > 0)


def test_logged_outcomes_match_the_trace(logged_run):
    log = logged_run["log"]
    branches = logged_run["branches"]
    for idx, pc, taken in zip(log.branch_index, log.pc, log.taken):
        assert branches[idx].pc == int(pc)
        assert branches[idx].taken == bool(taken)


# --------------------------------------------------------------------------- #
# misprediction bitmask
# --------------------------------------------------------------------------- #

def test_bitmask_agrees_with_the_per_pc_aggregate(logged_run, tmp_path):
    """Per-instance log and per-PC CSV must tell the same story."""
    import csv

    per_pc = tmp_path / "per_pc.csv"
    subprocess.run(
        [str(DRIVER), str(logged_run["trace"]), "--predictors", PREDICTORS,
         "--per-pc", str(per_pc)],
        capture_output=True, check=True,
    )
    aggregate = {
        int(r["pc"], 16): int(r["tage_sc_l_64KB_mispredictions"])
        for r in csv.DictReader(open(per_pc, newline=""))
    }

    log = logged_run["log"]
    for pc in log.pcs():
        from_log = int(log.for_pc(int(pc)).mispredicted("tage_sc_l_64KB").sum())
        assert from_log == aggregate[int(pc)]


def test_always_taken_bit_is_exactly_the_not_taken_instances(logged_run):
    """always_taken is wrong precisely when the branch was not taken."""
    log = logged_run["log"]
    assert log.mispredicted("always_taken").tolist() == (~log.taken).tolist()


def test_unknown_predictor_is_rejected(logged_run):
    with pytest.raises(KeyError, match="not in this log"):
        logged_run["log"].mispredicted("no_such_predictor")


# --------------------------------------------------------------------------- #
# hybrid arithmetic
# --------------------------------------------------------------------------- #

def _baseline_total(summary, name="tage_sc_l_64KB"):
    return [p for p in summary["predictors"] if p["name"] == name][0][
        "mispredictions"
    ]


def test_a_model_that_matches_the_baseline_changes_nothing(logged_run):
    log, summary = logged_run["log"], logged_run["summary"]
    total = _baseline_total(summary)
    model = {
        int(pc): ~log.for_pc(int(pc)).mispredicted("tage_sc_l_64KB")
        for pc in log.pcs()
    }
    result = hybrid_mispredictions(log, "tage_sc_l_64KB", total, model)
    assert result.hybrid_mispredictions == total
    assert result.mpki_reduction == 0.0


def test_a_perfect_model_removes_exactly_the_covered_mispredictions(logged_run):
    log, summary = logged_run["log"], logged_run["summary"]
    total = _baseline_total(summary)
    result = oracle_ceiling(log, "tage_sc_l_64KB", total)

    covered_misses = int(log.mispredicted("tage_sc_l_64KB").sum())
    assert result.model_mispredictions_on_covered == 0
    assert result.baseline_mispredictions_on_covered == covered_misses
    assert result.hybrid_mispredictions == total - covered_misses


def test_a_model_that_gets_everything_wrong_is_reported_as_worse(logged_run):
    """A hybrid can lose. The metric must be able to say so."""
    log, summary = logged_run["log"], logged_run["summary"]
    total = _baseline_total(summary)
    model = {
        int(pc): np.zeros(len(log.for_pc(int(pc))), dtype=bool)
        for pc in log.pcs()
    }
    result = hybrid_mispredictions(log, "tage_sc_l_64KB", total, model)
    assert result.hybrid_mispredictions > total
    assert result.mpki_reduction < 0


def test_mpki_uses_the_runs_instruction_count(logged_run):
    log, summary = logged_run["log"], logged_run["summary"]
    total = _baseline_total(summary)
    result = oracle_ceiling(log, "tage_sc_l_64KB", total)
    assert result.metric_instructions == summary["metadata"]["metric_instructions"]
    assert result.baseline_mpki == pytest.approx(
        1000.0 * total / result.metric_instructions
    )


def test_misaligned_model_output_is_rejected(logged_run):
    """The check that stops a silently wrong evaluation."""
    log, summary = logged_run["log"], logged_run["summary"]
    total = _baseline_total(summary)
    pc = int(log.pcs()[0])
    short = np.ones(len(log.for_pc(pc)) - 1, dtype=bool)
    with pytest.raises(ValueError, match="misaligned"):
        hybrid_mispredictions(log, "tage_sc_l_64KB", total, {pc: short})


def test_unlogged_pc_is_rejected(logged_run):
    log, summary = logged_run["log"], logged_run["summary"]
    total = _baseline_total(summary)
    with pytest.raises(ValueError, match="no instances in the log"):
        hybrid_mispredictions(
            log, "tage_sc_l_64KB", total, {0xDEAD: np.ones(1, dtype=bool)}
        )


# --------------------------------------------------------------------------- #
# corruption
# --------------------------------------------------------------------------- #

def test_truncated_log_is_rejected(logged_run):
    prefix = logged_run["prefix"]
    data = prefix.with_suffix(".bin").read_bytes()
    prefix.with_suffix(".bin").write_bytes(data[: -LOG_DTYPE.itemsize])
    with pytest.raises(PredictionLogError, match="truncated"):
        load_prediction_log(prefix)


def test_version_mismatch_is_rejected(logged_run):
    prefix = logged_run["prefix"]
    meta = json.loads(prefix.with_suffix(".json").read_text())
    meta["version"] = 99
    prefix.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(PredictionLogError, match="unsupported log version"):
        load_prediction_log(prefix)


def test_record_size_mismatch_is_rejected(logged_run):
    prefix = logged_run["prefix"]
    meta = json.loads(prefix.with_suffix(".json").read_text())
    meta["record_bytes"] = 24
    prefix.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(PredictionLogError, match="record size mismatch"):
        load_prediction_log(prefix)


def test_log_pcs_without_log_is_an_error(tmp_path):
    trace = make_trace(tmp_path, pseudorandom_branches(100))
    pcs = tmp_path / "pcs.txt"
    pcs.write_text("400000\n")
    out = subprocess.run(
        [str(DRIVER), str(trace), "--log-pcs", str(pcs)], capture_output=True
    )
    assert out.returncode == 2
    assert b"requires --log" in out.stderr


def test_pc_list_accepts_comments_and_whitespace(tmp_path):
    branches = pseudorandom_branches(5_000)
    trace = make_trace(tmp_path, branches)
    pc = sorted({b.pc for b in branches if b.conditional})[0]

    pcs = tmp_path / "pcs.txt"
    pcs.write_text(f"# a comment\n\n  {pc:x}  # trailing\n\n")
    prefix = tmp_path / "log"
    subprocess.run(
        [str(DRIVER), str(trace), "--predictors", PREDICTORS,
         "--log", str(prefix), "--log-pcs", str(pcs)],
        capture_output=True, check=True,
    )
    assert load_prediction_log(prefix).pcs().tolist() == [pc]


# --------------------------------------------------------------------------- #
# per-instance TAGE state (log v2)
# --------------------------------------------------------------------------- #

def test_tage_state_is_present_when_tage_runs(logged_run):
    assert logged_run["log"].has_tage_state is True


def test_tage_features_shape_and_range(logged_run):
    log = logged_run["log"]
    features = log.tage_features()
    assert features.shape == (len(log), len(log.tage_feature_names()))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_tage_features_vary_per_instance(logged_run):
    """The whole point: per-PC aggregates would be constant and useless.

    A per-branch model only ever sees one PC, so a feature that does not vary
    within that PC contributes zero gradient. Several of these must vary even
    after restricting to a single branch.
    """
    log = logged_run["log"]
    pc = int(log.pcs()[0])
    features = log.for_pc(pc).tage_features()
    varying = int((features.std(axis=0) > 1e-6).sum())
    assert varying >= 3, (
        f"only {varying}/{features.shape[1]} features vary within one branch")


def test_final_prediction_is_the_prediction_not_the_label(logged_run):
    """The recorded prediction must be what the predictor OUTPUT, not whether
    it was right. Prediction XOR correctness reconstructs the label exactly --
    which also proves we stored the final prediction and not the TAGE
    component's pre-corrector guess."""
    log = logged_run["log"]
    predicted = log.flag("final_prediction")
    wrong = log.mispredicted("tage_sc_l_64KB")
    assert not np.array_equal(predicted, log.taken)   # not the label itself
    assert np.array_equal(predicted ^ wrong, log.taken)


def test_tage_component_can_disagree_with_the_final_prediction(logged_run):
    """The statistical corrector overrides TAGE sometimes; that disagreement is
    itself a useful per-instance signal, so both are recorded."""
    log = logged_run["log"]
    disagree = log.flag("tage_prediction") != log.flag("final_prediction")
    column = log.tage_feature_names().index("corrector_overrode_tage")
    assert np.array_equal(log.tage_features()[:, column] > 0.5, disagree)


def test_unknown_tage_flag_rejected(logged_run):
    with pytest.raises(KeyError, match="not a TAGE flag"):
        logged_run["log"].flag("definitely_not_a_flag")


def test_feature_names_match_feature_columns(logged_run):
    """Names and columns must stay in lockstep -- a checkpoint's first FC
    layer is indexed by this order."""
    log = logged_run["log"]
    for groups in [("structural",), ("confidence",), ("directional",),
                   ("structural", "directional")]:
        assert log.tage_features(groups).shape[1] == \
            len(log.tage_feature_names(groups))


def test_unknown_feature_group_rejected(logged_run):
    with pytest.raises(KeyError, match="not feature groups"):
        logged_run["log"].tage_features(("structural", "bogus"))


def test_hit_bank_zero_means_bimodal_fallback(logged_run):
    log = logged_run["log"]
    features = log.tage_features()
    column = log.tage_feature_names().index("bimodal_fallback")
    assert np.array_equal(features[:, column] > 0.5,
                          log.records["hit_bank"] == 0)
