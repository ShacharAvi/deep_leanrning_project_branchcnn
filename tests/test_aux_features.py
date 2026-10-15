"""Per-instance feature join and the feature-fusion model.

The join in :mod:`src.dataset.aux_features` is the dangerous part of this
feature: an off-by-one attaches every feature row to the neighbouring dynamic
instance, and the result still trains, still converges, and still reports a
plausible accuracy. Nothing downstream would notice. So these tests check the
join against an independently computed answer rather than against itself, and
check that a *deliberately* broken join is rejected rather than tolerated.
"""

import subprocess

import numpy as np
import pytest
import torch

from src.dataset import (
    DIRECTION_FREE_GROUPS,
    AuxFeatureError,
    AuxFeatureSpec,
    BranchDataset,
)
from src.evaluation import TAGE_FEATURE_GROUPS, load_prediction_log, log_paths
from src.models import BranchNetWithFeatures, build_model, needs_features

from .conftest import REPO_ROOT

DRIVER = REPO_ROOT / "build" / "run_predictors"


# --------------------------------------------------------------------------- #
# log_paths -- the dotted-filename hazard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("prefix", [
    "logs/605.mcf_s-1152B",
    "logs/605.mcf_s-1152B.bin",
    "logs/605.mcf_s-1152B.json",
])
def test_log_paths_survives_dots_in_the_name(prefix):
    """``Path.with_suffix`` would turn ``605.mcf_s-1152B`` into ``605``.

    Every trace id in this project contains dots, so this is not a corner
    case -- it is the normal case. The bug was invisible in tests using
    dot-free names like ``tmp_path / "log"``.
    """
    bin_path, json_path = log_paths(prefix)
    assert bin_path.name == "605.mcf_s-1152B.bin"
    assert json_path.name == "605.mcf_s-1152B.json"


def test_log_paths_leaves_dot_free_names_alone():
    bin_path, json_path = log_paths("/tmp/v2_log")
    assert (bin_path.name, json_path.name) == ("v2_log.bin", "v2_log.json")


# --------------------------------------------------------------------------- #
# feature groups
# --------------------------------------------------------------------------- #

def test_group_order_is_stable():
    """A checkpoint's first FC layer is indexed by this order."""
    assert list(TAGE_FEATURE_GROUPS) == [
        "structural", "confidence", "directional"]


def test_direction_free_groups_exclude_tage_answer():
    """The scientific claim depends on this: a direction-free model cannot
    simply learn 'flip TAGE'."""
    names = [n for g in DIRECTION_FREE_GROUPS
             for n, _ in TAGE_FEATURE_GROUPS[g]]
    assert "final_prediction" not in names
    assert "tage_prediction" not in names
    assert "sc_sum_signed" not in names
    # The direction-free margin must be the magnitude, not the signed value.
    assert "sc_margin" in names


def test_unknown_group_rejected():
    with pytest.raises(KeyError, match="not feature groups"):
        AuxFeatureSpec(log_dir="/tmp", groups=("structural", "nope"))


def test_spec_requires_a_source():
    with pytest.raises(ValueError, match="log_dir or mapping"):
        AuxFeatureSpec()


def test_spec_reports_whether_it_reveals_tage_direction():
    assert not AuxFeatureSpec(log_dir="/tmp",
                              groups=DIRECTION_FREE_GROUPS
                              ).reveals_tage_direction
    assert AuxFeatureSpec(log_dir="/tmp",
                          groups=("structural", "directional")
                          ).reveals_tage_direction


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

def _model(num_features=11, **kw):
    return BranchNetWithFeatures(vocabulary_size=8192,
                                 num_features=num_features, **kw)


def test_concat_width_is_pooled_plus_kept_features():
    model = _model(feature_mask=[True] * 10 + [False])
    assert model.num_features_used == 10
    assert model.concat_size == model.trunk.pooled_size + 10


def test_forward_returns_one_logit_per_sample():
    model = _model().eval()
    out = model(torch.randint(0, 8192, (4, 582)), torch.randn(4, 11))
    assert out.shape == (4,)


def test_features_are_required():
    with pytest.raises(ValueError, match="requires per-instance features"):
        _model().eval()(torch.randint(0, 8192, (4, 582)))


def test_feature_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="expected 11 features"):
        _model().eval()(torch.randint(0, 8192, (2, 582)), torch.randn(2, 5))


def test_batch_mismatch_is_rejected():
    with pytest.raises(ValueError, match="batch mismatch"):
        _model().eval()(torch.randint(0, 8192, (4, 582)), torch.randn(3, 11))


def test_all_constant_features_is_an_error_not_a_silent_no_op():
    """If every feature is constant for this branch there is nothing to learn
    from them, and a model that pretends otherwise just wastes parameters."""
    with pytest.raises(ValueError, match="drops every feature"):
        _model(feature_mask=[False] * 11)


def test_feature_mask_length_is_checked():
    with pytest.raises(ValueError, match="feature_mask has 3 entries"):
        _model(feature_mask=[True, True, True])


def test_features_actually_change_the_output():
    """A concat that is wired up wrong can still produce the right shape."""
    model = _model().eval()
    tokens = torch.randint(0, 8192, (8, 582))
    torch.manual_seed(0)
    a = model(tokens, torch.zeros(8, 11))
    b = model(tokens, torch.ones(8, 11) * 5)
    assert not torch.allclose(a, b), "features are not reaching the head"


def test_masked_features_are_ignored():
    model = _model(feature_mask=[True] * 10 + [False]).eval()
    tokens = torch.randint(0, 8192, (8, 582))
    base = torch.randn(8, 11)
    other = base.clone()
    other[:, 10] = 999.0            # only the masked-out column differs
    assert torch.allclose(model(tokens, base), model(tokens, other))


def test_build_model_knows_the_feature_variant():
    model = build_model("branchnet_features", vocabulary_size=8192,
                        history_length=582, num_features=11)
    assert isinstance(model, BranchNetWithFeatures)
    assert needs_features("branchnet_features")
    assert not needs_features("branchnet_big")


def test_trunk_matches_plain_branchnet_pooled_size():
    """The ablation is only clean if the trunk is unchanged."""
    plain = build_model("branchnet_big", vocabulary_size=8192,
                        history_length=582)
    assert _model().trunk.pooled_size == plain.pooled_size


# --------------------------------------------------------------------------- #
# the join, against a real driver run
# --------------------------------------------------------------------------- #

pytestmark_driver = pytest.mark.skipif(
    not DRIVER.exists(), reason="build/run_predictors not built")


@pytest.fixture
def joined(tmp_path):
    """Build a trace, a dataset and a log from it, then join them.

    Uses a **dotted** prefix on purpose, so the dotted-filename hazard is
    covered end to end and not only in the unit test above.
    """
    from .test_predictor_crossvalidation import make_trace, pseudorandom_branches
    from src.dataset import build_trace_dataset

    branches = pseudorandom_branches(20_000)
    trace = make_trace(tmp_path, branches)

    counts: dict[int, int] = {}
    for b in branches:
        if b.conditional:
            counts[b.pc] = counts.get(b.pc, 0) + 1
    target = max(counts, key=counts.get)

    pc_file = tmp_path / "hard_brs.txt"
    pc_file.write_text(f"{target:x}\n")

    prefix = tmp_path / "605.mcf_s-999B"
    subprocess.run(
        [str(DRIVER), str(trace), "--predictors", "tage_sc_l_64KB",
         "--log", str(prefix), "--log-pcs", str(pc_file)],
        capture_output=True, check=True)

    dataset_path = tmp_path / "ds.hdf5"
    build_trace_dataset(trace, dataset_path, [target],
                        pc_bits=30, split_role="train",
                        extra_attrs={"trace_id": "605.mcf_s-999B"})
    return {"dir": tmp_path, "pc": target, "dataset": dataset_path,
            "prefix": prefix, "history_length": 32, "trace": trace,
            "pc_file": pc_file}


@pytestmark_driver
def test_join_is_exact_against_an_independent_recompute(joined):
    spec = AuxFeatureSpec(log_dir=joined["dir"], groups=DIRECTION_FREE_GROUPS)
    ds = BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                       history_length=joined["history_length"],
                       pc_bits=12, aux_features=spec)
    log = load_prediction_log(joined["prefix"]).for_pc(joined["pc"])
    source = ds.sources[0]

    slot = np.searchsorted(log.branch_index, source.indices)
    assert np.array_equal(log.branch_index[slot], source.indices)
    # The label the dataset serves must be the outcome the log recorded for
    # the same instance. This is what an off-by-one would break.
    labels = (source.history[source.indices] & 1).astype(bool)
    assert np.array_equal(labels, log.taken[slot])
    expected = log.tage_features(DIRECTION_FREE_GROUPS)[slot]
    assert np.allclose(source.features, expected)


@pytestmark_driver
def test_iter_batches_yields_features_aligned_to_their_own_labels(joined):
    spec = AuxFeatureSpec(log_dir=joined["dir"], groups=("structural",
                                                         "directional"))
    ds = BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                       history_length=joined["history_length"],
                       pc_bits=12, aux_features=spec)
    names = ds.aux_feature_names
    pred_col = names.index("final_prediction")

    log = load_prediction_log(joined["prefix"]).for_pc(joined["pc"])
    by_index = dict(zip(log.branch_index.tolist(),
                        log.flag("final_prediction").tolist()))

    # Shuffled, because shuffling is where a misalignment would appear.
    seen = 0
    for tokens, features, labels in ds.iter_batches(128, shuffle=True, seed=3):
        assert features.shape == (tokens.shape[0], len(names))
        seen += tokens.shape[0]
    assert seen == len(ds)

    # Spot-check alignment through the source arrays, which is what the
    # batcher indexes into.
    source = ds.sources[0]
    for i in range(0, source.indices.size, max(1, source.indices.size // 50)):
        assert bool(source.features[i, pred_col]) == by_index[
            int(source.indices[i])]


@pytestmark_driver
def test_subsampling_keeps_features_with_their_instances(joined):
    """Reindexing `indices` without reindexing `features` identically is a
    silent corruption -- it trains fine and reports a plausible number."""
    spec = AuxFeatureSpec(log_dir=joined["dir"], groups=DIRECTION_FREE_GROUPS)
    full = BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                         history_length=joined["history_length"],
                         pc_bits=12, aux_features=spec)
    cap = max(10, len(full) // 3)
    small = BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                          history_length=joined["history_length"],
                          pc_bits=12, aux_features=spec, max_samples=cap)
    assert len(small) < len(full)

    lookup = dict(zip(full.sources[0].indices.tolist(),
                      map(tuple, full.sources[0].features.tolist())))
    for idx, row in zip(small.sources[0].indices,
                        small.sources[0].features):
        assert np.allclose(row, lookup[int(idx)])


@pytestmark_driver
def test_a_log_covering_fewer_instances_is_refused_not_silently_dropped(joined,
                                                                        tmp_path):
    """A partial join must raise.

    Built with ``--limit``, so the log is *internally consistent* -- its
    sidecar matches its records -- and only covers a prefix of the trace. That
    is the dangerous case: nothing is corrupt, there is simply less coverage
    than the dataset has, and silently dropping the unmatched occurrences
    would bias training towards whatever the log happens to contain.
    """
    short_dir = tmp_path / "short"
    short_dir.mkdir()
    subprocess.run(
        [str(DRIVER), str(joined["trace"]), "--predictors", "tage_sc_l_64KB",
         "--limit", "2000",
         "--log", str(short_dir / "605.mcf_s-999B"),
         "--log-pcs", str(joined["pc_file"])],
        capture_output=True, check=True)

    spec = AuxFeatureSpec(log_dir=short_dir, groups=DIRECTION_FREE_GROUPS)
    with pytest.raises(AuxFeatureError, match="no row in the prediction log"):
        BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                      history_length=joined["history_length"],
                      pc_bits=12, aux_features=spec)


@pytestmark_driver
def test_a_truncated_log_is_caught_at_load(joined):
    """Separate failure mode, caught earlier: the sidecar says N records and
    the binary holds fewer."""
    from src.evaluation import PredictionLogError

    bin_path, _ = log_paths(joined["prefix"])
    data = bin_path.read_bytes()
    bin_path.write_bytes(data[: len(data) // 2])

    spec = AuxFeatureSpec(log_dir=joined["dir"], groups=DIRECTION_FREE_GROUPS)
    with pytest.raises(PredictionLogError, match="truncated"):
        BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                      history_length=joined["history_length"],
                      pc_bits=12, aux_features=spec)


@pytestmark_driver
def test_missing_log_file_names_the_command_to_fix_it(joined):
    spec = AuxFeatureSpec(log_dir=joined["dir"] / "nope",
                          groups=DIRECTION_FREE_GROUPS)
    with pytest.raises(AuxFeatureError, match="run_predictors"):
        BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                      history_length=joined["history_length"],
                      pc_bits=12, aux_features=spec)


@pytestmark_driver
def test_feature_mask_is_measured_on_the_dataset(joined):
    spec = AuxFeatureSpec(log_dir=joined["dir"], groups=DIRECTION_FREE_GROUPS)
    ds = BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                       history_length=joined["history_length"],
                       pc_bits=12, aux_features=spec)
    mask = ds.aux_feature_mask()
    assert mask.shape == (ds.num_aux_features,)
    recomputed = ds.sources[0].features.std(axis=0) > 1e-6
    assert np.array_equal(mask, recomputed)


@pytestmark_driver
def test_a_model_trains_end_to_end_on_joined_features(joined):
    """Shape contract through one optimiser step -- not a quality claim."""
    spec = AuxFeatureSpec(log_dir=joined["dir"], groups=DIRECTION_FREE_GROUPS)
    ds = BranchDataset([joined["dataset"]], br_pc=joined["pc"],
                       history_length=joined["history_length"],
                       pc_bits=12, aux_features=spec)
    model = BranchNetWithFeatures(
        vocabulary_size=ds.vocabulary_size,
        num_features=ds.num_aux_features,
        feature_mask=ds.aux_feature_mask(),
        history_lengths=(8, 16, 32), channels=(4, 4, 4),
        pooling_widths=(2, 2, 2), conv_widths=(3, 3, 3), hidden=(8,))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    tokens, features, labels = next(iter(ds.iter_batches(64)))
    before = float(criterion(model(tokens, features), labels).detach())
    for _ in range(5):
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(tokens, features), labels)
        loss.backward()
        optimiser.step()
    assert float(loss.detach()) < before, \
        "loss did not move; features may be detached from the graph"
