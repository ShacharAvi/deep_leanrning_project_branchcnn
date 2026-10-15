"""Predictor tests (spec section 27, "Predictor tests").

The spec asks for behaviour to be verified against hand-designed patterns:
``1111111111``, ``0000000000``, ``1010101010``, ``1100110011``. Those patterns
are chosen well -- each isolates a specific capability, and the predictors
separate cleanly on them:

    pattern   bimodal   gshare(H>=2)   perceptron
    1111...   learns    learns         learns
    0000...   learns    learns         learns
    1010...   FAILS     learns         learns
    1100...   FAILS     learns         learns

Bimodal failing the last two is the entire motivation for correlating history,
so it is asserted rather than merely observed.
"""

import pytest

from src.predictors import (
    AlwaysNotTaken,
    AlwaysTaken,
    Bimodal,
    Branch,
    GShare,
    Perceptron,
    SaturatingCounter,
    pattern_to_branches,
    simulate,
    xor_fold,
)

PC = 0x1000


def steady_state_mispredictions(predictor, pattern, repeat=200, warmup_fraction=0.5):
    """Mispredictions after the predictor has had time to converge."""
    branches = pattern_to_branches(pattern, pc=PC, repeat=repeat)
    split = int(len(branches) * warmup_fraction)
    simulate(predictor, branches[:split])
    return simulate(predictor, branches[split:])


# --------------------------------------------------------------------------- #
# saturating counter
# --------------------------------------------------------------------------- #

def test_two_bit_counter_range_matches_mbplib():
    """mbp::SatCtr<2> spans -2..1 and predicts taken when non-negative."""
    counter = SaturatingCounter(2)
    assert (counter.min_value, counter.max_value) == (-2, 1)
    assert counter.value == 0 and counter.predicts_taken


def test_counter_saturates_at_both_ends():
    counter = SaturatingCounter(2)
    for _ in range(10):
        counter.update(True)
    assert counter.value == 1
    for _ in range(10):
        counter.update(False)
    assert counter.value == -2


def test_counter_needs_two_updates_to_flip_prediction():
    """The hysteresis that makes a 2-bit counter better than a 1-bit one."""
    counter = SaturatingCounter(2)
    counter.update(True)  # -> 1, strongly taken
    counter.update(False)  # -> 0, still predicts taken
    assert counter.predicts_taken
    counter.update(False)  # -> -1
    assert not counter.predicts_taken


# --------------------------------------------------------------------------- #
# always taken / not taken
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "predictor_cls,pattern,expected_mispredictions",
    [
        (AlwaysTaken, "1111111111", 0),
        (AlwaysTaken, "0000000000", 10),
        (AlwaysNotTaken, "0000000000", 0),
        (AlwaysNotTaken, "1111111111", 10),
    ],
)
def test_always_predictors(predictor_cls, pattern, expected_mispredictions):
    result = simulate(predictor_cls(), pattern_to_branches(pattern))
    assert result.mispredictions == expected_mispredictions


def test_always_predictors_are_complementary():
    """Their misprediction counts must sum to the branch count, by definition."""
    branches = pattern_to_branches("1100101110100011")
    taken = simulate(AlwaysTaken(), branches)
    not_taken = simulate(AlwaysNotTaken(), branches)
    assert taken.mispredictions + not_taken.mispredictions == len(branches)


# --------------------------------------------------------------------------- #
# bimodal
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pattern", ["1111111111", "0000000000"])
def test_bimodal_learns_constant_patterns(pattern):
    assert steady_state_mispredictions(Bimodal(8), pattern).mispredictions == 0


def test_bimodal_converges_within_two_branches():
    """From weakly-taken, at most two updates are needed to lock on."""
    predictor = Bimodal(8)
    result = simulate(predictor, pattern_to_branches("0", repeat=10),
                      record_predictions=True)
    assert result.predictions[0] is True  # initial state is weakly taken
    assert all(p is False for p in result.predictions[2:])


@pytest.mark.parametrize("pattern", ["1010101010", "1100110011"])
def test_bimodal_cannot_learn_history_dependent_patterns(pattern):
    """The motivation for GShare: no history means no correlation."""
    result = steady_state_mispredictions(Bimodal(8), pattern)
    assert result.misprediction_rate > 0.3


def test_bimodal_separates_branches_by_pc():
    """Two PCs with opposite behaviour must not interfere at distinct indices."""
    predictor = Bimodal(8)
    hot, cold = 0x1000, 0x1040  # differ within the 8 index bits
    assert predictor.index(hot) != predictor.index(cold)

    branches = []
    for _ in range(50):
        branches.append(Branch(pc=hot, taken=True))
        branches.append(Branch(pc=cold, taken=False))
    simulate(predictor, branches)
    assert predictor.predict(hot) is True
    assert predictor.predict(cold) is False


def test_bimodal_aliases_pcs_that_share_index_bits():
    """Indexing is pc & mask, so PCs differing only above the mask collide."""
    predictor = Bimodal(8)
    assert predictor.index(0x1000) == predictor.index(0x1000 + (1 << 8))


# --------------------------------------------------------------------------- #
# gshare
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pattern", ["1111111111", "0000000000", "1010101010"])
def test_gshare_learns_patterns_within_its_history(pattern):
    assert steady_state_mispredictions(GShare(4, 8), pattern).mispredictions == 0


def test_gshare_learns_period_four_pattern():
    """1100 needs at least 2 bits of history to disambiguate."""
    assert steady_state_mispredictions(GShare(4, 10), "1100").mispredictions == 0


def test_gshare_with_no_usable_history_degenerates_to_bimodal():
    """H=1 cannot separate the two states of a period-4 pattern."""
    result = steady_state_mispredictions(GShare(1, 10), "1100")
    assert result.misprediction_rate > 0.1


def test_gshare_history_tracks_all_branches_by_default():
    """Matches mbp::Gshare's IGNORE_UCD=false default."""
    predictor = GShare(4, 8)
    predictor.track(Branch(pc=0, taken=True, conditional=False))
    assert predictor.ghist == 0b1


def test_gshare_can_ignore_unconditional_branches():
    predictor = GShare(4, 8, ignore_unconditional=True)
    predictor.track(Branch(pc=0, taken=True, conditional=False))
    assert predictor.ghist == 0


def test_gshare_history_is_bounded_by_history_length():
    predictor = GShare(3, 8)
    for _ in range(10):
        predictor.track(Branch(pc=0, taken=True))
    assert predictor.ghist == 0b111


# --------------------------------------------------------------------------- #
# xor_fold -- must match mbp::XorFold exactly for cross-validation to mean anything
# --------------------------------------------------------------------------- #

def test_xor_fold_width_zero():
    assert xor_fold(0xDEADBEEF, 0) == 0


def test_xor_fold_is_identity_for_values_inside_one_window():
    assert xor_fold(0b1011, 8) == 0b1011


def test_xor_fold_combines_successive_windows():
    # Two 4-bit windows: 0xA ^ 0x3 == 0x9
    assert xor_fold(0x3A, 4) == (0xA ^ 0x3)


def test_xor_fold_result_fits_in_width():
    for width in (1, 3, 7, 12, 18):
        assert xor_fold(0xFFFF_FFFF_FFFF_FFFF, width) < (1 << width)


# --------------------------------------------------------------------------- #
# perceptron
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pattern", ["1111111111", "0000000000", "1010101010"])
def test_perceptron_learns_linearly_separable_patterns(pattern):
    assert steady_state_mispredictions(
        Perceptron(8, 4), pattern
    ).mispredictions == 0


def test_perceptron_theta_matches_the_paper():
    """Jimenez & Lin's heuristic: theta = floor(1.93h + 14)."""
    assert Perceptron(62, 10).theta == int(1.93 * 62 + 14)
    assert Perceptron(24, 10).theta == int(1.93 * 24 + 14)


def test_perceptron_bias_alone_learns_a_constant_branch():
    """With all-taken history the bias saturates; weights need not do the work."""
    predictor = Perceptron(8, 4)
    simulate(predictor, pattern_to_branches("1", repeat=100))
    assert predictor.weights[predictor.index(PC)][0] > 0


def test_perceptron_stops_training_once_confident():
    """Training halts above theta -- that is what bounds weight growth."""
    predictor = Perceptron(8, 4, weight_bits=8)
    simulate(predictor, pattern_to_branches("1", repeat=500))
    before = list(predictor.weights[predictor.index(PC)])
    simulate(predictor, pattern_to_branches("1", repeat=50))
    assert predictor.weights[predictor.index(PC)] == before


def test_perceptron_weights_saturate():
    predictor = Perceptron(4, 4, weight_bits=4)
    simulate(predictor, pattern_to_branches("10", repeat=2000))
    flat = [w for row in predictor.weights for w in row]
    assert min(flat) >= predictor.weight_min
    assert max(flat) <= predictor.weight_max


def test_perceptron_scales_linearly_with_history():
    """The paper's central claim, versus GShare's exponential table growth."""
    small = Perceptron(16, 10).storage_bits
    large = Perceptron(32, 10).storage_bits
    assert large / small == pytest.approx(33 / 17, rel=1e-6)


# --------------------------------------------------------------------------- #
# the canonical simulation loop
# --------------------------------------------------------------------------- #

def test_unconditional_branches_are_not_counted_but_are_tracked():
    """They never get predicted, yet they still shape the global history."""
    predictor = GShare(4, 8)
    branches = [
        Branch(pc=0x1000, taken=True, conditional=False),
        Branch(pc=0x1004, taken=True, conditional=True),
    ]
    result = simulate(predictor, branches)
    assert result.conditional_branches == 1
    assert predictor.ghist == 0b11  # both branches entered the history


def test_storage_bits_are_reported():
    assert Bimodal(18).storage_bits == (1 << 18) * 2
    assert Perceptron(62, 10, 8).storage_bits == 1024 * 63 * 8
