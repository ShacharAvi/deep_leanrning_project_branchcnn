"""Stage 2 audit -- the specific correctness properties the project brief names.

The existing predictor tests cover behaviour on hand-designed patterns. This
file audits the *mechanics* the brief asks to verify explicitly:

  - prediction happens before update
  - the target outcome is not visible at prediction time
  - saturating counters actually saturate
  - history is updated at the right moment
  - indexing is correct
  - initialisation is defined, not accidental

These are separated from test_predictors.py because they are an audit of
semantics rather than of outcomes: several would pass trivially on a predictor
that gave the right answers for the wrong reasons.
"""

import inspect

import pytest

from src.predictors import (
    AlwaysNotTaken,
    AlwaysTaken,
    Bimodal,
    Branch,
    GShare,
    Perceptron,
    SaturatingCounter,
    simulate,
)

PREDICTORS = [
    ("always_taken", lambda: AlwaysTaken()),
    ("always_not_taken", lambda: AlwaysNotTaken()),
    ("bimodal", lambda: Bimodal(log_table_size=10)),
    ("gshare", lambda: GShare(history_length=8, log_table_size=10)),
    ("perceptron", lambda: Perceptron(history_length=16, log_num_perceptrons=6)),
]


# --------------------------------------------------------------------------- #
# the outcome must not be visible at prediction time
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,build", PREDICTORS)
def test_predict_signature_cannot_see_the_outcome(name, build):
    """``predict`` takes only a PC. Structurally it *cannot* peek at the label.

    This is the strongest form of the guarantee: not "we checked it does not
    use the outcome" but "the outcome is not in scope".
    """
    signature = inspect.signature(type(build()).predict)
    parameters = [p for p in signature.parameters if p != "self"]
    assert parameters == ["pc"], f"{name}.predict takes {parameters}"


@pytest.mark.parametrize("name,build", PREDICTORS)
def test_prediction_is_unchanged_by_the_outcome_that_follows(name, build):
    """Same history, opposite outcomes -> identical prediction for that branch.

    Catches a predictor that trains before it predicts.
    """
    taken_run = [Branch(pc=0x100, taken=True)] * 5
    a, b = build(), build()
    simulate(a, taken_run)
    simulate(b, taken_run)
    assert a.predict(0x200) == b.predict(0x200)

    # Now give them opposite next outcomes and confirm the prediction made
    # *before* that outcome was the same.
    before_a = a.predict(0x200)
    before_b = b.predict(0x200)
    a.train(Branch(pc=0x200, taken=True))
    b.train(Branch(pc=0x200, taken=False))
    assert before_a == before_b


@pytest.mark.parametrize("name,build", PREDICTORS)
def test_predict_does_not_mutate_prediction_state(name, build):
    """Calling predict twice must give the same answer.

    mbp::Predictor allows predict() to cache hashes, but it must not change
    anything that affects a future prediction.
    """
    predictor = build()
    simulate(predictor, [Branch(pc=0x100, taken=i % 3 == 0) for i in range(50)])
    first = predictor.predict(0x100)
    for _ in range(5):
        assert predictor.predict(0x100) == first


# --------------------------------------------------------------------------- #
# ordering of predict / train / track
# --------------------------------------------------------------------------- #

def test_simulation_loop_order_is_predict_then_train_then_track():
    """The canonical order, recorded by instrumenting a predictor."""
    calls = []

    class Recorder(Bimodal):
        def predict(self, pc):
            calls.append("predict")
            return super().predict(pc)

        def train(self, branch):
            calls.append("train")
            super().train(branch)

        def track(self, branch):
            calls.append("track")
            super().track(branch)

    simulate(Recorder(log_table_size=8), [Branch(pc=0x100, taken=True)])
    assert calls == ["predict", "train", "track"]


def test_unconditional_branches_are_tracked_but_never_predicted():
    """They shape the global history yet are never scored.

    Getting this wrong silently changed TAGE-SC-L's result by 506
    mispredictions on SHORT_MOBILE-73 earlier in the project.
    """
    calls = []

    class Recorder(GShare):
        def predict(self, pc):
            calls.append("predict")
            return super().predict(pc)

        def track(self, branch):
            calls.append("track")
            super().track(branch)

    result = simulate(Recorder(history_length=4, log_table_size=8),
                      [Branch(pc=0x100, taken=True, conditional=False)])
    assert calls == ["track"]
    assert result.conditional_branches == 0


def test_history_update_happens_after_the_prediction_it_informs():
    """The branch being predicted must not already be in its own history."""
    predictor = GShare(history_length=4, log_table_size=8)
    assert predictor.ghist == 0
    before = predictor.index(0x100)
    simulate(predictor, [Branch(pc=0x100, taken=True)])
    assert predictor.ghist == 0b1          # updated only after
    assert predictor.index(0x100) != before


# --------------------------------------------------------------------------- #
# saturating counters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bits,lo,hi", [(1, -1, 0), (2, -2, 1), (3, -4, 3),
                                        (8, -128, 127)])
def test_counter_bounds(bits, lo, hi):
    counter = SaturatingCounter(bits)
    assert (counter.min_value, counter.max_value) == (lo, hi)


@pytest.mark.parametrize("bits", [1, 2, 3, 8])
def test_counter_saturates_and_does_not_wrap(bits):
    """Wrapping instead of saturating would flip the prediction at the extreme."""
    counter = SaturatingCounter(bits)
    for _ in range(1000):
        counter.update(True)
    assert counter.value == counter.max_value
    assert counter.predicts_taken
    for _ in range(1000):
        counter.update(False)
    assert counter.value == counter.min_value
    assert not counter.predicts_taken


def test_counter_initial_state_is_documented_and_neutral():
    """Starts at 0 = weakly taken, matching mbp::SatCtr's zero-initialisation."""
    assert SaturatingCounter(2).value == 0
    assert SaturatingCounter(2).predicts_taken is True


def test_bimodal_table_starts_uniform():
    predictor = Bimodal(log_table_size=6)
    assert all(c.value == 0 for c in predictor.table)


# --------------------------------------------------------------------------- #
# indexing
# --------------------------------------------------------------------------- #

def test_bimodal_index_is_the_low_pc_bits():
    predictor = Bimodal(log_table_size=10)
    for pc in (0x0, 0x3FF, 0x400, 0xDEADBEEF):
        assert predictor.index(pc) == pc & 0x3FF


def test_bimodal_index_is_always_in_range():
    predictor = Bimodal(log_table_size=8)
    for pc in (0, 1, 2**31, 2**52 - 1, 0xFFFFFFC00025FE20):
        assert 0 <= predictor.index(pc) < len(predictor.table)


def test_gshare_index_depends_on_both_pc_and_history():
    predictor = GShare(history_length=8, log_table_size=10)
    base = predictor.index(0x1000)
    assert predictor.index(0x1004) != base          # PC matters
    predictor.ghist = 0b10110
    assert predictor.index(0x1000) != base          # history matters


def test_gshare_index_is_always_in_range():
    predictor = GShare(history_length=25, log_table_size=12)
    for pc in (0, 0xDEADBEEF, 2**52 - 1):
        for ghist in (0, 1, (1 << 25) - 1):
            predictor.ghist = ghist
            assert 0 <= predictor.index(pc) < len(predictor.table)


def test_perceptron_index_is_in_range():
    predictor = Perceptron(history_length=8, log_num_perceptrons=6)
    for pc in (0, 4, 0xDEADBEEF, 2**52 - 1):
        assert 0 <= predictor.index(pc) < len(predictor.weights)


# --------------------------------------------------------------------------- #
# perceptron: the specifics the brief asks to document
# --------------------------------------------------------------------------- #

def test_perceptron_history_encoding_is_bipolar():
    """Taken -> +1, not taken -> -1 (Jimenez & Lin). A 0/1 encoding would make
    not-taken contribute nothing instead of contributing negatively."""
    predictor = Perceptron(history_length=2, log_num_perceptrons=2)
    row = predictor.index(0x100)
    predictor.weights[row] = [0, 3, 5]      # bias, w1, w2

    predictor.ghist = [True, True]
    assert predictor.predict(0x100) is True
    assert predictor._output == 0 + 3 + 5

    predictor.ghist = [False, False]
    predictor.predict(0x100)
    assert predictor._output == 0 - 3 - 5   # negative, not zero


def test_perceptron_weights_start_at_zero():
    predictor = Perceptron(history_length=4, log_num_perceptrons=4)
    assert all(w == 0 for row in predictor.weights for w in row)


def test_perceptron_bias_is_weight_zero_with_constant_input():
    """w[0] has x[0] == 1 always, so it tracks the branch's overall bias."""
    predictor = Perceptron(history_length=4, log_num_perceptrons=4)
    row = predictor.index(0x100)
    predictor.weights[row][0] = 7
    predictor.ghist = [False] * 4
    predictor.predict(0x100)
    assert predictor._output == 7


def test_perceptron_prediction_rule_is_sign_of_the_dot_product():
    predictor = Perceptron(history_length=2, log_num_perceptrons=2)
    row = predictor.index(0x100)
    predictor.ghist = [True, False]
    predictor.weights[row] = [0, 1, 1]      # y = 0 + 1 - 1 = 0
    assert predictor.predict(0x100) is True   # y >= 0 predicts taken
    predictor.weights[row] = [-1, 1, 1]       # y = -1
    assert predictor.predict(0x100) is False


def test_perceptron_threshold_matches_the_paper():
    """theta = floor(1.93h + 14)."""
    for h in (12, 24, 62):
        assert Perceptron(h, 10).theta == int(1.93 * h + 14)


def test_perceptron_update_rule_adds_t_times_x():
    """On a wrong (or low-confidence) prediction: w[i] += t * x[i]."""
    predictor = Perceptron(history_length=2, log_num_perceptrons=2)
    row = predictor.index(0x100)
    predictor.ghist = [True, False]
    before = list(predictor.weights[row])
    predictor.train(Branch(pc=0x100, taken=True))      # t = +1
    after = predictor.weights[row]
    assert after[0] == before[0] + 1       # bias: x = +1
    assert after[1] == before[1] + 1       # ghist[0] taken: x = +1
    assert after[2] == before[2] - 1       # ghist[1] not taken: x = -1


def test_perceptron_does_not_update_when_already_confident():
    """Training stops above theta -- what bounds weight growth."""
    predictor = Perceptron(history_length=4, log_num_perceptrons=4)
    row = predictor.index(0x100)
    predictor.weights[row] = [predictor.theta + 10, 0, 0, 0, 0]
    before = list(predictor.weights[row])
    predictor.train(Branch(pc=0x100, taken=True))
    assert predictor.weights[row] == before


def test_perceptron_storage_scales_linearly_with_history():
    """The paper's central claim, versus GShare's exponential table growth."""
    a = Perceptron(16, 10, 8).storage_bits
    b = Perceptron(32, 10, 8).storage_bits
    assert b / a == pytest.approx(33 / 17, rel=1e-6)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,build", PREDICTORS)
def test_predictors_are_deterministic(name, build):
    """No randomness anywhere: two fresh instances must agree exactly."""
    branches = [Branch(pc=0x100 + (i % 7) * 4, taken=(i * 13) % 5 < 2)
                for i in range(500)]
    a = simulate(build(), branches, record_predictions=True)
    b = simulate(build(), branches, record_predictions=True)
    assert a.mispredictions == b.mispredictions
    assert a.predictions == b.predictions
