"""The five classical baselines of spec section 9 (baselines 0-4).

``Bimodal`` and ``GShare`` reproduce MBPlib's indexing bit-for-bit;
``Perceptron`` reproduces Jimenez & Lin, HPCA 2001. Each has a C++ counterpart
used for full-trace runs -- see ``tools/run_predictors.cpp``.
"""

from __future__ import annotations

from .base import Branch, Predictor, SaturatingCounter, xor_fold


class AlwaysTaken(Predictor):
    """Spec baseline 0. A sanity check, not a predictor."""

    name = "always_taken"

    def predict(self, pc: int) -> bool:
        return True

    def train(self, branch: Branch) -> None:
        pass


class AlwaysNotTaken(Predictor):
    """Spec baseline 1. Reveals strong class imbalance."""

    name = "always_not_taken"

    def predict(self, pc: int) -> bool:
        return False

    def train(self, branch: Branch) -> None:
        pass


class Bimodal(Predictor):
    """Spec baseline 2: a table of saturating counters indexed by PC bits.

    Matches ``mbp::Bimodal<T>``: ``index = pc & (2^T - 1)``, and taken is
    predicted when the counter is non-negative.

    Has no history, so it cannot learn any pattern that depends on preceding
    outcomes -- on ``1010...`` it settles into one direction and is wrong half
    the time. That limitation is the motivation for GShare.
    """

    name = "bimodal"

    def __init__(self, log_table_size: int = 14, counter_bits: int = 2):
        self.log_table_size = log_table_size
        self.counter_bits = counter_bits
        self.table = [
            SaturatingCounter(counter_bits) for _ in range(1 << log_table_size)
        ]

    def index(self, pc: int) -> int:
        return pc & ((1 << self.log_table_size) - 1)

    def predict(self, pc: int) -> bool:
        return self.table[self.index(pc)].predicts_taken

    def train(self, branch: Branch) -> None:
        self.table[self.index(branch.pc)].update(branch.taken)

    @property
    def storage_bits(self) -> int:
        return (1 << self.log_table_size) * self.counter_bits


class GShare(Predictor):
    """Spec baseline 3: PC xor global history, indexing saturating counters.

    Matches ``mbp::Gshare<H, T, IGNORE_UCD>``, including the left-shift that
    aligns history against the low PC bits::

        index = XorFold(pc ^ (ghist << (T - H % T)), T)

    ``ignore_unconditional=False`` by default, matching MBPlib: *all* branches
    enter the global history, not only conditional ones.
    """

    name = "gshare"

    def __init__(
        self,
        history_length: int = 15,
        log_table_size: int = 14,
        counter_bits: int = 2,
        ignore_unconditional: bool = False,
    ):
        self.history_length = history_length
        self.log_table_size = log_table_size
        self.counter_bits = counter_bits
        self.ignore_unconditional = ignore_unconditional
        self.table = [
            SaturatingCounter(counter_bits) for _ in range(1 << log_table_size)
        ]
        self.ghist = 0
        self._shift = log_table_size - (history_length % log_table_size)

    def index(self, pc: int) -> int:
        return xor_fold(pc ^ (self.ghist << self._shift), self.log_table_size)

    def predict(self, pc: int) -> bool:
        return self.table[self.index(pc)].predicts_taken

    def train(self, branch: Branch) -> None:
        self.table[self.index(branch.pc)].update(branch.taken)

    def track(self, branch: Branch) -> None:
        if self.ignore_unconditional and not branch.conditional:
            return
        self.ghist = ((self.ghist << 1) | int(branch.taken)) & (
            (1 << self.history_length) - 1
        )

    @property
    def storage_bits(self) -> int:
        return (1 << self.log_table_size) * self.counter_bits + self.history_length


class Perceptron(Predictor):
    """Spec baseline 4: Jimenez & Lin, HPCA 2001 (articles/article3.pdf).

    One perceptron per PC bucket, bipolar history (taken = +1)::

        y = w[0] + sum_i w[i] * x[i]      prediction = (y >= 0)

    Trained on misprediction *or* low confidence (``|y| <= theta``), with the
    paper's threshold heuristic ``theta = floor(1.93 * h + 14)``.

    Unlike GShare it scales linearly with history length rather than
    exponentially, which is the paper's central claim and the reason it belongs
    in this project's narrative.
    """

    name = "perceptron"

    def __init__(
        self,
        history_length: int = 62,
        log_num_perceptrons: int = 10,
        weight_bits: int = 8,
        ignore_unconditional: bool = False,
    ):
        self.history_length = history_length
        self.log_num_perceptrons = log_num_perceptrons
        self.weight_bits = weight_bits
        self.ignore_unconditional = ignore_unconditional
        self.theta = int(1.93 * history_length + 14)
        self.weight_max = (1 << (weight_bits - 1)) - 1
        self.weight_min = -(1 << (weight_bits - 1))
        self.weights = [
            [0] * (history_length + 1) for _ in range(1 << log_num_perceptrons)
        ]
        self.ghist = [False] * history_length  # index 0 == most recent
        self._output = 0
        self._row = 0

    def index(self, pc: int) -> int:
        return (pc >> 2) & ((1 << self.log_num_perceptrons) - 1)

    def predict(self, pc: int) -> bool:
        self._row = self.index(pc)
        weights = self.weights[self._row]
        total = weights[0]  # bias; x[0] == 1
        for i, bit in enumerate(self.ghist):
            total += weights[i + 1] if bit else -weights[i + 1]
        self._output = total
        return total >= 0

    def train(self, branch: Branch) -> None:
        predicted = self.predict(branch.pc)
        if predicted == branch.taken and abs(self._output) > self.theta:
            return

        weights = self.weights[self._row]
        t = 1 if branch.taken else -1
        weights[0] = max(self.weight_min, min(self.weight_max, weights[0] + t))
        for i, bit in enumerate(self.ghist):
            delta = t if bit else -t
            weights[i + 1] = max(
                self.weight_min, min(self.weight_max, weights[i + 1] + delta)
            )

    def track(self, branch: Branch) -> None:
        if self.ignore_unconditional and not branch.conditional:
            return
        self.ghist.insert(0, branch.taken)
        self.ghist.pop()

    @property
    def storage_bits(self) -> int:
        return (
            (1 << self.log_num_perceptrons)
            * (self.history_length + 1)
            * self.weight_bits
        )
