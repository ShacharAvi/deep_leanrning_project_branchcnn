"""Reference branch predictors in Python.

These exist for two reasons:

1. **Testable semantics.** The spec (section 27) asks for predictor behaviour to
   be verified against hand-designed patterns like ``1010101010``. Doing that in
   Python keeps the tests readable.
2. **A cross-check on the C++ driver.** ``Bimodal`` and ``GShare`` reproduce
   MBPlib's hashing bit-for-bit, so the same synthetic trace must yield identical
   misprediction counts from ``build/run_predictors`` and from here. Two
   independent implementations agreeing is much stronger evidence than either
   alone.

Full-trace runs use the C++ driver -- it is several orders of magnitude faster.
These are correctness oracles, not production predictors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class Branch:
    """One branch event, mirroring ``mbp::Branch``."""

    pc: int
    taken: bool
    conditional: bool = True
    indirect: bool = False


class SaturatingCounter:
    """Signed saturating counter, matching ``mbp::SatCtr<N>``.

    Range is ``[-2^(N-1), 2^(N-1) - 1]``; a 2-bit counter spans -2..1 and
    predicts taken when the value is non-negative.
    """

    __slots__ = ("value", "min_value", "max_value")

    def __init__(self, bits: int, value: int = 0):
        self.min_value = -(1 << (bits - 1))
        self.max_value = (1 << (bits - 1)) - 1
        self.value = value

    def update(self, increment: bool) -> None:
        delta = 1 if increment else -1
        self.value = max(self.min_value, min(self.max_value, self.value + delta))

    @property
    def predicts_taken(self) -> bool:
        return self.value >= 0


def xor_fold(value: int, width: int) -> int:
    """Port of ``mbp::XorFold``: fold a 64-bit value down to ``width`` bits."""
    if width == 0:
        return 0
    value &= UINT64_MASK
    fold = 0
    for _ in range(0, 64, width):
        fold ^= value
        value >>= width
    return fold & ((1 << width) - 1)


class Predictor(ABC):
    """Interface mirroring ``mbp::Predictor``.

    The simulation protocol is::

        if branch.conditional:
            predictor.predict(branch.pc)
            predictor.train(branch)
        predictor.track(branch)
    """

    name = "predictor"

    @abstractmethod
    def predict(self, pc: int) -> bool:
        """Predicted direction. Must not change state that affects future predictions."""

    @abstractmethod
    def train(self, branch: Branch) -> None:
        """Update prediction state for this branch."""

    def track(self, branch: Branch) -> None:
        """Update global history state. Called for *every* branch."""

    @property
    def storage_bits(self) -> int:
        return 0


@dataclass
class SimulationResult:
    conditional_branches: int
    mispredictions: int
    predictions: list[bool]

    @property
    def misprediction_rate(self) -> float:
        if not self.conditional_branches:
            return 0.0
        return self.mispredictions / self.conditional_branches

    @property
    def accuracy(self) -> float:
        return 1.0 - self.misprediction_rate


def simulate(
    predictor: Predictor, branches, *, record_predictions: bool = False
) -> SimulationResult:
    """Run a predictor over branches using the canonical MBPlib loop.

    ``track`` is called for every branch, conditional or not -- that is what
    ``mbp::Simulate`` does, and it matters: omitting it changes which branches
    reach the global history, and with it the result.
    """
    conditional = mispredictions = 0
    predictions: list[bool] = []

    for branch in branches:
        if branch.conditional:
            predicted = predictor.predict(branch.pc)
            predictor.train(branch)
            conditional += 1
            if predicted != branch.taken:
                mispredictions += 1
            if record_predictions:
                predictions.append(predicted)
        predictor.track(branch)

    return SimulationResult(conditional, mispredictions, predictions)


def pattern_to_branches(pattern: str, pc: int = 0x1000, repeat: int = 1):
    """``"1010"`` -> alternating taken/not-taken branches at one PC.

    The hand-designed sequences of spec section 27.
    """
    bits = [c == "1" for c in pattern.strip()] * repeat
    return [Branch(pc=pc, taken=bit) for bit in bits]
