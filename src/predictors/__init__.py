"""Reference branch predictors (spec section 9, baselines 0-4)."""

from .base import (
    Branch,
    Predictor,
    SaturatingCounter,
    SimulationResult,
    pattern_to_branches,
    simulate,
    xor_fold,
)
from .classical import AlwaysNotTaken, AlwaysTaken, Bimodal, GShare, Perceptron

__all__ = [
    "AlwaysNotTaken",
    "AlwaysTaken",
    "Bimodal",
    "Branch",
    "GShare",
    "Perceptron",
    "Predictor",
    "SaturatingCounter",
    "SimulationResult",
    "pattern_to_branches",
    "simulate",
    "xor_fold",
]
