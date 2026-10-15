"""Hybrid arbitration between TAGE-SC-L and a per-branch neural model.

Three policies, all scored on the same per-execution arrays:

``branch``  (BranchNet, spec §23 step 6)
    Attach the model to the branch -- every execution -- iff it beat TAGE on
    validation significantly (one-sided exact McNemar on the discordant
    executions, p < alpha).

``confidence``  (this project's contribution)
    Decide per EXECUTION: use the model only when TAGE's own confidence is in
    a set S (e.g. "low", "low+medium") AND the model is confident,
    ``|sigmoid(logit) - 0.5| >= tau``. (S, tau) are fitted per branch on
    validation, and the chosen rule must itself pass McNemar on the
    executions it overrides, at a Bonferroni-corrected level because the grid
    is searched. Even on hard branches TAGE is right 85-95% of the time;
    attaching a model to all of them gives back on confident-TAGE executions
    what it gains on uncertain ones.

``oracle``
    Whichever of the two is right when they disagree -- the ceiling of any
    arbiter over these two predictors. Upper bound only; never deployable.

Nothing here sees test data while fitting: :func:`fit_confidence_rule` and
:func:`branch_gate` take validation arrays only, and the test arrays are
passed only to :func:`score`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: TAGE confidence levels, from the prediction log's flags.
HIGH, MEDIUM, LOW = 0, 1, 2

#: Candidate TAGE-confidence sets: name -> minimum level overridden.
CONFIDENCE_SETS = {"low": LOW, "low+medium": MEDIUM, "all": HIGH}
TAU_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45)


@dataclass
class Executions:
    """Per-execution arrays for one branch on one split, all aligned."""

    logits: np.ndarray        # model logits
    labels: np.ndarray        # actual direction (0/1)
    tage_correct: np.ndarray  # bool
    tage_conf: np.ndarray     # HIGH / MEDIUM / LOW

    def __post_init__(self):
        n = self.logits.shape[0]
        for name in ("labels", "tage_correct", "tage_conf"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} has {getattr(self, name).shape[0]} "
                                 f"rows, logits {n}: misaligned")
        self.tage_correct = self.tage_correct.astype(bool)

    @property
    def model_correct(self) -> np.ndarray:
        return (self.logits > 0) == (self.labels > 0.5)

    @property
    def model_confidence(self) -> np.ndarray:
        return np.abs(1.0 / (1.0 + np.exp(-self.logits)) - 0.5)

    def __len__(self) -> int:
        return int(self.logits.shape[0])


def mcnemar_one_sided(model_only: int, tage_only: int) -> float:
    """P(model no better than TAGE), exact test on discordant executions."""
    from scipy.stats import binomtest
    n = model_only + tage_only
    if n == 0:
        return 1.0
    return float(binomtest(model_only, n, 0.5, alternative="greater").pvalue)


def _discordant(ex: Executions, use: np.ndarray) -> tuple[int, int]:
    m, t = ex.model_correct, ex.tage_correct
    return int((use & m & ~t).sum()), int((use & ~m & t).sum())


@dataclass
class Rule:
    """A fitted confidence rule; ``None`` set means never override."""

    conf_set: str | None
    tau: float = 0.0
    val_gain: int = 0
    val_p: float = 1.0
    meta: dict = field(default_factory=dict)

    def use_model(self, ex: Executions) -> np.ndarray:
        if self.conf_set is None:
            return np.zeros(len(ex), dtype=bool)
        level = CONFIDENCE_SETS[self.conf_set]
        return (ex.tage_conf >= level) & (ex.model_confidence >= self.tau)


def branch_gate(val: Executions, alpha: float = 0.05) -> Rule:
    """BranchNet policy: all executions, or none."""
    use = np.ones(len(val), dtype=bool)
    mo, to = _discordant(val, use)
    p = mcnemar_one_sided(mo, to)
    if p < alpha:
        return Rule("all", 0.0, mo - to, p)
    return Rule(None, val_gain=0, val_p=p)


def fit_confidence_rule(val: Executions, alpha: float = 0.05) -> Rule:
    """Best (S, tau) on validation; kept only if significant after correction."""
    grid = [(s, tau) for s in CONFIDENCE_SETS for tau in TAU_GRID]
    best = None
    for s, tau in grid:
        rule = Rule(s, tau)
        mo, to = _discordant(val, rule.use_model(val))
        gain = mo - to
        if best is None or gain > best[0]:
            best = (gain, s, tau, mo, to)
    gain, s, tau, mo, to = best
    p = mcnemar_one_sided(mo, to)
    if gain > 0 and p < alpha / len(grid):
        return Rule(s, tau, gain, p, {"grid": len(grid)})
    return Rule(None, val_gain=0, val_p=p, meta={"grid": len(grid),
                                                 "best_rejected": [s, tau, gain]})


def score(ex: Executions, rule: Rule | str) -> dict:
    """Mispredictions on ``ex`` for TAGE alone and for the hybrid.

    ``rule`` is a fitted :class:`Rule`, or ``"oracle"``.
    """
    tage_miss = int((~ex.tage_correct).sum())
    if rule == "oracle":
        hybrid_miss = int((~ex.tage_correct & ~ex.model_correct).sum())
        overridden = int((ex.tage_correct != ex.model_correct).sum())
    else:
        use = rule.use_model(ex)
        wrong = np.where(use, ~ex.model_correct, ~ex.tage_correct)
        hybrid_miss = int(wrong.sum())
        overridden = int(use.sum())
    return {"executions": len(ex), "tage_misses": tage_miss,
            "hybrid_misses": hybrid_miss, "overridden": overridden}
