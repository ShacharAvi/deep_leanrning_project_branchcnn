"""Prediction logs and hybrid MPKI.

A *prediction log* records, for every dynamic instance of a selected branch,
whether each reference predictor got it right. It is written by
``tools/run_predictors.cpp --log PREFIX --log-pcs FILE`` as a packed binary
plus a JSON sidecar describing the layout.

It exists to make hybrid evaluation exact rather than estimated. The hybrid
predictor of spec §17.4 is "conventional predictor everywhere, learned model on
a few selected branches", so its misprediction count is::

    hybrid = baseline_total
           - baseline_mispredictions_on_the_selected_branches
           + model_mispredictions_on_the_selected_branches

That middle term can only come from per-instance data: a per-PC aggregate tells
you how often TAGE-SC-L was wrong at a PC, but attributing a *specific* dynamic
instance to the model requires knowing which instance it was.

**Alignment is the thing to get right.** ``branch_index`` is the absolute
position among *all* branches in the trace -- conditional and not -- which is
exactly ``chunk.index`` from :mod:`src.traces`. A neural model's predictions must
be joined to the log on that index, never on "the n-th occurrence of this PC",
because the two diverge the moment a warmup region is skipped.

The log is normally restricted to the selected H2P PCs. Logging every
conditional branch costs 18 bytes each -- over 5 GB on a large trace -- and
nothing downstream needs it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Must match ``LogRecord`` in tools/run_predictors.cpp (packed, 25 bytes).
#:
#: The trailing fields are TAGE-SC-L's state at *this* prediction. They are
#: computed inside get_prediction(), so they are available at fetch in real
#: hardware and are legitimate model inputs -- unlike the per-PC aggregates in
#: the screening CSVs, which are constants inside a per-branch model and
#: therefore carry no gradient.
LOG_DTYPE = np.dtype(
    [
        ("branch_index", "<i8"),
        ("pc", "<u8"),
        ("taken", "u1"),
        ("mispredicted", "u1"),
        ("hit_bank", "i1"),       # tagged component that answered; 0 == bimodal
        ("alt_bank", "i1"),
        ("tage_flags", "u1"),     # see TAGE_FLAGS
        ("sc_sum", "<i2"),        # statistical corrector weighted sum
        ("sc_threshold", "<i2"),  # its adaptive threshold
    ],
    align=False,
)

#: Bit positions within ``tage_flags``.
TAGE_FLAGS = {
    "high_confidence": 1, "medium_confidence": 2, "low_confidence": 4,
    "longest_match_prediction": 8, "alt_prediction": 16,
    "loop_valid": 32, "final_prediction": 64, "tage_prediction": 128,
}

LOG_FORMAT = "branch-prediction-log"
LOG_VERSION = 2


def as_unsigned_pc(pc: int) -> np.uint64:
    """A PC as the log stores it: the 64-bit pattern, unsigned.

    The dataset side keeps PCs sign-extended as int64 (kernel-space PCs such
    as ``0xffff8000...`` are negative there); the C++ driver writes the same
    bit pattern as uint64. Comparing the two needs the bit pattern, not the
    numeric value.
    """
    return np.uint64(int(pc) & 0xFFFF_FFFF_FFFF_FFFF)


class PredictionLogError(ValueError):
    """Raised when a log file is malformed or inconsistent with its sidecar."""


@dataclass(frozen=True)
class PredictionLog:
    """Per-instance outcomes for the logged branches.

    Attributes:
        records: Structured array with ``branch_index``, ``pc``, ``taken`` and
            ``mispredicted`` (a bitmask over ``predictors``).
        predictors: Predictor names, in bit order.
        metric_instructions: MPKI denominator for the run that produced this log.
        meta: The full JSON sidecar.
    """

    records: np.ndarray
    predictors: list[str]
    metric_instructions: int
    meta: dict

    def __len__(self) -> int:
        return int(self.records.shape[0])

    @property
    def branch_index(self) -> np.ndarray:
        return self.records["branch_index"]

    @property
    def pc(self) -> np.ndarray:
        return self.records["pc"]

    @property
    def taken(self) -> np.ndarray:
        return self.records["taken"].astype(bool)

    def mispredicted(self, predictor: str) -> np.ndarray:
        """Boolean mask: did ``predictor`` get each logged instance wrong?"""
        if predictor not in self.predictors:
            raise KeyError(
                f"{predictor!r} not in this log; have {self.predictors}"
            )
        bit = np.uint8(1 << self.predictors.index(predictor))
        return (self.records["mispredicted"] & bit) != 0

    def for_pc(self, pc: int) -> "PredictionLog":
        """Restrict to one static branch, preserving absolute indices."""
        return PredictionLog(
            records=self.records[self.records["pc"] == as_unsigned_pc(pc)],
            predictors=self.predictors,
            metric_instructions=self.metric_instructions,
            meta=self.meta,
        )

    def pcs(self) -> np.ndarray:
        return np.unique(self.records["pc"])

    @property
    def has_tage_state(self) -> bool:
        return bool(self.meta.get("has_tage_state", False))

    def flag(self, name: str) -> np.ndarray:
        """Boolean mask for one ``tage_flags`` bit."""
        if name not in TAGE_FLAGS:
            raise KeyError(f"{name!r} not a TAGE flag; have {sorted(TAGE_FLAGS)}")
        return (self.records["tage_flags"] & np.uint8(TAGE_FLAGS[name])) != 0

    def tage_features(self, groups=("structural", "confidence", "directional")
                      ) -> np.ndarray:
        """Per-instance TAGE state as a float32 ``(N, k)`` matrix.

        All of these are available **at fetch**, when a predictor must answer,
        so none of them is a time-travel leak. But they are not scientifically
        equivalent, so they are selectable by group -- see
        :data:`TAGE_FEATURE_GROUPS`:

        ``structural``
            Which history length answered, and whether TAGE's components
            agreed. Carries no predicted *direction*. A model given only this
            is still answering "can a NN learn this branch's behaviour", with
            TAGE's state as a hint about *how hard* this instance is.
        ``confidence``
            TAGE's own uncertainty, direction-free (``|sc_sum|``, not
            ``sc_sum``). Same question as above.
        ``directional``
            TAGE's actual answer. A model given this is answering a
            **different** question -- "can a NN correct TAGE" -- because it can
            learn "flip TAGE when it looks unsure" without learning the branch
            at all. Legal, realistic for a hybrid, and genuinely interesting,
            but it must be reported as a different experiment. See
            the feature ablation in the report.

        Column order within a group is fixed and documented in
        :data:`TAGE_FEATURE_GROUPS`, because a checkpoint's weights are
        meaningless if it changes. Use :meth:`tage_feature_names` to get the
        resolved order for a given selection.

        Scaled to roughly unit range so the FC head need not learn the
        normalisation.
        """
        if not self.has_tage_state:
            raise PredictionLogError(
                "this log has no TAGE state (written by an older driver); "
                "re-run build/run_predictors to get a version-2 log"
            )
        unknown = [g for g in groups if g not in TAGE_FEATURE_GROUPS]
        if unknown:
            raise KeyError(
                f"{unknown} not feature groups; "
                f"choose from {sorted(TAGE_FEATURE_GROUPS)}"
            )
        columns = [fn(self) for g in TAGE_FEATURE_GROUPS   # fixed group order
                   if g in groups
                   for _, fn in TAGE_FEATURE_GROUPS[g]]
        return np.stack(columns, axis=1).astype(np.float32)

    def tage_feature_names(self, groups=("structural", "confidence",
                                         "directional")) -> list[str]:
        """Resolved column names for :meth:`tage_features`, in order."""
        return [name for g in TAGE_FEATURE_GROUPS if g in groups
                for name, _ in TAGE_FEATURE_GROUPS[g]]


#: Feature groups, in canonical column order. Each entry is
#: ``(name, log -> float array)``. Keep this ordering stable: a trained
#: checkpoint's first FC layer is indexed by it.
TAGE_FEATURE_GROUPS: dict[str, tuple] = {
    "structural": (
        ("hit_bank", lambda s: s.records["hit_bank"].astype(np.float32) / 36.0),
        ("alt_bank", lambda s: s.records["alt_bank"].astype(np.float32) / 36.0),
        ("bimodal_fallback",
         lambda s: (s.records["hit_bank"] == 0).astype(np.float32)),
        ("provider_alt_disagree",
         lambda s: (s.flag("longest_match_prediction")
                    != s.flag("alt_prediction")).astype(np.float32)),
        ("loop_valid", lambda s: s.flag("loop_valid").astype(np.float32)),
        ("corrector_overrode_tage",
         lambda s: (s.flag("tage_prediction")
                    != s.flag("final_prediction")).astype(np.float32)),
    ),
    "confidence": (
        ("high_confidence", lambda s: s.flag("high_confidence").astype(np.float32)),
        ("medium_confidence",
         lambda s: s.flag("medium_confidence").astype(np.float32)),
        ("low_confidence", lambda s: s.flag("low_confidence").astype(np.float32)),
        # |sc_sum|, not sc_sum: magnitude is confidence, sign is the answer.
        ("sc_margin",
         lambda s: np.abs(s.records["sc_sum"].astype(np.float32)) / 256.0),
        ("sc_over_threshold",
         lambda s: (np.abs(s.records["sc_sum"].astype(np.int32))
                    >= np.abs(s.records["sc_threshold"].astype(np.int32))
                    ).astype(np.float32)),
    ),
    "directional": (
        ("final_prediction",
         lambda s: s.flag("final_prediction").astype(np.float32)),
        ("tage_prediction",
         lambda s: s.flag("tage_prediction").astype(np.float32)),
        ("sc_sum_signed",
         lambda s: s.records["sc_sum"].astype(np.float32) / 256.0),
    ),
}


def log_paths(prefix: str | Path) -> tuple[Path, Path]:
    """``(bin_path, json_path)`` for a log prefix, as plain string suffixes.

    **Do not use ``Path.with_suffix`` here.** Our trace ids contain dots
    (``605.mcf_s-1152B``), and ``with_suffix`` treats everything after the
    *first* dot as an existing extension to replace -- so
    ``Path("605.mcf_s-1152B").with_suffix(".json")`` is ``605.json``, a
    different file. That silently broke every log lookup for a real trace
    while passing for dot-free names like ``/tmp/v2_log``.
    """
    prefix = Path(prefix)
    if prefix.name.endswith((".bin", ".json")):
        prefix = prefix.with_name(prefix.name.rsplit(".", 1)[0])
    return (prefix.with_name(prefix.name + ".bin"),
            prefix.with_name(prefix.name + ".json"))


def load_prediction_log(prefix: str | Path) -> PredictionLog:
    """Load ``PREFIX.bin`` + ``PREFIX.json``.

    ``prefix`` may be given with or without an extension.
    """
    bin_path, json_path = log_paths(prefix)

    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)

    meta = json.loads(json_path.read_text())
    if meta.get("format") != LOG_FORMAT:
        raise PredictionLogError(
            f"{json_path} is not a {LOG_FORMAT} (got {meta.get('format')!r})"
        )
    if meta.get("version") != LOG_VERSION:
        raise PredictionLogError(
            f"unsupported log version {meta.get('version')} (expected {LOG_VERSION})"
        )
    if meta.get("record_bytes") != LOG_DTYPE.itemsize:
        raise PredictionLogError(
            f"record size mismatch: sidecar says {meta.get('record_bytes')}, "
            f"this reader expects {LOG_DTYPE.itemsize}"
        )

    records = np.fromfile(bin_path, dtype=LOG_DTYPE)
    if records.shape[0] != meta["records"]:
        raise PredictionLogError(
            f"{bin_path.name} holds {records.shape[0]:,} records but the "
            f"sidecar declares {meta['records']:,} -- file is truncated"
        )
    if records.shape[0] and np.any(np.diff(records["branch_index"]) <= 0):
        raise PredictionLogError(
            "branch_index is not strictly increasing; the log is corrupt or "
            "was concatenated from separate runs"
        )

    return PredictionLog(
        records=records,
        predictors=list(meta["predictors"]),
        metric_instructions=int(meta["metric_instructions"]),
        meta=meta,
    )


@dataclass(frozen=True)
class HybridResult:
    """Outcome of swapping a learned model in for selected branches."""

    baseline_mispredictions: int
    hybrid_mispredictions: int
    metric_instructions: int
    covered_instances: int
    baseline_mispredictions_on_covered: int
    model_mispredictions_on_covered: int

    @property
    def baseline_mpki(self) -> float:
        return 1000.0 * self.baseline_mispredictions / self.metric_instructions

    @property
    def hybrid_mpki(self) -> float:
        return 1000.0 * self.hybrid_mispredictions / self.metric_instructions

    @property
    def mpki_reduction(self) -> float:
        """Relative reduction, the metric spec §17.4 asks for. Negative = worse."""
        if not self.baseline_mispredictions:
            return 0.0
        return (
            self.baseline_mispredictions - self.hybrid_mispredictions
        ) / self.baseline_mispredictions

    def as_dict(self) -> dict:
        return {
            "baseline_mispredictions": self.baseline_mispredictions,
            "hybrid_mispredictions": self.hybrid_mispredictions,
            "baseline_mpki": self.baseline_mpki,
            "hybrid_mpki": self.hybrid_mpki,
            "mpki_reduction": self.mpki_reduction,
            "covered_instances": self.covered_instances,
            "baseline_mispredictions_on_covered":
                self.baseline_mispredictions_on_covered,
            "model_mispredictions_on_covered":
                self.model_mispredictions_on_covered,
        }


def hybrid_mispredictions(
    log: PredictionLog,
    baseline: str,
    baseline_total_mispredictions: int,
    model_correct: dict[int, np.ndarray],
    *,
    metric_instructions: int | None = None,
) -> HybridResult:
    """Score a hybrid predictor against per-instance ground truth.

    Args:
        log: Prediction log covering at least the branches in ``model_correct``.
        baseline: Which logged predictor the model is displacing, e.g.
            ``"tage_sc_l_64KB"``.
        baseline_total_mispredictions: The baseline's misprediction count over
            the *whole* trace, from the run's summary JSON. The log only covers
            selected branches, so this cannot be derived from it.
        model_correct: ``{branch_index: bool array}`` is **not** the shape here --
            pass ``{pc: correct}``, where ``correct`` is a boolean array aligned
            with ``log.for_pc(pc)`` in ``branch_index`` order.
        metric_instructions: MPKI denominator. Defaults to the log's.

    Returns:
        A :class:`HybridResult`.

    Raises:
        ValueError: If a model's prediction array does not line up with the
            log's instances for that PC. This is the check that stops a
            silently misaligned evaluation from producing a plausible number.
    """
    instructions = (
        metric_instructions
        if metric_instructions is not None
        else log.metric_instructions
    )
    baseline_wrong_all = log.mispredicted(baseline)

    covered = 0
    baseline_wrong_on_covered = 0
    model_wrong_on_covered = 0

    for pc, correct in model_correct.items():
        mask = log.records["pc"] == as_unsigned_pc(pc)
        n = int(mask.sum())
        if n == 0:
            raise ValueError(
                f"pc {pc:#x} has no instances in the log; it was not selected "
                f"when the log was written"
            )
        correct = np.asarray(correct, dtype=bool)
        if correct.shape[0] != n:
            raise ValueError(
                f"pc {pc:#x}: model produced {correct.shape[0]:,} predictions "
                f"but the log holds {n:,} instances -- the evaluation is "
                f"misaligned"
            )
        covered += n
        baseline_wrong_on_covered += int(baseline_wrong_all[mask].sum())
        model_wrong_on_covered += int((~correct).sum())

    hybrid = (
        baseline_total_mispredictions
        - baseline_wrong_on_covered
        + model_wrong_on_covered
    )
    return HybridResult(
        baseline_mispredictions=baseline_total_mispredictions,
        hybrid_mispredictions=hybrid,
        metric_instructions=instructions,
        covered_instances=covered,
        baseline_mispredictions_on_covered=baseline_wrong_on_covered,
        model_mispredictions_on_covered=model_wrong_on_covered,
    )


def oracle_ceiling(
    log: PredictionLog, baseline: str, baseline_total_mispredictions: int
) -> HybridResult:
    """Upper bound: a model that is perfect on every logged branch.

    Useful before training anything -- if the ceiling is small, no model will
    rescue the result, and that is worth knowing early.
    """
    perfect = {
        int(pc): np.ones(int((log.records["pc"] == pc).sum()), dtype=bool)
        for pc in log.pcs()
    }
    return hybrid_mispredictions(
        log, baseline, baseline_total_mispredictions, perfect
    )
