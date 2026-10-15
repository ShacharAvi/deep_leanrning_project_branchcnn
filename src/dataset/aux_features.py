"""Per-instance auxiliary features, joined to per-branch datasets.

The whole point of this module is one join, done exactly:

    prediction log `branch_index`  ==  HDF5 `br_indices_<pc>` value

Both are **the position of a branch among all branches in the trace** -- the
invariant documented in ``README.md`` and pinned by tests in
``tests/test_prediction_log.py``. Because they are the same coordinate, the
join is an equality match, not a nearest-neighbour or an interpolation. That
matters more than it sounds: an off-by-one here attaches every feature row to
the *neighbouring* dynamic instance, which still trains, still converges, and
still reports a plausible accuracy. So this module refuses to guess -- a
missing or duplicated index raises.

Why these features at all: on branch ``0x461e70`` across 16 cross-trace folds
our model sits in a 1.2 pt accuracy band while TAGE ranges over 4.1 pt. The
model is a *consistent* predictor whose ceiling is set by what it can see --
and it sees only ``(pc_low, direction)`` pairs. We are not data-starved (the
within-phase generalisation gap is 0.11 pt). So the cheap test of the plateau
hypothesis is to add *information per example* rather than more examples.

See :data:`src.evaluation.TAGE_FEATURE_GROUPS` for what each group means and,
importantly, which research question each one implies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from src.evaluation import (
    TAGE_FEATURE_GROUPS,
    load_prediction_log,
    log_paths,
)

#: Feature groups that do NOT reveal TAGE's predicted direction. A model given
#: only these is still answering "can a neural net learn this branch", with
#: TAGE's state as a difficulty hint.
DIRECTION_FREE_GROUPS = ("structural", "confidence")


class AuxFeatureError(RuntimeError):
    """Raised when features cannot be joined to a dataset exactly."""


@dataclass
class AuxFeatureSpec:
    """Where to find per-instance features, and which ones to use.

    Args:
        log_dir: Directory holding prediction logs named ``<trace_id>.bin`` /
            ``.json``, as written by ``build/run_predictors --log``.
        groups: Which :data:`TAGE_FEATURE_GROUPS` to include. Defaults to the
            direction-free groups, because including ``directional`` changes
            the research question (see module docstring) and that should be an
            explicit choice rather than a default.
        mapping: Optional explicit ``trace_id -> log prefix`` override, for
            when logs are not named after the trace.
    """

    log_dir: Path | None = None
    groups: tuple[str, ...] = DIRECTION_FREE_GROUPS
    mapping: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir) if self.log_dir is not None else None
        self.groups = tuple(self.groups)
        unknown = [g for g in self.groups if g not in TAGE_FEATURE_GROUPS]
        if unknown:
            raise KeyError(f"{unknown} not feature groups; "
                           f"choose from {sorted(TAGE_FEATURE_GROUPS)}")
        if self.log_dir is None and not self.mapping:
            raise ValueError("give either log_dir or mapping")

    @property
    def reveals_tage_direction(self) -> bool:
        """True if a model using these features can see TAGE's answer.

        Callers should record this in their results: it is the difference
        between "a neural net learned this branch" and "a neural net learned
        when to override TAGE".
        """
        return "directional" in self.groups

    def prefix_for(self, trace_id: str) -> Path:
        if trace_id in self.mapping:
            return Path(self.mapping[trace_id])
        return self.log_dir / trace_id

    def load(self, dataset_path: Path, br_pc: int,
             indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Features for ``indices``, in the same order, plus column names.

        ``indices`` are positions among all branches, taken straight from the
        dataset's ``br_indices_<pc>``.
        """
        with h5py.File(dataset_path, "r") as fh:
            trace_id = str(fh.attrs["trace_id"])

        prefix = self.prefix_for(trace_id)
        _, json_path = log_paths(prefix)
        if not json_path.exists():
            raise AuxFeatureError(
                f"no prediction log for trace {trace_id!r} at {json_path}\n"
                f"Generate it with:\n"
                f"  ./build/run_predictors <trace> --predictors tage_sc_l_64KB"
                f" \\\n      --log {prefix} --log-pcs <hard_brs.txt>"
            )

        log = load_prediction_log(prefix).for_pc(int(br_pc))
        if len(log) == 0:
            raise AuxFeatureError(
                f"prediction log {prefix} has no records for branch "
                f"{br_pc:#x} -- was it in --log-pcs?")

        log_index = log.branch_index
        if np.any(np.diff(log_index) <= 0):
            raise AuxFeatureError(
                f"{prefix} branch_index is not strictly increasing; the join "
                f"below assumes it is sorted")

        slot = np.searchsorted(log_index, indices)
        # searchsorted can return len(log_index) for values past the end.
        in_range = slot < log_index.size
        matched = np.zeros(indices.shape, dtype=bool)
        matched[in_range] = log_index[slot[in_range]] == indices[in_range]
        if not matched.all():
            missing = int((~matched).sum())
            first = int(indices[~matched][0])
            raise AuxFeatureError(
                f"{missing} of {indices.size} occurrences of {br_pc:#x} in "
                f"{trace_id} have no row in the prediction log (first: branch "
                f"index {first}).\n"
                f"This means the log and the dataset were built from different "
                f"traces, or the log was truncated (--limit / --warmup). "
                f"Refusing to train on a partial join: silently dropping these "
                f"would bias the training set towards whatever the log "
                f"happens to cover."
            )

        features = log.tage_features(self.groups)[slot]
        names = log.tage_feature_names(self.groups)
        if features.shape != (indices.size, len(names)):
            raise AuxFeatureError(
                f"joined features have shape {features.shape}, expected "
                f"{(indices.size, len(names))}")
        return np.ascontiguousarray(features, dtype=np.float32), names

    def describe(self) -> dict:
        return {
            "groups": list(self.groups),
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "reveals_tage_direction": self.reveals_tage_direction,
        }
