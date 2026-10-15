"""TAGE-SC-L decision as a per-instance input (and a learnable prior weight).

Source: ``tools/tage_cbp2025.cpp`` replays the CBP2016 64 KB TAGE-SC-L over each trace and writes
4 bytes per conditional branch, in conditional-branch order:

    byte 0  mispredicted         <-- DERIVED FROM THE LABEL. Used only to verify the join and
                                     to score the baseline. NEVER an input.
    byte 1  predicted direction  (made before the outcome is known)
    byte 2  HitBank              (provider table, 0 = bimodal)
    byte 3  confidence bits      high | low<<1 | alt-available<<2

Why this is not a leak: TAGE's answer for branch k is a function of branches < k only, and the
replay is independent of the neural model. The join is checked exactly:
``predicted XOR mispredicted == taken`` for every sample, and the log length must equal the
number of samples (so ``sample_stride`` must be 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .feature_extraction import TraceFeatures

__all__ = [
    "TAGE_FEATURE_NAMES",
    "NUM_TAGE_FEATURES",
    "NUM_TAGE_STATES",
    "TageLog",
    "load_tage_log",
    "tage_feature_matrix",
    "tage_state",
]

TAGE_FEATURE_NAMES: List[str] = [
    "tage_pred_sign",   # +1 taken / -1 not taken
    "tage_hit_bank",    # HitBank / 36
    "tage_bimodal",     # 1 when no tagged table hit
    "tage_high_conf",
    "tage_low_conf",
    "tage_alt_avail",
]
NUM_TAGE_FEATURES = len(TAGE_FEATURE_NAMES)

#: 6 HitBank buckets x 8 confidence patterns. The learnable TAGE weight is indexed by this state.
HIT_BUCKETS = 6
NUM_TAGE_STATES = HIT_BUCKETS * 8
_MAX_HIT = 36


@dataclass
class TageLog:
    pred: np.ndarray   # uint8 0/1
    hit: np.ndarray    # uint8
    conf: np.ndarray   # uint8 (3 bits)
    # The misprediction byte is deliberately NOT stored on the object.


def load_tage_log(path: Path, features: TraceFeatures) -> TageLog:
    """Load and verify the log against one trace's samples. Refuses a partial or shifted join."""
    if features.config.sample_stride != 1:
        raise ValueError("TAGE join needs sample_stride == 1 (one log row per conditional branch)")
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 4:
        raise ValueError(f"{path}: size {raw.size} is not a multiple of 4")
    log = raw.reshape(-1, 4)
    if log.shape[0] != features.num_samples:
        raise ValueError(
            f"{path}: {log.shape[0]} TAGE rows but {features.num_samples} samples for "
            f"{features.trace_id}; refusing a partial join"
        )
    miss, pred = log[:, 0], log[:, 1]
    if not np.array_equal(pred ^ miss, features.s_out):
        raise ValueError(f"{path}: predicted XOR mispredicted != taken; log is misaligned")
    return TageLog(pred=pred.copy(), hit=log[:, 2].copy(), conf=log[:, 3].copy())


def tage_state(hit: np.ndarray, conf: np.ndarray) -> np.ndarray:
    bucket = np.minimum(hit.astype(np.int64) // 6, HIT_BUCKETS - 1)
    return bucket * 8 + (conf.astype(np.int64) & 7)


def tage_feature_matrix(log: TageLog, rows: np.ndarray) -> np.ndarray:
    pred = log.pred[rows].astype(np.float32)
    hit = log.hit[rows].astype(np.float32)
    conf = log.conf[rows].astype(np.int64)
    out = np.empty((rows.shape[0], NUM_TAGE_FEATURES), np.float32)
    out[:, 0] = 2.0 * pred - 1.0
    out[:, 1] = np.minimum(hit, _MAX_HIT) / _MAX_HIT
    out[:, 2] = (hit == 0).astype(np.float32)
    out[:, 3] = (conf & 1).astype(np.float32)
    out[:, 4] = ((conf >> 1) & 1).astype(np.float32)
    out[:, 5] = ((conf >> 2) & 1).astype(np.float32)
    return out
