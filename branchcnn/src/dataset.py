"""Splits, windowing and feature assembly for the branch-direction CNN.

Three things live here.

**Splits.** The 14 media traces are not 14 independent samples: by PC-set
containment they collapse into 7 families, and two traces of one family are near
duplicates. Splitting by trace would therefore put near-copies of the training
data into test. Splits are by *family*, and :func:`check_split` fails loudly if
that is ever violated.

**Windowing.** Sequence inputs are sliced from the stored branch history on
demand: ``history[pos - H : pos]``. Materialising windows would cost H times the
memory for no extra information. ``pos`` is the branch's own slot, and the slice
is half-open, so a branch never sees itself.

**Feature assembly.** Compact per-sample integers are turned into float vectors
one batch at a time. The full matrix would be ~5 GB at 15.7M samples; a batch of
it is a few hundred kilobytes.

Normalisation statistics are computed on the training split only and then
applied unchanged to validation and test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .tage_features import (
    NUM_TAGE_FEATURES,
    TageLog,
    load_tage_log,
    tage_feature_matrix,
    tage_state,
)
from .feature_extraction import (
    LOCAL_HISTORY_BITS,
    MIX_CLASSES,
    SLICE_CLASSES,
    TraceFeatures,
    load_features,
)

__all__ = [
    "TRACE_FAMILIES",
    "DEFAULT_SPLIT",
    "SplitSpec",
    "DatasetConfig",
    "SplitData",
    "BranchDataset",
    "check_split",
    "fold_pc",
    "scalar_feature_names",
]

#: Trace -> family, from PC-set containment >= 0.9 measured across the 14 media
#: traces. Members of a family share almost all of their static branches.
TRACE_FAMILIES: Dict[str, str] = {
    # CBP2025 web, cluster W1 (same binary; each trace is its own "family" because they are
    # independent runs, NOT near-duplicates: PC Jaccard ~0.47).
    "web_3_trace": "W3", "web_5_trace": "W5", "web_8_trace": "W8", "web_27_trace": "W27",
    "web_31_trace": "W31", "web_32_trace": "W32", "web_33_trace": "W33",
    # cross-program test traces: a DIFFERENT binary (cluster W2), hard (TAGE MPKI 6-8)
    "web_7_trace": "W7", "web_93_trace": "W93",
    "media_0_trace": "A",
    "media_1_trace": "A",
    "media_3_trace": "A",
    "media_7_trace": "A",
    "media_12_trace": "A",
    "media_13_trace": "A",
    "media_10_trace": "B",
    "media_11_trace": "B",
    "media_2_trace": "C",
    "media_6_trace": "C",
    "media_4_trace": "D",
    "media_5_trace": "E",
    "media_8_trace": "F",
    "media_9_trace": "G",
}

# Plan (b) and other web runs: every CBP2025 web trace is its own family
# (independent runs of one binary, as for W1 above). Existing entries are kept.
for _n in range(0, 256):
    TRACE_FAMILIES.setdefault(f"web_{_n}_trace", f"W{_n}")


@dataclass(frozen=True)
class SplitSpec:
    """Which families go to which role."""

    train: Tuple[str, ...]
    validation: Tuple[str, ...]
    test: Tuple[str, ...]

    def role_of(self, family: str) -> Optional[str]:
        if family in self.train:
            return "train"
        if family in self.validation:
            return "validation"
        if family in self.test:
            return "test"
        return None

    def traces(self, role: str) -> List[str]:
        families = getattr(self, role)
        return sorted(t for t, f in TRACE_FAMILIES.items() if f in families)


#: Families A/B/C train (10 traces), D/E validate (2), F/G test (2).
DEFAULT_SPLIT = SplitSpec(
    train=("A", "B", "C"), validation=("D", "E"), test=("F", "G")
)


def check_split(spec: SplitSpec) -> None:
    """Fail if any family appears in more than one role, or none at all."""
    groups = {"train": set(spec.train), "validation": set(spec.validation), "test": set(spec.test)}
    for left in groups:
        for right in groups:
            if left >= right:
                continue
            shared = groups[left] & groups[right]
            if shared:
                raise ValueError(
                    f"families {sorted(shared)} appear in both {left} and {right}; "
                    f"splits must be family-disjoint"
                )
    known = set(TRACE_FAMILIES.values())
    assigned = groups["train"] | groups["validation"] | groups["test"]
    unknown = assigned - known
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}")
    for role, families in groups.items():
        if not families:
            raise ValueError(f"split role {role!r} has no families")


def fold_pc(values: np.ndarray, pc_bits: int) -> np.ndarray:
    """XOR-fold a PC down to ``pc_bits``.

    Masking alone would collide any two branches whose addresses differ only
    above the mask, which happens between shared libraries. Folding mixes the
    high bits back in.
    """
    out = np.asarray(values, dtype=np.uint64).copy()
    mask = np.uint64((1 << pc_bits) - 1)
    shift = np.uint64(pc_bits)
    result = out & mask
    out = out >> shift
    while bool(np.any(out)):
        result = result ^ (out & mask)
        out = out >> shift
    return result.astype(np.int64)


@dataclass(frozen=True)
class DatasetConfig:
    """Representation knobs. One extraction serves every setting here."""

    history_length: int = 64
    pc_bits: int = 14
    #: Cap on training samples per trace; validation and test are never capped,
    #: because a thinned test set makes any reported rate fiction.
    max_train_samples_per_trace: Optional[int] = None
    seed: int = 0
    #: Join the TAGE-SC-L per-instance log (direction, hit bank, confidence) to every sample.
    use_tage: bool = False
    tage_dir: Optional[Path] = None
    #: Control: permute the TAGE rows across instances of each trace (labels and inputs fixed).
    shuffle_tage: bool = False


def scalar_feature_names() -> List[str]:
    """Names of the scalar block, in column order."""
    names = [f"local_hist_{i}" for i in range(LOCAL_HISTORY_BITS)]
    names += [
        "log_nseen",
        "dep_depth",
        "log_dep_ninstr",
        "log_dep_npcs",
        "log_nload",
        "log_near_load",
        "log_oldest",
        "log_near_writer",
        "nsrc",
        "has_near_load",
        "has_oldest",
        "has_near_writer",
        "reads_flags",
        "slice_incomplete",
    ]
    names += [f"fwclass_{i}" for i in range(13)]
    names += [f"dwclass_bit_{i}" for i in range(12)]
    names += [f"regkind_bit_{i}" for i in range(5)]
    names += [f"slice_mix_{c}" for c in SLICE_CLASSES]
    return names


_NONE_U32 = int(np.iinfo(np.uint32).max)
NUM_SCALARS = len(scalar_feature_names())
NUM_SEQ_NUMERIC = 2 + len(MIX_CLASSES)  # backward, log ninstr, mix counts


def _build_scalars(features: TraceFeatures, rows: np.ndarray) -> np.ndarray:
    """Assemble the float scalar block for ``rows`` of one trace."""
    count = rows.shape[0]
    out = np.zeros((count, NUM_SCALARS), dtype=np.float32)

    local = features.s_local[rows].astype(np.uint32)
    for bit in range(LOCAL_HISTORY_BITS):
        out[:, bit] = ((local >> np.uint32(bit)) & np.uint32(1)).astype(np.float32)
    col = LOCAL_HISTORY_BITS

    def put(values: np.ndarray) -> None:
        nonlocal col
        out[:, col] = values.astype(np.float32)
        col += 1

    put(np.log1p(features.s_nseen[rows].astype(np.float64)))
    put(features.s_dep_depth[rows])
    put(np.log1p(features.s_dep_ninstr[rows].astype(np.float64)))
    put(np.log1p(features.s_dep_npcs[rows].astype(np.float64)))
    put(np.log1p(features.s_nload[rows].astype(np.float64)))

    for name in ("s_near_load", "s_oldest", "s_near_writer"):
        raw = getattr(features, name)[rows].astype(np.int64)
        missing = raw == _NONE_U32
        safe = np.where(missing, 0, raw).astype(np.float64)
        put(np.log1p(safe))
    put(features.s_nsrc[rows])
    for name in ("s_near_load", "s_oldest", "s_near_writer"):
        raw = getattr(features, name)[rows].astype(np.int64)
        put((raw != _NONE_U32).astype(np.float32))

    put(features.s_flags[rows])
    put(features.s_incomplete[rows])

    # Flag-writer instruction class: 12 classes plus a "no flag writer" slot.
    fwclass = features.s_fwclass[rows].astype(np.int64)
    fwclass = np.where(fwclass > 11, 12, fwclass)
    out[np.arange(count), col + fwclass] = 1.0
    col += 13

    dwclass = features.s_dwclass[rows].astype(np.int64)
    for bit in range(12):
        out[:, col + bit] = ((dwclass >> bit) & 1).astype(np.float32)
    col += 12

    regkinds = features.s_regkinds[rows].astype(np.int64)
    for bit in range(5):
        out[:, col + bit] = ((regkinds >> bit) & 1).astype(np.float32)
    col += 5

    mix = features.s_slice_mix[rows].astype(np.float64)
    out[:, col : col + len(SLICE_CLASSES)] = np.log1p(mix).astype(np.float32)
    col += len(SLICE_CLASSES)

    assert col == NUM_SCALARS, (col, NUM_SCALARS)
    return out


@dataclass
class _Trace:
    """One loaded trace, with its sample rows already filtered."""

    features: TraceFeatures
    family: str
    rows: np.ndarray  # indices into the sample arrays
    #: Precomputed sequence channels, shared by every window of this trace.
    token: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    klass: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    numeric: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    #: Folded PC of the branch being predicted, one per sample row.
    sample_pc: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    #: TAGE log aligned to the sample arrays (indexed by ``rows`` values), or None.
    tage: Optional[TageLog] = None


@dataclass
class SplitData:
    """Loaded traces for one split role."""

    role: str
    config: DatasetConfig
    traces: List[_Trace] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return int(sum(t.rows.shape[0] for t in self.traces))

    @property
    def taken_rate(self) -> float:
        total = self.num_samples
        if not total:
            return 0.0
        taken = sum(int(t.features.s_out[t.rows].sum()) for t in self.traces)
        return taken / total

    def families(self) -> List[str]:
        return sorted({t.family for t in self.traces})


class BranchDataset:
    """Loads feature files, builds splits, and yields batches.

    Not a ``torch.utils.data.Dataset``: windows are gathered a whole batch at a
    time with one fancy-index, which is far faster than a Python call per
    sample at these volumes.
    """

    def __init__(
        self,
        feature_dir: Path,
        config: DatasetConfig = DatasetConfig(),
        split: SplitSpec = DEFAULT_SPLIT,
    ) -> None:
        check_split(split)
        self.feature_dir = Path(feature_dir)
        self.config = config
        self.split = split
        self.splits: Dict[str, SplitData] = {}
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    # -- loading -----------------------------------------------------------

    def load(self, roles: Sequence[str] = ("train", "validation", "test")) -> None:
        rng = np.random.default_rng(self.config.seed)
        for role in roles:
            data = SplitData(role=role, config=self.config)
            for trace_id in self.split.traces(role):
                path = self.feature_dir / f"{trace_id}.npz"
                if not path.exists():
                    continue
                features = load_features(path)
                rows = self._select_rows(features, role, rng)
                if rows.size == 0:
                    continue
                entry = _Trace(
                    features=features,
                    family=TRACE_FAMILIES[trace_id],
                    rows=rows,
                )
                self._prepare_sequence(entry)
                if self.config.use_tage:
                    entry.tage = self._load_tage(trace_id, features)
                data.traces.append(entry)
            self.splits[role] = data

        if "train" in self.splits:
            self._fit_normaliser(self.splits["train"])

    def _load_tage(self, trace_id: str, features: TraceFeatures) -> TageLog:
        if self.config.tage_dir is None:
            raise ValueError("use_tage needs DatasetConfig.tage_dir")
        log = load_tage_log(Path(self.config.tage_dir) / f"{trace_id}.log.bin", features)
        if self.config.shuffle_tage:
            perm = np.random.default_rng(self.config.seed + 7).permutation(log.pred.shape[0])
            log = TageLog(pred=log.pred[perm], hit=log.hit[perm], conf=log.conf[perm])
        return log

    def _select_rows(
        self, features: TraceFeatures, role: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Drop samples without a full history window, then optionally thin."""
        valid = np.flatnonzero(features.s_pos >= self.config.history_length)
        cap = self.config.max_train_samples_per_trace
        if role == "train" and cap is not None and valid.size > cap:
            chosen = rng.choice(valid.size, size=cap, replace=False)
            valid = np.sort(valid[chosen])
        return valid

    def _prepare_sequence(self, entry: _Trace) -> None:
        """Precompute per-branch-slot channels once per trace."""
        features = entry.features
        folded = fold_pc(features.h_pc, self.config.pc_bits)
        entry.token = (folded << 1) | features.h_dir.astype(np.int64)
        entry.klass = features.h_class.astype(np.int64)

        numeric = np.empty((features.num_branches, NUM_SEQ_NUMERIC), dtype=np.float32)
        numeric[:, 0] = features.h_back.astype(np.float32)
        numeric[:, 1] = np.log1p(features.h_ninstr.astype(np.float64))
        numeric[:, 2:] = np.log1p(features.h_mix.astype(np.float64))
        entry.numeric = numeric

        entry.sample_pc = fold_pc(
            features.s_pc >> np.uint64(2), self.config.pc_bits
        )

    # -- normalisation -----------------------------------------------------

    def _fit_normaliser(self, data: SplitData, max_rows: int = 200_000) -> None:
        """Mean/std over a sample of the training split only."""
        rng = np.random.default_rng(self.config.seed)
        blocks = []
        per_trace = max(1, max_rows // max(1, len(data.traces)))
        for entry in data.traces:
            rows = entry.rows
            if rows.size > per_trace:
                rows = np.sort(rng.choice(rows, size=per_trace, replace=False))
            blocks.append(_build_scalars(entry.features, rows))
        if not blocks:
            return
        stacked = np.concatenate(blocks, axis=0)
        self._mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        # Constant columns would divide by zero; leave them at zero instead.
        std[std < 1e-6] = 1.0
        self._std = std

    @property
    def vocabulary_size(self) -> int:
        return 1 << (self.config.pc_bits + 1)

    @property
    def num_pc_ids(self) -> int:
        return 1 << self.config.pc_bits

    def normaliser(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._mean is None or self._std is None:
            raise RuntimeError("normaliser not fitted; call load() with a train split")
        return self._mean, self._std

    # -- batching ----------------------------------------------------------

    def iter_batches(
        self,
        role: str,
        batch_size: int = 512,
        shuffle: bool = False,
        seed: int = 0,
        shuffle_labels: bool = False,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Yield batches of assembled features.

        ``shuffle_labels`` is the leakage control: it permutes labels within the
        split while leaving inputs untouched, so a clean pipeline can do no
        better than the majority class.
        """
        data = self.splits[role]
        if not data.traces:
            return
        mean, std = self.normaliser()
        rng = np.random.default_rng(seed)

        pairs = np.concatenate(
            [
                np.stack(
                    [np.full(t.rows.shape[0], i, dtype=np.int64), t.rows]
                )
                for i, t in enumerate(data.traces)
            ],
            axis=1,
        )
        order = (
            rng.permutation(pairs.shape[1]) if shuffle else np.arange(pairs.shape[1])
        )

        offsets = np.arange(-self.config.history_length, 0)
        for start in range(0, order.size, batch_size):
            chunk = order[start : start + batch_size]
            trace_ids, rows = pairs[0, chunk], pairs[1, chunk]
            size = chunk.size

            tokens = np.empty((size, self.config.history_length), np.int64)
            classes = np.empty((size, self.config.history_length), np.int64)
            numeric = np.empty(
                (size, self.config.history_length, NUM_SEQ_NUMERIC), np.float32
            )
            scalars = np.empty((size, NUM_SCALARS), np.float32)
            pcs = np.empty(size, np.int64)
            labels = np.empty(size, np.float32)
            if self.config.use_tage:
                tage = np.empty((size, NUM_TAGE_FEATURES), np.float32)
                tstate = np.empty(size, np.int64)

            for trace_id in np.unique(trace_ids):
                mask = trace_ids == trace_id
                entry = data.traces[trace_id]
                positions = entry.features.s_pos[rows[mask]]
                window = positions[:, None] + offsets
                tokens[mask] = entry.token[window]
                classes[mask] = entry.klass[window]
                numeric[mask] = entry.numeric[window]
                scalars[mask] = _build_scalars(entry.features, rows[mask])
                pcs[mask] = entry.sample_pc[rows[mask]]
                labels[mask] = entry.features.s_out[rows[mask]].astype(np.float32)
                if self.config.use_tage:
                    tage[mask] = tage_feature_matrix(entry.tage, rows[mask])
                    tstate[mask] = tage_state(entry.tage.hit[rows[mask]], entry.tage.conf[rows[mask]])

            if shuffle_labels:
                labels = rng.permutation(labels)

            scalars = (scalars - mean) / std
            extra = {}
            if self.config.use_tage:
                extra = {"tage": tage, "tage_state": tstate, "tage_sign": tage[:, 0].copy()}
            yield {
                **extra,
                "tokens": tokens,
                "classes": classes,
                "seq_numeric": numeric,
                "scalars": scalars,
                "pc": pcs,
                "label": labels,
            }

    # -- reporting ---------------------------------------------------------

    def describe(self) -> dict:
        return {
            "history_length": self.config.history_length,
            "pc_bits": self.config.pc_bits,
            "vocabulary_size": self.vocabulary_size,
            "num_scalars": NUM_SCALARS,
            "num_seq_numeric": NUM_SEQ_NUMERIC,
            "splits": {
                role: {
                    "families": data.families(),
                    "traces": [t.features.trace_id for t in data.traces],
                    "samples": data.num_samples,
                    "taken_rate": round(data.taken_rate, 6),
                }
                for role, data in self.splits.items()
            },
        }

    def save_description(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.describe(), indent=2), encoding="utf-8")
