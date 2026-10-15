"""Fetch-safe raw trace features (CBP2025), joined to per-branch datasets.

Written by ``build/cbp2025_convert --features PREFIX``: ``PREFIX.bin`` holds one
packed record per dynamic instance of a selected branch -- ``branch_index``
(<i8), ``pc`` (<u8), then float32 features -- and ``PREFIX.json`` names the
columns and their groups. See ``tools/cbp2025_convert.cpp`` for the fetch-safety
rule every column obeys; the short version is that nothing here depends on the
branch's own execution or on any instruction that may still be in flight.

The join is the same exact equality on ``branch_index`` that
:class:`~src.dataset.aux_features.AuxFeatureSpec` uses for the TAGE log, and it
refuses a partial match for the same reason: a misaligned row still trains.

:class:`CombinedFeatureSpec` concatenates several specs column-wise, so a model
can see TAGE-SC-L's internal state and the raw features together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from ..evaluation.prediction_log import as_unsigned_pc
from .aux_features import AuxFeatureError

#: Column groups the converter emits (path, inst_mix, committed) and the
#: producer tool emits (producers, inst_hist; tools/cbp2025_producers.cpp).
RAW_FEATURE_GROUPS = ("path", "inst_mix", "committed", "producers", "inst_hist")

_SIDE_FORMAT = "cbp2025-fetch-safe-features"


def load_raw_features(prefix: str | Path) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, dict]:
    """``(branch_index, pc, features, sidecar)`` for one feature file."""
    prefix = Path(prefix)
    sidecar = json.loads(prefix.with_name(prefix.name + ".json").read_text())
    if sidecar.get("format") != _SIDE_FORMAT:
        raise AuxFeatureError(f"{prefix}.json is not a {_SIDE_FORMAT} sidecar")
    names = sidecar["feature_names"]
    dtype = np.dtype([("branch_index", "<i8"), ("pc", "<u8"),
                      ("features", "<f4", (len(names),))])
    if dtype.itemsize != sidecar["record_bytes"]:
        raise AuxFeatureError(
            f"{prefix}: record is {sidecar['record_bytes']} bytes, sidecar "
            f"columns imply {dtype.itemsize}")
    raw = np.fromfile(prefix.with_name(prefix.name + ".bin"), dtype=dtype)
    if raw.shape[0] != sidecar["records"]:
        raise AuxFeatureError(
            f"{prefix}.bin has {raw.shape[0]} records, sidecar says "
            f"{sidecar['records']} -- truncated write?")
    return raw["branch_index"], raw["pc"], raw["features"], sidecar


@dataclass
class RawFeatureSpec:
    """Where the raw feature files live and which groups to use.

    Args:
        feature_dir: Directory of ``<trace_id>.feat.bin`` / ``.feat.json``.
        groups: Subset of :data:`RAW_FEATURE_GROUPS`.
    """

    feature_dir: Path
    groups: tuple[str, ...] = RAW_FEATURE_GROUPS
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.feature_dir = Path(self.feature_dir)
        self.groups = tuple(self.groups)
        unknown = [g for g in self.groups if g not in RAW_FEATURE_GROUPS]
        if unknown:
            raise KeyError(f"{unknown} not raw feature groups; choose from "
                           f"{RAW_FEATURE_GROUPS}")

    reveals_tage_direction = False

    def _file(self, trace_id: str):
        if trace_id not in self._cache:
            prefix = self.feature_dir / f"{trace_id}.feat"
            if not prefix.with_name(prefix.name + ".json").exists():
                raise AuxFeatureError(
                    f"no raw feature file for {trace_id!r} at {prefix}.json; "
                    f"generate it with build/cbp2025_convert --features")
            self._cache[trace_id] = load_raw_features(prefix)
        return self._cache[trace_id]

    def load(self, dataset_path: Path, br_pc: int,
             indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
        with h5py.File(dataset_path, "r") as fh:
            trace_id = str(fh.attrs["trace_id"])
        branch_index, pc, features, sidecar = self._file(trace_id)

        keep_cols = [i for i, g in enumerate(sidecar["feature_groups"])
                     if g in self.groups]
        names = [sidecar["feature_names"][i] for i in keep_cols]

        mine = pc == as_unsigned_pc(br_pc)
        log_index = branch_index[mine]
        if log_index.size == 0:
            raise AuxFeatureError(
                f"raw features for {trace_id} have no rows for {br_pc:#x} -- "
                f"was it in --feature-pcs?")
        if np.any(np.diff(log_index) <= 0):
            raise AuxFeatureError(f"{trace_id}: feature branch_index not sorted")

        slot = np.searchsorted(log_index, indices)
        in_range = slot < log_index.size
        matched = np.zeros(indices.shape, dtype=bool)
        matched[in_range] = log_index[slot[in_range]] == indices[in_range]
        if not matched.all():
            raise AuxFeatureError(
                f"{int((~matched).sum())} of {indices.size} occurrences of "
                f"{br_pc:#x} in {trace_id} have no raw feature row (first: "
                f"branch index {int(indices[~matched][0])}). Refusing a partial "
                f"join.")
        rows = features[mine][slot][:, keep_cols]
        return np.ascontiguousarray(rows, dtype=np.float32), names

    def describe(self) -> dict:
        return {"raw_groups": list(self.groups),
                "raw_feature_dir": str(self.feature_dir)}


@dataclass
class CombinedFeatureSpec:
    """Column-wise concatenation of several feature specs.

    Order is the order of ``parts``; names are prefixed per part so a column
    is never ambiguous in saved metrics.
    """

    parts: list

    @property
    def groups(self) -> tuple[str, ...]:
        out: list[str] = []
        for p in self.parts:
            out += list(getattr(p, "groups", ()))
        return tuple(out)

    @property
    def reveals_tage_direction(self) -> bool:
        return any(getattr(p, "reveals_tage_direction", False)
                   for p in self.parts)

    def load(self, dataset_path: Path, br_pc: int,
             indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
        blocks, names = [], []
        for p in self.parts:
            block, part_names = p.load(dataset_path, br_pc, indices)
            prefix = "tage." if not isinstance(p, RawFeatureSpec) else "raw."
            blocks.append(block)
            names += [prefix + n for n in part_names]
        return np.concatenate(blocks, axis=1), names

    def describe(self) -> dict:
        out: dict = {"reveals_tage_direction": self.reveals_tage_direction}
        for p in self.parts:
            out.update(p.describe())
        return out
