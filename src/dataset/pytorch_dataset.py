"""PyTorch datasets over the per-branch HDF5 files.

One :class:`BranchDataset` serves one static branch, reading from one or more
trace files that share a split role. A sample is::

    inputs = history[i-H : i]        # H tokens, target EXCLUDED
    label  = history[i] & 1          # the target's direction

which is BranchNet's contract (``dataset_loader.py``: ``inputs = chunk[:-1]``,
``label = chunk[-1] & 1``), verified against their source.

Windows are sliced on demand rather than materialised -- at H=582 a
materialised set would be ~600x larger for no extra information.

``pc_bits`` / ``pc_hash_bits`` are applied here, not at build time, so a single
built dataset serves every representation ablation (spec §19).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def preprocess_history(
    history: np.ndarray,
    pc_bits: int,
    pc_hash_bits: int,
    hash_dir_with_pc: bool,
    dtype,
) -> np.ndarray:
    """Mask tokens to ``pc_bits``, optionally folding the PC to ``pc_hash_bits``.

    Port of BranchNet's ``preprocess_history``. The fold is an XOR of successive
    ``pc_hash_bits``-wide slices, which is what lets a Mini-BranchNet index a
    small table without a full embedding over 2^(pc_bits+1) tokens.

    Operates on a copy: the input may be a shared in-memory array.
    """
    if pc_hash_bits > pc_bits:
        raise ValueError(f"pc_hash_bits ({pc_hash_bits}) > pc_bits ({pc_bits})")

    history = np.array(history, dtype=np.int64, copy=True)
    np.bitwise_and(history, (1 << (1 + pc_bits)) - 1, out=history)

    if hash_dir_with_pc:
        if pc_hash_bits < pc_bits + 1:
            unprocessed = pc_bits + 1 - pc_hash_bits
            mask = (1 << pc_hash_bits) - 1
            shift = 1
            temp = np.empty_like(history)
            while unprocessed > 0:
                np.right_shift(history, shift * pc_hash_bits, out=temp)
                np.bitwise_and(temp, mask, out=temp)
                np.bitwise_xor(history, temp, out=history)
                shift += 1
                unprocessed -= pc_hash_bits
            np.bitwise_and(history, mask, out=history)
    else:
        if pc_hash_bits < pc_bits:
            unprocessed = pc_bits - pc_hash_bits
            mask = ((1 << pc_hash_bits) - 1) << 1  # leave the direction bit alone
            shift = 1
            temp = np.empty_like(history)
            while unprocessed > 0:
                np.right_shift(history, shift * pc_hash_bits, out=temp)
                np.bitwise_and(temp, mask, out=temp)
                np.bitwise_xor(history, temp, out=history)
                shift += 1
                unprocessed -= pc_hash_bits
            np.bitwise_and(history, (1 << (pc_hash_bits + 1)) - 1, out=history)

    return history.astype(dtype)


def vocabulary_size(pc_bits: int, pc_hash_bits: int, hash_dir_with_pc: bool) -> int:
    """Number of distinct token ids the embedding must cover."""
    if hash_dir_with_pc:
        return 1 << min(pc_hash_bits, pc_bits + 1)
    return 1 << (min(pc_hash_bits, pc_bits) + 1)


def remove_incomplete_histories(indices: np.ndarray, history_length: int) -> np.ndarray:
    """Drop occurrences too early in the trace to have a full window.

    Without this, the first few occurrences would be padded with whatever
    precedes them -- or indexed out of bounds.
    """
    if indices.size == 0:
        return indices
    return indices[indices >= history_length]


#: Preprocessed histories, keyed by (path, encoding). Decompressing and folding
#: a trace costs seconds; leave-one-out over 16 members would otherwise redo it
#: 256 times. Process-local, and backed by an on-disk .npy cache.
_HISTORY_CACHE: dict[tuple, np.ndarray] = {}


def _cache_path(h5_path: Path, stored_pc_bits: int, pc_hash_bits: int,
                hash_dir_with_pc: bool, dtype) -> Path:
    key = (f"{stored_pc_bits}_{pc_hash_bits}_{int(hash_dir_with_pc)}"
           f"_{np.dtype(dtype).name}")
    return h5_path.with_name(h5_path.name + f".hist_{key}.npy")


def load_history(h5_path: Path, fh, stored_pc_bits: int, pc_hash_bits: int,
                 hash_dir_with_pc: bool, dtype, *, use_disk_cache: bool = True):
    """Preprocessed history for a trace, from cache where possible.

    The result depends only on the trace and the token encoding -- not on which
    branch or which fold -- so one cache entry serves every model and every
    leave-one-out split.
    """
    key = (str(h5_path), stored_pc_bits, pc_hash_bits, bool(hash_dir_with_pc),
           np.dtype(dtype).name)
    cached = _HISTORY_CACHE.get(key)
    if cached is not None:
        return cached

    disk = _cache_path(h5_path, stored_pc_bits, pc_hash_bits,
                       hash_dir_with_pc, dtype)
    if use_disk_cache and disk.exists():
        history = np.load(disk)
    else:
        history = preprocess_history(fh["history"][:], stored_pc_bits,
                                     pc_hash_bits, hash_dir_with_pc, dtype)
        if use_disk_cache:
            # Write via a handle: np.save(path, ...) appends ".npy" when the
            # name does not already end in it, which silently moves the file.
            tmp = disk.with_name(disk.name + ".part")
            with open(tmp, "wb") as out:
                np.save(out, history)
            tmp.replace(disk)

    _HISTORY_CACHE[key] = history
    return history


def clear_history_cache() -> None:
    """Drop the in-process cache (the on-disk one is left alone)."""
    _HISTORY_CACHE.clear()


@dataclass
class _Source:
    path: Path
    indices: np.ndarray
    history: np.ndarray | None  # in-memory, already preprocessed
    num_taken: int
    num_not_taken: int
    #: Per-instance aux features, row i aligned to ``indices[i]``. None unless
    #: the dataset was built with ``aux_features=``.
    features: np.ndarray | None = None
    #: Per-instance loss weights, aligned like ``features``. Training only.
    weights: np.ndarray | None = None


class BranchDataset(Dataset):
    """Samples for one static branch, pooled over traces of one split role.

    Args:
        paths: HDF5 files built by :mod:`src.dataset.builder`.
        br_pc: The static branch to serve.
        history_length: ``H``. Window is ``history[i-H : i]``.
        pc_bits: PC bits used by the model (<= the stored ``pc_bits``).
        pc_hash_bits: Fold width; equal to ``pc_bits`` means no folding.
        hash_dir_with_pc: Fold the direction bit into the hash too.
        in_memory: Preprocess and hold each trace's history in RAM. Much faster;
            costs ~2 bytes per branch per trace at typical settings.
        expected_split_role: If given, every file must carry this role. This is
            the guard that stops a test trace being used for training.
    """

    def __init__(
        self,
        paths,
        *,
        br_pc: int,
        history_length: int,
        pc_bits: int = 12,
        pc_hash_bits: int | None = None,
        hash_dir_with_pc: bool = False,
        in_memory: bool = True,
        expected_split_role: str | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        cache_histories: bool = True,
        time_slice: tuple[str, float] | None = None,
        aux_features: "AuxFeatureSpec | None" = None,
        index_range: tuple[int, int] | None = None,
        sample_weights=None,
        allow_missing: bool = False,
    ):
        self.br_pc = int(br_pc)
        self.history_length = int(history_length)
        self.pc_bits = int(pc_bits)
        self.pc_hash_bits = int(pc_hash_bits if pc_hash_bits is not None else pc_bits)
        self.hash_dir_with_pc = bool(hash_dir_with_pc)
        self.in_memory = bool(in_memory)
        self.max_samples = max_samples
        self.seed = int(seed)
        self.vocabulary_size = vocabulary_size(
            self.pc_bits, self.pc_hash_bits, self.hash_dir_with_pc)

        if max_samples is not None and expected_split_role == "test":
            raise ValueError(
                "refusing to subsample a test set: hybrid MPKI requires a "
                "prediction for every dynamic instance, so dropping any makes "
                "the reported MPKI fiction. Cap training data only."
            )

        if time_slice is not None:
            side, fraction = time_slice
            if side not in ("head", "tail"):
                raise ValueError(f"time_slice side must be head/tail, got {side!r}")
            if not 0.0 < fraction < 1.0:
                raise ValueError(f"time_slice fraction must be in (0,1), got {fraction}")
        self.time_slice = time_slice
        if index_range is not None:
            lo, hi = (int(index_range[0]), int(index_range[1]))
            if not 0 <= lo < hi:
                raise ValueError(f"index_range must satisfy 0 <= lo < hi, got "
                                 f"{index_range}")
            index_range = (lo, hi)
        self.index_range = index_range
        self.aux_features = aux_features
        self.aux_feature_names: list[str] = []

        self._dtype = np.uint16 if self.vocabulary_size <= (1 << 16) else np.int64
        self.sources: list[_Source] = []
        self.missing_in: list[str] = []
        self.split_roles: set[str] = set()
        self.total_available = 0

        for path in [Path(p) for p in paths]:
            with h5py.File(path, "r") as fh:
                role = fh.attrs.get("split_role", "unknown")
                if expected_split_role is not None and role != expected_split_role:
                    raise ValueError(
                        f"{path.name} has split_role={role!r} but this dataset "
                        f"expects {expected_split_role!r} -- refusing to mix splits"
                    )
                self.split_roles.add(str(role))
                if role == "chronological" and index_range is None:
                    raise ValueError(
                        f"{path.name} holds train, validation AND test as "
                        f"time ranges of one trace; pass index_range= so a "
                        f"dataset can never see the whole trace")

                stored_pc_bits = int(fh.attrs["pc_bits"])
                if self.pc_bits > stored_pc_bits:
                    raise ValueError(
                        f"{path.name} stores {stored_pc_bits} PC bits; cannot "
                        f"serve pc_bits={self.pc_bits}"
                    )

                key = f"br_indices_{hex(self.br_pc)}"
                if key not in fh and allow_missing:
                    # Cross-trace pools: a branch may simply not execute in
                    # some inputs of the program. That trace contributes no
                    # samples; it is not an error.
                    self.missing_in.append(path.name)
                    continue
                if key not in fh:
                    raise KeyError(
                        f"{path.name} has no index for branch {self.br_pc:#x}; "
                        f"it was not among the build's target PCs"
                    )
                indices = remove_incomplete_histories(fh[key][:], self.history_length)
                if index_range is not None:
                    # Positions are absolute branch indices, so one range
                    # applies identically to every static branch.
                    indices = indices[(indices >= index_range[0])
                                      & (indices < index_range[1])]
                if time_slice is not None:
                    indices = self._apply_time_slice(indices)
                if indices.size == 0:
                    continue

                history = None
                if self.in_memory:
                    history = load_history(
                        path, fh, stored_pc_bits, self.pc_hash_bits,
                        self.hash_dir_with_pc, self._dtype,
                        use_disk_cache=cache_histories)

                features = None
                if aux_features is not None:
                    features, names = aux_features.load(
                        path, self.br_pc, indices)
                    if self.aux_feature_names and names != self.aux_feature_names:
                        raise ValueError(
                            "aux feature columns differ between traces: "
                            f"{self.aux_feature_names} vs {names}")
                    self.aux_feature_names = names

                weights = None
                if sample_weights is not None:
                    # Same exact branch_index join as the features.
                    weights = np.asarray(
                        sample_weights.load(path, self.br_pc, indices),
                        dtype=np.float32)
                    if weights.shape != indices.shape:
                        raise ValueError(
                            f"sample weights shape {weights.shape} != "
                            f"occurrences {indices.shape}")

                self.sources.append(_Source(
                    path=path,
                    indices=indices,
                    weights=weights,
                    features=features,
                    history=history,
                    num_taken=int(fh.attrs.get(f"num_taken_{hex(self.br_pc)}", 0)),
                    num_not_taken=int(
                        fh.attrs.get(f"num_not_taken_{hex(self.br_pc)}", 0)),
                ))
                self._stored_pc_bits = stored_pc_bits

        self.has_sample_weights = sample_weights is not None
        self.total_available = sum(s.indices.size for s in self.sources)
        if max_samples is not None and self.total_available > max_samples:
            self._subsample(max_samples)

        self._offsets = np.cumsum([0] + [s.indices.size for s in self.sources])
        self._handles: dict[Path, h5py.File] = {}

    def _apply_time_slice(self, indices: np.ndarray) -> np.ndarray:
        """Take a chronologically contiguous head or tail of the occurrences.

        Used to carve an *in-distribution* validation set out of the training
        traces. The separate-trace validation we started with turned out to be
        unusable on mcf: trace 994B is 99.13% taken on 0x406337, so no epoch
        could beat its bias floor and checkpoint selection was effectively
        random.

        A gap of ``history_length`` occurrences is dropped between the two
        parts, so no validation sample's window can overlap a training
        sample's window. Contiguous rather than random: random would make the
        two halves interleave in time, which is the leakage spec §7.1 forbids.
        """
        side, fraction = self.time_slice
        gap = self.history_length
        cut = int(round(indices.size * (1.0 - fraction)))
        if side == "head":
            return indices[:max(0, cut - gap)]
        return indices[cut:]

    def _subsample(self, max_samples: int) -> None:
        """Thin the training set to at most ``max_samples``, deterministically.

        This is **not** a train/test split -- it only reduces how many of the
        training occurrences are visited. Each surviving sample still carries
        its true preceding history, so no temporal structure is broken. The
        thing spec §7.1 forbids is randomly splitting windows *across* splits,
        which happens at trace level and is untouched here.

        Occurrences are drawn without replacement across all source traces in
        proportion to their size, then re-sorted so HDF5 reads stay sequential.
        """
        rng = np.random.default_rng(self.seed)
        keep = max_samples / self.total_available
        for source in self.sources:
            take = max(1, int(round(source.indices.size * keep)))
            if take >= source.indices.size:
                continue
            chosen = np.sort(
                rng.choice(source.indices.size, size=take, replace=False))
            source.indices = source.indices[chosen]
            if source.weights is not None:
                source.weights = source.weights[chosen]
            if source.features is not None:
                # Must be reindexed identically or every feature row would be
                # attached to the wrong dynamic instance -- a silent corruption
                # that still trains and still reports a plausible number.
                source.features = source.features[chosen]

    def __len__(self) -> int:
        return int(self._offsets[-1]) if self.sources else 0

    @property
    def num_aux_features(self) -> int:
        return len(self.aux_feature_names)

    def aux_feature_mask(self, *, min_std: float = 1e-6) -> np.ndarray:
        """Which aux features actually vary in *this* dataset.

        We train one model per static branch, so a feature that is constant
        across this branch's instances has zero variance and therefore zero
        gradient -- it can only add parameters and dilute the head. Measured
        on the training split and passed to the model, never on test.
        """
        if self.aux_features is None:
            raise RuntimeError("dataset has no aux features")
        stacked = np.concatenate(
            [s.features for s in self.sources if s.features is not None])
        return stacked.std(axis=0) > min_std

    @property
    def pos_weight(self) -> float:
        """``num_not_taken / num_taken``, for ``BCEWithLogitsLoss``.

        Computed over the *full* occurrence set, not the subsample -- the
        subsample is uniform, so this is the same ratio with less noise.

        Measure imbalance before using this (spec §16): compare unweighted BCE
        first, and only weight if the imbalance is severe.
        """
        taken = sum(s.num_taken for s in self.sources)
        not_taken = sum(s.num_not_taken for s in self.sources)
        return (not_taken / taken) if taken else 1.0

    @property
    def taken_rate(self) -> float:
        taken = sum(s.num_taken for s in self.sources)
        total = taken + sum(s.num_not_taken for s in self.sources)
        return (taken / total) if total else 0.0

    def _window(self, source: _Source, index: int) -> np.ndarray:
        """``history[index-H : index+1]`` -- H inputs plus the target."""
        lo, hi = index - self.history_length, index + 1
        if source.history is not None:
            return source.history[lo:hi]
        handle = self._handles.get(source.path)
        if handle is None:
            handle = self._handles[source.path] = h5py.File(source.path, "r")
        return preprocess_history(
            handle["history"][lo:hi], self._stored_pc_bits, self.pc_hash_bits,
            self.hash_dir_with_pc, self._dtype)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        source_idx = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        source = self.sources[source_idx]
        position = int(source.indices[idx - self._offsets[source_idx]])

        chunk = self._window(source, position)
        inputs = chunk[:-1]          # the target itself is never an input
        label = float(chunk[-1] & 1)  # the target's direction
        return (torch.from_numpy(np.ascontiguousarray(inputs)).long(),
                torch.tensor(label, dtype=torch.float32))

    def shuffle_labels(self, seed: int = 0) -> None:
        """Randomly permute the labels. **Leakage control, never for real runs.**

        Inputs are untouched, so any genuine input->label relationship is
        destroyed. A clean pipeline can then do no better than the majority
        class; anything above the bias floor means information is reaching the
        model through a path other than the history window.
        """
        rng = np.random.default_rng(seed)
        self._label_permutation = []
        for source in self.sources:
            labels = (source.history[source.indices] & 1).astype(np.float32)
            self._label_permutation.append(rng.permutation(labels))
        self.labels_shuffled = True

    def iter_batches(self, batch_size: int, *, shuffle: bool = False,
                     seed: int = 0, drop_last: bool = False,
                     with_weights: bool = False):
        """Yield ``(inputs, labels)`` batches, gathering windows vectorised.

        The per-sample ``__getitem__`` path costs a Python call per sample,
        which dominates at millions of samples per epoch. Here a whole batch of
        windows is gathered with one fancy-index on the in-memory history
        (``positions[:, None] + arange(-H, 0)``), which is ~20x faster and is
        what makes an epoch practical.

        Requires ``in_memory=True``; falls back to a DataLoader otherwise.
        """
        if any(s.history is None for s in self.sources):
            raise RuntimeError("iter_batches requires in_memory=True")

        offsets = np.arange(-self.history_length, 0)
        rng = np.random.default_rng(seed)

        # (source_index, position) pairs, shuffled together so batches mix traces.
        pairs = np.concatenate([
            np.stack([np.full(s.indices.size, i, dtype=np.int64), s.indices])
            for i, s in enumerate(self.sources)
        ], axis=1)
        order = rng.permutation(pairs.shape[1]) if shuffle else \
            np.arange(pairs.shape[1])

        for start in range(0, order.size, batch_size):
            chunk = order[start:start + batch_size]
            if drop_last and chunk.size < batch_size:
                break
            source_ids, positions = pairs[0, chunk], pairs[1, chunk]

            inputs = np.empty((chunk.size, self.history_length), dtype=np.int64)
            labels = np.empty(chunk.size, dtype=np.float32)
            weights = (np.empty(chunk.size, dtype=np.float32)
                       if with_weights else None)
            features = (None if self.aux_features is None else
                        np.empty((chunk.size, self.num_aux_features),
                                 dtype=np.float32))
            for source_id in np.unique(source_ids):
                mask = source_ids == source_id
                pos = positions[mask]
                source = self.sources[source_id]
                history = source.history
                inputs[mask] = history[pos[:, None] + offsets]
                if features is not None or weights is not None:
                    # `pos` is a subset of source.indices; recover each row's
                    # slot so features follow their own instance exactly.
                    slot = np.searchsorted(source.indices, pos)
                    if features is not None:
                        features[mask] = source.features[slot]
                    if weights is not None:
                        if source.weights is None:
                            raise RuntimeError("with_weights=True but the "
                                               "dataset has no sample_weights")
                        weights[mask] = source.weights[slot]
                if getattr(self, "labels_shuffled", False):
                    # Positions are a subset of source.indices; map back to
                    # the permuted label for each.
                    where = np.searchsorted(self.sources[source_id].indices, pos)
                    labels[mask] = self._label_permutation[source_id][where]
                else:
                    labels[mask] = (history[pos] & 1).astype(np.float32)

            out = [torch.from_numpy(inputs)]
            if features is not None:
                out.append(torch.from_numpy(features))
            out.append(torch.from_numpy(labels))
            if weights is not None:
                out.append(torch.from_numpy(weights))   # after the labels
            yield tuple(out)

    def describe(self) -> dict:
        return {
            "br_pc": hex(self.br_pc),
            "samples": len(self),
            "available": self.total_available,
            "subsampled": len(self) < self.total_available,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "time_slice": self.time_slice,
            "index_range": self.index_range,
            "traces": [s.path.name for s in self.sources],
            "split_roles": sorted(self.split_roles),
            "history_length": self.history_length,
            "pc_bits": self.pc_bits,
            "pc_hash_bits": self.pc_hash_bits,
            "vocabulary_size": self.vocabulary_size,
            "taken_rate": self.taken_rate,
            "pos_weight": self.pos_weight,
        }

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
