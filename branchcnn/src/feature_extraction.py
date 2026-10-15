"""Causal feature extraction: CBP2025 trace -> CNN-ready arrays.

Independent of ``branch_extraction`` (Stage 3) and ``run_analysis`` (the analysis
pipeline); both stay untouched. Only the generic streaming state in
``register_state`` is reused.

Two groups of arrays come out of each trace.

**Branch history** -- one entry per *branch* instruction (conditional and not),
in execution order. The CNN slides a window over these. Stored unfolded so one
extraction serves every ``pc_bits`` setting.

**Samples** -- one entry per *conditional* branch, carrying a pointer into the
history plus per-occurrence scalars.

Causality
---------
Every field is computed from instructions that executed strictly before the
branch being labelled:

* the history window is sliced ``[pos - H, pos)``, so the branch's own slot is
  never an input;
* ``s_local`` holds earlier outcomes of the same PC, shifted *before* the current
  outcome is folded in;
* dependency edges are resolved when each instruction is seen, so a writer that
  executes after the branch can never be linked to it;
* nothing here is a whole-trace aggregate.

Deliberately excluded
---------------------
Predictor-derived features (TAGE state, 2-bit counter state, running taken rate)
are left for a later stage. Register *values* and load *addresses* are excluded
because they are not available at fetch time in real hardware, which makes them
an oracle study rather than a feature.

Opcode features
---------------
CBP2025 carries an ``InstClass`` for every instruction, which SBBT does not. That
is used three ways: the class of each history branch, the instruction mix of the
basic block before it, and the class composition of the branch's dependency
slice.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .logical_instructions import LogicalInstruction, iter_logical_instructions
from .register_state import InstructionWindow, ShadowRegisterFile, WindowEntry
from .trace_inspection import (
    ALU,
    BRANCH_CLASSES,
    CC_OFFSET,
    COND_BR,
    FP_OP,
    LOAD,
    SLOW_ALU,
    STORE,
    UNCOND_DIRECT,
    UNCOND_INDIRECT,
    VEC_OFFSET,
    ZERO_OFFSET,
)

__all__ = [
    "ExtractionConfig",
    "TraceFeatures",
    "FeatureExtractor",
    "extract_trace",
    "save_features",
    "load_features",
    "MIX_CLASSES",
    "SLICE_CLASSES",
    "LOCAL_HISTORY_BITS",
    "REG_KIND_BITS",
]

#: Instruction classes counted in the basic block preceding each branch.
MIX_CLASSES: Tuple[int, ...] = (ALU, LOAD, STORE, FP_OP, SLOW_ALU)

#: Instruction classes counted inside a branch's dependency slice.
SLICE_CLASSES: Tuple[int, ...] = (
    ALU,
    LOAD,
    STORE,
    COND_BR,
    UNCOND_DIRECT,
    UNCOND_INDIRECT,
    FP_OP,
    SLOW_ALU,
)

#: Width of the per-PC local outcome history, bit 0 = most recent.
LOCAL_HISTORY_BITS = 32

#: Bit positions in ``s_regkinds``.
REG_KIND_BITS = {"gpr": 1, "stack_pointer": 2, "simd": 4, "flags": 8, "zero": 16}

_NONE_U32 = int(np.iinfo(np.uint32).max)
_CLIP_U32 = _NONE_U32 - 1
_CLIP_U16 = int(np.iinfo(np.uint16).max)
_CLIP_U8 = int(np.iinfo(np.uint8).max)
_NO_CLASS = _CLIP_U8

_MIX_SLOT = {cls: slot for slot, cls in enumerate(MIX_CLASSES)}
_SLICE_SLOT = {cls: slot for slot, cls in enumerate(SLICE_CLASSES)}


def _reg_kind_bit(reg: int) -> int:
    if reg == CC_OFFSET:
        return REG_KIND_BITS["flags"]
    if reg == ZERO_OFFSET:
        return REG_KIND_BITS["zero"]
    if reg == 31:
        return REG_KIND_BITS["stack_pointer"]
    if reg < VEC_OFFSET:
        return REG_KIND_BITS["gpr"]
    return REG_KIND_BITS["simd"]


@dataclass(frozen=True)
class ExtractionConfig:
    """Knobs for one extraction run.

    ``max_depth``/``max_nodes`` bound the dependency slice. They are much tighter
    than the analysis pipeline's 32/512 because this runs on *every* conditional
    branch rather than a 1-in-20 sample. Both are recorded in the output so a
    later run cannot be compared against incomparable settings.
    """

    max_depth: int = 8
    max_nodes: int = 64
    window_capacity: int = 1 << 16
    #: Skip this many leading instructions before emitting samples.
    warmup_instructions: int = 0
    #: Emit every Nth conditional branch as a sample. History stays complete, so
    #: windows are unaffected; this only thins the training set.
    sample_stride: int = 1
    instruction_limit: Optional[int] = None


@dataclass
class TraceFeatures:
    """Extraction output for one trace."""

    workload_id: str
    trace_id: str
    config: ExtractionConfig
    instructions_seen: int = 0

    # --- branch history, one entry per branch instruction ---
    h_pc: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint32))
    h_dir: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    h_class: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    h_back: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    h_ninstr: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint16))
    h_mix: np.ndarray = field(
        default_factory=lambda: np.empty((0, len(MIX_CLASSES)), np.uint8)
    )

    # --- samples, one entry per conditional branch ---
    s_pos: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    s_pc: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint64))
    s_out: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    s_local: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint32))
    s_nseen: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint32))
    s_dep_depth: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    s_dep_ninstr: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint16))
    s_dep_npcs: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint16))
    s_nload: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint16))
    s_near_load: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint32))
    s_oldest: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint32))
    s_flags: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    s_fwclass: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    s_nsrc: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    s_near_writer: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint32))
    s_dwclass: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint16))
    s_regkinds: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    s_slice_mix: np.ndarray = field(
        default_factory=lambda: np.empty((0, len(SLICE_CLASSES)), np.uint8)
    )
    s_incomplete: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))

    @property
    def num_branches(self) -> int:
        return int(self.h_pc.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.s_pos.shape[0])

    @property
    def taken_rate(self) -> float:
        return float(self.s_out.mean()) if self.num_samples else 0.0


#: numpy dtype -> ``array`` typecode of the same width.
_TYPECODE = {
    np.uint8: "B",
    np.uint16: "H",
    np.uint32: "I",
    np.uint64: "Q",
    np.int64: "q",
}


class _Accum:
    """Growable packed column.

    Backed by ``array.array`` rather than a list: at 15.7M samples across ~19
    columns, boxed Python ints cost over a gigabyte of object overhead, while a
    packed buffer costs only the bytes themselves.
    """

    __slots__ = ("values", "dtype", "clip", "width")

    def __init__(self, dtype, clip: Optional[int] = None, width: int = 1) -> None:
        self.dtype = dtype
        self.clip = clip
        self.width = width
        self.values = array(_TYPECODE[dtype])

    def add(self, value: int) -> None:
        if self.clip is not None and value > self.clip:
            value = self.clip
        self.values.append(value)

    def extend(self, values) -> None:
        """Append one row of a multi-column accumulator."""
        clip = self.clip
        if clip is None:
            self.values.extend(values)
        else:
            self.values.extend(v if v <= clip else clip for v in values)

    def finish(self) -> np.ndarray:
        # Copied so the result owns its memory and stays valid and writable once
        # the accumulator is released.
        out = np.frombuffer(self.values, dtype=self.dtype).copy()
        if self.width > 1:
            return out.reshape(-1, self.width)
        return out


class FeatureExtractor:
    """Streams one trace and accumulates history and sample arrays.

    Ordering inside :meth:`observe` is the part that must not be reordered: an
    instruction's own producers are resolved, then a conditional branch is
    analysed, and only then does the register file record the instruction's
    writes. That is what makes "a later writer is never a producer" structural
    rather than something we check for afterwards.
    """

    def __init__(self, config: ExtractionConfig = ExtractionConfig()) -> None:
        self.config = config
        self.registers = ShadowRegisterFile()
        self.window = InstructionWindow(config.window_capacity)

        self._h_pc = _Accum(np.uint32)
        self._h_dir = _Accum(np.uint8)
        self._h_class = _Accum(np.uint8)
        self._h_back = _Accum(np.uint8)
        self._h_ninstr = _Accum(np.uint16, _CLIP_U16)
        self._h_mix = _Accum(np.uint8, _CLIP_U8, width=len(MIX_CLASSES))

        self._s_pos = _Accum(np.int64)
        self._s_pc = _Accum(np.uint64)
        self._s_out = _Accum(np.uint8)
        self._s_local = _Accum(np.uint32)
        self._s_nseen = _Accum(np.uint32, _CLIP_U32)
        self._s_dep_depth = _Accum(np.uint8, _CLIP_U8)
        self._s_dep_ninstr = _Accum(np.uint16, _CLIP_U16)
        self._s_dep_npcs = _Accum(np.uint16, _CLIP_U16)
        self._s_nload = _Accum(np.uint16, _CLIP_U16)
        self._s_near_load = _Accum(np.uint32)
        self._s_oldest = _Accum(np.uint32)
        self._s_flags = _Accum(np.uint8)
        self._s_fwclass = _Accum(np.uint8)
        self._s_nsrc = _Accum(np.uint8, _CLIP_U8)
        self._s_near_writer = _Accum(np.uint32)
        self._s_dwclass = _Accum(np.uint16)
        self._s_regkinds = _Accum(np.uint8)
        self._s_slice_mix = _Accum(np.uint8, _CLIP_U8, width=len(SLICE_CLASSES))
        self._s_incomplete = _Accum(np.uint8)

        self._branches = 0
        self._cond_branches = 0
        self._instructions = 0
        self._block_mix = [0] * len(MIX_CLASSES)
        self._block_instructions = 0
        #: pc -> [packed local history, occurrences seen]
        self._local: Dict[int, List[int]] = {}

    # -- streaming ---------------------------------------------------------

    def observe(self, instr: LogicalInstruction) -> None:
        producers, _, _ = self.registers.resolve_sources(instr.in_regs)
        self._instructions = instr.index + 1

        if instr.inst_class in BRANCH_CLASSES:
            if instr.inst_class == COND_BR:
                if (
                    instr.index >= self.config.warmup_instructions
                    and self._cond_branches % self.config.sample_stride == 0
                ):
                    self._emit_sample(instr, producers)
                self._cond_branches += 1
            self._append_history(instr)

        self.window.add(
            WindowEntry(
                index=instr.index,
                pc=instr.pc,
                inst_class=instr.inst_class,
                is_load=instr.inst_class == LOAD,
                mem_address=instr.eff_addr,
                in_regs=instr.in_regs,
                src_writers=producers,
            )
        )
        self.registers.apply(instr)

        slot = _MIX_SLOT.get(instr.inst_class)
        if slot is not None:
            self._block_mix[slot] += 1
        self._block_instructions += 1

    def _append_history(self, instr: LogicalInstruction) -> None:
        """Record a branch slot.

        Called after the sample is emitted, so a sample's window ends strictly
        before its own branch's slot.
        """
        self._h_pc.add(int((instr.pc >> 2) & 0xFFFFFFFF))
        self._h_dir.add(1 if instr.taken else 0)
        self._h_class.add(instr.inst_class)
        backward = (
            instr.taken
            and instr.taken_target is not None
            and instr.taken_target < instr.pc
        )
        self._h_back.add(1 if backward else 0)
        self._h_ninstr.add(self._block_instructions)
        self._h_mix.extend(self._block_mix)

        self._block_mix = [0] * len(MIX_CLASSES)
        self._block_instructions = 0
        self._branches += 1

    # -- per-sample features ----------------------------------------------

    def _emit_sample(
        self, branch: LogicalInstruction, producers: Tuple[Optional[int], ...]
    ) -> None:
        pc = branch.pc
        # Position in the history arrays that this branch *will* occupy. The
        # model window is [pos - H, pos), so the branch's own slot is excluded.
        self._s_pos.add(self._branches)
        self._s_pc.add(pc)
        outcome = 1 if branch.taken else 0
        self._s_out.add(outcome)

        state = self._local.get(pc)
        if state is None:
            state = self._local[pc] = [0, 0]
        # Appended before the current outcome is folded in: strictly past.
        self._s_local.add(state[0])
        self._s_nseen.add(state[1])
        state[0] = ((state[0] << 1) | outcome) & ((1 << LOCAL_HISTORY_BITS) - 1)
        state[1] += 1

        regkinds = 0
        for reg in branch.in_regs:
            regkinds |= _reg_kind_bit(reg)
        self._s_regkinds.add(regkinds)
        self._s_nsrc.add(len(branch.in_regs))
        reads_flags = CC_OFFSET in branch.in_regs
        self._s_flags.add(1 if reads_flags else 0)

        flag_writer = self.registers.writer_of(CC_OFFSET)
        if reads_flags and flag_writer is not None:
            self._s_fwclass.add(flag_writer.inst_class)
        else:
            self._s_fwclass.add(_NO_CLASS)

        known = [p for p in producers if p is not None]
        self._s_near_writer.add(
            min(branch.index - max(known), _CLIP_U32) if known else _NONE_U32
        )
        dwclass = 0
        for index in known:
            entry = self.window.get(index)
            if entry is not None:
                dwclass |= 1 << entry.inst_class
        self._s_dwclass.add(dwclass)

        self._slice_features(branch, producers)

    def _slice_features(
        self, branch: LogicalInstruction, producers: Tuple[Optional[int], ...]
    ) -> None:
        """Bounded backward walk over the dynamic dataflow graph.

        A separate implementation from ``branch_dependencies._walk`` so that
        module stays untouched; this one also tallies instruction classes, which
        is the opcode signal the analysis pipeline does not record.
        """
        limits = self.config
        visited: set = set()
        static_pcs: set = set()
        class_counts = [0] * len(SLICE_CLASSES)
        loads: List[int] = []
        incomplete = 0

        frontier = [index for index in producers if index is not None]
        depth = 0
        exhausted = False

        while frontier and not exhausted:
            if depth >= limits.max_depth:
                incomplete = 1
                break
            depth += 1
            next_frontier: List[int] = []
            for index in frontier:
                if index in visited:
                    continue
                if len(visited) >= limits.max_nodes:
                    incomplete = 1
                    exhausted = True
                    break
                entry = self.window.get(index)
                if entry is None:
                    incomplete = 1
                    continue
                visited.add(index)
                static_pcs.add(entry.pc)
                slot = _SLICE_SLOT.get(entry.inst_class)
                if slot is not None:
                    class_counts[slot] += 1
                if entry.is_load:
                    loads.append(index)
                for parent in entry.src_writers:
                    # A producer index is always strictly older; guard anyway.
                    if parent is None or parent >= index or parent in visited:
                        continue
                    next_frontier.append(parent)
            frontier = next_frontier

        self._s_dep_depth.add(depth)
        self._s_dep_ninstr.add(len(visited))
        self._s_dep_npcs.add(len(static_pcs))
        self._s_nload.add(len(loads))
        self._s_near_load.add(
            min(branch.index - max(loads), _CLIP_U32) if loads else _NONE_U32
        )
        self._s_oldest.add(
            min(branch.index - min(visited), _CLIP_U32) if visited else _NONE_U32
        )
        self._s_slice_mix.extend(class_counts)
        self._s_incomplete.add(incomplete)

    # -- finalise ----------------------------------------------------------

    def result(self, workload_id: str, trace_id: str) -> TraceFeatures:
        mix = self._h_mix.finish()
        slice_mix = self._s_slice_mix.finish()
        return TraceFeatures(
            workload_id=workload_id,
            trace_id=trace_id,
            config=self.config,
            instructions_seen=self._instructions,
            h_pc=self._h_pc.finish(),
            h_dir=self._h_dir.finish(),
            h_class=self._h_class.finish(),
            h_back=self._h_back.finish(),
            h_ninstr=self._h_ninstr.finish(),
            h_mix=mix,
            s_pos=self._s_pos.finish(),
            s_pc=self._s_pc.finish(),
            s_out=self._s_out.finish(),
            s_local=self._s_local.finish(),
            s_nseen=self._s_nseen.finish(),
            s_dep_depth=self._s_dep_depth.finish(),
            s_dep_ninstr=self._s_dep_ninstr.finish(),
            s_dep_npcs=self._s_dep_npcs.finish(),
            s_nload=self._s_nload.finish(),
            s_near_load=self._s_near_load.finish(),
            s_oldest=self._s_oldest.finish(),
            s_flags=self._s_flags.finish(),
            s_fwclass=self._s_fwclass.finish(),
            s_nsrc=self._s_nsrc.finish(),
            s_near_writer=self._s_near_writer.finish(),
            s_dwclass=self._s_dwclass.finish(),
            s_regkinds=self._s_regkinds.finish(),
            s_slice_mix=slice_mix,
            s_incomplete=self._s_incomplete.finish(),
        )


def extract_trace(
    path: Path,
    workload_id: str,
    trace_id: str,
    config: ExtractionConfig = ExtractionConfig(),
    progress_every: int = 0,
) -> TraceFeatures:
    """Stream a trace from disk and return its feature arrays."""
    extractor = FeatureExtractor(config)
    for instr in iter_logical_instructions(path, limit=config.instruction_limit):
        extractor.observe(instr)
        if progress_every and instr.index and instr.index % progress_every == 0:
            print(
                f"  {trace_id}: {instr.index:,} instructions, "
                f"{extractor._cond_branches:,} cond branches",
                flush=True,
            )
    return extractor.result(workload_id, trace_id)


_ARRAY_FIELDS = (
    "h_pc",
    "h_dir",
    "h_class",
    "h_back",
    "h_ninstr",
    "h_mix",
    "s_pos",
    "s_pc",
    "s_out",
    "s_local",
    "s_nseen",
    "s_dep_depth",
    "s_dep_ninstr",
    "s_dep_npcs",
    "s_nload",
    "s_near_load",
    "s_oldest",
    "s_flags",
    "s_fwclass",
    "s_nsrc",
    "s_near_writer",
    "s_dwclass",
    "s_regkinds",
    "s_slice_mix",
    "s_incomplete",
)


def save_features(features: TraceFeatures, path: Path) -> None:
    """Write one trace's arrays atomically, so a reader never sees a part file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(features, name) for name in _ARRAY_FIELDS}
    payload["meta_workload_id"] = np.array(features.workload_id)
    payload["meta_trace_id"] = np.array(features.trace_id)
    payload["meta_instructions"] = np.array(features.instructions_seen, np.int64)
    payload["meta_max_depth"] = np.array(features.config.max_depth, np.int64)
    payload["meta_max_nodes"] = np.array(features.config.max_nodes, np.int64)
    payload["meta_sample_stride"] = np.array(features.config.sample_stride, np.int64)
    payload["meta_window_capacity"] = np.array(
        features.config.window_capacity, np.int64
    )
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(path)


def load_features(path: Path) -> TraceFeatures:
    """Read back a file written by :func:`save_features`."""
    with np.load(path, allow_pickle=False) as data:
        config = ExtractionConfig(
            max_depth=int(data["meta_max_depth"]),
            max_nodes=int(data["meta_max_nodes"]),
            window_capacity=int(data["meta_window_capacity"]),
            sample_stride=int(data["meta_sample_stride"]),
        )
        features = TraceFeatures(
            workload_id=str(data["meta_workload_id"]),
            trace_id=str(data["meta_trace_id"]),
            config=config,
            instructions_seen=int(data["meta_instructions"]),
        )
        for name in _ARRAY_FIELDS:
            setattr(features, name, data[name])
    return features
