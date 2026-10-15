"""Part 2: shadow architectural register state and a bounded instruction window.

Two structures are maintained while streaming a trace chronologically.

``ShadowRegisterFile`` holds, per architectural register, the latest known value
and metadata about its most recent writer. It is tiny (at most 66 registers) so
it is never evicted.

``InstructionWindow`` holds the most recent N logical instructions so that the
backward dependency slice can walk real dataflow edges.

Why edges are recorded at execution time
----------------------------------------
A naive slice would ask the register file "who wrote r3?" while walking
backwards, but by then r3 may have been overwritten many times. So as each
instruction is processed we resolve its *own* source producers immediately and
store those instruction indices on the window entry. The slice then walks a
fixed dynamic dataflow graph instead of re-querying mutable state, which is what
makes "a writer after the branch is never linked to that branch" true by
construction.

Register id groups are taken from ``enum Offset`` in trace_reader.h: 0-30 GPR,
31 stack pointer, 32-63 SIMD, 64 flags, 65 zero.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .logical_instructions import LogicalInstruction
from .trace_inspection import CC_OFFSET, ZERO_OFFSET

__all__ = [
    "RegisterWriter",
    "ShadowRegisterFile",
    "InstructionWindow",
    "WindowEntry",
    "UNKNOWN_NEVER_WRITTEN",
    "UNKNOWN_ZERO_REGISTER",
]

UNKNOWN_NEVER_WRITTEN = "never_written_in_trace"
UNKNOWN_ZERO_REGISTER = "zero_register_constant"
UNKNOWN_SIMD_HIGH = "simd_high_lane_not_tracked"


class RegisterWriter:
    """Metadata about the most recent writer of one register."""

    __slots__ = (
        "instruction_index",
        "pc",
        "inst_class",
        "is_load",
        "mem_address",
        "value",
        "source_registers",
    )

    def __init__(
        self,
        instruction_index: int,
        pc: int,
        inst_class: int,
        is_load: bool,
        mem_address: Optional[int],
        value: Optional[int],
        source_registers: Tuple[int, ...],
    ) -> None:
        self.instruction_index = instruction_index
        self.pc = pc
        self.inst_class = inst_class
        self.is_load = is_load
        self.mem_address = mem_address
        self.value = value
        self.source_registers = source_registers


class WindowEntry:
    """Minimal record of a past instruction, with resolved producer edges."""

    __slots__ = (
        "index",
        "pc",
        "inst_class",
        "is_load",
        "mem_address",
        "in_regs",
        "src_writers",
    )

    def __init__(
        self,
        index: int,
        pc: int,
        inst_class: int,
        is_load: bool,
        mem_address: Optional[int],
        in_regs: Tuple[int, ...],
        src_writers: Tuple[Optional[int], ...],
    ) -> None:
        self.index = index
        self.pc = pc
        self.inst_class = inst_class
        self.is_load = is_load
        self.mem_address = mem_address
        self.in_regs = in_regs
        self.src_writers = src_writers


class InstructionWindow:
    """Fixed-capacity ring of recent instructions, indexed by instruction index."""

    def __init__(self, capacity: int = 1 << 16) -> None:
        if capacity <= 0:
            raise ValueError("window capacity must be positive")
        self.capacity = capacity
        self._slots: List[Optional[WindowEntry]] = [None] * capacity
        self.evictions = 0

    def add(self, entry: WindowEntry) -> None:
        slot = entry.index % self.capacity
        if self._slots[slot] is not None:
            self.evictions += 1
        self._slots[slot] = entry

    def get(self, index: int) -> Optional[WindowEntry]:
        entry = self._slots[index % self.capacity]
        if entry is not None and entry.index == index:
            return entry
        return None

    def __contains__(self, index: int) -> bool:
        return self.get(index) is not None


class ShadowRegisterFile:
    """Latest known value and writer metadata per architectural register."""

    def __init__(self) -> None:
        self._writers: Dict[int, RegisterWriter] = {}

    def writer_of(self, reg: int) -> Optional[RegisterWriter]:
        return self._writers.get(reg)

    def value_of(self, reg: int) -> Tuple[Optional[int], Optional[str]]:
        """Return (value, unknown_reason). ``reason`` is None when known."""
        if reg == ZERO_OFFSET:
            # The zero register reads as 0 and has no producer.
            return 0, None
        writer = self._writers.get(reg)
        if writer is None:
            return None, UNKNOWN_NEVER_WRITTEN
        if writer.value is None:
            return None, UNKNOWN_SIMD_HIGH
        return writer.value, None

    def producer_of(self, reg: int) -> Optional[int]:
        """Instruction index that last wrote ``reg``, if any."""
        if reg == ZERO_OFFSET:
            return None
        writer = self._writers.get(reg)
        return writer.instruction_index if writer is not None else None

    def resolve_sources(
        self, in_regs: Tuple[int, ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...], Tuple[Optional[str], ...]]:
        """Resolve producers and values for one instruction's sources."""
        producers: List[Optional[int]] = []
        values: List[Optional[int]] = []
        reasons: List[Optional[str]] = []
        for reg in in_regs:
            producers.append(self.producer_of(reg))
            value, reason = self.value_of(reg)
            values.append(value)
            reasons.append(reason)
        return tuple(producers), tuple(values), tuple(reasons)

    def apply(self, instr: LogicalInstruction) -> None:
        """Record the destination writes of ``instr``.

        Must be called *after* the instruction's own sources are resolved, so a
        self-modifying register (base update) does not become its own producer.
        """
        for position, reg in enumerate(instr.out_regs):
            if reg == ZERO_OFFSET:
                # Writes to the zero register are discarded by the architecture.
                continue
            self._writers[reg] = RegisterWriter(
                instruction_index=instr.index,
                pc=instr.pc,
                inst_class=instr.inst_class,
                is_load=instr.is_load,
                mem_address=instr.eff_addr if instr.is_load else None,
                value=instr.out_values_lo[position],
                source_registers=instr.in_regs,
            )

    @property
    def tracked_registers(self) -> int:
        return len(self._writers)

    def flag_writer(self) -> Optional[RegisterWriter]:
        return self._writers.get(CC_OFFSET)
