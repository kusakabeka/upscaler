"""Bytecode instruction set for the stack VM."""

from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Optional


class Op(IntEnum):
    # Stack manipulation
    PUSH = auto()       # PUSH <value>
    POP = auto()        # discard top

    # Variables
    LOAD = auto()       # LOAD <name>  → push value
    STORE = auto()      # STORE <name> ← pop value
    DEF_GLOBAL = auto() # define in global scope

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()        # unary minus

    # Comparison  (result: True / False on stack)
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()

    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()        # unary !

    # Control flow
    JUMP = auto()           # JUMP <abs_addr>
    JUMP_IF_FALSE = auto()  # JUMP_IF_FALSE <abs_addr>  (pops condition)

    # Functions
    CALL = auto()    # CALL <name> <argc>
    RETURN = auto()  # RETURN (value already on stack; null pushed if missing)

    # I/O
    PRINT = auto()   # pop & print top of stack

    # Debug
    HALT = auto()


@dataclass
class Instruction:
    op: Op
    arg1: object = None   # primary operand
    arg2: object = None   # secondary operand (e.g. argc for CALL)
    line: int = 0

    def __repr__(self) -> str:
        parts = [self.op.name]
        if self.arg1 is not None:
            parts.append(repr(self.arg1))
        if self.arg2 is not None:
            parts.append(repr(self.arg2))
        return " ".join(parts)


@dataclass
class Chunk:
    """Compiled bytecode for a single function (or top-level program)."""
    name: str
    instructions: list[Instruction]
    params: list[str]

    def disassemble(self) -> str:
        lines = [f"=== {self.name} ==="]
        for i, instr in enumerate(self.instructions):
            lines.append(f"  {i:04d}  {instr!r:<30}  ; line {instr.line}")
        return "\n".join(lines)
