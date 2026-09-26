"""Stack-based virtual machine."""

from __future__ import annotations
from .bytecode import Op, Instruction, Chunk


class VMError(Exception):
    def __init__(self, msg: str, line: int = 0):
        super().__init__(f"[line {line}] {msg}")
        self.line = line


class CallFrame:
    """Activation record for a function call."""
    __slots__ = ("chunk", "ip", "locals")

    def __init__(self, chunk: Chunk, locals_: dict):
        self.chunk = chunk
        self.ip = 0
        self.locals: dict[str, object] = locals_

    def fetch(self) -> Instruction:
        instr = self.chunk.instructions[self.ip]
        self.ip += 1
        return instr


class VM:
    """
    Stack VM execution engine.

    Architecture:
    - Value stack  : operands + temporaries
    - Call stack   : CallFrame objects (function activation records)
    - Globals      : shared across all frames
    - Builtins     : native functions (len, str, int, …)
    """

    MAX_CALL_DEPTH = 256

    def __init__(self, functions: dict[str, Chunk] | None = None):
        self._functions: dict[str, Chunk] = functions or {}
        self._globals: dict[str, object] = {}
        self._stack: list[object] = []
        self._frames: list[CallFrame] = []
        self._builtins = _register_builtins()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, main: Chunk) -> None:
        self._functions.update({main.name: main})
        frame = CallFrame(main, {})
        self._frames.append(frame)
        self._exec()

    def add_function(self, chunk: Chunk) -> None:
        self._functions[chunk.name] = chunk

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    def _exec(self) -> None:
        while self._frames:
            frame = self._frames[-1]
            instr = frame.fetch()
            op = instr.op

            if op == Op.HALT:
                self._frames.pop()

            elif op == Op.PUSH:
                self._stack.append(instr.arg1)

            elif op == Op.POP:
                self._pop()

            # --- Variables ---
            elif op == Op.DEF_GLOBAL:
                self._globals[instr.arg1] = self._pop()

            elif op == Op.STORE:
                val = self._pop()
                name = instr.arg1
                if name in frame.locals:
                    frame.locals[name] = val
                elif name in self._globals:
                    self._globals[name] = val
                else:
                    raise VMError(f"Undefined variable '{name}'", instr.line)

            elif op == Op.LOAD:
                name = instr.arg1
                if name in frame.locals:
                    self._stack.append(frame.locals[name])
                elif name in self._globals:
                    self._stack.append(self._globals[name])
                else:
                    raise VMError(f"Undefined variable '{name}'", instr.line)

            # --- Arithmetic ---
            elif op == Op.ADD:
                b, a = self._pop(), self._pop()
                self._stack.append(a + b)
            elif op == Op.SUB:
                b, a = self._pop(), self._pop()
                self._stack.append(a - b)
            elif op == Op.MUL:
                b, a = self._pop(), self._pop()
                self._stack.append(a * b)
            elif op == Op.DIV:
                b, a = self._pop(), self._pop()
                if b == 0:
                    raise VMError("Division by zero", instr.line)
                result = a / b
                self._stack.append(int(result) if isinstance(a, int) and isinstance(b, int) and result == int(result) else result)
            elif op == Op.MOD:
                b, a = self._pop(), self._pop()
                if b == 0:
                    raise VMError("Modulo by zero", instr.line)
                self._stack.append(a % b)
            elif op == Op.NEG:
                self._stack.append(-self._pop())

            # --- Comparison ---
            elif op == Op.EQ:
                b, a = self._pop(), self._pop()
                self._stack.append(a == b)
            elif op == Op.NEQ:
                b, a = self._pop(), self._pop()
                self._stack.append(a != b)
            elif op == Op.LT:
                b, a = self._pop(), self._pop()
                self._stack.append(a < b)
            elif op == Op.GT:
                b, a = self._pop(), self._pop()
                self._stack.append(a > b)
            elif op == Op.LTE:
                b, a = self._pop(), self._pop()
                self._stack.append(a <= b)
            elif op == Op.GTE:
                b, a = self._pop(), self._pop()
                self._stack.append(a >= b)

            # --- Logical ---
            elif op == Op.AND:
                b, a = self._pop(), self._pop()
                self._stack.append(bool(a) and bool(b))
            elif op == Op.OR:
                b, a = self._pop(), self._pop()
                self._stack.append(bool(a) or bool(b))
            elif op == Op.NOT:
                self._stack.append(not self._pop())

            # --- Control flow ---
            elif op == Op.JUMP:
                frame.ip = instr.arg1

            elif op == Op.JUMP_IF_FALSE:
                if not self._pop():
                    frame.ip = instr.arg1

            # --- Functions ---
            elif op == Op.CALL:
                name: str = instr.arg1
                argc: int = instr.arg2
                args = list(reversed([self._pop() for _ in range(argc)]))

                if name in self._builtins:
                    result = self._builtins[name](args, instr.line)
                    self._stack.append(result)
                elif name in self._functions:
                    fn = self._functions[name]
                    if len(fn.params) != argc:
                        raise VMError(
                            f"'{name}' expects {len(fn.params)} args, got {argc}",
                            instr.line,
                        )
                    if len(self._frames) >= self.MAX_CALL_DEPTH:
                        raise VMError("Max call depth exceeded", instr.line)
                    locals_ = dict(zip(fn.params, args))
                    self._frames.append(CallFrame(fn, locals_))
                else:
                    raise VMError(f"Undefined function '{name}'", instr.line)

            elif op == Op.RETURN:
                ret_val = self._pop()
                self._frames.pop()
                self._stack.append(ret_val)

            # --- I/O ---
            elif op == Op.PRINT:
                val = self._pop()
                print(_format_value(val))

            else:
                raise VMError(f"Unknown opcode: {op}", instr.line)

    # ------------------------------------------------------------------
    # Stack helpers
    # ------------------------------------------------------------------

    def _pop(self) -> object:
        if not self._stack:
            raise VMError("Stack underflow")
        return self._stack.pop()


# ------------------------------------------------------------------
# Built-in functions
# ------------------------------------------------------------------

def _format_value(val: object) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _register_builtins() -> dict:
    def _len(args, line):
        if len(args) != 1:
            raise VMError("len() expects 1 argument", line)
        try:
            return len(args[0])
        except TypeError:
            raise VMError("len() argument has no length", line)

    def _str(args, line):
        if len(args) != 1:
            raise VMError("str() expects 1 argument", line)
        return _format_value(args[0])

    def _int(args, line):
        if len(args) != 1:
            raise VMError("int() expects 1 argument", line)
        try:
            return int(args[0])
        except (ValueError, TypeError):
            raise VMError(f"Cannot convert {args[0]!r} to int", line)

    def _float_fn(args, line):
        if len(args) != 1:
            raise VMError("float() expects 1 argument", line)
        try:
            return float(args[0])
        except (ValueError, TypeError):
            raise VMError(f"Cannot convert {args[0]!r} to float", line)

    def _type_fn(args, line):
        if len(args) != 1:
            raise VMError("type() expects 1 argument", line)
        v = args[0]
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "string"
        return "unknown"

    return {
        "len": _len,
        "str": _str,
        "int": _int,
        "float": _float_fn,
        "type": _type_fn,
    }
