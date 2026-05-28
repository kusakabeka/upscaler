"""AST → bytecode compiler."""

from .ast_nodes import (
    Node, Program, Block,
    IntLiteral, FloatLiteral, StringLiteral, BoolLiteral, NullLiteral,
    Identifier, BinaryOp, UnaryOp, CallExpr,
    LetStmt, AssignStmt, PrintStmt, ReturnStmt, ExprStmt,
    IfStmt, WhileStmt, FuncDef,
)
from .bytecode import Op, Instruction, Chunk


class CompileError(Exception):
    def __init__(self, msg: str, line: int = 0):
        super().__init__(f"[line {line}] {msg}")
        self.line = line


class Compiler:
    def __init__(self):
        self._functions: dict[str, Chunk] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, program: Program) -> tuple[Chunk, dict[str, Chunk]]:
        """Return (main_chunk, function_chunks)."""
        main = self._compile_block_stmts("<main>", [], program.body)
        return main, self._functions

    # ------------------------------------------------------------------
    # Chunk builders
    # ------------------------------------------------------------------

    def _compile_block_stmts(
        self, name: str, params: list[str], stmts: list[Node], is_func: bool = False
    ) -> Chunk:
        chunk = Chunk(name=name, instructions=[], params=params)
        for stmt in stmts:
            self._stmt(chunk, stmt)
        if is_func:
            # Implicit return null for functions that fall off the end
            chunk.instructions.append(Instruction(Op.PUSH, None))
            chunk.instructions.append(Instruction(Op.RETURN))
        else:
            chunk.instructions.append(Instruction(Op.HALT))
        return chunk

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _stmt(self, chunk: Chunk, node: Node) -> None:
        match node:
            case LetStmt():
                self._expr(chunk, node.value)
                chunk.instructions.append(
                    Instruction(Op.DEF_GLOBAL, node.name, line=node.line)
                )
            case AssignStmt():
                self._expr(chunk, node.value)
                chunk.instructions.append(
                    Instruction(Op.STORE, node.name, line=node.line)
                )
            case PrintStmt():
                self._expr(chunk, node.expr)
                chunk.instructions.append(Instruction(Op.PRINT, line=node.line))
            case ReturnStmt():
                if node.value is not None:
                    self._expr(chunk, node.value)
                else:
                    chunk.instructions.append(
                        Instruction(Op.PUSH, None, line=node.line)
                    )
                chunk.instructions.append(Instruction(Op.RETURN, line=node.line))
            case ExprStmt():
                self._expr(chunk, node.expr)
                chunk.instructions.append(Instruction(Op.POP, line=node.line))
            case IfStmt():
                self._if_stmt(chunk, node)
            case WhileStmt():
                self._while_stmt(chunk, node)
            case FuncDef():
                self._func_def(node)
            case Block():
                for s in node.stmts:
                    self._stmt(chunk, s)
            case _:
                raise CompileError(f"Unknown statement type: {type(node)}", node.line)

    def _if_stmt(self, chunk: Chunk, node: IfStmt) -> None:
        self._expr(chunk, node.condition)
        # Placeholder for JUMP_IF_FALSE
        jif_idx = len(chunk.instructions)
        chunk.instructions.append(Instruction(Op.JUMP_IF_FALSE, None, line=node.line))

        for s in node.then_block.stmts:
            self._stmt(chunk, s)

        if node.else_block:
            # Placeholder for JUMP over else
            jmp_idx = len(chunk.instructions)
            chunk.instructions.append(Instruction(Op.JUMP, None, line=node.line))
            # Patch JUMP_IF_FALSE → here (start of else)
            chunk.instructions[jif_idx].arg1 = len(chunk.instructions)
            for s in node.else_block.stmts:
                self._stmt(chunk, s)
            # Patch JUMP → after else
            chunk.instructions[jmp_idx].arg1 = len(chunk.instructions)
        else:
            # Patch JUMP_IF_FALSE → after then
            chunk.instructions[jif_idx].arg1 = len(chunk.instructions)

    def _while_stmt(self, chunk: Chunk, node: WhileStmt) -> None:
        loop_start = len(chunk.instructions)
        self._expr(chunk, node.condition)
        jif_idx = len(chunk.instructions)
        chunk.instructions.append(Instruction(Op.JUMP_IF_FALSE, None, line=node.line))
        for s in node.body.stmts:
            self._stmt(chunk, s)
        # Loop back
        chunk.instructions.append(
            Instruction(Op.JUMP, loop_start, line=node.line)
        )
        # Patch exit
        chunk.instructions[jif_idx].arg1 = len(chunk.instructions)

    def _func_def(self, node: FuncDef) -> None:
        func_chunk = self._compile_block_stmts(
            node.name, node.params, node.body.stmts, is_func=True
        )
        self._functions[node.name] = func_chunk

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def _expr(self, chunk: Chunk, node: Node) -> None:
        match node:
            case IntLiteral() | FloatLiteral() | StringLiteral() | BoolLiteral():
                chunk.instructions.append(
                    Instruction(Op.PUSH, node.value, line=node.line)
                )
            case NullLiteral():
                chunk.instructions.append(
                    Instruction(Op.PUSH, None, line=node.line)
                )
            case Identifier():
                chunk.instructions.append(
                    Instruction(Op.LOAD, node.name, line=node.line)
                )
            case BinaryOp():
                self._binary(chunk, node)
            case UnaryOp():
                self._unary(chunk, node)
            case CallExpr():
                for arg in node.args:
                    self._expr(chunk, arg)
                chunk.instructions.append(
                    Instruction(Op.CALL, node.callee, len(node.args), line=node.line)
                )
            case _:
                raise CompileError(f"Unknown expression type: {type(node)}", node.line)

    _BINARY_OPS: dict[str, Op] = {
        "+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "%": Op.MOD,
        "==": Op.EQ, "!=": Op.NEQ, "<": Op.LT, ">": Op.GT,
        "<=": Op.LTE, ">=": Op.GTE,
        "&&": Op.AND, "||": Op.OR,
    }

    def _binary(self, chunk: Chunk, node: BinaryOp) -> None:
        op = self._BINARY_OPS.get(node.op)
        if op is None:
            raise CompileError(f"Unknown binary operator: {node.op}", node.line)
        self._expr(chunk, node.left)
        self._expr(chunk, node.right)
        chunk.instructions.append(Instruction(op, line=node.line))

    def _unary(self, chunk: Chunk, node: UnaryOp) -> None:
        self._expr(chunk, node.operand)
        if node.op == "-":
            chunk.instructions.append(Instruction(Op.NEG, line=node.line))
        elif node.op == "!":
            chunk.instructions.append(Instruction(Op.NOT, line=node.line))
        else:
            raise CompileError(f"Unknown unary operator: {node.op}", node.line)
