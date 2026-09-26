"""AST node definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------
# Base
# ------------------------------------------------------------------

@dataclass
class Node:
    # kw_only so subclass positional fields don't conflict with a defaulted parent field
    line: int = field(default=0, kw_only=True, compare=False, repr=False)


# ------------------------------------------------------------------
# Expressions
# ------------------------------------------------------------------

@dataclass
class IntLiteral(Node):
    value: int


@dataclass
class FloatLiteral(Node):
    value: float


@dataclass
class StringLiteral(Node):
    value: str


@dataclass
class BoolLiteral(Node):
    value: bool


@dataclass
class NullLiteral(Node):
    pass


@dataclass
class Identifier(Node):
    name: str


@dataclass
class BinaryOp(Node):
    op: str
    left: Node
    right: Node


@dataclass
class UnaryOp(Node):
    op: str
    operand: Node


@dataclass
class CallExpr(Node):
    callee: str
    args: list[Node]


# ------------------------------------------------------------------
# Statements
# ------------------------------------------------------------------

@dataclass
class LetStmt(Node):
    name: str
    value: Node


@dataclass
class AssignStmt(Node):
    name: str
    value: Node


@dataclass
class PrintStmt(Node):
    expr: Node


@dataclass
class ReturnStmt(Node):
    value: Optional[Node]


@dataclass
class ExprStmt(Node):
    expr: Node


@dataclass
class Block(Node):
    stmts: list[Node]


@dataclass
class IfStmt(Node):
    condition: Node
    then_block: Block
    else_block: Optional[Block]


@dataclass
class WhileStmt(Node):
    condition: Node
    body: Block


@dataclass
class FuncDef(Node):
    name: str
    params: list[str]
    body: Block


@dataclass
class Program(Node):
    body: list[Node]
