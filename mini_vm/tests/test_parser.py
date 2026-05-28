import pytest
from mini_vm.lexer import Lexer
from mini_vm.parser import Parser, ParseError
from mini_vm.ast_nodes import (
    Program, LetStmt, AssignStmt, PrintStmt, ReturnStmt,
    IfStmt, WhileStmt, FuncDef,
    BinaryOp, UnaryOp, CallExpr, Identifier,
    IntLiteral, StringLiteral, BoolLiteral, NullLiteral,
)


def parse(src):
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()


class TestLiterals:
    def test_int(self):
        prog = parse("1")
        assert isinstance(prog.body[0].expr, IntLiteral)
        assert prog.body[0].expr.value == 1

    def test_string(self):
        prog = parse('"hi"')
        assert isinstance(prog.body[0].expr, StringLiteral)

    def test_bool(self):
        prog = parse("true")
        assert isinstance(prog.body[0].expr, BoolLiteral)
        assert prog.body[0].expr.value is True

    def test_null(self):
        prog = parse("null")
        assert isinstance(prog.body[0].expr, NullLiteral)


class TestStatements:
    def test_let(self):
        prog = parse("let x = 5")
        stmt = prog.body[0]
        assert isinstance(stmt, LetStmt)
        assert stmt.name == "x"
        assert stmt.value.value == 5

    def test_assign(self):
        prog = parse("let x = 0 x = 10")
        assert isinstance(prog.body[1], AssignStmt)
        assert prog.body[1].name == "x"

    def test_print(self):
        prog = parse("print 42")
        assert isinstance(prog.body[0], PrintStmt)

    def test_if(self):
        prog = parse("if true { print 1 }")
        stmt = prog.body[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.else_block is None

    def test_if_else(self):
        prog = parse("if false { print 1 } else { print 2 }")
        stmt = prog.body[0]
        assert stmt.else_block is not None

    def test_while(self):
        prog = parse("while false { print 0 }")
        assert isinstance(prog.body[0], WhileStmt)

    def test_func_def(self):
        prog = parse("func add(a, b) { return a + b }")
        fn = prog.body[0]
        assert isinstance(fn, FuncDef)
        assert fn.name == "add"
        assert fn.params == ["a", "b"]


class TestExpressions:
    def test_binary_precedence(self):
        prog = parse("2 + 3 * 4")
        expr = prog.body[0].expr
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "*"

    def test_unary_neg(self):
        prog = parse("-5")
        expr = prog.body[0].expr
        assert isinstance(expr, UnaryOp)
        assert expr.op == "-"

    def test_call_expr(self):
        prog = parse("add(1, 2)")
        expr = prog.body[0].expr
        assert isinstance(expr, CallExpr)
        assert expr.callee == "add"
        assert len(expr.args) == 2

    def test_grouped(self):
        prog = parse("(1 + 2) * 3")
        expr = prog.body[0].expr
        assert expr.op == "*"


class TestErrors:
    def test_missing_closing_brace(self):
        with pytest.raises(ParseError):
            parse("if true { print 1 ")

    def test_unexpected_token(self):
        with pytest.raises(ParseError):
            parse("let = 5")
