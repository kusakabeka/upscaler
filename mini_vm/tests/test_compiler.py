import pytest
from mini_vm.lexer import Lexer
from mini_vm.parser import Parser
from mini_vm.compiler import Compiler
from mini_vm.bytecode import Op


def compile_src(src):
    tokens = Lexer(src).tokenize()
    ast = Parser(tokens).parse()
    return Compiler().compile(ast)


def ops(chunk):
    return [i.op for i in chunk.instructions]


class TestExpressions:
    def test_push_int(self):
        chunk, _ = compile_src("5")
        assert Op.PUSH in ops(chunk)

    def test_arithmetic(self):
        chunk, _ = compile_src("2 + 3")
        assert Op.ADD in ops(chunk)

    def test_subtraction(self):
        chunk, _ = compile_src("10 - 4")
        assert Op.SUB in ops(chunk)

    def test_multiply(self):
        chunk, _ = compile_src("3 * 4")
        assert Op.MUL in ops(chunk)

    def test_unary_neg(self):
        chunk, _ = compile_src("-1")
        assert Op.NEG in ops(chunk)

    def test_comparison(self):
        chunk, _ = compile_src("1 < 2")
        assert Op.LT in ops(chunk)


class TestStatements:
    def test_let_produces_def_global(self):
        chunk, _ = compile_src("let x = 10")
        assert Op.DEF_GLOBAL in ops(chunk)

    def test_print_produces_print(self):
        chunk, _ = compile_src("print 42")
        assert Op.PRINT in ops(chunk)

    def test_if_produces_jump_if_false(self):
        chunk, _ = compile_src("if true { print 1 }")
        assert Op.JUMP_IF_FALSE in ops(chunk)

    def test_if_else_produces_jump(self):
        chunk, _ = compile_src("if true { print 1 } else { print 2 }")
        assert Op.JUMP in ops(chunk)

    def test_while_produces_loop(self):
        chunk, _ = compile_src("while false { print 0 }")
        assert Op.JUMP in ops(chunk)
        assert Op.JUMP_IF_FALSE in ops(chunk)

    def test_func_def_stored_separately(self):
        _, fns = compile_src("func sq(x) { return x * x }")
        assert "sq" in fns

    def test_call_produces_call_op(self):
        chunk, _ = compile_src("func sq(x) { return x * x } sq(5)")
        assert Op.CALL in ops(chunk)


class TestPatching:
    def test_if_jump_target_is_valid(self):
        chunk, _ = compile_src("if true { print 1 }")
        jif = next(i for i in chunk.instructions if i.op == Op.JUMP_IF_FALSE)
        assert 0 <= jif.arg1 < len(chunk.instructions)

    def test_while_jump_target_is_valid(self):
        chunk, _ = compile_src("while false { }")
        jmp = next(i for i in chunk.instructions if i.op == Op.JUMP)
        assert 0 <= jmp.arg1 < len(chunk.instructions)
