import pytest
from io import StringIO
from contextlib import redirect_stdout
from mini_vm import run
from mini_vm.vm import VMError


def capture(src, **kw):
    buf = StringIO()
    with redirect_stdout(buf):
        run(src, **kw)
    return buf.getvalue().strip()


class TestArithmetic:
    def test_add(self):
        assert capture("print 1 + 2") == "3"

    def test_sub(self):
        assert capture("print 10 - 4") == "6"

    def test_mul(self):
        assert capture("print 3 * 4") == "12"

    def test_div_int(self):
        assert capture("print 10 / 2") == "5"

    def test_mod(self):
        assert capture("print 7 % 3") == "1"

    def test_unary_neg(self):
        assert capture("print -5") == "-5"

    def test_precedence(self):
        assert capture("print 2 + 3 * 4") == "14"

    def test_parens(self):
        assert capture("print (2 + 3) * 4") == "20"

    def test_div_by_zero(self):
        with pytest.raises(VMError):
            run("print 1 / 0")


class TestVariables:
    def test_let_and_load(self):
        assert capture("let x = 42\nprint x") == "42"

    def test_reassign(self):
        assert capture("let x = 1\nx = 99\nprint x") == "99"

    def test_undefined_variable(self):
        with pytest.raises(VMError):
            run("print x")


class TestControlFlow:
    def test_if_true(self):
        assert capture("if true { print 1 }") == "1"

    def test_if_false(self):
        assert capture("if false { print 1 }") == ""

    def test_if_else_true(self):
        assert capture("if true { print 1 } else { print 2 }") == "1"

    def test_if_else_false(self):
        assert capture("if false { print 1 } else { print 2 }") == "2"

    def test_while_loop(self):
        src = "let i = 0\nwhile i < 3 { print i\ni = i + 1 }"
        assert capture(src) == "0\n1\n2"

    def test_nested_if(self):
        src = "let x = 5\nif x > 3 { if x > 4 { print \"big\" } }"
        assert capture(src) == "big"


class TestFunctions:
    def test_simple_func(self):
        src = "func greet() { print \"hello\" }\ngreet()"
        assert capture(src) == "hello"

    def test_func_with_args(self):
        src = "func add(a, b) { return a + b }\nprint add(3, 4)"
        assert capture(src) == "7"

    def test_func_recursion(self):
        src = """
func fact(n) {
    if n <= 1 { return 1 }
    return n * fact(n - 1)
}
print fact(5)
"""
        assert capture(src) == "120"

    def test_func_wrong_argc(self):
        with pytest.raises(VMError):
            run("func f(x) { return x }\nf(1, 2)")


class TestLogical:
    def test_and_true(self):
        assert capture("print true && true") == "true"

    def test_and_false(self):
        assert capture("print true && false") == "false"

    def test_or(self):
        assert capture("print false || true") == "true"

    def test_not(self):
        assert capture("print !false") == "true"


class TestBuiltins:
    def test_str(self):
        assert capture('print str(42)') == "42"

    def test_int(self):
        assert capture('print int("10")') == "10"

    def test_type_int(self):
        assert capture('print type(1)') == "int"

    def test_type_string(self):
        assert capture('print type("hi")') == "string"

    def test_type_null(self):
        assert capture("print type(null)") == "null"
