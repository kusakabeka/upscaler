import pytest
from mini_vm.lexer import Lexer, LexerError
from mini_vm.tokens import TokenType


def lex(src):
    return Lexer(src).tokenize()


def types(src):
    return [t.type for t in lex(src) if t.type != TokenType.EOF]


class TestLiterals:
    def test_integer(self):
        tokens = lex("42")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == 42

    def test_float(self):
        tokens = lex("3.14")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == pytest.approx(3.14)

    def test_string(self):
        tokens = lex('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_string_escape(self):
        tokens = lex(r'"line\nnext"')
        assert tokens[0].value == "line\nnext"

    def test_bool_true(self):
        tokens = lex("true")
        assert tokens[0].type == TokenType.TRUE
        assert tokens[0].value is True

    def test_bool_false(self):
        tokens = lex("false")
        assert tokens[0].type == TokenType.FALSE
        assert tokens[0].value is False

    def test_null(self):
        tokens = lex("null")
        assert tokens[0].type == TokenType.NULL


class TestKeywords:
    def test_keywords(self):
        src = "let func return if else while print"
        expected = [
            TokenType.LET, TokenType.FUNC, TokenType.RETURN,
            TokenType.IF, TokenType.ELSE, TokenType.WHILE, TokenType.PRINT,
        ]
        assert types(src) == expected


class TestOperators:
    def test_arithmetic(self):
        assert types("+ - * / %") == [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
            TokenType.SLASH, TokenType.PERCENT,
        ]

    def test_comparison(self):
        assert types("== != < > <= >=") == [
            TokenType.EQ, TokenType.NEQ, TokenType.LT,
            TokenType.GT, TokenType.LTE, TokenType.GTE,
        ]

    def test_logical(self):
        assert types("&& || !") == [TokenType.AND, TokenType.OR, TokenType.BANG]


class TestComments:
    def test_line_comment_ignored(self):
        assert types("# this is a comment\n42") == [TokenType.INTEGER]

    def test_inline_comment(self):
        assert types("1 + 2 # add") == [
            TokenType.INTEGER, TokenType.PLUS, TokenType.INTEGER,
        ]


class TestErrors:
    def test_unexpected_char(self):
        with pytest.raises(LexerError):
            lex("@")

    def test_unterminated_string(self):
        with pytest.raises(LexerError):
            lex('"no closing')
