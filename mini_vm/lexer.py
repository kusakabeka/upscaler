"""Lexer: source text → token stream."""

from .tokens import Token, TokenType, KEYWORDS


class LexerError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"[{line}:{col}] {msg}")
        self.line = line
        self.col = col


class Lexer:
    def __init__(self, source: str):
        self._src = source
        self._pos = 0
        self._line = 1
        self._col = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            tok = self._next_token()
            tokens.append(tok)
            if tok.type == TokenType.EOF:
                break
        return tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        idx = self._pos + offset
        return self._src[idx] if idx < len(self._src) else "\0"

    def _advance(self) -> str:
        ch = self._src[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    def _skip_whitespace_and_comments(self) -> None:
        while self._pos < len(self._src):
            ch = self._peek()
            if ch in (" ", "\t", "\r", "\n"):
                self._advance()
            elif ch == "#":  # line comment
                while self._peek() not in ("\n", "\0"):
                    self._advance()
            else:
                break

    def _read_number(self) -> Token:
        line, col = self._line, self._col
        buf = []
        is_float = False
        while self._peek().isdigit():
            buf.append(self._advance())
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            buf.append(self._advance())  # consume '.'
            while self._peek().isdigit():
                buf.append(self._advance())
        text = "".join(buf)
        value = float(text) if is_float else int(text)
        tt = TokenType.FLOAT if is_float else TokenType.INTEGER
        return Token(tt, value, line, col)

    def _read_string(self) -> Token:
        line, col = self._line, self._col
        self._advance()  # opening quote
        buf = []
        while self._peek() not in ('"', "\0"):
            ch = self._advance()
            if ch == "\\" :
                esc = self._advance()
                ch = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}.get(esc, esc)
            buf.append(ch)
        if self._peek() == "\0":
            raise LexerError("Unterminated string literal", line, col)
        self._advance()  # closing quote
        return Token(TokenType.STRING, "".join(buf), line, col)

    def _read_ident_or_keyword(self) -> Token:
        line, col = self._line, self._col
        buf = []
        while self._peek().isalnum() or self._peek() == "_":
            buf.append(self._advance())
        text = "".join(buf)
        tt = KEYWORDS.get(text, TokenType.IDENT)
        value: object = text
        if tt == TokenType.TRUE:
            value = True
        elif tt == TokenType.FALSE:
            value = False
        elif tt == TokenType.NULL:
            value = None
        return Token(tt, value, line, col)

    def _next_token(self) -> Token:
        self._skip_whitespace_and_comments()
        line, col = self._line, self._col

        if self._pos >= len(self._src):
            return Token(TokenType.EOF, None, line, col)

        ch = self._peek()

        if ch.isdigit():
            return self._read_number()
        if ch == '"':
            return self._read_string()
        if ch.isalpha() or ch == "_":
            return self._read_ident_or_keyword()

        self._advance()

        # Two-character tokens
        nxt = self._peek()
        two = ch + nxt
        TWO: dict[str, TokenType] = {
            "==": TokenType.EQ,
            "!=": TokenType.NEQ,
            "<=": TokenType.LTE,
            ">=": TokenType.GTE,
            "&&": TokenType.AND,
            "||": TokenType.OR,
        }
        if two in TWO:
            self._advance()
            return Token(TWO[two], two, line, col)

        # Single-character tokens
        ONE: dict[str, TokenType] = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "<": TokenType.LT,
            ">": TokenType.GT,
            "!": TokenType.BANG,
            "=": TokenType.ASSIGN,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            ",": TokenType.COMMA,
            ";": TokenType.SEMICOLON,
        }
        if ch in ONE:
            return Token(ONE[ch], ch, line, col)

        raise LexerError(f"Unexpected character {ch!r}", line, col)
