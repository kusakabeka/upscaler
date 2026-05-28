from enum import Enum, auto


class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    NULL = auto()

    # Identifiers & keywords
    IDENT = auto()
    LET = auto()
    FUNC = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    PRINT = auto()
    TRUE = auto()
    FALSE = auto()

    # Arithmetic
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()

    # Comparison
    EQ = auto()       # ==
    NEQ = auto()      # !=
    LT = auto()       # <
    GT = auto()       # >
    LTE = auto()      # <=
    GTE = auto()      # >=

    # Logical
    AND = auto()      # &&
    OR = auto()       # ||
    BANG = auto()     # !

    # Assignment
    ASSIGN = auto()   # =

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMICOLON = auto()

    # Special
    EOF = auto()
    NEWLINE = auto()


KEYWORDS: dict[str, TokenType] = {
    "let": TokenType.LET,
    "func": TokenType.FUNC,
    "return": TokenType.RETURN,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "print": TokenType.PRINT,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "null": TokenType.NULL,
}


class Token:
    __slots__ = ("type", "value", "line", "col")

    def __init__(self, type: TokenType, value: object, line: int, col: int):
        self.type = type
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"
