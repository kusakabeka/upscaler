"""Recursive-descent parser: token stream → AST."""

from .tokens import Token, TokenType
from .ast_nodes import (
    Node, Program, Block,
    IntLiteral, FloatLiteral, StringLiteral, BoolLiteral, NullLiteral,
    Identifier, BinaryOp, UnaryOp, CallExpr,
    LetStmt, AssignStmt, PrintStmt, ReturnStmt, ExprStmt,
    IfStmt, WhileStmt, FuncDef,
)


class ParseError(Exception):
    def __init__(self, msg: str, token: Token):
        super().__init__(f"[{token.line}:{token.col}] {msg} (got {token.type.name})")
        self.token = token


class Parser:
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> Program:
        stmts: list[Node] = []
        while not self._at(TokenType.EOF):
            stmts.append(self._statement())
        return Program(body=stmts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _at(self, *types: TokenType) -> bool:
        return self._peek().type in types

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.type != TokenType.EOF:
            self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._peek()
        if tok.type != tt:
            raise ParseError(f"Expected {tt.name}", tok)
        return self._advance()

    def _match(self, *types: TokenType) -> Token | None:
        if self._at(*types):
            return self._advance()
        return None

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _statement(self) -> Node:
        tok = self._peek()

        if tok.type == TokenType.LET:
            return self._let_stmt()
        if tok.type == TokenType.FUNC:
            return self._func_def()
        if tok.type == TokenType.RETURN:
            return self._return_stmt()
        if tok.type == TokenType.IF:
            return self._if_stmt()
        if tok.type == TokenType.WHILE:
            return self._while_stmt()
        if tok.type == TokenType.PRINT:
            return self._print_stmt()

        # Assignment or expression
        return self._expr_or_assign_stmt()

    def _let_stmt(self) -> LetStmt:
        line = self._peek().line
        self._advance()  # consume 'let'
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.ASSIGN)
        value = self._expression()
        self._match(TokenType.SEMICOLON)
        return LetStmt(name=name, value=value, line=line)

    def _func_def(self) -> FuncDef:
        line = self._peek().line
        self._advance()  # consume 'func'
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.LPAREN)
        params: list[str] = []
        if not self._at(TokenType.RPAREN):
            params.append(self._expect(TokenType.IDENT).value)
            while self._match(TokenType.COMMA):
                params.append(self._expect(TokenType.IDENT).value)
        self._expect(TokenType.RPAREN)
        body = self._block()
        return FuncDef(name=name, params=params, body=body, line=line)

    def _return_stmt(self) -> ReturnStmt:
        line = self._peek().line
        self._advance()  # consume 'return'
        value = None
        if not self._at(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
            value = self._expression()
        self._match(TokenType.SEMICOLON)
        return ReturnStmt(value=value, line=line)

    def _if_stmt(self) -> IfStmt:
        line = self._peek().line
        self._advance()  # consume 'if'
        condition = self._expression()
        then_block = self._block()
        else_block = None
        if self._match(TokenType.ELSE):
            else_block = self._block()
        return IfStmt(condition=condition, then_block=then_block,
                      else_block=else_block, line=line)

    def _while_stmt(self) -> WhileStmt:
        line = self._peek().line
        self._advance()  # consume 'while'
        condition = self._expression()
        body = self._block()
        return WhileStmt(condition=condition, body=body, line=line)

    def _print_stmt(self) -> PrintStmt:
        line = self._peek().line
        self._advance()  # consume 'print'
        expr = self._expression()
        self._match(TokenType.SEMICOLON)
        return PrintStmt(expr=expr, line=line)

    def _expr_or_assign_stmt(self) -> Node:
        line = self._peek().line
        expr = self._expression()
        # Check for assignment: ident = expr
        if isinstance(expr, Identifier) and self._at(TokenType.ASSIGN):
            self._advance()  # consume '='
            value = self._expression()
            self._match(TokenType.SEMICOLON)
            return AssignStmt(name=expr.name, value=value, line=line)
        self._match(TokenType.SEMICOLON)
        return ExprStmt(expr=expr, line=line)

    def _block(self) -> Block:
        line = self._peek().line
        self._expect(TokenType.LBRACE)
        stmts: list[Node] = []
        while not self._at(TokenType.RBRACE, TokenType.EOF):
            stmts.append(self._statement())
        self._expect(TokenType.RBRACE)
        return Block(stmts=stmts, line=line)

    # ------------------------------------------------------------------
    # Expressions  (Pratt / precedence climbing)
    # ------------------------------------------------------------------

    def _expression(self) -> Node:
        return self._or_expr()

    def _or_expr(self) -> Node:
        left = self._and_expr()
        while self._at(TokenType.OR):
            op = self._advance().value
            right = self._and_expr()
            left = BinaryOp(op=op, left=left, right=right, line=left.line)
        return left

    def _and_expr(self) -> Node:
        left = self._equality()
        while self._at(TokenType.AND):
            op = self._advance().value
            right = self._equality()
            left = BinaryOp(op=op, left=left, right=right, line=left.line)
        return left

    def _equality(self) -> Node:
        left = self._comparison()
        while self._at(TokenType.EQ, TokenType.NEQ):
            op = self._advance().value
            right = self._comparison()
            left = BinaryOp(op=op, left=left, right=right, line=left.line)
        return left

    def _comparison(self) -> Node:
        left = self._additive()
        while self._at(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            op = self._advance().value
            right = self._additive()
            left = BinaryOp(op=op, left=left, right=right, line=left.line)
        return left

    def _additive(self) -> Node:
        left = self._multiplicative()
        while self._at(TokenType.PLUS, TokenType.MINUS):
            op = self._advance().value
            right = self._multiplicative()
            left = BinaryOp(op=op, left=left, right=right, line=left.line)
        return left

    def _multiplicative(self) -> Node:
        left = self._unary()
        while self._at(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self._advance().value
            right = self._unary()
            left = BinaryOp(op=op, left=left, right=right, line=left.line)
        return left

    def _unary(self) -> Node:
        if self._at(TokenType.MINUS, TokenType.BANG):
            tok = self._advance()
            operand = self._unary()
            return UnaryOp(op=tok.value, operand=operand, line=tok.line)
        return self._call_or_primary()

    def _call_or_primary(self) -> Node:
        node = self._primary()
        if isinstance(node, Identifier) and self._at(TokenType.LPAREN):
            line = self._peek().line
            self._advance()  # consume '('
            args: list[Node] = []
            if not self._at(TokenType.RPAREN):
                args.append(self._expression())
                while self._match(TokenType.COMMA):
                    args.append(self._expression())
            self._expect(TokenType.RPAREN)
            return CallExpr(callee=node.name, args=args, line=line)
        return node

    def _primary(self) -> Node:
        tok = self._peek()

        if tok.type == TokenType.INTEGER:
            self._advance()
            return IntLiteral(value=tok.value, line=tok.line)
        if tok.type == TokenType.FLOAT:
            self._advance()
            return FloatLiteral(value=tok.value, line=tok.line)
        if tok.type == TokenType.STRING:
            self._advance()
            return StringLiteral(value=tok.value, line=tok.line)
        if tok.type in (TokenType.TRUE, TokenType.FALSE):
            self._advance()
            return BoolLiteral(value=tok.value, line=tok.line)
        if tok.type == TokenType.NULL:
            self._advance()
            return NullLiteral(line=tok.line)
        if tok.type == TokenType.IDENT:
            self._advance()
            return Identifier(name=tok.value, line=tok.line)
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._expression()
            self._expect(TokenType.RPAREN)
            return expr

        raise ParseError("Unexpected token in expression", tok)
