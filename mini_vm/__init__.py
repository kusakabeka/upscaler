"""mini_vm — a tiny stack-based language interpreter."""

from .lexer import Lexer, LexerError
from .parser import Parser, ParseError
from .compiler import Compiler, CompileError
from .vm import VM, VMError

__all__ = [
    "Lexer", "LexerError",
    "Parser", "ParseError",
    "Compiler", "CompileError",
    "VM", "VMError",
    "run",
]


def run(source: str, *, dump_bytecode: bool = False) -> None:
    """Lex → Parse → Compile → Execute a source string."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    compiler = Compiler()
    main_chunk, fn_chunks = compiler.compile(ast)

    if dump_bytecode:
        print(main_chunk.disassemble())
        for chunk in fn_chunks.values():
            print(chunk.disassemble())

    vm = VM(fn_chunks)
    vm.run(main_chunk)
