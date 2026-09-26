#!/usr/bin/env python3
"""Entry point: run a .tiny file or start the REPL."""

import sys
import argparse
from . import run, Lexer, LexerError, Parser, ParseError, Compiler, CompileError, VM, VMError


def run_file(path: str, dump: bool) -> int:
    try:
        source = open(path).read()
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1
    try:
        run(source, dump_bytecode=dump)
    except (LexerError, ParseError, CompileError, VMError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def repl(dump: bool) -> None:
    print("mini_vm REPL  (Ctrl-D to quit)")
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            continue
        try:
            run(line, dump_bytecode=dump)
        except (LexerError, ParseError, CompileError, VMError) as e:
            print(f"Error: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="mini_vm", description="Run .tiny source files")
    ap.add_argument("file", nargs="?", help="Source file to run")
    ap.add_argument("--dump", action="store_true", help="Dump bytecode before execution")
    args = ap.parse_args()

    if args.file:
        sys.exit(run_file(args.file, args.dump))
    else:
        repl(args.dump)


if __name__ == "__main__":
    main()
