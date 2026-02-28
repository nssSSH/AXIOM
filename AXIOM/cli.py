"""
AXIOM v0.1 - Command Line Interface

Usage:
    axiom input.ax -o output.py [--opt-level 0|1] [--debug] [--emit-ast]
    python -m axiom input.ax -o output.py
"""

import sys
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="axiom",
        description="AXIOM v0.1 — Adaptive eXpressive Intent-Oriented Matrix Language compiler",
    )
    parser.add_argument("input", help="Path to the .ax source file")
    parser.add_argument("-o", "--output", help="Path for the generated .py file")
    parser.add_argument(
        "--opt-level",
        type=int,
        choices=[0, 1],
        default=1,
        dest="opt_level",
        help="Optimization level: 0 = none, 1 = standard (default: 1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print compilation phase info to stderr",
    )
    parser.add_argument(
        "--emit-ast",
        action="store_true",
        dest="emit_ast",
        help="Emit the parsed AST as JSON instead of Python code",
    )

    args = parser.parse_args(argv)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.emit_ast:
        base = os.path.splitext(args.input)[0]
        output_path = base + ".ast.json"
    else:
        base = os.path.splitext(args.input)[0]
        output_path = base + ".py"

    # Compile
    from .compiler import compile_file, CompilationError

    try:
        compile_file(
            args.input,
            output_path,
            opt_level=args.opt_level,
            emit_ast=args.emit_ast,
            debug=args.debug,
        )
        print(f"[axiom] Compiled {args.input!r} → {output_path!r}")
    except FileNotFoundError:
        print(f"[axiom] Error: Input file not found: {args.input!r}", file=sys.stderr)
        sys.exit(1)
    except CompilationError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
