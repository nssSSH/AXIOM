"""
AXIOM v0.1 - Compiler Orchestrator
Runs all compiler phases in sequence and returns the Python output.
"""

import json
from .lexer import tokenize, LexerError
from .parser import Parser, ParseError
from .semantic import SemanticAnalyzer, SemanticError
from .optimizer import Optimizer
from .codegen import CodeGenerator, CodeGenError


class CompilationError(Exception):
    """Unified compilation error wrapper."""
    pass


def compile_source(
    source: str,
    opt_level: int = 1,
    emit_ast: bool = False,
    debug: bool = False,
) -> str:
    """
    Compile AXIOM source text to Python source text.

    Parameters
    ----------
    source     : AXIOM source code string
    opt_level  : 0 = no optimizations, 1 = all optimizations
    emit_ast   : if True, return a JSON representation of the AST instead of Python
    debug      : print each phase summary to stderr

    Returns
    -------
    Python source code string (or JSON AST if emit_ast=True)

    Raises
    ------
    CompilationError on any phase failure
    """
    import sys

    def log(msg):
        if debug:
            print(f"[AXIOM] {msg}", file=sys.stderr)

    # ── Phase 1: Lexical Analysis ─────────────────────────────────────────────
    log("Phase 1: Lexical analysis")
    try:
        tokens = tokenize(source)
    except LexerError as e:
        raise CompilationError(str(e)) from e

    log(f"  {len(tokens)-1} tokens produced")

    # ── Phase 2: Parsing ──────────────────────────────────────────────────────
    log("Phase 2: Parsing")
    try:
        parser = Parser(tokens)
        ast = parser.parse()
    except ParseError as e:
        raise CompilationError(str(e)) from e

    log(f"  {len(ast.statements)} top-level statements")

    if emit_ast:
        return _ast_to_json(ast)

    # ── Phase 3: Semantic Analysis ────────────────────────────────────────────
    log("Phase 3: Semantic analysis")
    try:
        SemanticAnalyzer().analyze(ast)
    except SemanticError as e:
        raise CompilationError(str(e)) from e

    # ── Phase 4: Optimization ─────────────────────────────────────────────────
    log(f"Phase 4: Optimization (level {opt_level})")
    ast = Optimizer(opt_level=opt_level).optimize(ast)

    # ── Phase 5: Code Generation ──────────────────────────────────────────────
    log("Phase 5: Code generation")
    try:
        python_code = CodeGenerator().generate(ast)
    except CodeGenError as e:
        raise CompilationError(str(e)) from e

    log("  Compilation successful")
    return python_code


def compile_file(
    input_path: str,
    output_path: str,
    opt_level: int = 1,
    emit_ast: bool = False,
    debug: bool = False,
) -> None:
    """Read an .ax file and write the compiled .py to output_path."""
    with open(input_path, "r", encoding="utf-8") as f:
        source = f.read()

    result = compile_source(source, opt_level=opt_level, emit_ast=emit_ast, debug=debug)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)


# ── AST serialization (for --emit-ast) ────────────────────────────────────────

def _ast_to_json(node) -> str:
    return json.dumps(_node_to_dict(node), indent=2)


def _node_to_dict(node):
    if node is None:
        return None
    if isinstance(node, list):
        return [_node_to_dict(n) for n in node]
    if not hasattr(node, '__dataclass_fields__'):
        return node  # primitive
    d = {"_type": type(node).__name__}
    for field_name in node.__dataclass_fields__:
        val = getattr(node, field_name)
        d[field_name] = _node_to_dict(val)
    return d
