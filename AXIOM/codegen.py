"""
AXIOM v0.1 - Code Generator
Converts an optimized AST into clean, readable Python 3 source code.
"""

from typing import Set
from .ast_nodes import (
    ProgramNode, AssignmentNode, PrintNode, IntentNode,
    PipeNode, MapNode, FilterNode, ReduceNode, LambdaNode,
    RangeNode, ArrayNode, IdentifierNode, NumberNode,
    BinaryOpNode, UnaryOpNode, ASTNode
)


class CodeGenError(Exception):
    def __init__(self, message: str, line: int = 0):
        super().__init__(f"[CodeGenError] Line {line}: {message}")
        self.line = line


class CodeGenerator:
    def __init__(self):
        self._needs_reduce = False
        self._lines: list = []

    def generate(self, program: ProgramNode) -> str:
        """Return Python source code string for the given program AST."""
        self._needs_reduce = self._check_needs_reduce(program)
        self._lines = []

        if self._needs_reduce:
            self._lines.append("from functools import reduce")
            self._lines.append("")

        for stmt in program.statements:
            self._lines.append(self._emit_statement(stmt))

        return "\n".join(self._lines) + "\n"

    # ------------------------------------------------------------------ statements

    def _emit_statement(self, node: ASTNode) -> str:
        if isinstance(node, AssignmentNode):
            return f"{node.name} = {self._emit_expr(node.value)}"
        if isinstance(node, PrintNode):
            return f"print({self._emit_expr(node.value)})"
        # bare expression statement
        return self._emit_expr(node)

    # ------------------------------------------------------------------ expressions

    def _emit_expr(self, node: ASTNode, intents: set = None) -> str:
        if intents is None:
            intents = getattr(node, '_intents', set())

        if isinstance(node, IntentNode):
            merged = intents | {node.modifier}
            return self._emit_expr(node.expr, merged)

        if isinstance(node, NumberNode):
            v = node.value
            return str(int(v)) if isinstance(v, float) and v == int(v) else str(v)

        if isinstance(node, IdentifierNode):
            return node.name

        if isinstance(node, RangeNode):
            return self._emit_range(node, intents)

        if isinstance(node, ArrayNode):
            elems = ", ".join(self._emit_expr(e, intents) for e in node.elements)
            return f"[{elems}]"

        if isinstance(node, LambdaNode):
            body = self._emit_expr(node.body)
            return f"lambda {node.param}: {body}"

        if isinstance(node, PipeNode):
            return self._emit_pipe(node, intents)

        if isinstance(node, MapNode):
            return self._emit_map(node, intents)

        if isinstance(node, FilterNode):
            return self._emit_filter(node, intents)

        if isinstance(node, ReduceNode):
            return self._emit_reduce(node, intents)

        if isinstance(node, BinaryOpNode):
            left  = self._emit_expr(node.left)
            right = self._emit_expr(node.right)
            op    = node.op
            # Wrap in parens if needed for precedence clarity
            return f"{self._maybe_paren(node.left, left, op)} {op} {self._maybe_paren(node.right, right, op)}"

        if isinstance(node, UnaryOpNode):
            operand = self._emit_expr(node.operand)
            if node.op == 'not':
                return f"not {operand}"
            return f"{node.op}{operand}"

        raise CodeGenError(f"Unknown AST node type: {type(node).__name__}", getattr(node, 'line', 0))

    def _maybe_paren(self, node: ASTNode, code: str, parent_op: str) -> str:
        """Add parentheses around binary sub-expressions with lower precedence."""
        if isinstance(node, BinaryOpNode):
            prec = {'+': 1, '-': 1, '*': 2, '/': 2, '%': 2,
                    'and': 0, 'or': 0,
                    '==': 0, '!=': 0, '<': 0, '>': 0, '<=': 0, '>=': 0}
            if prec.get(node.op, 0) < prec.get(parent_op, 0):
                return f"({code})"
        return code

    # ------------------------------------------------------------------ range

    def _emit_range(self, node: RangeNode, intents: set) -> str:
        start = self._emit_expr(node.start)
        end   = self._emit_expr(node.end)
        # end is inclusive → range(start, end+1)
        if isinstance(node.end, NumberNode):
            end_expr = str(node.end.value + 1) if isinstance(node.end.value, int) else f"{end} + 1"
        else:
            end_expr = f"{end} + 1"

        range_call = f"range({start}, {end_expr})"

        # $mem → keep as generator (range object)
        if 'mem' in intents:
            return range_call
        return f"list({range_call})"

    # ------------------------------------------------------------------ pipe

    def _emit_pipe(self, node: PipeNode, intents: set) -> str:
        """
        Pipe flattening: collect the full pipe chain, then emit as nested calls.
        a |> f |> g  →  g(f(a))
        """
        chain = self._collect_pipe_chain(node)
        # chain[0] is innermost value, rest are functions
        result = self._emit_expr(chain[0], intents)
        for fn in chain[1:]:
            fn_code = self._emit_expr(fn, intents)
            result = f"{fn_code}({result})"
        return result

    def _collect_pipe_chain(self, node: ASTNode) -> list:
        """Return [value, fn1, fn2, ...] in left-to-right order."""
        if isinstance(node, PipeNode):
            left_chain = self._collect_pipe_chain(node.left)
            return left_chain + [node.right]
        return [node]

    # ------------------------------------------------------------------ map / filter / reduce

    def _emit_map(self, node: MapNode, intents: set) -> str:
        array   = self._emit_expr(node.array, intents)
        lam     = self._emit_expr(node.lambda_)
        if 'mem' in intents:
            return f"map({lam}, {array})"
        return f"list(map({lam}, {array}))"

    def _emit_filter(self, node: FilterNode, intents: set) -> str:
        array = self._emit_expr(node.array, intents)
        lam   = self._emit_expr(node.lambda_)
        if 'mem' in intents:
            return f"filter({lam}, {array})"
        return f"list(filter({lam}, {array}))"

    def _emit_reduce(self, node: ReduceNode, intents: set) -> str:
        array = self._emit_expr(node.array, intents)
        op    = node.operator

        # Optimization: reduce(+) → sum()
        if op == '+':
            return f"sum({array})"

        # map operator names to lambda expressions
        op_map = {
            '-':   'lambda a, b: a - b',
            '*':   'lambda a, b: a * b',
            '/':   'lambda a, b: a / b',
            'min': 'lambda a, b: a if a < b else b',
            'max': 'lambda a, b: a if a > b else b',
        }
        lam = op_map.get(op)
        if lam is None:
            raise CodeGenError(f"Unsupported reduce operator: {op}", node.line)
        return f"reduce({lam}, {array})"

    # ------------------------------------------------------------------ util

    def _check_needs_reduce(self, program: ProgramNode) -> bool:
        """Walk the AST to see if reduce is used (and not just for +)."""
        return self._any_reduce(program)

    def _any_reduce(self, node: ASTNode) -> bool:
        if isinstance(node, ReduceNode) and node.operator != '+':
            return True
        for attr in vars(node).values():
            if isinstance(attr, ASTNode) and self._any_reduce(attr):
                return True
            if isinstance(attr, list):
                for item in attr:
                    if isinstance(item, ASTNode) and self._any_reduce(item):
                        return True
        return False
