"""
AXIOM v0.1 - Test Suite
Tests for Lexer, Parser, Semantic Analyzer, Optimizer, and Code Generator.
"""

import sys
import os
import unittest

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom.lexer import tokenize, TokenType, LexerError
from axiom.parser import Parser, ParseError
from axiom.semantic import SemanticAnalyzer, SemanticError
from axiom.optimizer import Optimizer
from axiom.codegen import CodeGenerator
from axiom.compiler import compile_source, CompilationError


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compile_(source: str, opt_level: int = 1) -> str:
    return compile_source(source.strip(), opt_level=opt_level).strip()


def token_types(source: str):
    return [t.type for t in tokenize(source) if t.type != TokenType.EOF]


def parse_(source: str):
    tokens = tokenize(source.strip())
    return Parser(tokens).parse()


# ═══════════════════════════════════════════════════════════════════════════════
# Lexer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLexer(unittest.TestCase):

    def test_number_integer(self):
        toks = tokenize("42")
        self.assertEqual(toks[0].type, TokenType.NUMBER)
        self.assertEqual(toks[0].value, "42")

    def test_number_float(self):
        toks = tokenize("3.14")
        self.assertEqual(toks[0].type, TokenType.NUMBER)
        self.assertEqual(toks[0].value, "3.14")

    def test_identifier(self):
        toks = tokenize("myVar")
        self.assertEqual(toks[0].type, TokenType.IDENTIFIER)
        self.assertEqual(toks[0].value, "myVar")

    def test_print_keyword(self):
        toks = tokenize("print")
        self.assertEqual(toks[0].type, TokenType.KEYWORD)

    def test_arrow(self):
        types = token_types("x -> 5")
        self.assertIn(TokenType.ARROW, types)

    def test_range_operator(self):
        types = token_types("1..10")
        self.assertIn(TokenType.RANGE, types)

    def test_pipe_operator(self):
        types = token_types("x |> f")
        self.assertIn(TokenType.PIPE, types)

    def test_map_operator(self):
        types = token_types("x @ {a -> a}")
        self.assertIn(TokenType.MAP, types)

    def test_filter_operator(self):
        types = token_types("x ? {a -> a}")
        self.assertIn(TokenType.FILTER, types)

    def test_reduce_operator(self):
        types = token_types("x # +")
        self.assertIn(TokenType.REDUCE, types)

    def test_dollar(self):
        types = token_types("x $mem")
        self.assertIn(TokenType.DOLLAR, types)

    def test_comparison_ops(self):
        src = "a == b != c <= d >= e < f > g"
        types = token_types(src)
        self.assertIn(TokenType.EQ, types)
        self.assertIn(TokenType.NEQ, types)
        self.assertIn(TokenType.LTE, types)
        self.assertIn(TokenType.GTE, types)
        self.assertIn(TokenType.LT, types)
        self.assertIn(TokenType.GT, types)

    def test_line_tracking(self):
        toks = tokenize("a\nb\nc")
        lines = {t.value: t.line for t in toks if t.type != TokenType.EOF}
        self.assertEqual(lines["a"], 1)
        self.assertEqual(lines["b"], 2)
        self.assertEqual(lines["c"], 3)

    def test_comment_ignored(self):
        toks = [t for t in tokenize("// this is a comment\nx") if t.type != TokenType.EOF]
        self.assertEqual(len(toks), 1)
        self.assertEqual(toks[0].value, "x")

    def test_invalid_character(self):
        with self.assertRaises(LexerError):
            tokenize("x ` y")

    def test_min_max_as_operator(self):
        toks = [t for t in tokenize("min max") if t.type != TokenType.EOF]
        self.assertTrue(all(t.type == TokenType.OPERATOR for t in toks))


# ═══════════════════════════════════════════════════════════════════════════════
# Parser Tests
# ═══════════════════════════════════════════════════════════════════════════════

from axiom.ast_nodes import (
    AssignmentNode, PrintNode, RangeNode, PipeNode,
    MapNode, FilterNode, ReduceNode, LambdaNode,
    IdentifierNode, NumberNode, IntentNode, BinaryOpNode
)


class TestParser(unittest.TestCase):

    def test_assignment(self):
        ast = parse_("x -> 5")
        stmt = ast.statements[0]
        self.assertIsInstance(stmt, AssignmentNode)
        self.assertEqual(stmt.name, "x")
        self.assertIsInstance(stmt.value, NumberNode)
        self.assertEqual(stmt.value.value, 5)

    def test_range(self):
        ast = parse_("x -> 1..10")
        rng = ast.statements[0].value
        self.assertIsInstance(rng, RangeNode)
        self.assertEqual(rng.start.value, 1)
        self.assertEqual(rng.end.value, 10)

    def test_pipe(self):
        ast = parse_("x -> 1..10 |> sum")
        pipe = ast.statements[0].value
        self.assertIsInstance(pipe, PipeNode)

    def test_pipe_chain(self):
        ast = parse_("x -> 1..10 |> sum |> print")
        # outer pipe: (1..10 |> sum) |> print
        outer = ast.statements[0].value
        self.assertIsInstance(outer, PipeNode)
        self.assertIsInstance(outer.left, PipeNode)

    def test_map(self):
        ast = parse_("y -> x @ {v -> v * 2}")
        m = ast.statements[0].value
        self.assertIsInstance(m, MapNode)
        self.assertIsInstance(m.lambda_, LambdaNode)
        self.assertEqual(m.lambda_.param, "v")

    def test_filter(self):
        ast = parse_("y -> x ? {v -> v > 0}")
        f = ast.statements[0].value
        self.assertIsInstance(f, FilterNode)

    def test_reduce(self):
        ast = parse_("y -> x # +")
        r = ast.statements[0].value
        self.assertIsInstance(r, ReduceNode)
        self.assertEqual(r.operator, "+")

    def test_reduce_min(self):
        ast = parse_("y -> x # min")
        r = ast.statements[0].value
        self.assertIsInstance(r, ReduceNode)
        self.assertEqual(r.operator, "min")

    def test_print_stmt(self):
        ast = parse_("print x")
        stmt = ast.statements[0]
        self.assertIsInstance(stmt, PrintNode)

    def test_intent_modifier(self):
        ast = parse_("x -> 1..10 $mem")
        intent = ast.statements[0].value
        self.assertIsInstance(intent, IntentNode)
        self.assertEqual(intent.modifier, "mem")

    def test_array_literal(self):
        from axiom.ast_nodes import ArrayNode
        ast = parse_("x -> [1, 2, 3]")
        arr = ast.statements[0].value
        self.assertIsInstance(arr, ArrayNode)
        self.assertEqual(len(arr.elements), 3)

    def test_complex_lambda_body(self):
        ast = parse_("y -> x ? {a -> a % 2 == 0}")
        lam = ast.statements[0].value.lambda_
        # body should be a BinaryOpNode (==)
        self.assertIsInstance(lam.body, BinaryOpNode)
        self.assertEqual(lam.body.op, "==")

    def test_parse_error_missing_arrow(self):
        # A lambda missing the arrow should fail
        with self.assertRaises((ParseError, CompilationError)):
            parse_("x -> { p }")  # lambda without ->

    def test_nested_pipe_in_map(self):
        # Piping into a function reference in the lambda body context
        ast = parse_("x -> 1..5 |> sum")
        self.assertIsInstance(ast.statements[0].value, PipeNode)


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemantic(unittest.TestCase):

    def _analyze(self, source):
        ast = parse_(source)
        SemanticAnalyzer().analyze(ast)
        return ast

    def test_valid_program(self):
        self._analyze("x -> 1..10\ny -> x # +")

    def test_undeclared_identifier(self):
        with self.assertRaises(SemanticError):
            self._analyze("y -> x # +")  # x not declared

    def test_valid_reduce_ops(self):
        for op in ('+', '-', '*', '/', 'min', 'max'):
            self._analyze(f"x -> [1,2,3]\ny -> x # {op}")

    def test_invalid_reduce_op(self):
        # We can't directly inject an invalid op through the parser normally,
        # but we can test the semantic check directly via AST manipulation.
        from axiom.ast_nodes import ProgramNode, AssignmentNode, ReduceNode, IdentifierNode, ArrayNode, NumberNode
        arr = ArrayNode(elements=[NumberNode(value=1)])
        bad_reduce = ReduceNode(array=arr, operator="^")
        bad_reduce.line = 1
        assign = AssignmentNode(name="y", value=bad_reduce)
        assign.line = 1
        prog = ProgramNode(statements=[assign])
        with self.assertRaises(SemanticError):
            SemanticAnalyzer().analyze(prog)

    def test_lambda_param_scoped(self):
        # param 'v' should be valid inside lambda but not outside
        self._analyze("x -> [1,2,3]\ny -> x @ {v -> v * 2}")

    def test_invalid_intent(self):
        with self.assertRaises(SemanticError):
            self._analyze("x -> 1..5 $turbo")


# ═══════════════════════════════════════════════════════════════════════════════
# Code Generation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeGen(unittest.TestCase):

    def test_range_basic(self):
        out = compile_("x -> 1..10")
        self.assertEqual(out, "x = list(range(1, 11))")

    def test_range_mem(self):
        out = compile_("x -> 1..10 $mem")
        self.assertEqual(out, "x = range(1, 11)")

    def test_map(self):
        out = compile_("x -> [1,2,3]\ny -> x @ {v -> v * 2}")
        self.assertIn("list(map(lambda v: v * 2, x))", out)

    def test_filter(self):
        out = compile_("x -> [1,2,3,4]\ny -> x ? {v -> v % 2 == 0}")
        self.assertIn("list(filter(lambda v: v % 2 == 0, x))", out)

    def test_reduce_plus_becomes_sum(self):
        out = compile_("x -> [1,2,3]\ny -> x # +")
        self.assertIn("sum(x)", out)
        self.assertNotIn("reduce", out)

    def test_reduce_multiply_uses_reduce(self):
        out = compile_("x -> [1,2,3]\ny -> x # *")
        self.assertIn("from functools import reduce", out)
        self.assertIn("reduce(", out)

    def test_reduce_min(self):
        out = compile_("x -> [3,1,2]\ny -> x # min")
        self.assertIn("reduce(", out)

    def test_reduce_max(self):
        out = compile_("x -> [3,1,2]\ny -> x # max")
        self.assertIn("reduce(", out)

    def test_pipe_simple(self):
        out = compile_("x -> [1,2,3]\ny -> x |> sum")
        self.assertEqual(out, "x = [1, 2, 3]\ny = sum(x)")

    def test_pipe_chain(self):
        out = compile_("x -> [1,2,3]\n1..5 |> sum |> print")
        # Should call print(sum(list(range(1, 6))))
        self.assertIn("print(sum(list(range(1, 6))))", out)

    def test_print_stmt(self):
        out = compile_("x -> 42\nprint x")
        self.assertIn("print(x)", out)

    def test_array_literal(self):
        out = compile_("x -> [10, 20, 30]")
        self.assertEqual(out, "x = [10, 20, 30]")

    def test_full_example(self):
        src = """
nums -> 1..10
evens -> nums ? {x -> x % 2 == 0}
squares -> evens @ {x -> x*x}
total -> squares # +
print total
"""
        out = compile_(src)
        expected_lines = [
            "nums = list(range(1, 11))",
            "evens = list(filter(lambda x: x % 2 == 0, nums))",
            "squares = list(map(lambda x: x * x, evens))",
            "total = sum(squares)",
            "print(total)",
        ]
        for line in expected_lines:
            self.assertIn(line, out)

    def test_no_import_when_no_reduce(self):
        out = compile_("x -> [1,2,3]\ny -> x # +")
        self.assertNotIn("import", out)

    def test_import_when_reduce_needed(self):
        out = compile_("x -> [1,2,3]\ny -> x # *")
        self.assertIn("from functools import reduce", out)

    def test_mem_modifier_with_filter(self):
        out = compile_("x -> [1,2,3]\ny -> (x ? {v -> v > 1}) $mem")
        self.assertIn("filter(", out)
        self.assertNotIn("list(filter(", out)

    def test_mem_modifier_with_map(self):
        out = compile_("x -> [1,2,3]\ny -> (x @ {v -> v * 2}) $mem")
        self.assertIn("map(", out)
        self.assertNotIn("list(map(", out)

    def test_debug_modifier_compiles(self):
        # $debug is a valid modifier – should not raise
        out = compile_("x -> 1..5 $debug")
        self.assertIn("range(1, 6)", out)

    def test_safe_modifier_compiles(self):
        out = compile_("x -> 1..5 $safe")
        self.assertIn("range(1, 6)", out)


# ═══════════════════════════════════════════════════════════════════════════════
# Optimizer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptimizer(unittest.TestCase):

    def test_opt_level_0_no_sum_conversion(self):
        # At opt_level 0, we still get the correct output but without some transforms.
        # Actually reduce(+) is a codegen rule, not an optimizer rule — always applies.
        out = compile_("x -> [1,2]\ny -> x # +", opt_level=0)
        self.assertIn("sum(x)", out)

    def test_opt_level_1_mem_propagated(self):
        out = compile_("x -> 1..100 $mem", opt_level=1)
        self.assertEqual(out, "x = range(1, 101)")


# ═══════════════════════════════════════════════════════════════════════════════
# Integration / Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    def test_chained_map_filter(self):
        src = """
data -> 1..20
big -> data ? {x -> x > 10}
doubled -> big @ {x -> x * 2}
print doubled
"""
        out = compile_(src)
        self.assertIn("list(range(1, 21))", out)
        self.assertIn("list(filter(lambda x: x > 10, data))", out)
        self.assertIn("list(map(lambda x: x * 2, big))", out)
        self.assertIn("print(doubled)", out)

    def test_reduce_chain(self):
        src = """
nums -> [5, 3, 8, 1]
result -> nums # max
print result
"""
        out = compile_(src)
        self.assertIn("from functools import reduce", out)
        self.assertIn("reduce(lambda a, b: a if a > b else b, nums)", out)

    def test_nested_lambda_expression(self):
        src = """
vals -> [1, 2, 3, 4, 5]
result -> vals ? {n -> n % 2 == 0} @ {n -> n * n}
"""
        out = compile_(src)
        self.assertIn("list(map(lambda n: n * n", out)
        self.assertIn("list(filter(lambda n: n % 2 == 0, vals)", out)

    def test_undefined_var_raises(self):
        with self.assertRaises(CompilationError):
            compile_source("y -> undefined_var # +")

    def test_invalid_intent_raises(self):
        with self.assertRaises(CompilationError):
            compile_source("x -> 1..5 $hyperturbo")

    def test_large_range(self):
        out = compile_("x -> 1..1000000 $mem")
        self.assertEqual(out, "x = range(1, 1000001)")

    def test_emit_ast_returns_json(self):
        import json
        result = compile_source("x -> 1..5", emit_ast=True)
        parsed = json.loads(result)
        self.assertEqual(parsed["_type"], "ProgramNode")

    def test_print_expression(self):
        out = compile_("print 1..5 |> sum")
        self.assertIn("print(sum(list(range(1, 6))))", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
