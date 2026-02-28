"""
AXIOM v0.1 - AST Node Definitions
Backend-agnostic typed AST for the AXIOM compiler.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    line: int = 0


@dataclass
class ProgramNode(ASTNode):
    """Root node of the program."""
    statements: List[ASTNode] = field(default_factory=list)


@dataclass
class AssignmentNode(ASTNode):
    """identifier -> expression"""
    name: str = ""
    value: ASTNode = None


@dataclass
class PrintNode(ASTNode):
    """print expression"""
    value: ASTNode = None


@dataclass
class IntentNode(ASTNode):
    """expression $modifier — wraps an expression with a compile-time hint."""
    expr: ASTNode = None
    modifier: str = ""


@dataclass
class PipeNode(ASTNode):
    """left |> right"""
    left: ASTNode = None
    right: ASTNode = None


@dataclass
class MapNode(ASTNode):
    """array @ {param -> expr}"""
    array: ASTNode = None
    lambda_: ASTNode = None


@dataclass
class FilterNode(ASTNode):
    """array ? {param -> condition}"""
    array: ASTNode = None
    lambda_: ASTNode = None


@dataclass
class ReduceNode(ASTNode):
    """array # operator"""
    array: ASTNode = None
    operator: str = ""


@dataclass
class LambdaNode(ASTNode):
    """{param -> expression}"""
    param: str = ""
    body: ASTNode = None


@dataclass
class RangeNode(ASTNode):
    """a..b (inclusive)"""
    start: ASTNode = None
    end: ASTNode = None


@dataclass
class ArrayNode(ASTNode):
    """[e1, e2, ...]"""
    elements: List[ASTNode] = field(default_factory=list)


@dataclass
class IdentifierNode(ASTNode):
    """A variable reference."""
    name: str = ""


@dataclass
class NumberNode(ASTNode):
    """A numeric literal (int or float)."""
    value: Any = 0


@dataclass
class BinaryOpNode(ASTNode):
    """Generic binary operation for expressions inside lambdas."""
    left: ASTNode = None
    op: str = ""
    right: ASTNode = None


@dataclass
class UnaryOpNode(ASTNode):
    """Unary operation."""
    op: str = ""
    operand: ASTNode = None
