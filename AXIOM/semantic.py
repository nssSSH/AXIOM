"""
AXIOM v0.1 - Semantic Analyzer
Validates the AST:
  - Identifiers declared before use
  - Reduce operators are valid
  - Lambda parameters are properly scoped
  - Intent modifiers are recognized
Attaches semantic metadata in-place.
"""

from typing import Set
from .ast_nodes import (
    ProgramNode, AssignmentNode, PrintNode, IntentNode,
    PipeNode, MapNode, FilterNode, ReduceNode, LambdaNode,
    RangeNode, ArrayNode, IdentifierNode, NumberNode,
    BinaryOpNode, UnaryOpNode, ASTNode
)

VALID_REDUCE_OPS = {'+', '-', '*', '/', 'min', 'max'}
VALID_INTENTS    = {'fast', 'mem', 'safe', 'debug'}

# Python built-in functions that are always in scope
PYTHON_BUILTINS = {
    'print', 'sum', 'min', 'max', 'len', 'abs', 'round', 'range',
    'list', 'tuple', 'set', 'dict', 'int', 'float', 'str', 'bool',
    'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter',
    'any', 'all', 'input', 'type', 'isinstance', 'repr',
}


class SemanticError(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"[SemanticError] Line {line}: {message}")
        self.line = line


class SemanticAnalyzer:
    def __init__(self):
        self._declared: Set[str] = set()
        self._lambda_scope: Set[str] = set()

    def analyze(self, program: ProgramNode) -> None:
        for stmt in program.statements:
            self._visit(stmt)

    # ------------------------------------------------------------------ visitor

    def _visit(self, node: ASTNode) -> None:
        method = f"_visit_{type(node).__name__}"
        visitor = getattr(self, method, self._visit_generic)
        visitor(node)

    def _visit_generic(self, node: ASTNode) -> None:
        # Fallback: recurse into all ASTNode fields
        for attr in vars(node).values():
            if isinstance(attr, ASTNode):
                self._visit(attr)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, ASTNode):
                        self._visit(item)

    def _visit_ProgramNode(self, node: ProgramNode) -> None:
        for stmt in node.statements:
            self._visit(stmt)

    def _visit_AssignmentNode(self, node: AssignmentNode) -> None:
        self._visit(node.value)
        self._declared.add(node.name)

    def _visit_PrintNode(self, node: PrintNode) -> None:
        self._visit(node.value)

    def _visit_IntentNode(self, node: IntentNode) -> None:
        if node.modifier not in VALID_INTENTS:
            raise SemanticError(
                f"Unknown intent modifier '${node.modifier}'. "
                f"Valid: {', '.join('$'+m for m in sorted(VALID_INTENTS))}",
                node.line
            )
        self._visit(node.expr)

    def _visit_IdentifierNode(self, node: IdentifierNode) -> None:
        # Allow if declared globally, is in current lambda scope, or is a Python builtin
        if (node.name not in self._declared
                and node.name not in self._lambda_scope
                and node.name not in PYTHON_BUILTINS):
            raise SemanticError(
                f"Identifier '{node.name}' used before declaration",
                node.line
            )

    def _visit_NumberNode(self, node: NumberNode) -> None:
        pass  # always valid

    def _visit_RangeNode(self, node: RangeNode) -> None:
        self._visit(node.start)
        self._visit(node.end)

    def _visit_ArrayNode(self, node: ArrayNode) -> None:
        for elem in node.elements:
            self._visit(elem)

    def _visit_PipeNode(self, node: PipeNode) -> None:
        self._visit(node.left)
        self._visit(node.right)

    def _visit_MapNode(self, node: MapNode) -> None:
        self._visit(node.array)
        self._visit(node.lambda_)

    def _visit_FilterNode(self, node: FilterNode) -> None:
        self._visit(node.array)
        self._visit(node.lambda_)

    def _visit_ReduceNode(self, node: ReduceNode) -> None:
        if node.operator not in VALID_REDUCE_OPS:
            raise SemanticError(
                f"Invalid reduce operator '{node.operator}'. "
                f"Valid: {', '.join(sorted(VALID_REDUCE_OPS))}",
                node.line
            )
        self._visit(node.array)

    def _visit_LambdaNode(self, node: LambdaNode) -> None:
        # Push parameter into lambda scope
        prev = self._lambda_scope.copy()
        self._lambda_scope.add(node.param)
        self._visit(node.body)
        self._lambda_scope = prev

    def _visit_BinaryOpNode(self, node: BinaryOpNode) -> None:
        self._visit(node.left)
        self._visit(node.right)

    def _visit_UnaryOpNode(self, node: UnaryOpNode) -> None:
        self._visit(node.operand)
