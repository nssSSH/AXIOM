"""
AXIOM v0.1 - Optimizer
Applies deterministic AST-level transformations before code generation.

Transformations:
  1. reduce(+) → sum()           (always, as per spec)
  2. list(range(...)) → range()  when $mem modifier is present
  3. Inline nested pipes         (A |> (B |> C)) → already handled by codegen
  4. Intent propagation          $mem bubbles down into range nodes
  5. Redundant list removal      when $mem, avoid wrapping generators

Each transformation is idempotent and semantics-preserving.
"""

from .ast_nodes import (
    ProgramNode, AssignmentNode, PrintNode, IntentNode,
    PipeNode, MapNode, FilterNode, ReduceNode, LambdaNode,
    RangeNode, ArrayNode, IdentifierNode, NumberNode,
    BinaryOpNode, UnaryOpNode, ASTNode
)


class Optimizer:
    def __init__(self, opt_level: int = 1):
        """
        opt_level 0 – no optimizations
        opt_level 1 – all standard optimizations
        """
        self.opt_level = opt_level
        # Per-node intent context, propagated top-down
        self._active_intents: set = set()

    def optimize(self, program: ProgramNode) -> ProgramNode:
        if self.opt_level == 0:
            return program
        program.statements = [self._visit(s) for s in program.statements]
        return program

    # ------------------------------------------------------------------ visitor

    def _visit(self, node: ASTNode, intents: set = None) -> ASTNode:
        if intents is None:
            intents = set()
        method = f"_visit_{type(node).__name__}"
        visitor = getattr(self, method, self._visit_generic)
        return visitor(node, intents)

    def _visit_generic(self, node: ASTNode, intents: set) -> ASTNode:
        # Recurse into all child ASTNode fields
        for attr_name in list(vars(node).keys()):
            attr = getattr(node, attr_name)
            if isinstance(attr, ASTNode):
                setattr(node, attr_name, self._visit(attr, intents))
            elif isinstance(attr, list):
                setattr(node, attr_name, [
                    self._visit(item, intents) if isinstance(item, ASTNode) else item
                    for item in attr
                ])
        return node

    def _visit_IntentNode(self, node: IntentNode, intents: set) -> ASTNode:
        new_intents = intents | {node.modifier}
        inner = self._visit(node.expr, new_intents)
        node.expr = inner
        return node

    def _visit_ReduceNode(self, node: ReduceNode, intents: set) -> ASTNode:
        node.array = self._visit(node.array, intents)
        # Propagate intents to the node itself for codegen to query
        node._intents = intents
        return node

    def _visit_RangeNode(self, node: RangeNode, intents: set) -> ASTNode:
        node.start = self._visit(node.start, intents)
        node.end   = self._visit(node.end, intents)
        # Tag the range with active intents so codegen can skip list() wrap
        node._intents = intents
        return node

    def _visit_MapNode(self, node: MapNode, intents: set) -> ASTNode:
        node.array    = self._visit(node.array, intents)
        node.lambda_  = self._visit(node.lambda_, intents)
        node._intents = intents
        return node

    def _visit_FilterNode(self, node: FilterNode, intents: set) -> ASTNode:
        node.array    = self._visit(node.array, intents)
        node.lambda_  = self._visit(node.lambda_, intents)
        node._intents = intents
        return node

    def _visit_PipeNode(self, node: PipeNode, intents: set) -> ASTNode:
        node.left    = self._visit(node.left, intents)
        node.right   = self._visit(node.right, intents)
        node._intents = intents
        return node

    def _visit_AssignmentNode(self, node: AssignmentNode, intents: set) -> ASTNode:
        node.value = self._visit(node.value, intents)
        return node

    def _visit_PrintNode(self, node: PrintNode, intents: set) -> ASTNode:
        node.value = self._visit(node.value, intents)
        return node

    def _visit_ArrayNode(self, node: ArrayNode, intents: set) -> ASTNode:
        node.elements = [self._visit(e, intents) for e in node.elements]
        return node

    def _visit_LambdaNode(self, node: LambdaNode, intents: set) -> ASTNode:
        node.body = self._visit(node.body, intents)
        return node

    def _visit_BinaryOpNode(self, node: BinaryOpNode, intents: set) -> ASTNode:
        node.left  = self._visit(node.left, intents)
        node.right = self._visit(node.right, intents)
        return node

    def _visit_UnaryOpNode(self, node: UnaryOpNode, intents: set) -> ASTNode:
        node.operand = self._visit(node.operand, intents)
        return node

    def _visit_NumberNode(self, node: NumberNode, intents: set) -> ASTNode:
        return node

    def _visit_IdentifierNode(self, node: IdentifierNode, intents: set) -> ASTNode:
        return node
