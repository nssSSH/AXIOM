"""
AXIOM v0.1 - Recursive Descent Parser
Converts a token stream into a typed AST.
"""

from typing import List, Optional
from .lexer import Token, TokenType, LexerError
from .ast_nodes import (
    ProgramNode, AssignmentNode, PrintNode, IntentNode,
    PipeNode, MapNode, FilterNode, ReduceNode, LambdaNode,
    RangeNode, ArrayNode, IdentifierNode, NumberNode,
    BinaryOpNode, UnaryOpNode, ASTNode
)


class ParseError(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"[ParseError] Line {line}: {message}")
        self.line = line


class Parser:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------ helpers

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _peek2(self) -> Optional[Token]:
        if self._pos + 1 < len(self._tokens):
            return self._tokens[self._pos + 1]
        return None

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, ttype: TokenType) -> Token:
        tok = self._peek()
        if tok.type != ttype:
            raise ParseError(
                f"Expected {ttype.name} but got {tok.type.name} ({tok.value!r})",
                tok.line
            )
        return self._advance()

    def _match(self, *types: TokenType) -> bool:
        return self._peek().type in types

    def _at_statement_boundary(self) -> bool:
        """True if we've reached EOF or a token that starts a new statement."""
        tok = self._peek()
        return tok.type == TokenType.EOF

    # ------------------------------------------------------------------ public

    def parse(self) -> ProgramNode:
        stmts = []
        while not self._match(TokenType.EOF):
            stmts.append(self._parse_statement())
        node = ProgramNode(statements=stmts)
        node.line = 1
        return node

    # ------------------------------------------------------------------ statements

    def _parse_statement(self) -> ASTNode:
        tok = self._peek()

        # print statement
        if tok.type == TokenType.KEYWORD and tok.value == 'print':
            return self._parse_print()

        # assignment: IDENTIFIER -> expr
        if tok.type == TokenType.IDENTIFIER and self._peek2() and self._peek2().type == TokenType.ARROW:
            return self._parse_assignment()

        # bare expression statement
        return self._parse_expression()

    def _parse_print(self) -> PrintNode:
        tok = self._advance()  # consume 'print'
        value = self._parse_expression()
        node = PrintNode(value=value)
        node.line = tok.line
        return node

    def _parse_assignment(self) -> AssignmentNode:
        name_tok = self._advance()           # IDENTIFIER
        self._expect(TokenType.ARROW)        # ->
        value = self._parse_expression()
        node = AssignmentNode(name=name_tok.value, value=value)
        node.line = name_tok.line
        return node

    # ------------------------------------------------------------------ expressions

    def _parse_expression(self) -> ASTNode:
        expr = self._parse_pipe_expr()

        # Intent modifier: expr $modifier
        if self._match(TokenType.DOLLAR):
            dollar_tok = self._advance()
            mod_tok = self._expect(TokenType.IDENTIFIER)
            node = IntentNode(expr=expr, modifier=mod_tok.value)
            node.line = dollar_tok.line
            return node

        return expr

    def _parse_pipe_expr(self) -> ASTNode:
        left = self._parse_logical_expr()

        while self._match(TokenType.PIPE):
            pipe_tok = self._advance()  # consume |>
            right = self._parse_logical_expr()
            node = PipeNode(left=left, right=right)
            node.line = pipe_tok.line
            left = node

        return left

    def _parse_logical_expr(self) -> ASTNode:
        # Check for map, filter, reduce first (they bind to a primary on the left)
        left = self._parse_comparison()

        while True:
            if self._match(TokenType.MAP):
                at_tok = self._advance()
                lam = self._parse_lambda()
                node = MapNode(array=left, lambda_=lam)
                node.line = at_tok.line
                left = node

            elif self._match(TokenType.FILTER):
                q_tok = self._advance()
                lam = self._parse_lambda()
                node = FilterNode(array=left, lambda_=lam)
                node.line = q_tok.line
                left = node

            elif self._match(TokenType.REDUCE):
                hash_tok = self._advance()
                op_tok = self._peek()
                if op_tok.type not in (TokenType.OPERATOR,):
                    raise ParseError(
                        f"Expected reduce operator (+,-,*,/,min,max) but got {op_tok.value!r}",
                        op_tok.line
                    )
                self._advance()
                node = ReduceNode(array=left, operator=op_tok.value)
                node.line = hash_tok.line
                left = node

            elif self._match(TokenType.AND):
                op_tok = self._advance()
                right = self._parse_comparison()
                node = BinaryOpNode(left=left, op='and', right=right)
                node.line = op_tok.line
                left = node

            elif self._match(TokenType.OR):
                op_tok = self._advance()
                right = self._parse_comparison()
                node = BinaryOpNode(left=left, op='or', right=right)
                node.line = op_tok.line
                left = node

            else:
                break

        return left

    def _parse_comparison(self) -> ASTNode:
        left = self._parse_additive()

        _CMP = {
            TokenType.EQ:  '==',
            TokenType.NEQ: '!=',
            TokenType.LTE: '<=',
            TokenType.GTE: '>=',
            TokenType.LT:  '<',
            TokenType.GT:  '>',
        }
        while self._peek().type in _CMP:
            op_tok = self._advance()
            right = self._parse_additive()
            node = BinaryOpNode(left=left, op=_CMP[op_tok.type], right=right)
            node.line = op_tok.line
            left = node

        return left

    def _parse_additive(self) -> ASTNode:
        left = self._parse_multiplicative()

        while self._match(TokenType.OPERATOR) and self._peek().value in ('+', '-'):
            op_tok = self._advance()
            right = self._parse_multiplicative()
            node = BinaryOpNode(left=left, op=op_tok.value, right=right)
            node.line = op_tok.line
            left = node

        return left

    def _parse_multiplicative(self) -> ASTNode:
        left = self._parse_unary()

        while (
            (self._match(TokenType.OPERATOR) and self._peek().value in ('*', '/'))
            or self._match(TokenType.MOD)
        ):
            op_tok = self._advance()
            right = self._parse_unary()
            node = BinaryOpNode(left=left, op=op_tok.value, right=right)
            node.line = op_tok.line
            left = node

        return left

    def _parse_unary(self) -> ASTNode:
        if self._match(TokenType.NOT):
            op_tok = self._advance()
            operand = self._parse_unary()
            node = UnaryOpNode(op='not', operand=operand)
            node.line = op_tok.line
            return node
        if self._match(TokenType.OPERATOR) and self._peek().value == '-':
            op_tok = self._advance()
            operand = self._parse_unary()
            node = UnaryOpNode(op='-', operand=operand)
            node.line = op_tok.line
            return node
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        tok = self._peek()

        # Parenthesised expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr

        # Lambda literal
        if tok.type == TokenType.LBRACE:
            return self._parse_lambda()

        # Array literal
        if tok.type == TokenType.LBRACKET:
            return self._parse_array()

        # Number (possibly followed by range op)
        if tok.type == TokenType.NUMBER:
            num_tok = self._advance()
            num_node = self._make_number(num_tok)

            # Range: NUMBER .. NUMBER
            if self._match(TokenType.RANGE):
                self._advance()  # consume ..
                end_tok = self._expect(TokenType.NUMBER)
                end_node = self._make_number(end_tok)
                node = RangeNode(start=num_node, end=end_node)
                node.line = num_tok.line
                return node

            return num_node

        # Identifier or keyword-as-identifier (e.g. 'print' used as a function ref in pipe)
        if tok.type in (TokenType.IDENTIFIER, TokenType.KEYWORD):
            id_tok = self._advance()
            id_node = IdentifierNode(name=id_tok.value)
            id_node.line = id_tok.line

            if self._match(TokenType.RANGE):
                self._advance()
                end_tok = self._expect(TokenType.NUMBER)
                end_node = self._make_number(end_tok)
                node = RangeNode(start=id_node, end=end_node)
                node.line = id_tok.line
                return node

            return id_node

        raise ParseError(
            f"Unexpected token {tok.type.name} ({tok.value!r}) in expression",
            tok.line
        )

    def _parse_lambda(self) -> LambdaNode:
        open_tok = self._expect(TokenType.LBRACE)
        param_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.ARROW)
        body = self._parse_expression()
        self._expect(TokenType.RBRACE)
        node = LambdaNode(param=param_tok.value, body=body)
        node.line = open_tok.line
        return node

    def _parse_array(self) -> ArrayNode:
        open_tok = self._expect(TokenType.LBRACKET)
        elements = []
        if not self._match(TokenType.RBRACKET):
            elements.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                self._advance()
                elements.append(self._parse_expression())
        self._expect(TokenType.RBRACKET)
        node = ArrayNode(elements=elements)
        node.line = open_tok.line
        return node

    # ------------------------------------------------------------------ helpers

    def _make_number(self, tok: Token) -> NumberNode:
        val = float(tok.value) if '.' in tok.value else int(tok.value)
        node = NumberNode(value=val)
        node.line = tok.line
        return node
