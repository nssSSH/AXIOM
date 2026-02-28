"""
AXIOM v0.1 - Lexer
Tokenizes AXIOM source code into a flat token stream.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum, auto


class TokenType(Enum):
    # Literals
    NUMBER     = auto()
    IDENTIFIER = auto()
    # Operators / punctuation
    ARROW      = auto()   # ->
    RANGE      = auto()   # ..
    PIPE       = auto()   # |>
    MAP        = auto()   # @
    FILTER     = auto()   # ?
    REDUCE     = auto()   # #
    DOLLAR     = auto()   # $
    OPERATOR   = auto()   # + - * / min max
    # Brackets
    LBRACE     = auto()   # {
    RBRACE     = auto()   # }
    LBRACKET   = auto()   # [
    RBRACKET   = auto()   # ]
    LPAREN     = auto()   # (
    RPAREN     = auto()   # )
    COMMA      = auto()   # ,
    # Comparisons & logical (used inside lambdas)
    EQ         = auto()   # ==
    NEQ        = auto()   # !=
    LTE        = auto()   # <=
    GTE        = auto()   # >=
    LT         = auto()   # <
    GT         = auto()   # >
    AND        = auto()   # and
    OR         = auto()   # or
    NOT        = auto()   # not
    MOD        = auto()   # %
    # Keywords
    KEYWORD    = auto()   # print
    # Sentinel
    EOF        = auto()


KEYWORDS = {"print", "and", "or", "not", "min", "max"}
OPERATOR_KEYWORDS = {"min", "max"}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


class LexerError(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"[LexerError] Line {line}: {message}")
        self.line = line


# Token specification: ordered list of (TokenType, regex) pairs
_TOKEN_SPEC = [
    (TokenType.PIPE,       r'\|>'),
    (TokenType.ARROW,      r'->'),
    (TokenType.RANGE,      r'\.\.'),
    (TokenType.EQ,         r'=='),
    (TokenType.NEQ,        r'!='),
    (TokenType.LTE,        r'<='),
    (TokenType.GTE,        r'>='),
    (TokenType.LT,         r'<'),
    (TokenType.GT,         r'>'),
    (TokenType.NUMBER,     r'\d+(?:\.\d+)?'),
    (TokenType.IDENTIFIER, r'[A-Za-z_][A-Za-z0-9_]*'),
    (TokenType.MAP,        r'@'),
    (TokenType.FILTER,     r'\?'),
    (TokenType.REDUCE,     r'#'),
    (TokenType.DOLLAR,     r'\$'),
    (TokenType.OPERATOR,   r'[+\-*/]'),
    (TokenType.MOD,        r'%'),
    (TokenType.LBRACE,     r'\{'),
    (TokenType.RBRACE,     r'\}'),
    (TokenType.LBRACKET,   r'\['),
    (TokenType.RBRACKET,   r'\]'),
    (TokenType.LPAREN,     r'\('),
    (TokenType.RPAREN,     r'\)'),
    (TokenType.COMMA,      r','),
]

_MASTER_RE = re.compile(
    r'(?:' + '|'.join(f'(?P<T{i}>{spec[1]})' for i, spec in enumerate(_TOKEN_SPEC)) + r')',
    re.ASCII
)

_WHITESPACE_RE = re.compile(r'[ \t\r]+')
_COMMENT_RE    = re.compile(r'//[^\n]*')
_NEWLINE_RE    = re.compile(r'\n')


def tokenize(source: str) -> List[Token]:
    """
    Convert AXIOM source string into a list of Tokens.
    Raises LexerError on unrecognized characters.
    """
    tokens: List[Token] = []
    line = 1
    pos = 0
    length = len(source)

    while pos < length:
        # Skip whitespace (not newlines)
        m = _WHITESPACE_RE.match(source, pos)
        if m:
            pos = m.end()
            continue

        # Skip comments
        m = _COMMENT_RE.match(source, pos)
        if m:
            pos = m.end()
            continue

        # Newlines
        m = _NEWLINE_RE.match(source, pos)
        if m:
            line += 1
            pos = m.end()
            continue

        # Try all token patterns
        m = _MASTER_RE.match(source, pos)
        if not m:
            raise LexerError(f"Unexpected character: {source[pos]!r}", line)

        raw = m.group(0)
        # Determine which group matched
        tok_type = None
        for i, (ttype, _) in enumerate(_TOKEN_SPEC):
            if m.group(f'T{i}') is not None:
                tok_type = ttype
                break

        # Reclassify identifiers that are keywords or operators
        if tok_type == TokenType.IDENTIFIER:
            if raw == 'print':
                tok_type = TokenType.KEYWORD
            elif raw in ('and',):
                tok_type = TokenType.AND
            elif raw in ('or',):
                tok_type = TokenType.OR
            elif raw in ('not',):
                tok_type = TokenType.NOT
            elif raw in ('min', 'max'):
                tok_type = TokenType.OPERATOR  # reduce operators

        tokens.append(Token(tok_type, raw, line))
        pos = m.end()

    tokens.append(Token(TokenType.EOF, '', line))
    return tokens
