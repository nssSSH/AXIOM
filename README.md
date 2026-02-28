# AXIOM v0.1

**Adaptive eXpressive Intent-Oriented Matrix Language**

AXIOM is an ASCII-only, array-first, expression-based DSL that compiles
deterministically to clean Python 3 code.

```
nums -> 1..10
evens -> nums ? {x -> x % 2 == 0}
squares -> evens @ {x -> x*x}
total -> squares # +
print total
```

Compiles to:

```python
nums = list(range(1, 11))
evens = list(filter(lambda x: x % 2 == 0, nums))
squares = list(map(lambda x: x * x, evens))
total = sum(squares)
print(total)
```

---

## Installation

```bash
pip install -e .
```

## Usage

```bash
axiom input.ax -o output.py
axiom input.ax --opt-level 0   # disable optimizations
axiom input.ax --emit-ast       # dump AST as JSON
axiom input.ax --debug          # verbose phase output
```

---

## Language Reference

### Assignment
```
identifier -> expression
```

### Range (inclusive)
```
1..10    →  list(range(1, 11))
```

### Pipe
```
value |> function           →  function(value)
1..10 |> sum |> print       →  print(sum(list(range(1, 11))))
```

### Map
```
array @ {x -> expr}         →  list(map(lambda x: expr, array))
```

### Filter
```
array ? {x -> condition}    →  list(filter(lambda x: condition, array))
```

### Reduce
```
array # +     →  sum(array)
array # *     →  reduce(lambda a,b: a*b, array)
array # min   →  reduce(lambda a,b: a if a<b else b, array)
array # max   →  reduce(lambda a,b: a if a>b else b, array)
```

### Array Literal
```
[1, 2, 3]    →  [1, 2, 3]
```

### Intent Modifiers (compile-time hints)

Modifiers go at the **end** of a full expression:

| Modifier | Effect |
|----------|--------|
| `$mem`   | Use generators instead of lists (avoids `list()` wrapper on ranges/map/filter) |
| `$fast`  | Prefer built-in functions |
| `$safe`  | (reserved: type assertions in future versions) |
| `$debug` | (reserved: debug injection in future versions) |

Example:
```
1..1000000 |> sum $mem   →  sum(range(1, 1000001))
```

### Comments
```
// This is a comment
```

---

## Architecture

```
axiom/
 ├── __init__.py      Public API
 ├── lexer.py         Phase 1: Tokenization
 ├── parser.py        Phase 2: Recursive descent parser → AST
 ├── ast_nodes.py     Typed dataclass AST nodes
 ├── semantic.py      Phase 3: Semantic validation
 ├── optimizer.py     Phase 4: AST-level optimizations
 ├── codegen.py       Phase 5: Python code generation
 ├── compiler.py      Pipeline orchestrator
 └── cli.py           Command-line interface
```

Each phase is independently testable. The AST is backend-agnostic,
designed to support future NumPy, parallel, or JIT backends.

---

## Running Tests

```bash
python tests.py
```

65 tests covering lexer, parser, semantic analysis, optimizer, and code generation.

---

## Design Principles

- **No `eval()` or `exec()`** — pure source-to-source compilation
- **Deterministic** — same input always produces identical output
- **Readable output** — generated Python is human-readable
- **Clear errors** — all errors include line numbers
- **Modular** — each compiler phase is independently replaceable
