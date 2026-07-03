# Example 2 — Quadratic

**Target formula:** `y = x^2 + 3`

Demonstrates the `squared` unary operator (`squared(x)` = `x^2`). The solver
finds the exact formula at complexity 4 with MAPE = 0%.

## Running

From this directory:

```bash
xcelerator-solver solver.toml
```

## What to expect

```
 Rank  Expression      Train MAPE %   Val MAPE %   Complexity
    1  3 + (x)^2       0.000000       0.000000     4
```

## Key config settings

| Setting | Value | Why |
|---|---|---|
| `unary` | `["squared"]` | Adds `x^2` to the search vocabulary |
| `binary` | `["add"]` | Only addition is needed to combine `x^2` and `3` |
| `constants` | `["3"]` | The additive constant to discover |

## Expanding the search

To search for higher-degree polynomials or roots, add more unary operators:

```toml
[operators]
binary = ["add", "multiply"]
unary  = ["squared", "cubed", "sqrt"]
```
