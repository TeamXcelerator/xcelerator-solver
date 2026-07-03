# Example 4 — Pinned Terms

**Target formula:** `y = ln(x) + 2`

Demonstrates `pinned_terms`: a constraint that forces every accepted candidate
to contain a specific sub-expression. Here, `ln(x)` is pinned — any formula
that does not contain `ln(x)` as a sub-tree is rejected, regardless of how well
it fits the data.

This is useful when domain knowledge tells you a certain mathematical structure
must be present. Pinning it eliminates irrelevant candidates and speeds up the
search.

## Running

From this directory:

```bash
xcelerator-solver solver.toml
```

## What to expect

```
 Rank  Expression      Train MAPE %   Val MAPE %   Complexity
    1  2 + ln(x)       0.000007       0.000008     4
```

The tiny non-zero error is floating-point rounding on the CSV input values —
the formula is exact to the precision of the training data.

## Key config settings

| Setting | Value | Why |
|---|---|---|
| `pinned_terms` | `["ln(x)"]` | Every result must contain `ln(x)` as a sub-tree |
| `unary` | `["ln"]` | Makes `ln` available to the search |
| `binary` | `["add"]` | Only addition needed to combine `ln(x)` with `2` |
| `constants` | `["2"]` | The additive offset to discover |

## Pinned terms notation

Pinned terms use the same function-call notation as the config's operator names:

```toml
pinned_terms = [
    "ln(x)",                         # require ln(x) as a sub-tree
    "multiply(2, Pi)",               # require 2*Pi
    "divide(multiply(4, Pi), ln(N))" # require 4*Pi/ln(N)
]
```

Multiple pinned terms are combined with AND — every one must be present.
