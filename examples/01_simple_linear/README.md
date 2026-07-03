# Example 1 — Simple Linear

**Target formula:** `y = 2 * x + 1`

The simplest possible demonstration. Given six training points, the solver
finds the exact formula at complexity 5 with MAPE = 0%.

## Running

From this directory:

```bash
xcelerator-solver solver.toml
```

## What to expect

```
 Rank  Expression      Train MAPE %   Val MAPE %   Complexity
    1  1 + 2 * x       0.000000       0.000000     5
    2  1 + x + x       0.000000       0.000000     5
```

The top-ranked results are all equivalent forms of `2*x + 1` — the solver
explores the full expression space and surfaces every formula that fits.

## Key config settings

| Setting | Value | Why |
|---|---|---|
| `max_complexity` | 5 | The target formula has 5 AST nodes |
| `max_error_pct` | 1.0 | 1% MAPE — admits the exact formula and small rounding variants |
| `constants` | `["1", "2"]` | Only the constants actually needed |
| `binary` | `["add", "multiply"]` | Only the operators actually needed |
| `unary` | `[]` | No unary operators required; omitted from search entirely |
