# Examples

Each subdirectory is a self-contained example with its own `solver.toml`,
`data/train.csv`, `data/val.csv`, and a README explaining what it demonstrates.

## Running an example

Build the solver first (from the repository root):

```bash
cargo build --release
```

Then run from the example directory:

```bash
cd examples/01_simple_linear
../../target/release/xcelerator-solver solver.toml
```

Or pass the binary path explicitly from any directory:

```bash
xcelerator-solver examples/01_simple_linear/solver.toml
```

## Examples at a glance

| Example | Target formula | Key feature |
|---|---|---|
| [`01_simple_linear`](01_simple_linear/) | `y = 2*x + 1` | Minimal config; single variable |
| [`02_quadratic`](02_quadratic/) | `y = x^2 + 3` | `squared` unary operator |
| [`03_multivariate`](03_multivariate/) | `y = a * b` | Two variables; MAE metric |
| [`04_pinned_terms`](04_pinned_terms/) | `y = ln(x) + 2` | Required sub-expressions |
| [`05_special_functions`](05_special_functions/) | `y = erf(x)` | `erf`, `tgamma`, `lgamma` operators |

## Building on an example

Each `solver.toml` is intentionally minimal — only the constants and operators
the target formula actually needs. When adapting to your own data:

1. Replace `data/train.csv` and `data/val.csv` with your data.
2. Set `variables` to the column names in your CSV.
3. Set `constants` to the numeric or named constants you want in the search pool.
4. Set `binary`/`unary` to the operators you want to allow.
5. Increase `max_complexity` to allow more complex formulas.
6. Tune `max_error_pct` to control how strict the acceptance threshold is.
