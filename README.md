# Xcelerator Solver

A deterministic, precision-configurable symbolic regression engine written in
Rust. Given training data and an explicit pool of allowed terms and operators,
it searches for mathematical expressions that explain the data within a
user-specified error threshold.

**Author:** Ronnie Andrews, Jr.  
**ORCID:** [0009-0003-9724-3104](https://orcid.org/0009-0003-9724-3104)  
**Contact:** randrewsmath@gmail.com  
**Organization:** Team Xcelerator Inc.®  
**Date:** June 2026

---

## Key Features

- **Deterministic search** — Bottom-Up Enumeration (BUE) guarantees the
  simplest formula at any error level is always found first. Identical inputs
  always produce identical results.

- **Fixed pool** — only the terms and operators you declare are used. No
  implicit constants, no open-ended search. Results are always interpretable.

- **Pinned sub-components** — require that every candidate formula contains a
  specific mathematical sub-structure (e.g., `4π/ln(N)`). Focus the search on
  structurally relevant formulas without reducing generality.

- **Configurable high precision** — optionally evaluates all expressions using
  MPFR arbitrary-precision arithmetic (`rug` crate) at a user-specified number
  of decimal digits. Named constants such as `Pi` and `e` are computed at full
  target precision, not promoted from `f64`.

- **Train / validate split** — trains against one CSV, ranks the top candidates
  by training error, then scores each against a separate validation CSV.
  Final output is ordered by validation performance.

- **Parallel evaluation** — rayon-based worker pool; configurable thread cap
  to avoid overloading shared machines.

- **File + console output** — results written simultaneously to a file (for
  version control and diffing) and to the console.

---

## Reporting issues & feature requests

Found a bug, hit a limitation, or have an idea for a new capability? Please
reach out directly rather than forking the repository or starting an
independent project:

- Open an issue: https://github.com/TeamXcelerator/xcelerator-solver/issues
- Or email: randrewsmath@gmail.com

This keeps fixes and improvements consolidated in one place, so everyone who
depends on the solver benefits from them. The license (see below) does not
permit modification or redistribution — the intended path for any change,
however small, is to report it here so it can be reviewed and fixed or added
upstream.

## Citing this work

If you use the Xcelerator Solver in your research, we'd appreciate a citation.
Knowing the solver is being used helps justify continued development and makes
it easier for others to discover and benefit from it.

```bibtex
@software{AndrewsXceleratorSolver2026,
  author = {Andrews, Ronnie, Jr.},
  title  = {Xcelerator Solver: Deterministic, Precision-Configurable
            Symbolic Regression Engine},
  year   = {2026},
  doi    = {10.5281/zenodo.21051073},
  url    = {https://github.com/TeamXcelerator/xcelerator-solver}
}
```

A note in the methods section along the lines of *"symbolic regression was
performed using the Xcelerator Solver (github.com/TeamXcelerator/xcelerator-solver)"*
is equally welcome. Thank you.

---

## Building

### Standard precision (f64, builds natively on Windows and Linux)

```bash
cargo build --release
```

### High-precision mode (requires GMP/MPFR)

**Linux:**
```bash
sudo apt install build-essential libgmp-dev libmpfr-dev libmpc-dev
cargo build --release --features hp
```

**Windows:** use WSL2 with the Ubuntu GMP/MPFR packages, then build inside
WSL using the path to the repository.

---

## Usage

```
xcelerator-solver <config.toml> [--json]
```

All parameters are supplied via a TOML configuration file. Pass `--json` to
emit results as JSON instead of a formatted table.

---

## Configuration

A full annotated example:

```toml
# ── Data ────────────────────────────────────────────────────────────────────
training_csv   = "data/train.csv"      # CSV with header row; used for search
validation_csv = "data/validate.csv"   # CSV with header row; used for scoring
target_column  = "D"                   # column the solver tries to predict

# ── Search limits ────────────────────────────────────────────────────────────
max_error_pct  = 5.0    # reject any formula with training MAPE above this %
max_complexity = 9      # maximum expression tree node count
max_time_secs  = 120.0  # wall-clock timeout

# ── Output ───────────────────────────────────────────────────────────────────
output_file    = "results/run_001.txt"  # written alongside console output
top_candidates = 20                     # how many top training results to validate

# ── Performance ──────────────────────────────────────────────────────────────
max_threads      = 4    # cap rayon thread pool (omit to use all cores)
precision_digits = 50   # HP decimal digits; omit or set 0 for standard f64
                        # requires --features hp build when > 0
error_metric     = "mape"  # "mape" (default), "mae", or "rmse"
                            # use mae/rmse when the target is near-zero or negative

# ── Term pool ─────────────────────────────────────────────────────────────────
[terms]
variables = ["L2", "N"]               # must match column headers in the CSVs
constants = ["Pi", "e", "1", "2", "4"] # named constants or numeric literals
# Optional: pre-built sub-expressions treated as single atoms (complexity 1).
# Useful when a known sub-structure should be available without rediscovery.
composite = [
    "multiply(N, ln(N))",        # N*ln(N) — available as one atom
    "divide(1, multiply(N, N))", # 1/N^2   — available as one atom
]

# ── Operator pool ─────────────────────────────────────────────────────────────
# Operators are spelled out to avoid special-character ambiguity in config files.
[operators]
binary = ["add", "subtract", "multiply", "divide", "power"]
unary  = ["sqrt", "squared", "sine", "cosine", "ln", "exp", "negate", "abs"]

# ── Pinned sub-components (optional) ─────────────────────────────────────────
# Every accepted candidate must contain ALL listed sub-expressions as sub-trees.
# Notation: binary_op(left, right)  or  unary_op(term)
# Same operator names as [operators] above.
pinned_terms = [
    "multiply(4, Pi)",                      # 4π
    "divide(multiply(4, Pi), ln(N))",       # 4π / ln(N)
]
```

### Named constants

| Name   | Value                  |
|--------|------------------------|
| `Pi`     | π (full HP precision)  |
| `e`      | Euler's number         |
| `Tau`    | 2π                     |
| `Phi`    | Golden ratio ≈ 1.618…  |
| `gamma`  | Euler–Mascheroni γ ≈ 0.5772… |
| `Catalan` | Catalan's constant G ≈ 0.9159… |
| `"2"`    | literal 2.0            |
| `"0.5"`  | literal 0.5            |

### Operators

| Binary     | Unary        |
|------------|--------------|
| `add`      | `sqrt`       |
| `subtract` | `squared`    |
| `multiply` | `cubed`      |
| `divide`   | `sine`       |
| `power`    | `cosine`     |
|            | `tangent`    |
|            | `arcsine`    |
|            | `arccosine`  |
|            | `arctangent` |
|            | `ln`         |
|            | `log`        |
|            | `exp`        |
|            | `tanh`       |
|            | `sinh`       |
|            | `cosh`       |
|            | `erf`        |
|            | `tgamma`     |
|            | `lgamma`     |
|            | `negate`     |
|            | `abs`        |

`log` is base-10. `arcsine`/`arccosine` guard the domain: inputs outside
`[−1, 1]` produce no result (the expression is discarded).
`tgamma` is the Gamma function Γ(x) — **not** the `gamma` constant (Euler–Mascheroni γ).
Use `constants = ["gamma"]` for the constant; `unary = ["tgamma"]` for the function.
`lgamma` is ln|Γ(x)|, the log-gamma function (numerically stable for large x).

---

## Input CSV format

Both CSVs must have a header row. Column names in the header identify variables.
The `target_column` value names the column the solver is trying to predict.
All other columns listed in `[terms] variables` are treated as inputs.

```
L2,N,D
13.0,60,111.0
45.0,50,102.0
100.0,40,100.0
```

Rows with unparseable values are skipped with a warning. At least 2 valid rows
are required.

---

## Output

```
Xcelerator Solver — results
Precision: HP-50 (166 bits)   Threads: 4   Timeout: no
────────────────────────────────────────────────────────────────────────────────
 Rank  Expression                    Train MAPE %   Val MAPE %   Complexity
────────────────────────────────────────────────────────────────────────────────
    1  4 * Pi / ln(N) + L2           0.312          0.418        7
    2  4 * Pi / ln(N) + L2 + 1       0.891          0.953        9
────────────────────────────────────────────────────────────────────────────────
Evaluated: 84231  |  Elapsed: 47.3s
```

With `--json`:

```json
{
  "precision_mode": "hp-50",
  "results": [
    {
      "expression":     "4 * Pi / ln(N) + L2",
      "train_mape_pct": 0.312,
      "val_mape_pct":   0.418,
      "complexity":     7
    }
  ],
  "stats": {
    "expressions_evaluated": 84231,
    "elapsed_secs": 47.3,
    "timed_out": false
  }
}
```

---

## How the search works

The solver uses **Bottom-Up Enumeration (BUE)**: it builds expressions
complexity-level by complexity-level, starting at the simplest (a single term)
and expanding outward. At each complexity level K, it combines all previously
built sub-expressions of lower complexity using the declared operators. This
means compound sub-structures emerge naturally as cached intermediate values
that are reused in larger expressions.

A bounded dedup set prevents the same expression from appearing in the results
more than once. The top `top_candidates` expressions by training error are
retained throughout the search. After the search, those candidates are scored
against the validation data and the final table is sorted by validation error.

---

## Examples

The `examples/` directory contains four self-contained worked examples, each
with a `solver.toml`, training/validation CSVs, and a README:

| Example | Target formula | Demonstrates |
|---|---|---|
| [`01_simple_linear`](examples/01_simple_linear/) | `y = 2*x + 1` | Minimal single-variable config |
| [`02_quadratic`](examples/02_quadratic/) | `y = x² + 3` | `squared` unary operator |
| [`03_multivariate`](examples/03_multivariate/) | `y = a * b` | Two variables; MAE metric |
| [`04_pinned_terms`](examples/04_pinned_terms/) | `y = ln(x) + 2` | Required sub-expressions |
| [`05_special_functions`](examples/05_special_functions/) | `y = erf(x)` | `erf`, `tgamma`, `lgamma` operators |

Run any example from its directory:

```bash
# Build first (from repo root)
cargo build --release

# Then run from the example directory
cd examples/01_simple_linear
../../target/release/xcelerator-solver solver.toml
```

---

## Version History

- **v0.1.0** — Initial public release. Bottom-Up Enumeration (BUE) search
  engine with three-actor pipeline (Generator → Evaluator → Aggregator);
  MAPE/MAE/RMSE error metrics; configurable HP arithmetic via MPFR; full
  unary/binary operator set including trigonometric, hyperbolic, inverse
  trig, special functions (`erf`, `tgamma`, `lgamma`); composite and pinned
  terms; named constants `Pi`, `e`, `Tau`, `Phi`, `gamma` (Euler–Mascheroni),
  `Catalan`; 5 verified examples; 173 tests.

---

## License

See [LICENSE](LICENSE). Source-available for verification and study.
Not licensed for modification, redistribution, or commercial use.

## Trademarks

"Team Xcelerator Inc." is a registered trademark of Team Xcelerator Inc.
All other trademarks are the property of their respective owners.
