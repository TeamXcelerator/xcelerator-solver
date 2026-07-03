# Example 5 — Special Functions

**Target formula:** `y = erf(x)` (error function)

Demonstrates the special-function operators added in the solver:
`erf`, `tgamma`, and `lgamma`. This example uses `erf` to show how
the solver can discover transcendental functions that would otherwise
require a high-complexity polynomial approximation to fit.

## Running

From this directory:

```bash
xcelerator-solver solver.toml
```

## What to expect

```
 Rank  Expression      Train MAPE %   Val MAPE %   Complexity
    1  erf(x)          ~0.000         ~0.000         2
```

The solver finds `erf(x)` at complexity 2 — a single unary operator
applied to the variable. Any polynomial approximation would require
much higher complexity to match the same accuracy.

## Key config settings

| Setting | Value | Why |
|---|---|---|
| `unary` | `["erf", "tanh", "exp", "negate"]` | Includes erf alongside tanh (also S-shaped) so the solver competes fairly |
| `binary` | `["add", "multiply"]` | Allows additive/multiplicative combinations if needed |
| `constants` | `[]` | No constants needed for the pure erf(x) target |
| `max_complexity` | 5 | Target is at complexity 2; extra headroom shows alternatives |

## The three special-function operators

| Config name | Function | Domain | Use case |
|---|---|---|---|
| `erf` | Error function erf(x) | All reals | Probability / diffusion PDEs |
| `tgamma` | Gamma function Γ(x) | All reals (±∞ at 0, −1, −2, …) | Factorial generalization, combinatorics |
| `lgamma` | ln\|Γ(x)\| (log-gamma) | All reals (−∞ at poles) | Numerically stable for large x |

**Important:** `tgamma` is the Gamma *function* Γ(x), distinct from
the `gamma` *constant* (Euler–Mascheroni γ ≈ 0.5772).
Use `constants = ["gamma"]` for the constant;
use `unary = ["tgamma"]` for the function.
