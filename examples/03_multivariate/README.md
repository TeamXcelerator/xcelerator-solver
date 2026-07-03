# Example 3 — Multivariate

**Target formula:** `y = a * b`

Demonstrates a two-variable search. No constants appear in the target formula.
Also shows the `mae` error metric, which is a better choice than `mape` when
target values span a wide range (MAPE weights small-valued rows more heavily).

## Running

From this directory:

```bash
xcelerator-solver solver.toml
```

## What to expect

```
 Rank  Expression    Train MAE      Val MAE      Complexity
    1  a * b         0.000          0.000         3
```

## Key config settings

| Setting | Value | Why |
|---|---|---|
| `error_metric` | `"mae"` | Mean Absolute Error; more robust than MAPE for multi-scale data |
| `max_error_pct` | `0.1` | In MAE mode this is an absolute threshold in target units |
| `variables` | `["a", "b"]` | Two input columns from the CSV |
| `constants` | `[]` | No numeric constants needed |

## When to use MAE vs MAPE

- **MAPE** (default): best when the target is bounded away from zero and you
  care about relative accuracy. Units-free: 2% is 2% regardless of scale.
- **MAE**: best when the target can be near zero, negative, or spans several
  orders of magnitude. Threshold is in the same units as the target column.
- **RMSE**: like MAE but penalizes large errors more heavily.
