// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! MAPE, MAE, and RMSE error metrics — f64 and HP variants.
//!
//! Three metrics are supported and selected via the `error_metric` config field:
//!
//! - **MAPE** (default) — Mean Absolute Percentage Error. Best when the target is
//!   bounded away from zero. For targets near zero (`|actual| ≤ 1e-12`) falls back
//!   to an absolute error (`|predicted| × 100`) rather than dividing by near-zero.
//! - **MAE** — Mean Absolute Error in target units. Handles negative and near-zero
//!   targets cleanly; use when MAPE would explode or be misleading.
//! - **RMSE** — Root Mean Squared Error. Like MAE but penalizes large misses
//!   more heavily.
//!
//! Any non-finite predicted value causes the entire expression to be discarded.

use crate::csv_loader::DataPoint;
use crate::eval::eval_expr;
use crate::expr::Expr;

/// Denominator threshold: actuals smaller than this use absolute fallback.
const NEAR_ZERO: f64 = 1e-12;

/// Which error metric the solver uses to score candidates against the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorMetric {
    /// Mean Absolute Percentage Error (%). Default. Breaks down when the
    /// target is near zero or negative — use Mae/Rmse in those regimes.
    Mape,
    /// Mean Absolute Error (absolute units of the target). Handles
    /// negative / near-zero targets cleanly.
    Mae,
    /// Root Mean Squared Error (absolute units). Penalizes large misses more.
    Rmse,
}

impl ErrorMetric {
    /// Parse a config string into an ErrorMetric. Defaults handled by caller.
    pub fn from_name(s: &str) -> Result<ErrorMetric, String> {
        match s.to_lowercase().as_str() {
            "mape" => Ok(ErrorMetric::Mape),
            "mae"  => Ok(ErrorMetric::Mae),
            "rmse" => Ok(ErrorMetric::Rmse),
            _ => Err(format!(
                "Unknown error_metric '{}'. Valid: mape, mae, rmse", s
            )),
        }
    }

    /// Compute this metric from parallel predicted/actual slices.
    /// Returns `None` if any predicted value is non-finite.
    pub fn compute(&self, predicted: &[f64], actual: &[f64]) -> Option<f64> {
        match self {
            ErrorMetric::Mape => compute_mape(predicted, actual),
            ErrorMetric::Mae  => compute_mae(predicted, actual),
            ErrorMetric::Rmse => compute_rmse(predicted, actual),
        }
    }

    /// Short label for output column headers.
    pub fn label(&self) -> &'static str {
        match self {
            ErrorMetric::Mape => "MAPE %",
            ErrorMetric::Mae  => "MAE",
            ErrorMetric::Rmse => "RMSE",
        }
    }
}

// ---------------------------------------------------------------------------
// f64
// ---------------------------------------------------------------------------

/// Evaluate `expr` on every training data point.
/// Returns `None` if any point produces an invalid result (domain error,
/// non-finite, missing variable).
pub fn eval_all(expr: &Expr, data: &[DataPoint]) -> Option<Vec<f64>> {
    data.iter()
        .map(|point| eval_expr(expr, &point.inputs))
        .collect() // Option<Vec<_>> short-circuits on first None
}

/// Compute MAPE from parallel predicted/actual slices.
///
/// Per-point formula:
/// - `|actual| > NEAR_ZERO` → `|predicted - actual| / |actual| * 100`
/// - `|actual| ≤ NEAR_ZERO` → `|predicted| * 100`   (absolute fallback)
///
/// Returns `None` if any predicted value is non-finite.
pub fn compute_mape(predicted: &[f64], actual: &[f64]) -> Option<f64> {
    debug_assert_eq!(predicted.len(), actual.len());

    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let n = predicted.len() as f64;
    let sum: f64 = predicted.iter()
        .zip(actual.iter())
        .map(|(&p, &a)| {
            if a.abs() > NEAR_ZERO {
                (p - a).abs() / a.abs() * 100.0
            } else {
                p.abs() * 100.0
            }
        })
        .sum();

    Some(sum / n)
}

/// Mean Absolute Error: mean(|predicted - actual|), in absolute target units.
/// Handles negative and near-zero targets cleanly (no division).
/// Returns `None` if any predicted value is non-finite.
pub fn compute_mae(predicted: &[f64], actual: &[f64]) -> Option<f64> {
    debug_assert_eq!(predicted.len(), actual.len());
    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let n = predicted.len() as f64;
    let sum: f64 = predicted.iter()
        .zip(actual.iter())
        .map(|(&p, &a)| (p - a).abs())
        .sum();
    Some(sum / n)
}

/// Root Mean Squared Error: sqrt(mean((predicted - actual)^2)).
/// Absolute units; penalizes large misses more than MAE.
/// Returns `None` if any predicted value is non-finite.
pub fn compute_rmse(predicted: &[f64], actual: &[f64]) -> Option<f64> {
    debug_assert_eq!(predicted.len(), actual.len());
    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let n = predicted.len() as f64;
    let sum_sq: f64 = predicted.iter()
        .zip(actual.iter())
        .map(|(&p, &a)| (p - a) * (p - a))
        .sum();
    Some((sum_sq / n).sqrt())
}

// ---------------------------------------------------------------------------
// HP variants
// ---------------------------------------------------------------------------

#[cfg(feature = "hp")]
pub mod hp_metric {
    use super::NEAR_ZERO;
    use crate::csv_loader::hp_loader::HpDataPoint;
    use crate::eval_hp::eval_expr_hp;
    use crate::expr::Expr;
    use rug::Float;

    /// Evaluate `expr` on every HP training data point.
    pub fn eval_all_hp(
        expr: &Expr,
        data: &[HpDataPoint],
        prec: u32,
    ) -> Option<Vec<Float>> {
        data.iter()
            .map(|point| eval_expr_hp(expr, &point.inputs, prec))
            .collect()
    }

    /// Compute MAPE using full HP arithmetic; convert only the final mean to f64.
    pub fn compute_mape_hp(
        predicted: &[Float],
        actual: &[Float],
        prec: u32,
    ) -> Option<f64> {
        debug_assert_eq!(predicted.len(), actual.len());

        if predicted.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let near_zero = Float::with_val(prec, NEAR_ZERO);
        let hundred   = Float::with_val(prec, 100u32);
        let n         = Float::with_val(prec, predicted.len() as u64);

        let mut sum = Float::with_val(prec, 0u32);

        for (p, a) in predicted.iter().zip(actual.iter()) {
            let abs_a = a.clone().abs();
            let term = if abs_a > near_zero {
                let diff = Float::with_val(prec, p - a).abs();
                diff / abs_a * Float::with_val(prec, &hundred)
            } else {
                p.clone().abs() * Float::with_val(prec, &hundred)
            };
            sum += term;
        }

        Some((sum / n).to_f64())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csv_loader::DataPoint;
    use crate::expr::{BinOp, ConstSource, Expr};

    fn point(x: f64, y: f64) -> DataPoint {
        DataPoint { inputs: [("x".to_string(), x)].into(), output: y }
    }

    fn mul_2x() -> Expr {
        Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(2.0, ConstSource::Literal("2".to_string()))),
            Box::new(Expr::Var("x".to_string())),
        )
    }

    // --- eval_all ---
    #[test]
    fn eval_all_perfect() {
        let data = vec![point(1.0, 2.0), point(2.0, 4.0), point(3.0, 6.0)];
        let predicted = eval_all(&mul_2x(), &data).unwrap();
        assert_eq!(predicted, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn eval_all_domain_error_returns_none() {
        use crate::expr::UnaryOp;
        let bad = Expr::Unary(UnaryOp::Sqrt,
            Box::new(Expr::Const(-1.0, ConstSource::Literal("-1".to_string()))));
        let data = vec![point(1.0, 1.0), point(2.0, 2.0)];
        assert_eq!(eval_all(&bad, &data), None);
    }

    // --- compute_mape ---
    #[test]
    fn mape_perfect_prediction() {
        let p = vec![2.0, 4.0, 6.0];
        let a = vec![2.0, 4.0, 6.0];
        assert_eq!(compute_mape(&p, &a), Some(0.0));
    }

    #[test]
    fn mape_ten_percent() {
        // Each predicted is 10% above actual
        let a = vec![100.0, 200.0, 50.0];
        let p: Vec<f64> = a.iter().map(|v| v * 1.1).collect();
        let mape = compute_mape(&p, &a).unwrap();
        assert!((mape - 10.0).abs() < 1e-10, "MAPE = {mape}");
    }

    #[test]
    fn mape_near_zero_actual_uses_absolute_fallback() {
        // actual ≈ 0, predicted = 50 → term = |50| * 100 = 5000
        let p = vec![50.0];
        let a = vec![0.0];
        let mape = compute_mape(&p, &a).unwrap();
        assert!((mape - 5000.0).abs() < 1e-9);
    }

    #[test]
    fn mape_nan_predicted_returns_none() {
        let p = vec![f64::NAN, 2.0];
        let a = vec![1.0, 2.0];
        assert_eq!(compute_mape(&p, &a), None);
    }

    #[test]
    fn mape_inf_predicted_returns_none() {
        let p = vec![f64::INFINITY];
        let a = vec![1.0];
        assert_eq!(compute_mape(&p, &a), None);
    }

    #[test]
    fn mae_basic() {
        // |2-1| + |4-4| + |5-6| = 1 + 0 + 1 = 2; /3 = 0.6667
        let p = vec![2.0, 4.0, 5.0];
        let a = vec![1.0, 4.0, 6.0];
        let mae = compute_mae(&p, &a).unwrap();
        assert!((mae - 2.0/3.0).abs() < 1e-12, "MAE = {mae}");
    }

    #[test]
    fn mae_handles_negative_and_zero_targets() {
        // MAE must be finite when targets are negative or zero.
        let p = vec![0.0, 0.5, 1.0];
        let a = vec![-0.5, 0.0, 0.8];
        let mae = compute_mae(&p, &a).unwrap();
        assert!(mae.is_finite());
        // = (0.5 + 0.5 + 0.2)/3
        assert!((mae - 1.2/3.0).abs() < 1e-12, "MAE = {mae}");
    }

    #[test]
    fn rmse_basic() {
        // errors 1, 0, 1 -> sqrt((1+0+1)/3) = sqrt(0.6667)
        let p = vec![2.0, 4.0, 5.0];
        let a = vec![1.0, 4.0, 6.0];
        let rmse = compute_rmse(&p, &a).unwrap();
        assert!((rmse - (2.0f64/3.0).sqrt()).abs() < 1e-12, "RMSE = {rmse}");
    }

    #[test]
    fn metric_from_name() {
        assert_eq!(ErrorMetric::from_name("mape").unwrap(), ErrorMetric::Mape);
        assert_eq!(ErrorMetric::from_name("MAE").unwrap(),  ErrorMetric::Mae);
        assert_eq!(ErrorMetric::from_name("rmse").unwrap(), ErrorMetric::Rmse);
        assert!(ErrorMetric::from_name("bogus").is_err());
    }

    #[test]
    fn mae_perfect_prediction() {
        let v = vec![1.0, 5.0, 3.0];
        assert_eq!(compute_mae(&v, &v), Some(0.0));
    }

    #[test]
    fn rmse_perfect_prediction() {
        let v = vec![2.0, 7.0, 4.0];
        assert_eq!(compute_rmse(&v, &v), Some(0.0));
    }

    #[test]
    fn mae_nan_returns_none() {
        let p = vec![f64::NAN, 1.0];
        let a = vec![1.0, 1.0];
        assert_eq!(compute_mae(&p, &a), None);
    }

    #[test]
    fn rmse_nan_returns_none() {
        let p = vec![f64::NAN];
        let a = vec![1.0];
        assert_eq!(compute_rmse(&p, &a), None);
    }

    #[test]
    fn rmse_greater_than_mae_for_unequal_errors() {
        // RMSE penalizes large errors more: one large miss dominates
        let p = vec![10.0, 1.0];
        let a = vec![0.0, 0.0];
        let mae  = compute_mae(&p, &a).unwrap();
        let rmse = compute_rmse(&p, &a).unwrap();
        // mae = (10+1)/2 = 5.5; rmse = sqrt((100+1)/2) ≈ 7.106
        assert!(rmse > mae, "RMSE ({rmse}) should exceed MAE ({mae}) when errors are unequal");
    }
}
