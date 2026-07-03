// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Evaluator worker pool.
//!
//! Receives candidate Expr values from the candidate channel,
//! evaluates each against training data using rayon parallel workers,
//! and sends EvalJob results to the result channel.
//! Knows nothing about search structure or ranking.

use crate::csv_loader::DataPoint;
use crate::error_metric::{eval_all, ErrorMetric};
use crate::expr::Expr;

/// Result of evaluating one candidate against training data.
/// `train_error == f64::NAN` signals an invalid expression (domain error or
/// non-finite result on any data point); the aggregator discards these.
#[derive(Debug)]
pub struct EvalJob {
    pub expr: Expr,
    pub train_error: f64,
}

/// Run the evaluator pool.
///
/// Spawns `rayon::current_num_threads()` workers, each pulling from
/// `candidate_rx` (crossbeam Receiver is Clone — no Mutex needed).
/// Exits when `candidate_rx` is disconnected (generator dropped its sender).
/// The caller must drop `result_tx` after this returns.
pub fn run_evaluator(
    candidate_rx: crossbeam_channel::Receiver<Expr>,
    result_tx: crossbeam_channel::Sender<EvalJob>,
    training: &[DataPoint],
    metric: ErrorMetric,
) {
    let num_workers = rayon::current_num_threads().max(1);

    rayon::scope(|s| {
        for _ in 0..num_workers {
            let rx = candidate_rx.clone();
            let tx = result_tx.clone();
            s.spawn(move |_| {
                // Drain the channel until the generator closes it.
                for expr in &rx {
                    let train_error = evaluate_one(&expr, training, metric);
                    // Ignore send error (aggregator may have stopped).
                    tx.send(EvalJob { expr, train_error }).ok();
                }
            });
        }
        // result_tx dropped here when the scope ends, signalling the aggregator.
    });
}

/// Evaluate one expression and return its training error under `metric`.
/// Returns `f64::NAN` if evaluation fails on any data point or the metric
/// is invalid.
fn evaluate_one(expr: &Expr, training: &[DataPoint], metric: ErrorMetric) -> f64 {
    let actuals: Vec<f64> = training.iter().map(|p| p.output).collect();
    match eval_all(expr, training) {
        Some(predicted) => metric.compute(&predicted, &actuals).unwrap_or(f64::NAN),
        None => f64::NAN,
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

    fn data(pairs: &[(f64, f64)]) -> Vec<DataPoint> {
        pairs.iter().map(|&(x, y)| DataPoint {
            inputs: [("x".to_string(), x)].into(),
            output: y,
        }).collect()
    }

    fn two_x() -> Expr {
        Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(2.0, ConstSource::Literal("2".to_string()))),
            Box::new(Expr::Var("x".to_string())),
        )
    }

    fn bad_expr() -> Expr {
        // sqrt(-1) always returns None
        Expr::Unary(
            crate::expr::UnaryOp::Sqrt,
            Box::new(Expr::Const(-1.0, ConstSource::Literal("-1".to_string()))),
        )
    }

    #[test]
    fn valid_expression_gives_correct_error() {
        let training = data(&[(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]);
        let err = evaluate_one(&two_x(), &training, ErrorMetric::Mape);
        assert!(err.is_finite() && err < 1e-10, "MAPE should be ~0, got {err}");
    }

    #[test]
    fn domain_error_expression_gives_nan() {
        let training = data(&[(1.0, 1.0), (2.0, 2.0)]);
        let err = evaluate_one(&bad_expr(), &training, ErrorMetric::Mape);
        assert!(err.is_nan(), "expected NAN, got {err}");
    }

    #[test]
    fn mae_handles_negative_target() {
        // target can be negative — MAE must still produce a finite value.
        let training = data(&[(1.0, -0.5), (2.0, 0.2)]);
        let err = evaluate_one(&two_x(), &training, ErrorMetric::Mae);
        assert!(err.is_finite(), "MAE should be finite for negative targets, got {err}");
    }

    #[test]
    fn channel_drains_correctly() {
        let (candidate_tx, candidate_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let training = data(&[(1.0, 2.0), (2.0, 4.0)]);

        // Send two expressions then close the sender.
        candidate_tx.send(two_x()).unwrap();
        candidate_tx.send(bad_expr()).unwrap();
        drop(candidate_tx);

        run_evaluator(candidate_rx, result_tx, &training, ErrorMetric::Mape);

        let results: Vec<EvalJob> = result_rx.try_iter().collect();
        assert_eq!(results.len(), 2);

        let valid = results.iter().find(|j| !j.train_error.is_nan());
        let invalid = results.iter().find(|j| j.train_error.is_nan());
        assert!(valid.is_some(), "expected one valid result");
        assert!(invalid.is_some(), "expected one NAN result");
    }
}
