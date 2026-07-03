// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! f64 expression evaluator.
//!
//! Returns `None` on any domain error or non-finite result.
//! A single `None` result for any data point causes the whole expression
//! to be discarded by the pipeline.

use crate::expr::{BinOp, Expr, UnaryOp};
use std::collections::HashMap;
use libm::{erf, tgamma, lgamma};

/// Threshold below which a denominator is treated as zero.
const DIV_ZERO_GUARD: f64 = 1e-300;

/// Evaluate `expr` for a single data point.
/// Returns `None` on domain errors (div-by-zero, sqrt of negative, etc.)
/// or if any intermediate result is non-finite.
pub fn eval_expr(expr: &Expr, inputs: &HashMap<String, f64>) -> Option<f64> {
    let result = eval_inner(expr, inputs)?;
    if result.is_finite() { Some(result) } else { None }
}

fn eval_inner(expr: &Expr, inputs: &HashMap<String, f64>) -> Option<f64> {
    match expr {
        Expr::Const(v, _) => {
            if v.is_finite() { Some(*v) } else { None }
        }

        Expr::Var(s) => inputs.get(s.as_str()).copied(),

        // Composite is transparent for evaluation.
        Expr::Composite(inner) => eval_expr(inner, inputs),

        Expr::Unary(op, child) => {
            let v = eval_expr(child, inputs)?;
            let r = match op {
                UnaryOp::Sin     => v.sin(),
                UnaryOp::Cos     => v.cos(),
                UnaryOp::Tan     => v.tan(),
                UnaryOp::Asin    => { if !(-1.0..=1.0).contains(&v) { return None; } v.asin() }
                UnaryOp::Acos    => { if !(-1.0..=1.0).contains(&v) { return None; } v.acos() }
                UnaryOp::Atan    => v.atan(),
                UnaryOp::Sqrt    => { if v < 0.0 { return None; } v.sqrt() }
                UnaryOp::Squared => v * v,
                UnaryOp::Cubed   => v * v * v,
                UnaryOp::Ln      => { if v <= 0.0 { return None; } v.ln() }
                UnaryOp::Log10   => { if v <= 0.0 { return None; } v.log10() }
                UnaryOp::Exp     => v.exp(),
                UnaryOp::Neg     => -v,
                UnaryOp::Abs     => v.abs(),
                UnaryOp::Tanh    => v.tanh(),
                UnaryOp::Sinh    => v.sinh(),
                UnaryOp::Cosh    => v.cosh(),
                UnaryOp::Erf     => erf(v),
                UnaryOp::Tgamma  => tgamma(v), // Γ(x); returns ±Inf at poles; is_finite guard catches it
                UnaryOp::Lgamma  => lgamma(v), // ln|Γ(x)|; returns +Inf at poles; is_finite guard catches it
            };
            if r.is_finite() { Some(r) } else { None }
        }

        Expr::Binary(op, l, r) => {
            let lv = eval_expr(l, inputs)?;
            let rv = eval_expr(r, inputs)?;
            let result = match op {
                BinOp::Add => lv + rv,
                BinOp::Sub => lv - rv,
                BinOp::Mul => lv * rv,
                BinOp::Div => {
                    if rv.abs() < DIV_ZERO_GUARD { return None; }
                    lv / rv
                }
                BinOp::Pow => lv.powf(rv),
            };
            if result.is_finite() { Some(result) } else { None }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{ConstSource, Expr, BinOp, UnaryOp};

    fn inputs(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }
    fn var(s: &str) -> Expr { Expr::Var(s.to_string()) }
    fn con(v: f64) -> Expr { Expr::Const(v, ConstSource::Literal(format!("{v}"))) }
    fn binop(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary(op, Box::new(l), Box::new(r))
    }
    fn unop(op: UnaryOp, e: Expr) -> Expr { Expr::Unary(op, Box::new(e)) }

    #[test] fn add_two_plus_three() {
        let e = binop(BinOp::Add, con(2.0), con(3.0));
        assert_eq!(eval_expr(&e, &HashMap::new()), Some(5.0));
    }
    #[test] fn variable_lookup() {
        let e = var("x");
        assert_eq!(eval_expr(&e, &inputs(&[("x", 4.0)])), Some(4.0));
    }
    #[test] fn missing_variable_is_none() {
        assert_eq!(eval_expr(&var("z"), &HashMap::new()), None);
    }
    #[test] fn squared_three() {
        assert_eq!(eval_expr(&unop(UnaryOp::Squared, con(3.0)), &HashMap::new()), Some(9.0));
    }
    #[test] fn cubed_three() {
        assert_eq!(eval_expr(&unop(UnaryOp::Cubed, con(3.0)), &HashMap::new()), Some(27.0));
    }
    #[test] fn log10_hundred() {
        let r = eval_expr(&unop(UnaryOp::Log10, con(100.0)), &HashMap::new()).unwrap();
        assert!((r - 2.0).abs() < 1e-12, "log10(100) should be 2, got {r}");
    }
    #[test] fn log10_nonpositive_is_none() {
        assert_eq!(eval_expr(&unop(UnaryOp::Log10, con(0.0)), &HashMap::new()), None);
        assert_eq!(eval_expr(&unop(UnaryOp::Log10, con(-1.0)), &HashMap::new()), None);
    }
    #[test] fn sqrt_four() {
        assert_eq!(eval_expr(&unop(UnaryOp::Sqrt, con(4.0)), &HashMap::new()), Some(2.0));
    }
    #[test] fn sqrt_negative_is_none() {
        assert_eq!(eval_expr(&unop(UnaryOp::Sqrt, con(-1.0)), &HashMap::new()), None);
    }
    #[test] fn ln_one() {
        assert_eq!(eval_expr(&unop(UnaryOp::Ln, con(1.0)), &HashMap::new()), Some(0.0));
    }
    #[test] fn ln_zero_is_none() {
        assert_eq!(eval_expr(&unop(UnaryOp::Ln, con(0.0)), &HashMap::new()), None);
    }
    #[test] fn ln_negative_is_none() {
        assert_eq!(eval_expr(&unop(UnaryOp::Ln, con(-1.0)), &HashMap::new()), None);
    }
    #[test] fn div_by_zero_is_none() {
        let e = binop(BinOp::Div, con(1.0), con(0.0));
        assert_eq!(eval_expr(&e, &HashMap::new()), None);
    }
    #[test] fn negate() {
        assert_eq!(eval_expr(&unop(UnaryOp::Neg, con(5.0)), &HashMap::new()), Some(-5.0));
    }
    #[test] fn abs_negative() {
        assert_eq!(eval_expr(&unop(UnaryOp::Abs, con(-3.0)), &HashMap::new()), Some(3.0));
    }
    #[test] fn exp_zero() {
        assert_eq!(eval_expr(&unop(UnaryOp::Exp, con(0.0)), &HashMap::new()), Some(1.0));
    }
    #[test] fn multiply_with_variable() {
        let e = binop(BinOp::Mul, con(2.0), var("x"));
        assert_eq!(eval_expr(&e, &inputs(&[("x", 5.0)])), Some(10.0));
    }
    #[test] fn pow_two_three() {
        let e = binop(BinOp::Pow, con(2.0), con(3.0));
        assert_eq!(eval_expr(&e, &HashMap::new()), Some(8.0));
    }
    #[test] fn inf_const_is_none() {
        assert_eq!(eval_expr(&con(f64::INFINITY), &HashMap::new()), None);
    }

    // --- operators added in later commits (tanh/sinh/cosh/tan/asin/acos/atan) ---

    #[test] fn tanh_zero() {
        assert_eq!(eval_expr(&unop(UnaryOp::Tanh, con(0.0)), &HashMap::new()), Some(0.0));
    }
    #[test] fn tanh_large() {
        let r = eval_expr(&unop(UnaryOp::Tanh, con(100.0)), &HashMap::new()).unwrap();
        assert!((r - 1.0).abs() < 1e-10, "tanh(100) ≈ 1, got {r}");
    }
    #[test] fn sinh_zero() {
        assert_eq!(eval_expr(&unop(UnaryOp::Sinh, con(0.0)), &HashMap::new()), Some(0.0));
    }
    #[test] fn cosh_zero() {
        assert_eq!(eval_expr(&unop(UnaryOp::Cosh, con(0.0)), &HashMap::new()), Some(1.0));
    }
    #[test] fn tan_zero() {
        assert_eq!(eval_expr(&unop(UnaryOp::Tan, con(0.0)), &HashMap::new()), Some(0.0));
    }
    #[test] fn asin_valid() {
        let r = eval_expr(&unop(UnaryOp::Asin, con(0.5)), &HashMap::new()).unwrap();
        // asin(0.5) = π/6 ≈ 0.5236
        assert!((r - std::f64::consts::FRAC_PI_2 / 3.0).abs() < 1e-10, "asin(0.5) = {r}");
    }
    #[test] fn asin_out_of_range_is_none() {
        // |x| > 1 → domain error
        assert_eq!(eval_expr(&unop(UnaryOp::Asin, con(2.0)), &HashMap::new()), None);
        assert_eq!(eval_expr(&unop(UnaryOp::Asin, con(-1.5)), &HashMap::new()), None);
    }
    #[test] fn acos_valid() {
        let r = eval_expr(&unop(UnaryOp::Acos, con(0.0)), &HashMap::new()).unwrap();
        // acos(0) = π/2
        assert!((r - std::f64::consts::FRAC_PI_2).abs() < 1e-10, "acos(0) = {r}");
    }
    #[test] fn acos_out_of_range_is_none() {
        assert_eq!(eval_expr(&unop(UnaryOp::Acos, con(1.5)), &HashMap::new()), None);
        assert_eq!(eval_expr(&unop(UnaryOp::Acos, con(-2.0)), &HashMap::new()), None);
    }
    #[test] fn atan_valid() {
        let r = eval_expr(&unop(UnaryOp::Atan, con(1.0)), &HashMap::new()).unwrap();
        // atan(1) = π/4
        assert!((r - std::f64::consts::FRAC_PI_4).abs() < 1e-10, "atan(1) = {r}");
    }
    #[test] fn sin_and_cos_basic() {
        let r_sin = eval_expr(&unop(UnaryOp::Sin, con(0.0)), &HashMap::new()).unwrap();
        let r_cos = eval_expr(&unop(UnaryOp::Cos, con(0.0)), &HashMap::new()).unwrap();
        assert_eq!(r_sin, 0.0);
        assert_eq!(r_cos, 1.0);
    }

    // --- special functions ---
    #[test] fn erf_zero() {
        // erf(0) = 0 exactly
        assert_eq!(eval_expr(&unop(UnaryOp::Erf, con(0.0)), &HashMap::new()), Some(0.0));
    }
    #[test] fn erf_large_approaches_one() {
        let r = eval_expr(&unop(UnaryOp::Erf, con(5.0)), &HashMap::new()).unwrap();
        assert!((r - 1.0).abs() < 1e-10, "erf(5) ≈ 1, got {r}");
    }
    #[test] fn erf_odd_symmetry() {
        let pos = eval_expr(&unop(UnaryOp::Erf, con(1.0)), &HashMap::new()).unwrap();
        let neg = eval_expr(&unop(UnaryOp::Erf, con(-1.0)), &HashMap::new()).unwrap();
        assert!((pos + neg).abs() < 1e-14, "erf(-x) = -erf(x)");
    }
    #[test] fn tgamma_positive_int() {
        // Γ(5) = 4! = 24
        let r = eval_expr(&unop(UnaryOp::Tgamma, con(5.0)), &HashMap::new()).unwrap();
        assert!((r - 24.0).abs() < 1e-10, "Γ(5) = 24, got {r}");
    }
    #[test] fn tgamma_half() {
        // Γ(1/2) = √π
        let r = eval_expr(&unop(UnaryOp::Tgamma, con(0.5)), &HashMap::new()).unwrap();
        assert!((r - std::f64::consts::PI.sqrt()).abs() < 1e-10, "Γ(0.5) = √π, got {r}");
    }
    #[test] fn tgamma_zero_is_none() {
        // Γ(0) = ±Inf → discarded
        let r = eval_expr(&unop(UnaryOp::Tgamma, con(0.0)), &HashMap::new());
        assert!(r.is_none() || r.map(|v| v.is_infinite()).unwrap_or(false),
                "Γ(0) should be None or Inf");
    }
    #[test] fn lgamma_one_is_zero() {
        // ln|Γ(1)| = ln(1) = 0
        let r = eval_expr(&unop(UnaryOp::Lgamma, con(1.0)), &HashMap::new()).unwrap();
        assert!(r.abs() < 1e-14, "lgamma(1) = 0, got {r}");
    }
    #[test] fn lgamma_five() {
        // ln|Γ(5)| = ln(24)
        let r = eval_expr(&unop(UnaryOp::Lgamma, con(5.0)), &HashMap::new()).unwrap();
        assert!((r - 24.0_f64.ln()).abs() < 1e-10, "lgamma(5) = ln(24), got {r}");
    }
}
