// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! HP expression evaluator (requires --features hp).
//!
//! Identical logic to eval.rs but uses rug::Float arithmetic throughout.
//! Named constants are re-materialized via MPFR built-ins at full precision.

use crate::expr::{BinOp, ConstSource, Expr, UnaryOp};
use crate::hp::hp_constant;
use rug::ops::Pow;
use rug::Float;
use std::collections::HashMap;

/// Evaluate `expr` at `prec` MPFR bits for one HP data point.
/// Returns `None` on any domain error or non-finite result.
pub fn eval_expr_hp(
    expr: &Expr,
    inputs: &HashMap<String, Float>,
    prec: u32,
) -> Option<Float> {
    let result = eval_hp_inner(expr, inputs, prec)?;
    if result.is_finite() { Some(result) } else { None }
}

fn eval_hp_inner(
    expr: &Expr,
    inputs: &HashMap<String, Float>,
    prec: u32,
) -> Option<Float> {
    match expr {
        Expr::Const(_, ConstSource::Named(name)) => {
            // Reconstruct at full MPFR precision — NOT promoted from f64.
            hp_constant(name, prec)
        }
        Expr::Const(_, ConstSource::Literal(s)) => {
            let parsed = Float::parse(s).ok()?;
            let v = Float::with_val(prec, parsed);
            if v.is_finite() { Some(v) } else { None }
        }

        Expr::Var(s) => inputs.get(s.as_str()).cloned(),

        // Composite is transparent for evaluation.
        Expr::Composite(inner) => eval_expr_hp(inner, inputs, prec),

        Expr::Unary(op, child) => {
            let v = eval_expr_hp(child, inputs, prec)?;
            let r: Float = match op {
                UnaryOp::Sin     => v.sin(),
                UnaryOp::Cos     => v.cos(),
                UnaryOp::Tan     => v.tan(),
                UnaryOp::Asin    => {
                    if v < -1 || v > 1 { return None; }
                    v.asin()
                }
                UnaryOp::Acos    => {
                    if v < -1 || v > 1 { return None; }
                    v.acos()
                }
                UnaryOp::Atan    => v.atan(),
                UnaryOp::Sqrt    => {
                    if v < 0 { return None; }
                    v.sqrt()
                }
                UnaryOp::Squared => {
                    let v2 = v.clone();
                    v * v2
                }
                UnaryOp::Cubed => {
                    let v2 = v.clone();
                    let v3 = v.clone();
                    v * v2 * v3
                }
                UnaryOp::Ln => {
                    if v <= 0 { return None; }
                    v.ln()
                }
                UnaryOp::Log10 => {
                    if v <= 0 { return None; }
                    v.log10()
                }
                UnaryOp::Exp  => v.exp(),
                UnaryOp::Neg  => -v,
                UnaryOp::Abs  => v.abs(),
                UnaryOp::Tanh => v.tanh(),
                UnaryOp::Sinh => v.sinh(),
                UnaryOp::Cosh => v.cosh(),
                UnaryOp::Erf     => v.erf(),
                UnaryOp::Tgamma  => v.gamma(),           // Γ(x) via MPFR
                UnaryOp::Lgamma  => v.ln_gamma().0,      // ln|Γ(x)|; .0 is the value, .1 is the sign
            };
            if r.is_finite() { Some(r) } else { None }
        }

        Expr::Binary(op, l, r) => {
            let lv = eval_expr_hp(l, inputs, prec)?;
            let rv = eval_expr_hp(r, inputs, prec)?;
            let result: Float = match op {
                BinOp::Add => lv + rv,
                BinOp::Sub => lv - rv,
                BinOp::Mul => lv * rv,
                BinOp::Div => {
                    let abs_rv = rv.clone().abs();
                    let eps = Float::with_val(prec, 1.0e-300_f64);
                    if abs_rv < eps { return None; }
                    lv / rv
                }
                BinOp::Pow => lv.pow(rv),
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
    use crate::expr::{ConstSource, Expr};
    use crate::hp::HpConfig;
    use rug::float::Constant;

    fn prec() -> u32 { HpConfig::for_decimal_digits(50).precision_bits }

    fn inputs(pairs: &[(&str, f64)]) -> HashMap<String, Float> {
        let p = prec();
        pairs.iter()
            .map(|&(k, v)| (k.to_string(), Float::with_val(p, v)))
            .collect()
    }
    fn var(s: &str)  -> Expr { Expr::Var(s.to_string()) }
    fn con(v: f64)   -> Expr { Expr::Const(v, ConstSource::Literal(format!("{v}"))) }
    fn pi_const()    -> Expr { Expr::Const(std::f64::consts::PI,
                                 ConstSource::Named("Pi".to_string())) }
    fn binop(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary(op, Box::new(l), Box::new(r))
    }
    fn unop(op: UnaryOp, e: Expr) -> Expr { Expr::Unary(op, Box::new(e)) }

    fn to_f64(r: Option<Float>) -> Option<f64> { r.map(|v| v.to_f64()) }

    #[test] fn add_hp()      { assert_eq!(to_f64(eval_expr_hp(&binop(BinOp::Add, con(2.0), con(3.0)), &HashMap::new(), prec())), Some(5.0)); }
    #[test] fn sqrt_four()   { let r = to_f64(eval_expr_hp(&unop(UnaryOp::Sqrt, con(4.0)), &HashMap::new(), prec())); assert!((r.unwrap() - 2.0).abs() < 1e-14); }
    #[test] fn sqrt_neg_none() { assert_eq!(eval_expr_hp(&unop(UnaryOp::Sqrt, con(-1.0)), &HashMap::new(), prec()), None); }
    #[test] fn ln_one()      { let r = to_f64(eval_expr_hp(&unop(UnaryOp::Ln, con(1.0)), &HashMap::new(), prec())); assert!((r.unwrap()).abs() < 1e-14); }
    #[test] fn ln_zero_none(){ assert_eq!(eval_expr_hp(&unop(UnaryOp::Ln, con(0.0)), &HashMap::new(), prec()), None); }
    #[test] fn div_zero_none(){ assert_eq!(eval_expr_hp(&binop(BinOp::Div, con(1.0), con(0.0)), &HashMap::new(), prec()), None); }
    #[test] fn variable_hp() {
        let r = to_f64(eval_expr_hp(&var("x"), &inputs(&[("x", 7.0)]), prec()));
        assert_eq!(r, Some(7.0));
    }
    #[test] fn pi_const_hp() {
        let p = prec();
        let r = eval_expr_hp(&pi_const(), &HashMap::new(), p).unwrap();
        let expected = Float::with_val(p, Constant::Pi);
        assert_eq!(r, expected, "Pi must be exact MPFR value, not f64-promoted");
    }
    #[test] fn squared_hp() {
        let r = to_f64(eval_expr_hp(&unop(UnaryOp::Squared, con(5.0)), &HashMap::new(), prec()));
        assert_eq!(r, Some(25.0));
    }

    // --- special functions (HP) ---
    #[test] fn erf_zero_hp() {
        assert_eq!(to_f64(eval_expr_hp(&unop(UnaryOp::Erf, con(0.0)), &HashMap::new(), prec())), Some(0.0));
    }
    #[test] fn erf_large_hp() {
        let r = to_f64(eval_expr_hp(&unop(UnaryOp::Erf, con(5.0)), &HashMap::new(), prec())).unwrap();
        assert!((r - 1.0).abs() < 1e-14, "erf(5) ≈ 1 at HP, got {r}");
    }
    #[test] fn tgamma_five_hp() {
        // Γ(5) = 4! = 24 — exact at HP
        let r = to_f64(eval_expr_hp(&unop(UnaryOp::Tgamma, con(5.0)), &HashMap::new(), prec())).unwrap();
        assert!((r - 24.0).abs() < 1e-14, "Γ(5) = 24 at HP, got {r}");
    }
    #[test] fn lgamma_five_hp() {
        // ln|Γ(5)| = ln(24)
        let r = to_f64(eval_expr_hp(&unop(UnaryOp::Lgamma, con(5.0)), &HashMap::new(), prec())).unwrap();
        assert!((r - 24.0_f64.ln()).abs() < 1e-14, "lgamma(5) = ln(24) at HP, got {r}");
    }
}
