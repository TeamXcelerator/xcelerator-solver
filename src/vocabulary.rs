// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Vocabulary: allowed terms, operators, constants, and seed expressions.
//! Maps spelled-out config names ("add", "multiply", "Pi") to typed values.

use crate::config::SolverConfig;
use crate::expr::{BinOp, ConstSource, Expr, UnaryOp};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Named constant table (f64 approximations for display/canonical)
// ---------------------------------------------------------------------------

const NAMED_CONSTANTS_F64: &[(&str, f64)] = &[
    ("Pi",      std::f64::consts::PI),
    ("e",       std::f64::consts::E),
    ("E",       std::f64::consts::E),
    ("Tau",     std::f64::consts::TAU),
    ("Phi",     1.618033988749895_f64),
    ("gamma",   0.5772156649015329_f64),   // Euler-Mascheroni constant
    ("Catalan", 0.915_965_594_177_219_f64),   // Catalan's constant G
];

// ---------------------------------------------------------------------------
// BinOp / UnaryOp name resolution
// ---------------------------------------------------------------------------

impl BinOp {
    /// Parse a spelled-out config name into a `BinOp`.
    pub fn from_name(s: &str) -> Result<BinOp, String> {
        match s {
            "add"      => Ok(BinOp::Add),
            "subtract" => Ok(BinOp::Sub),
            "multiply" => Ok(BinOp::Mul),
            "divide"   => Ok(BinOp::Div),
            "power"    => Ok(BinOp::Pow),
            _ => Err(format!(
                "Unknown binary operator '{}'. Valid: add, subtract, multiply, divide, power",
                s
            )),
        }
    }
}

impl UnaryOp {
    /// Parse a spelled-out config name into a `UnaryOp`.
    pub fn from_name(s: &str) -> Result<UnaryOp, String> {
        match s {
            "sqrt"    => Ok(UnaryOp::Sqrt),
            "squared" => Ok(UnaryOp::Squared),
            "cubed"   => Ok(UnaryOp::Cubed),
            "sine"    => Ok(UnaryOp::Sin),
            "cosine"  => Ok(UnaryOp::Cos),
            "tangent"   => Ok(UnaryOp::Tan),
            "arcsine"   => Ok(UnaryOp::Asin),
            "arccosine" => Ok(UnaryOp::Acos),
            "arctangent" => Ok(UnaryOp::Atan),
            "ln"      => Ok(UnaryOp::Ln),
            "log"     => Ok(UnaryOp::Log10),
            "exp"     => Ok(UnaryOp::Exp),
            "negate"  => Ok(UnaryOp::Neg),
            "abs"     => Ok(UnaryOp::Abs),
            "tanh"    => Ok(UnaryOp::Tanh),
            "sinh"    => Ok(UnaryOp::Sinh),
            "cosh"    => Ok(UnaryOp::Cosh),
            // Special functions
            "erf"     => Ok(UnaryOp::Erf),
            // Gamma function Γ(x) — NOTE: distinct from the "gamma" *constant* (Euler-Mascheroni γ).
            // Use `constants = ["gamma"]` for the constant; use `unary = ["tgamma"]` for the function.
            "tgamma"  => Ok(UnaryOp::Tgamma),
            "lgamma"  => Ok(UnaryOp::Lgamma),
            _ => Err(format!(
                "Unknown unary operator '{}'. Valid: sqrt, squared, cubed, sine, cosine, tangent, arcsine, arccosine, arctangent, ln, log, exp, negate, abs, tanh, sinh, cosh, erf, tgamma, lgamma",
                s
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Constant resolution
// ---------------------------------------------------------------------------

/// Resolve a constant name or numeric literal to `(f64, ConstSource)`.
///
/// Checks the named-constant table first; falls through to `f64` parse.
/// Returns `Err` if the name is not recognised and does not parse as a number.
pub fn resolve_constant_f64(name: &str) -> Result<(f64, ConstSource), String> {
    // Named constant?
    for &(key, val) in NAMED_CONSTANTS_F64 {
        if key == name {
            return Ok((val, ConstSource::Named(name.to_string())));
        }
    }
    // Numeric literal?
    match name.parse::<f64>() {
        Ok(v) => Ok((v, ConstSource::Literal(name.to_string()))),
        Err(_) => Err(format!(
            "Unknown constant '{}'. Use a named constant (Pi, e, Tau, Phi) or a numeric literal.",
            name
        )),
    }
}


// ---------------------------------------------------------------------------
// Vocabulary struct
// ---------------------------------------------------------------------------

/// The complete set of building blocks the solver is allowed to use.
pub struct Vocabulary {
    /// Complexity-1 atomic terms: constants and variables only.
    /// Unary and binary compositions are generated on demand by the generator.
    pub atoms: Vec<Expr>,
    pub binary_ops: Vec<BinOp>,
    pub unary_ops: Vec<UnaryOp>,
    /// Parsed pinned sub-components (from pinned.rs).
    pub pinned: Vec<Expr>,
}

impl Vocabulary {
    /// The complexity-1 atoms (constants + variables) the generator seeds from.
    pub fn atoms(&self) -> &[Expr] {
        &self.atoms
    }

    /// Build a `Vocabulary` from the solver config, validating all names
    /// against the CSV headers actually present in the training data.
    ///
    /// `csv_headers` should exclude the target column (callers must strip it).
    pub fn from_config(
        cfg: &SolverConfig,
        csv_headers: &[String],
    ) -> Result<Vocabulary, String> {
        // --- resolve binary ops ---
        let binary_ops: Vec<BinOp> = cfg.operators.binary.iter()
            .map(|s| BinOp::from_name(s))
            .collect::<Result<_, _>>()?;

        // --- resolve unary ops ---
        let unary_ops: Vec<UnaryOp> = cfg.operators.unary.iter()
            .map(|s| UnaryOp::from_name(s))
            .collect::<Result<_, _>>()?;

        // --- validate variable names against CSV headers ---
        for var_name in &cfg.terms.variables {
            if !csv_headers.contains(var_name) {
                return Err(format!(
                    "Variable '{}' listed in [terms] variables was not found \
                     in the training CSV headers: {:?}",
                    var_name, csv_headers
                ));
            }
        }

        // --- resolve constants ---
        let constants: Vec<Expr> = cfg.terms.constants.iter()
            .map(|name| {
                let (v, src) = resolve_constant_f64(name)?;
                Ok(Expr::Const(v, src))
            })
            .collect::<Result<Vec<_>, String>>()?;

        // --- build complexity-1 atoms: constants + variables ---
        // Unary-of-var and all higher compositions are generated on demand,
        // so we do NOT pre-build them here (keeps memory bounded).
        let mut atoms: Vec<Expr> = Vec::new();
        atoms.extend(constants);
        for var_name in &cfg.terms.variables {
            atoms.push(Expr::Var(var_name.clone()));
        }

        // --- parse composite terms and add as additional atoms ---
        // Composite terms are pre-built sub-expressions (e.g. "multiply(2, Pi)")
        // treated as atomic seeds. The solver can use them anywhere a constant
        // or variable would go, without having to rediscover the structure.
        if !cfg.terms.composite.is_empty() {
            let composites = parse_expr_list(
                &cfg.terms.composite, cfg, &binary_ops, &unary_ops,
                "[terms] composite"
            )?;
            // Wrap each in Composite so it counts as a single atomic building block.
            for c in composites {
                atoms.push(Expr::Composite(Box::new(c)));
            }
        }

        // --- parse pinned terms ---
        let pinned = match &cfg.pinned_terms {
            None => Vec::new(),
            Some(v) if v.is_empty() => Vec::new(),
            Some(v) => parse_expr_list(v, cfg, &binary_ops, &unary_ops, "pinned_terms")?,
        };

        Ok(Vocabulary { atoms, binary_ops, unary_ops, pinned })
    }
}

/// Parse a list of function-call expression strings into `Expr` trees.
/// Used by both composite terms and pinned terms.
fn parse_expr_list(
    raw:        &[String],
    cfg:        &SolverConfig,
    binary_ops: &[BinOp],
    unary_ops:  &[UnaryOp],
    context:    &str,   // for error messages
) -> Result<Vec<Expr>, String> {
    // Build terminal lookup from constants and variables.
    let mut terminal_map: HashMap<String, Expr> = HashMap::new();
    for name in &cfg.terms.constants {
        let (v, src) = resolve_constant_f64(name)
            .map_err(|e| format!("{context}: {e}"))?;
        terminal_map.insert(name.clone(), Expr::Const(v, src));
    }
    for name in &cfg.terms.variables {
        terminal_map.insert(name.clone(), Expr::Var(name.clone()));
    }

    let binary_map: HashMap<String, BinOp> = cfg.operators.binary.iter()
        .zip(binary_ops.iter())
        .map(|(name, &op)| (name.clone(), op))
        .collect();

    let unary_map: HashMap<String, UnaryOp> = cfg.operators.unary.iter()
        .zip(unary_ops.iter())
        .map(|(name, &op)| (name.clone(), op))
        .collect();

    let parser = crate::pinned::PinnedTermParser::new(&terminal_map, &binary_map, &unary_map);

    raw.iter()
        .map(|s| parser.parse(s).map_err(|e| format!("{context} '{}': {e}", s)))
        .collect()
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ConstSource;

    #[test]
    fn pi_resolves_named() {
        let (v, src) = resolve_constant_f64("Pi").unwrap();
        assert!((v - std::f64::consts::PI).abs() < 1e-14);
        assert_eq!(src, ConstSource::Named("Pi".to_string()));
    }

    #[test]
    fn numeric_literal_resolves() {
        let (v, src) = resolve_constant_f64("2.5").unwrap();
        assert_eq!(v, 2.5);
        assert_eq!(src, ConstSource::Literal("2.5".to_string()));
    }

    #[test]
    fn integer_literal_resolves() {
        let (v, _) = resolve_constant_f64("1").unwrap();
        assert_eq!(v, 1.0);
    }

    #[test]
    fn unknown_constant_errors() {
        assert!(resolve_constant_f64("Omega").is_err());
        assert!(resolve_constant_f64("xyz").is_err());
    }

    #[test]
    fn binop_from_name_all() {
        assert_eq!(BinOp::from_name("add").unwrap(),      BinOp::Add);
        assert_eq!(BinOp::from_name("subtract").unwrap(), BinOp::Sub);
        assert_eq!(BinOp::from_name("multiply").unwrap(), BinOp::Mul);
        assert_eq!(BinOp::from_name("divide").unwrap(),   BinOp::Div);
        assert_eq!(BinOp::from_name("power").unwrap(),    BinOp::Pow);
        assert!(BinOp::from_name("plus").is_err());
    }

    #[test]
    fn unaryop_from_name_all() {
        assert_eq!(UnaryOp::from_name("sqrt").unwrap(),    UnaryOp::Sqrt);
        assert_eq!(UnaryOp::from_name("squared").unwrap(), UnaryOp::Squared);
        assert_eq!(UnaryOp::from_name("cubed").unwrap(),   UnaryOp::Cubed);
        assert_eq!(UnaryOp::from_name("sine").unwrap(),    UnaryOp::Sin);
        assert_eq!(UnaryOp::from_name("cosine").unwrap(),  UnaryOp::Cos);
        assert_eq!(UnaryOp::from_name("ln").unwrap(),      UnaryOp::Ln);
        assert_eq!(UnaryOp::from_name("log").unwrap(),     UnaryOp::Log10);
        assert_eq!(UnaryOp::from_name("exp").unwrap(),     UnaryOp::Exp);
        assert_eq!(UnaryOp::from_name("negate").unwrap(),  UnaryOp::Neg);
        assert_eq!(UnaryOp::from_name("abs").unwrap(),     UnaryOp::Abs);
        assert_eq!(UnaryOp::from_name("erf").unwrap(),     UnaryOp::Erf);
        assert_eq!(UnaryOp::from_name("tgamma").unwrap(),  UnaryOp::Tgamma);
        assert_eq!(UnaryOp::from_name("lgamma").unwrap(),  UnaryOp::Lgamma);
        assert!(UnaryOp::from_name("tan").is_err());
    }

    #[test]
    fn catalan_constant_resolves() {
        let (v, src) = resolve_constant_f64("Catalan").unwrap();
        assert!((v - 0.9159655941772190_f64).abs() < 1e-14);
        assert_eq!(src, ConstSource::Named("Catalan".to_string()));
    }

    // Helper: build a minimal SolverConfig-like structure for from_config tests.
    fn make_cfg(vars: &[&str], consts: &[&str], binary: &[&str], unary: &[&str])
        -> SolverConfig
    {
        use std::path::PathBuf;
        use crate::config::{OperatorsConfig, TermsConfig};
        SolverConfig {
            training_csv:    PathBuf::from("t.csv"),
            validation_csv:  PathBuf::from("v.csv"),
            target_column:   "y".to_string(),
            max_error_pct:   5.0,
            max_complexity:  7,
            max_time_secs:   60.0,
            output_file:     PathBuf::from("out.txt"),
            top_candidates:  None,
            max_threads:     None,
            precision_digits: None,
            pinned_terms:    None,
            error_metric:    None,
            terms: TermsConfig {
                variables: vars.iter().map(|s| s.to_string()).collect(),
                constants: consts.iter().map(|s| s.to_string()).collect(),
                composite: vec![],
            },
            operators: OperatorsConfig {
                binary: binary.iter().map(|s| s.to_string()).collect(),
                unary:  unary.iter().map(|s| s.to_string()).collect(),
            },
        }
    }

    #[test]
    fn from_config_atom_count() {
        // vars=["x"], consts=["1","2"] → 3 atoms (unary combos generated on demand)
        let cfg = make_cfg(&["x"], &["1", "2"], &["add"], &["sqrt", "ln"]);
        let headers = vec!["x".to_string()];
        let vocab = Vocabulary::from_config(&cfg, &headers).unwrap();
        assert_eq!(vocab.atoms().len(), 3, "expected 3 atoms, got {}", vocab.atoms().len());
    }

    #[test]
    fn from_config_unknown_variable_errors() {
        let cfg = make_cfg(&["z"], &["1"], &["add"], &[]);
        let headers = vec!["x".to_string()]; // "z" not present
        assert!(Vocabulary::from_config(&cfg, &headers).is_err());
    }

    #[test]
    fn from_config_unknown_binop_errors() {
        let cfg = make_cfg(&["x"], &["1"], &["plus"], &[]);
        let headers = vec!["x".to_string()];
        assert!(Vocabulary::from_config(&cfg, &headers).is_err());
    }

    #[test]
    fn from_config_no_pinned_ok() {
        let cfg = make_cfg(&["x"], &["1"], &["add"], &[]);
        let vocab = Vocabulary::from_config(&cfg, &["x".to_string()]).unwrap();
        assert!(vocab.pinned.is_empty());
    }
}
