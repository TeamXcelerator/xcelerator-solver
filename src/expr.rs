// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Expression AST: types, canonical form, display, subtree containment.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp { Sin, Cos, Tan, Asin, Acos, Atan, Sqrt, Squared, Cubed, Ln, Log10, Exp, Neg, Abs, Tanh, Sinh, Cosh, Erf, Tgamma, Lgamma }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp { Add, Sub, Mul, Div, Pow }

/// Source of a constant value — used by the HP evaluator to reconstruct
/// the value at full precision without going through f64.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstSource {
    Named(String),   // "Pi", "e", "Tau", "Phi"
    Literal(String), // "1.5", "2", etc. — parsed via rug::Float::parse in HP mode
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// f64 approximation for display/canonical; ConstSource for HP eval.
    Const(f64, ConstSource),
    Var(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// A pre-built composite term treated as a single atomic building block.
    /// Counts as complexity 1; transparent for eval/display/canonical/dedup.
    Composite(Box<Expr>),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Binary operator display precedence (higher = tighter binding).
fn bin_prec(op: BinOp) -> u8 {
    match op {
        BinOp::Add | BinOp::Sub => 1,
        BinOp::Mul | BinOp::Div => 2,
        BinOp::Pow => 3,
    }
}

/// Whether a binary op is commutative (used for canonical sorting).
fn is_commutative(op: BinOp) -> bool {
    matches!(op, BinOp::Add | BinOp::Mul)
}

/// Canonical name for a unary operator (used in S-expression keys).
fn unary_canonical_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Sin     => "sin",
        UnaryOp::Cos     => "cos",
        UnaryOp::Tan     => "tan",
        UnaryOp::Asin    => "asin",
        UnaryOp::Acos    => "acos",
        UnaryOp::Atan    => "atan",
        UnaryOp::Sqrt    => "sqrt",
        UnaryOp::Squared => "sq",
        UnaryOp::Cubed   => "cube",
        UnaryOp::Ln      => "ln",
        UnaryOp::Log10   => "log10",
        UnaryOp::Exp     => "exp",
        UnaryOp::Neg     => "neg",
        UnaryOp::Abs     => "abs",
        UnaryOp::Tanh    => "tanh",
        UnaryOp::Sinh    => "sinh",
        UnaryOp::Cosh    => "cosh",
        UnaryOp::Erf     => "erf",
        UnaryOp::Tgamma  => "tgamma",
        UnaryOp::Lgamma  => "lgamma",
    }
}

/// Canonical symbol for a binary operator.
fn bin_canonical_sym(op: BinOp) -> &'static str {
    match op { BinOp::Add => "+", BinOp::Sub => "-",
               BinOp::Mul => "*", BinOp::Div => "/", BinOp::Pow => "^" }
}

// ---------------------------------------------------------------------------
// Expr methods
// ---------------------------------------------------------------------------

impl Expr {
    // --- structural ---

    /// Total number of AST nodes (leaf = 1, each op node adds 1).
    pub fn complexity(&self) -> usize {
        match self {
            Expr::Const(_, _) | Expr::Var(_) => 1,
            // A composite term is a single atomic building block.
            Expr::Composite(_) => 1,
            Expr::Unary(_, child)    => 1 + child.complexity(),
            Expr::Binary(_, l, r)   => 1 + l.complexity() + r.complexity(),
        }
    }

    /// Collect all `Var` names referenced anywhere in the tree.
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Expr::Var(s)          => { vars.insert(s.clone()); }
            Expr::Const(_, _)     => {}
            Expr::Composite(inner) => inner.collect_vars(vars),
            Expr::Unary(_, c)     => c.collect_vars(vars),
            Expr::Binary(_, l, r) => { l.collect_vars(vars); r.collect_vars(vars); }
        }
    }

    // --- canonical form (visited-set key) ---

    /// Deterministic S-expression key.
    /// Add and Mul sort their children lexicographically so that
    /// `a + b` and `b + a` share one key.
    pub fn canonical(&self) -> String {
        match self {
            Expr::Const(v, _) => format!("{:.6}", v),
            Expr::Var(s)      => s.clone(),
            // Transparent: a composite dedups with the equivalent hand-built tree.
            Expr::Composite(inner) => inner.canonical(),
            Expr::Unary(op, child) => {
                format!("({} {})", unary_canonical_name(*op), child.canonical())
            }
            Expr::Binary(op, l, r) => {
                let lk = l.canonical();
                let rk = r.canonical();
                let sym = bin_canonical_sym(*op);
                if is_commutative(*op) {
                    let (a, b) = if lk <= rk { (&lk, &rk) } else { (&rk, &lk) };
                    format!("({sym} {a} {b})")
                } else {
                    format!("({sym} {lk} {rk})")
                }
            }
        }
    }

    // --- human-readable display ---

    /// Infix display with minimal correct parenthesisation.
    pub fn display(&self) -> String {
        display_inner(self, 0, false)
    }

    // --- subtree containment ---

    /// Returns true if this expression or any of its sub-trees has the
    /// same canonical form as `target_canonical`.
    pub fn contains_subtree(&self, target_canonical: &str) -> bool {
        if self.canonical() == target_canonical {
            return true;
        }
        match self {
            Expr::Unary(_, child)    => child.contains_subtree(target_canonical),
            Expr::Binary(_, l, r)   =>
                l.contains_subtree(target_canonical)
                || r.contains_subtree(target_canonical),
            Expr::Composite(inner)  => inner.contains_subtree(target_canonical),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// display_inner
// ---------------------------------------------------------------------------

/// Recursively build a display string.
/// `parent_prec` is the precedence of the enclosing binary operator (0 = top).
/// `is_right_of_noncomm` indicates whether this node is the RIGHT child of
/// a Sub, Div, or Pow — in those cases same-prec sub-expressions also need
/// parentheses to avoid changing semantics.
fn display_inner(expr: &Expr, parent_prec: u8, is_right_of_noncomm: bool) -> String {
    match expr {
        Expr::Const(_, ConstSource::Named(name)) => name.clone(),
        Expr::Const(_, ConstSource::Literal(s))  => s.clone(),
        Expr::Var(s)      => s.clone(),

        // Transparent for display: render the inner expression with the same
        // precedence context so embedded composites parenthesize correctly.
        Expr::Composite(inner) => display_inner(inner, parent_prec, is_right_of_noncomm),

        Expr::Unary(op, child) => {
            // Unary functions never need outer parens; give child a very high
            // parent_prec so it won't wrap unless it's itself a Binary inside
            // a function argument (function calls handle parens themselves).
            let inner = display_inner(child, 0, false);
            match op {
                UnaryOp::Sin     => format!("sin({inner})"),
                UnaryOp::Cos     => format!("cos({inner})"),
                UnaryOp::Tan     => format!("tan({inner})"),
                UnaryOp::Asin    => format!("asin({inner})"),
                UnaryOp::Acos    => format!("acos({inner})"),
                UnaryOp::Atan    => format!("atan({inner})"),
                UnaryOp::Sqrt    => format!("sqrt({inner})"),
                UnaryOp::Squared => format!("({inner})^2"),
                UnaryOp::Cubed   => format!("({inner})^3"),
                UnaryOp::Ln      => format!("ln({inner})"),
                UnaryOp::Log10   => format!("log({inner})"),
                UnaryOp::Exp     => format!("exp({inner})"),
                UnaryOp::Neg     => format!("-{inner}"),
                UnaryOp::Abs     => format!("abs({inner})"),
                UnaryOp::Tanh    => format!("tanh({inner})"),
                UnaryOp::Sinh    => format!("sinh({inner})"),
                UnaryOp::Cosh    => format!("cosh({inner})"),
                UnaryOp::Erf     => format!("erf({inner})"),
                UnaryOp::Tgamma  => format!("tgamma({inner})"),
                UnaryOp::Lgamma  => format!("lgamma({inner})"),
            }
        }

        Expr::Binary(op, l, r) => {
            let my_prec = bin_prec(*op);
            let sym = match op {
                BinOp::Add => "+", BinOp::Sub => "-",
                BinOp::Mul => "*", BinOp::Div => "/", BinOp::Pow => "^",
            };
            // Right child of Sub/Div needs parens even at same prec.
            let right_is_noncomm = matches!(op, BinOp::Sub | BinOp::Div);
            let ls = display_inner(l, my_prec, false);
            let rs = display_inner(r, my_prec, right_is_noncomm);

            let s = format!("{ls} {sym} {rs}");
            // Wrap if our prec is lower than parent, or we're on the right
            // side of a non-commutative op at the same precedence level.
            if my_prec < parent_prec || (is_right_of_noncomm && my_prec == parent_prec) {
                format!("({s})")
            } else {
                s
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- convenience constructors ---
    fn var(s: &str) -> Expr { Expr::Var(s.to_string()) }
    fn con(v: f64) -> Expr { Expr::Const(v, ConstSource::Literal(format!("{v}"))) }
    fn add(l: Expr, r: Expr) -> Expr { Expr::Binary(BinOp::Add, Box::new(l), Box::new(r)) }
    fn sub(l: Expr, r: Expr) -> Expr { Expr::Binary(BinOp::Sub, Box::new(l), Box::new(r)) }
    fn mul(l: Expr, r: Expr) -> Expr { Expr::Binary(BinOp::Mul, Box::new(l), Box::new(r)) }
    fn div(l: Expr, r: Expr) -> Expr { Expr::Binary(BinOp::Div, Box::new(l), Box::new(r)) }
    fn ln(e: Expr)  -> Expr { Expr::Unary(UnaryOp::Ln, Box::new(e)) }
    fn neg(e: Expr) -> Expr { Expr::Unary(UnaryOp::Neg, Box::new(e)) }

    // --- complexity ---
    #[test] fn complexity_leaf()    { assert_eq!(var("x").complexity(), 1); }
    #[test] fn complexity_const()   { assert_eq!(con(2.0).complexity(), 1); }
    #[test] fn complexity_binary()  { assert_eq!(add(var("x"), con(1.0)).complexity(), 3); }
    #[test] fn complexity_unary()   { assert_eq!(ln(var("x")).complexity(), 2); }
    #[test] fn complexity_nested()  {
        // (x + 1) * 2 → 5 nodes
        assert_eq!(mul(add(var("x"), con(1.0)), con(2.0)).complexity(), 5);
    }

    // --- canonical: commutative normalisation ---
    #[test] fn canonical_add_commutative() {
        assert_eq!(add(var("x"), con(1.0)).canonical(),
                   add(con(1.0), var("x")).canonical());
    }
    #[test] fn canonical_mul_commutative() {
        assert_eq!(mul(con(2.0), var("x")).canonical(),
                   mul(var("x"), con(2.0)).canonical());
    }
    #[test] fn canonical_sub_not_commutative() {
        assert_ne!(sub(var("x"), con(1.0)).canonical(),
                   sub(con(1.0), var("x")).canonical());
    }
    #[test] fn canonical_div_not_commutative() {
        assert_ne!(div(var("x"), con(2.0)).canonical(),
                   div(con(2.0), var("x")).canonical());
    }

    // --- canonical: nested commutative ---
    #[test] fn canonical_nested_commutative() {
        // (a + b) + c  vs  c + (b + a) — both should canonicalise the same way
        let a = var("a"); let b = var("b"); let c = var("c");
        let lhs = add(add(a.clone(), b.clone()), c.clone());
        let rhs = add(c, add(b, a));
        assert_eq!(lhs.canonical(), rhs.canonical());
    }

    // --- contains_subtree ---
    #[test] fn contains_self() {
        let e = mul(con(2.0), var("x"));
        assert!(e.contains_subtree(&e.canonical()));
    }
    #[test] fn contains_inner() {
        let inner = mul(con(2.0), var("x"));
        let outer = add(inner.clone(), con(1.0));
        assert!(outer.contains_subtree(&inner.canonical()));
    }
    #[test] fn not_contains_different() {
        let outer = add(mul(con(2.0), var("x")), con(1.0));
        let other = mul(con(3.0), var("x"));
        assert!(!outer.contains_subtree(&other.canonical()));
    }
    #[test] fn leaf_does_not_contain_binary() {
        let e = var("x");
        assert!(!e.contains_subtree(&mul(con(2.0), var("x")).canonical()));
    }

    // --- display: basic cases ---
    #[test] fn display_var()   { assert_eq!(var("x").display(), "x"); }
    #[test] fn display_const() { assert_eq!(con(2.0).display(), "2"); }
    #[test] fn display_add()   { assert_eq!(add(var("x"), con(1.0)).display(), "x + 1"); }
    #[test] fn display_sub_right_paren() {
        // a - (b - c) must parenthesize right side
        let e = sub(var("a"), sub(var("b"), var("c")));
        assert!(e.display().contains("(b - c)"), "got: {}", e.display());
    }
    #[test] fn display_mul_over_add() {
        // (a + b) * c must parenthesize left side
        let e = mul(add(var("a"), var("b")), var("c"));
        assert!(e.display().contains("(a + b)"), "got: {}", e.display());
    }
    #[test] fn display_unary_ln() {
        assert_eq!(ln(var("x")).display(), "ln(x)");
    }
    #[test] fn display_neg() {
        assert_eq!(neg(var("x")).display(), "-x");
    }
    #[test] fn display_named_constant() {
        // Named constants must show their name, not the f64 value.
        let pi = Expr::Const(std::f64::consts::PI, ConstSource::Named("Pi".to_string()));
        assert_eq!(pi.display(), "Pi");
        let e = Expr::Const(std::f64::consts::E, ConstSource::Named("e".to_string()));
        assert_eq!(e.display(), "e");
    }
    #[test] fn display_literal_constant() {
        let two = Expr::Const(2.0, ConstSource::Literal("2".to_string()));
        assert_eq!(two.display(), "2");
    }
    #[test] fn composite_counts_as_one() {
        // A composite wrapping a 3-node expression still has complexity 1.
        let inner = add(var("x"), con(1.0));         // complexity 3
        assert_eq!(inner.complexity(), 3);
        let comp = Expr::Composite(Box::new(inner.clone()));
        assert_eq!(comp.complexity(), 1, "composite must count as atomic");
        // But it's transparent for canonical/display/eval.
        assert_eq!(comp.canonical(), inner.canonical());
        assert_eq!(comp.display(), inner.display());
    }
    #[test] fn composite_embedded_complexity() {
        // ln(composite) = 1 (ln) + 1 (composite) = 2, regardless of inner size.
        let inner = add(mul(con(2.0), var("x")), con(1.0)); // complexity 5
        let comp = Expr::Composite(Box::new(inner));
        let wrapped = ln(comp);
        assert_eq!(wrapped.complexity(), 2);
    }

    // --- variables ---
    #[test] fn variables_single() {
        let vars = var("x").variables();
        assert_eq!(vars, ["x".to_string()].iter().cloned().collect());
    }
    #[test] fn variables_multi() {
        let e = add(mul(var("x"), var("y")), var("x"));
        let vars = e.variables();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert_eq!(vars.len(), 2);
    }
    #[test] fn variables_const_empty() {
        assert!(con(3.5).variables().is_empty());
    }
}
