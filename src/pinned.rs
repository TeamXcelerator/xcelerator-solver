// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Pinned sub-components: expression string parser and containment check.
//!
//! Notation accepted by the parser:
//!   binary:  op_name(left_expr, right_expr)
//!   unary:   op_name(expr)
//!   terminal: pool term name or numeric literal

use crate::expr::{BinOp, Expr, UnaryOp};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parses pinned sub-component strings into `Expr` trees.
///
/// Constructed once per run from the vocabulary's operator and terminal maps.
pub struct PinnedTermParser<'a> {
    terminals:   &'a HashMap<String, Expr>,
    binary_ops:  &'a HashMap<String, BinOp>,
    unary_ops:   &'a HashMap<String, UnaryOp>,
}

impl<'a> PinnedTermParser<'a> {
    pub fn new(
        terminals:  &'a HashMap<String, Expr>,
        binary_ops: &'a HashMap<String, BinOp>,
        unary_ops:  &'a HashMap<String, UnaryOp>,
    ) -> Self {
        Self { terminals, binary_ops, unary_ops }
    }

    /// Parse a pinned term string such as `"divide(multiply(4, Pi), ln(N))"`.
    pub fn parse(&self, s: &str) -> Result<Expr, String> {
        let tokens = tokenize(s);
        let mut state = ParseState {
            tokens: &tokens,
            pos: 0,
            terminals: self.terminals,
            binary_ops: self.binary_ops,
            unary_ops: self.unary_ops,
        };
        let expr = state.parse_expr()?;
        if state.pos != tokens.len() {
            return Err(format!(
                "Unexpected tokens after expression: {:?}",
                &tokens[state.pos..]
            ));
        }
        Ok(expr)
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Split input into a flat token list: identifiers, `(`, `)`, `,`.
fn tokenize(s: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();

    for c in s.chars() {
        match c {
            '(' | ')' | ',' => {
                let trimmed = cur.trim().to_string();
                if !trimmed.is_empty() {
                    tokens.push(trimmed);
                }
                cur.clear();
                tokens.push(c.to_string());
            }
            _ if c.is_whitespace() => {
                let trimmed = cur.trim().to_string();
                if !trimmed.is_empty() {
                    tokens.push(trimmed);
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    let trimmed = cur.trim().to_string();
    if !trimmed.is_empty() {
        tokens.push(trimmed);
    }
    tokens
}

// ---------------------------------------------------------------------------
// Recursive-descent parser state
// ---------------------------------------------------------------------------

struct ParseState<'a> {
    tokens:     &'a [String],
    pos:        usize,
    terminals:  &'a HashMap<String, Expr>,
    binary_ops: &'a HashMap<String, BinOp>,
    unary_ops:  &'a HashMap<String, UnaryOp>,
}

impl<'a> ParseState<'a> {
    #[allow(dead_code)]
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn consume(&mut self) -> Result<&str, String> {
        match self.tokens.get(self.pos) {
            Some(t) => { self.pos += 1; Ok(t.as_str()) }
            None    => Err("Unexpected end of expression".to_string()),
        }
    }

    fn expect(&mut self, expected: &str) -> Result<(), String> {
        let got = self.consume()?;
        if got == expected {
            Ok(())
        } else {
            Err(format!("Expected '{}' but got '{}'", expected, got))
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        let token = self.consume()?.to_string();

        // Binary operator call: op(left, right)
        if let Some(&op) = self.binary_ops.get(&token) {
            self.expect("(")?;
            let left = self.parse_expr()?;
            self.expect(",")?;
            let right = self.parse_expr()?;
            self.expect(")")?;
            return Ok(Expr::Binary(op, Box::new(left), Box::new(right)));
        }

        // Unary operator call: op(child)
        if let Some(&op) = self.unary_ops.get(&token) {
            self.expect("(")?;
            let child = self.parse_expr()?;
            self.expect(")")?;
            return Ok(Expr::Unary(op, Box::new(child)));
        }

        // Terminal: variable or constant name
        if let Some(expr) = self.terminals.get(&token) {
            return Ok(expr.clone());
        }

        // Check for stray structural tokens
        if token == "(" || token == ")" || token == "," {
            return Err(format!("Unexpected token '{}'", token));
        }

        Err(format!(
            "Unknown name '{}'. Must be a declared operator, variable, or constant.",
            token
        ))
    }
}

// ---------------------------------------------------------------------------
// Containment check
// ---------------------------------------------------------------------------

/// Returns true if `expr` contains all pinned sub-expressions as sub-trees.
pub fn passes_pinned(expr: &Expr, pinned: &[Expr]) -> bool {
    pinned.iter().all(|p| expr.contains_subtree(&p.canonical()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{BinOp, ConstSource, Expr, UnaryOp};

    fn make_parser() -> (
        HashMap<String, Expr>,
        HashMap<String, BinOp>,
        HashMap<String, UnaryOp>,
    ) {
        let mut terms: HashMap<String, Expr> = HashMap::new();
        terms.insert("Pi".to_string(),
            Expr::Const(std::f64::consts::PI, ConstSource::Named("Pi".to_string())));
        terms.insert("N".to_string(),  Expr::Var("N".to_string()));
        terms.insert("x".to_string(),  Expr::Var("x".to_string()));
        terms.insert("2".to_string(),  Expr::Const(2.0, ConstSource::Literal("2".to_string())));
        terms.insert("4".to_string(),  Expr::Const(4.0, ConstSource::Literal("4".to_string())));

        let mut binary: HashMap<String, BinOp> = HashMap::new();
        binary.insert("multiply".to_string(), BinOp::Mul);
        binary.insert("divide".to_string(),   BinOp::Div);
        binary.insert("add".to_string(),      BinOp::Add);

        let mut unary: HashMap<String, UnaryOp> = HashMap::new();
        unary.insert("ln".to_string(),   UnaryOp::Ln);
        unary.insert("sqrt".to_string(), UnaryOp::Sqrt);

        (terms, binary, unary)
    }

    fn parser<'a>(
        t: &'a HashMap<String, Expr>,
        b: &'a HashMap<String, BinOp>,
        u: &'a HashMap<String, UnaryOp>,
    ) -> PinnedTermParser<'a> {
        PinnedTermParser::new(t, b, u)
    }

    #[test]
    fn parse_terminal_const() {
        let (t, b, u) = make_parser();
        let e = parser(&t, &b, &u).parse("Pi").unwrap();
        assert!(matches!(e, Expr::Const(_, ConstSource::Named(_))));
    }

    #[test]
    fn parse_terminal_var() {
        let (t, b, u) = make_parser();
        let e = parser(&t, &b, &u).parse("N").unwrap();
        assert_eq!(e, Expr::Var("N".to_string()));
    }

    #[test]
    fn parse_unary() {
        let (t, b, u) = make_parser();
        let e = parser(&t, &b, &u).parse("ln(N)").unwrap();
        assert_eq!(e, Expr::Unary(UnaryOp::Ln, Box::new(Expr::Var("N".to_string()))));
    }

    #[test]
    fn parse_binary() {
        let (t, b, u) = make_parser();
        let e = parser(&t, &b, &u).parse("multiply(2, Pi)").unwrap();
        let expected = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(2.0, ConstSource::Literal("2".to_string()))),
            Box::new(Expr::Const(std::f64::consts::PI, ConstSource::Named("Pi".to_string()))),
        );
        // Compare via canonical form (avoids f64 equality noise)
        assert_eq!(e.canonical(), expected.canonical());
    }

    #[test]
    fn parse_nested() {
        let (t, b, u) = make_parser();
        // divide(multiply(4, Pi), ln(N))
        let e = parser(&t, &b, &u).parse("divide(multiply(4, Pi), ln(N))").unwrap();
        // Check structure: Binary(Div, Binary(Mul, 4, Pi), Unary(Ln, N))
        match &e {
            Expr::Binary(BinOp::Div, left, right) => {
                assert!(matches!(left.as_ref(), Expr::Binary(BinOp::Mul, _, _)));
                assert!(matches!(right.as_ref(), Expr::Unary(UnaryOp::Ln, _)));
            }
            _ => panic!("Expected Binary(Div,...), got {:?}", e),
        }
    }

    #[test]
    fn parse_unknown_terminal_errors() {
        let (t, b, u) = make_parser();
        assert!(parser(&t, &b, &u).parse("Omega").is_err());
    }

    #[test]
    fn parse_wrong_arg_count_errors() {
        let (t, b, u) = make_parser();
        // binary op with only one arg
        assert!(parser(&t, &b, &u).parse("multiply(2)").is_err());
    }

    #[test]
    fn parse_mismatched_parens_errors() {
        let (t, b, u) = make_parser();
        assert!(parser(&t, &b, &u).parse("ln(N").is_err());
    }

    #[test]
    fn passes_pinned_empty() {
        let e = Expr::Var("x".to_string());
        assert!(passes_pinned(&e, &[]));
    }

    #[test]
    fn passes_pinned_matches() {
        let (t, b, u) = make_parser();
        let sub = parser(&t, &b, &u).parse("multiply(2, Pi)").unwrap();
        let outer = Expr::Binary(
            BinOp::Add,
            Box::new(sub.clone()),
            Box::new(Expr::Var("x".to_string())),
        );
        assert!(passes_pinned(&outer, &[sub]));
    }

    #[test]
    fn passes_pinned_no_match() {
        let (t, b, u) = make_parser();
        let pin = parser(&t, &b, &u).parse("multiply(4, Pi)").unwrap();
        let expr = Expr::Var("x".to_string());
        assert!(!passes_pinned(&expr, &[pin]));
    }
}
