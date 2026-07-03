// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Xcelerator Solver — deterministic, precision-configurable symbolic
//! regression engine.
//!
//! Given tabular data (CSV) and an explicit vocabulary of allowed terms and
//! operators, the solver exhaustively searches for mathematical expressions
//! that explain the data within a user-specified error threshold.
//!
//! # Design principles
//! - **Deterministic** — Bottom-Up Enumeration (BUE) always finds the simplest
//!   formula at any error level first. Identical inputs produce identical results.
//! - **Explicit pool** — only declared terms and operators are used; no hidden
//!   defaults, no open-ended constant optimization.
//! - **Configurable precision** — expressions can be evaluated at arbitrary
//!   MPFR precision (`--features hp`); named constants (`Pi`, `e`, `gamma`, …)
//!   are materialized at full target precision via MPFR built-ins.
//!
//! # Quick start (library usage)
//! ```no_run
//! use xcelerator_solver::{SolverConfig, solve};
//! let config = SolverConfig::load("solver.toml".as_ref()).unwrap();
//! let result = solve(config).unwrap();
//! ```

pub mod aggregator;
pub mod config;
pub mod csv_loader;
pub mod error_metric;
pub mod eval;
pub mod evaluator;
pub mod expr;
pub mod generator;
pub mod output;
pub mod pipeline;
pub mod pinned;
pub mod vocabulary;

#[cfg(feature = "hp")]
pub mod eval_hp;
#[cfg(feature = "hp")]
pub mod hp;

pub use config::SolverConfig;
pub use output::{FinalEntry, run_validation, print_table, print_json, Tee};
pub use pipeline::{PipelineResult, SearchStats, run_pipeline};

/// Run the solver end-to-end.
///
/// Loads CSVs, builds vocabulary, runs the BUE pipeline, and returns the
/// top candidates before the validation step. Callers can then invoke
/// `run_validation` and `print_table` / `print_json` for output.
pub fn solve(config: SolverConfig) -> Result<PipelineResult, String> {
    use crate::csv_loader::load_csv;
    use crate::vocabulary::Vocabulary;

    let mut warnings: Vec<String> = Vec::new();

    // Load training CSV.
    let var_names = config.terms.variables.clone();
    let (training, train_warns) = load_csv(
        &config.training_csv,
        &var_names,
        &config.target_column,
    )?;
    warnings.extend(train_warns);

    // Derive CSV headers (variable column names, excluding target).
    let csv_headers = var_names.clone();

    // Build vocabulary (validates all names against CSV headers).
    let vocab = Vocabulary::from_config(&config, &csv_headers)?;

    Ok(run_pipeline(&vocab, &config, &training, warnings))
}
