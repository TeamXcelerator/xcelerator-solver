// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! End-to-end integration tests.

use std::path::PathBuf;
use xcelerator_solver::{
    config::{OperatorsConfig, SolverConfig, TermsConfig},
    csv_loader::{load_csv, DataPoint},
    output::run_validation,
    pipeline::run_pipeline,
    vocabulary::Vocabulary,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[allow(clippy::too_many_arguments)]
fn cfg(
    vars: &[&str], consts: &[&str], binary: &[&str], unary: &[&str],
    max_complexity: usize, max_time: f64, max_error: f64,
    pinned: Option<Vec<String>>,
    top: Option<u32>,
) -> SolverConfig {
    SolverConfig {
        training_csv:    fixture("linear_train.csv"),
        validation_csv:  fixture("linear_val.csv"),
        target_column:   "y".to_string(),
        max_error_pct:   max_error,
        max_complexity,
        max_time_secs:   max_time,
        output_file:     PathBuf::from("out_test.txt"),
        top_candidates:  top,
        max_threads:     Some(2),
        precision_digits: None,
        pinned_terms:    pinned,
        error_metric:   None,
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

fn load_training(path: &std::path::Path, vars: &[&str], target: &str) -> Vec<DataPoint> {
    let vv: Vec<String> = vars.iter().map(|s| s.to_string()).collect();
    load_csv(path, &vv, target).unwrap().0
}

fn run(cfg: &SolverConfig, training: &[DataPoint]) -> xcelerator_solver::pipeline::PipelineResult {
    let headers: Vec<String> = cfg.terms.variables.clone();
    let vocab = Vocabulary::from_config(cfg, &headers).unwrap();
    run_pipeline(&vocab, cfg, training, vec![])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn known_linear_solution_found() {
    // y = 2*x + 1 should be found with near-zero MAPE
    let c = cfg(&["x"], &["1","2"], &["add","multiply"], &[], 5, 30.0, 0.001, None, None);
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    assert!(!result.top_entries.is_empty(), "no solution found");
    let best = &result.top_entries[0];
    assert!(best.train_error < 0.001, "error too high: {}", best.train_error);
}

#[test]
fn streaming_evaluates_expressions() {
    let c = cfg(&["x"], &["1"], &["add","multiply"], &["ln"], 3, 30.0, 100.0, None, None);
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    assert!(result.stats.expressions_evaluated > 0, "expected expressions to be evaluated");
}

#[test]
fn timeout_respected() {
    let c = cfg(&["x"], &["1","2","3"], &["add","multiply","divide"], &["sqrt","ln"], 9, 0.001, 100.0, None, None);
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    assert!(result.stats.timed_out, "should have timed out");
}

#[test]
fn mape_filter_strict_no_results() {
    // Pool can only form "x", "1", "x+1", "1+1" etc. — cannot fit y=2x+1 within 0.0001%
    let c = cfg(&["x"], &["1"], &["add"], &[], 3, 30.0, 0.0001, None, None);
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    assert!(result.top_entries.is_empty(), "should find no results within tight threshold");
}

#[test]
fn pool_constraint_no_multiply_no_fit() {
    // y = x^2; pool has no "squared" or "power" → can't fit within 0.001%
    let c = SolverConfig {
        training_csv:   fixture("quadratic_train.csv"),
        validation_csv: fixture("quadratic_train.csv"),
        target_column:  "y".to_string(),
        max_error_pct:  0.001,
        max_complexity: 5,
        max_time_secs:  30.0,
        output_file:    PathBuf::from("out_test.txt"),
        top_candidates: None, max_threads: Some(2), precision_digits: None,
        pinned_terms:   None,
        error_metric:   None,
        terms: TermsConfig {
            variables: vec!["x".to_string()],
            constants: vec!["1".to_string(), "2".to_string()],
            composite: vec![],
        },
        operators: OperatorsConfig {
            binary: vec!["add".to_string(), "subtract".to_string()],
            unary:  vec![],  // no "squared" operator
        },
    };
    let vv = vec!["x".to_string()];
    let (training, _) = load_csv(&fixture("quadratic_train.csv"), &vv, "y").unwrap();
    let headers = vv;
    let vocab = Vocabulary::from_config(&c, &headers).unwrap();
    let result = run_pipeline(&vocab, &c, &training, vec![]);
    assert!(result.top_entries.is_empty(),
        "without squared/power, can't fit y=x^2 to 0.001% MAPE");
}

#[test]
fn multi_variable_found() {
    let c = SolverConfig {
        training_csv: fixture("multivar_train.csv"),
        validation_csv: fixture("multivar_train.csv"),
        target_column: "z".to_string(),
        max_error_pct: 0.001,
        max_complexity: 3,
        max_time_secs: 30.0,
        output_file: PathBuf::from("out_test.txt"),
        top_candidates: None, max_threads: Some(2), precision_digits: None,
        pinned_terms: None,
        error_metric:   None,
        terms: TermsConfig {
            variables: vec!["x".to_string(), "y".to_string()],
            constants: vec![],
            composite: vec![],
        },
        operators: OperatorsConfig {
            binary: vec!["add".to_string()],
            unary: vec![],
        },
    };
    let vv = vec!["x".to_string(), "y".to_string()];
    let (training, _) = load_csv(&fixture("multivar_train.csv"), &vv, "z").unwrap();
    let headers = vv.clone();
    let vocab = Vocabulary::from_config(&c, &headers).unwrap();
    let result = run_pipeline(&vocab, &c, &training, vec![]);
    assert!(!result.top_entries.is_empty(), "should find x + y");
    let best = &result.top_entries[0];
    assert!(best.train_error < 0.001, "error: {}", best.train_error);
}

#[test]
fn top_n_cap_respected() {
    let c = cfg(&["x"], &["1","2","3"], &["add","multiply","subtract"], &[], 5, 30.0, 100.0, None, Some(3));
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    assert!(result.top_entries.len() <= 3, "top_n cap violated: {}", result.top_entries.len());
}

#[test]
fn missing_validation_csv_gives_na() {
    let c = SolverConfig {
        training_csv:   fixture("linear_train.csv"),
        validation_csv: PathBuf::from("nonexistent_val_xyz.csv"),
        target_column:  "y".to_string(),
        max_error_pct:  0.001,
        max_complexity: 5,
        max_time_secs:  30.0,
        output_file:    PathBuf::from("out_test.txt"),
        top_candidates: None, max_threads: Some(2), precision_digits: None,
        pinned_terms:   None,
        error_metric:   None,
        terms: TermsConfig {
            variables: vec!["x".to_string()],
            constants: vec!["1".to_string(), "2".to_string()],
            composite: vec![],
        },
        operators: OperatorsConfig {
            binary: vec!["add".to_string(), "multiply".to_string()],
            unary: vec![],
        },
    };
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    // Run validation with no data (simulate missing val CSV)
    let final_entries = run_validation(result.top_entries, None,
        xcelerator_solver::error_metric::ErrorMetric::Mape);
    for e in &final_entries {
        assert_eq!(e.val_error, None, "expected N/A val_error");
    }
}

#[test]
fn bad_csv_row_skipped_with_warning() {
    // bad_row_train.csv has one non-parseable row
    let c = SolverConfig {
        training_csv:   fixture("bad_row_train.csv"),
        validation_csv: fixture("linear_val.csv"),
        target_column:  "y".to_string(),
        max_error_pct:  0.001,
        max_complexity: 5,
        max_time_secs:  30.0,
        output_file:    PathBuf::from("out_test.txt"),
        top_candidates: None, max_threads: Some(2), precision_digits: None,
        pinned_terms:   None,
        error_metric:   None,
        terms: TermsConfig {
            variables: vec!["x".to_string()],
            constants: vec!["1".to_string(), "2".to_string()],
            composite: vec![],
        },
        operators: OperatorsConfig {
            binary: vec!["add".to_string(), "multiply".to_string()],
            unary: vec![],
        },
    };
    let vv = vec!["x".to_string()];
    let (training, warns) = load_csv(&fixture("bad_row_train.csv"), &vv, "y").unwrap();
    assert!(!warns.is_empty(), "expected a warning about the bad row");
    assert!(training.len() >= 2, "should have at least 2 valid rows");

    let headers = vv.clone();
    let vocab = Vocabulary::from_config(&c, &headers).unwrap();
    let result = run_pipeline(&vocab, &c, &training, warns);
    assert!(!result.top_entries.is_empty(), "should find solution despite bad row");
    assert!(!result.warnings.is_empty(), "warning should be in result");
}

#[test]
fn output_file_written() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let c = SolverConfig {
        training_csv:   fixture("linear_train.csv"),
        validation_csv: fixture("linear_val.csv"),
        target_column:  "y".to_string(),
        max_error_pct:  0.001,
        max_complexity: 5,
        max_time_secs:  30.0,
        output_file:    tmp.path().to_path_buf(),
        top_candidates: None, max_threads: Some(2), precision_digits: None,
        pinned_terms:   None,
        error_metric:   None,
        terms: TermsConfig {
            variables: vec!["x".to_string()],
            constants: vec!["1".to_string(), "2".to_string()],
            composite: vec![],
        },
        operators: OperatorsConfig {
            binary: vec!["add".to_string(), "multiply".to_string()],
            unary: vec![],
        },
    };
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let headers = vec!["x".to_string()];
    let vocab = Vocabulary::from_config(&c, &headers).unwrap();
    let result = run_pipeline(&vocab, &c, &training, vec![]);
    let val_data = load_training(&fixture("linear_val.csv"), &["x"], "y");
    let final_entries = run_validation(result.top_entries, Some(&val_data),
        xcelerator_solver::error_metric::ErrorMetric::Mape);

    let mut tee = xcelerator_solver::output::Tee::new();
    xcelerator_solver::output::print_table(&mut tee, &final_entries, &result.stats, "f64",
        xcelerator_solver::error_metric::ErrorMetric::Mape, 20);
    tee.finalize(tmp.path()).unwrap();

    let content = std::fs::read_to_string(tmp.path()).unwrap();
    assert!(!content.is_empty(), "output file should not be empty");
    assert!(content.contains("Expression"), "output should contain table headers");
}

#[test]
fn pinned_term_found() {
    // y = 2*x + 1; pin "multiply(2, x)" — solution must contain 2*x
    let c = SolverConfig {
        training_csv:   fixture("linear_train.csv"),
        validation_csv: fixture("linear_val.csv"),
        target_column:  "y".to_string(),
        max_error_pct:  0.001,
        max_complexity: 5,
        max_time_secs:  30.0,
        output_file:    PathBuf::from("out_test.txt"),
        top_candidates: None, max_threads: Some(2), precision_digits: None,
        pinned_terms:   Some(vec!["multiply(2, x)".to_string()]),
        error_metric:   None,
        terms: TermsConfig {
            variables: vec!["x".to_string()],
            constants: vec!["1".to_string(), "2".to_string()],
            composite: vec![],
        },
        operators: OperatorsConfig {
            binary: vec!["add".to_string(), "multiply".to_string()],
            unary: vec![],
        },
    };
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    assert!(!result.top_entries.is_empty(), "should find solution with pin");
    for e in &result.top_entries {
        assert!(
            e.expr.contains_subtree(
                &xcelerator_solver::expr::Expr::Binary(
                    xcelerator_solver::expr::BinOp::Mul,
                    Box::new(xcelerator_solver::expr::Expr::Const(
                        2.0, xcelerator_solver::expr::ConstSource::Literal("2".to_string()))),
                    Box::new(xcelerator_solver::expr::Expr::Var("x".to_string())),
                ).canonical()
            ),
            "result '{}' must contain 2*x sub-tree", e.display
        );
    }
}

#[test]
fn pinned_term_excludes_non_matching() {
    // Pin "multiply(3, x)" (3*x can't be formed with pool {1,2}); no results expected
    let c = SolverConfig {
        training_csv:   fixture("linear_train.csv"),
        validation_csv: fixture("linear_val.csv"),
        target_column:  "y".to_string(),
        max_error_pct:  0.001,
        max_complexity: 5,
        max_time_secs:  30.0,
        output_file:    PathBuf::from("out_test.txt"),
        top_candidates: None, max_threads: Some(2), precision_digits: None,
        pinned_terms:   Some(vec!["multiply(2, 2)".to_string()]),  // 4, never in a near-zero MAPE formula for this data
        error_metric:   None,
        terms: TermsConfig {
            variables: vec!["x".to_string()],
            constants: vec!["1".to_string(), "2".to_string()],
            composite: vec![],
        },
        operators: OperatorsConfig {
            binary: vec!["add".to_string(), "multiply".to_string()],
            unary: vec![],
        },
    };
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    // The only formulas within 0.001% MAPE for y=2x+1 must contain "4" (2*2) — unlikely
    // This verifies pinned terms actively filter results.
    for e in &result.top_entries {
        let pin_canonical = xcelerator_solver::expr::Expr::Binary(
            xcelerator_solver::expr::BinOp::Mul,
            Box::new(xcelerator_solver::expr::Expr::Const(2.0, xcelerator_solver::expr::ConstSource::Literal("2".to_string()))),
            Box::new(xcelerator_solver::expr::Expr::Const(2.0, xcelerator_solver::expr::ConstSource::Literal("2".to_string()))),
        ).canonical();
        assert!(
            e.expr.contains_subtree(&pin_canonical),
            "result '{}' failed pinned check", e.display
        );
    }
}

#[test]
fn validation_ordering_by_val_mape() {
    // Run and confirm final output is sorted by val_mape (or train_mape if no val)
    let c = cfg(&["x"], &["1","2"], &["add","multiply"], &[], 5, 30.0, 5.0, None, Some(10));
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    let val_data = load_training(&fixture("linear_val.csv"), &["x"], "y");
    let final_entries = run_validation(result.top_entries, Some(&val_data),
        xcelerator_solver::error_metric::ErrorMetric::Mape);

    // Entries must be sorted: val_error ascending, None last
    for w in final_entries.windows(2) {
        match (w[0].val_error, w[1].val_error) {
            (Some(a), Some(b)) => assert!(a <= b, "not sorted: {a} > {b}"),
            (Some(_), None)    => {} // Some before None
            (None, Some(_))    => panic!("None before Some — wrong order"),
            (None, None)       => {} // both None, order by train_error is fine
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke test diagnostic
// ---------------------------------------------------------------------------

#[test]
fn smoke_config_finds_results() {
    // Exact mirror of smoke/solver.toml — if this passes but CLI doesn't,
    // the bug is in main.rs; if this also fails, it's in the pipeline.
    let c = SolverConfig {
        training_csv:    fixture("linear_train.csv"),
        validation_csv:  fixture("linear_val.csv"),
        target_column:   "y".to_string(),
        max_error_pct:   1.0,
        max_complexity:  5,
        max_time_secs:   30.0,
        output_file:     PathBuf::from("out_test.txt"),
        top_candidates:  Some(5),
        max_threads:     Some(2),
        precision_digits: None,
        pinned_terms:    None,
        error_metric:   None,
        terms: TermsConfig {
            variables: vec!["x".to_string()],
            constants: vec!["1".to_string(), "2".to_string()],
            composite: vec![],
        },
        operators: OperatorsConfig {
            binary: vec!["add".to_string(), "multiply".to_string(), "subtract".to_string()],
            unary:  vec![],
        },
    };
    let training = load_training(&fixture("linear_train.csv"), &["x"], "y");
    let result = run(&c, &training);
    assert!(
        !result.top_entries.is_empty(),
        "smoke config produced no results! evaluated={}",
        result.stats.expressions_evaluated,
    );
}
