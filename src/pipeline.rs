// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Pipeline wiring: owns VisitedSet, spawns threads, connects all actors.
//!
//! Orchestrates: Generator → [channel] → Evaluator pool → [channel] → Aggregator

use crate::aggregator::{Aggregator, ResultEntry};
use crate::config::SolverConfig;
use crate::csv_loader::DataPoint;
use crate::evaluator::{run_evaluator, EvalJob};
use crate::expr::Expr;
use crate::generator::run_generator;
use crate::vocabulary::Vocabulary;
use std::io::IsTerminal;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SearchStats {
    pub expressions_evaluated: u64,
    pub elapsed_secs:          f64,
    pub timed_out:             bool,
}

#[derive(Debug)]
pub struct PipelineResult {
    pub top_entries: Vec<ResultEntry>,
    pub stats:       SearchStats,
    pub warnings:    Vec<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the full search pipeline and return the top candidates.
///
/// `vocab`    — resolved vocabulary (terms, operators, pinned).
/// `config`   — solver configuration (complexity, time, error threshold, etc.)
/// `training` — loaded training data points (f64 path).
/// `warnings` — startup warnings accumulated before this call (passed through).
pub fn run_pipeline(
    vocab:    &Vocabulary,
    config:   &SolverConfig,
    training: &[DataPoint],
    warnings: Vec<String>,
) -> PipelineResult {
    // Configure rayon thread pool (best-effort; ignore if already initialised).
    if let Some(n) = config.max_threads {
        if n > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n as usize)
                .build_global()
                .ok();
        }
    }

    let top_n         = config.effective_top_candidates();
    let max_complexity = config.max_complexity;
    let metric        = config.effective_metric();
    let deadline       = Instant::now() + Duration::from_secs_f64(config.max_time_secs);

    // Bounded work channel: capacity scales with worker count. When full, the
    // generator's send blocks — this is the backpressure that throttles
    // generation to the workers' pace and keeps memory bounded.
    let num_workers = config.max_threads
        .filter(|&n| n > 0)
        .map(|n| n as usize)
        .unwrap_or_else(rayon::current_num_threads)
        .max(1);
    let channel_capacity = num_workers * 64;

    let mut aggregator = Aggregator::new(top_n, config.max_error_pct, vocab.pinned.clone());

    let start = Instant::now();
    let mut eval_count: u64 = 0;

    // Use std::thread::scope so the generator and evaluator can borrow
    // stack-local data (&vocab, training) without requiring 'static lifetimes.
    std::thread::scope(|s| {
        let (candidate_tx, candidate_rx) = crossbeam_channel::bounded::<Expr>(channel_capacity);
        let (result_tx,    result_rx)    = crossbeam_channel::bounded::<EvalJob>(channel_capacity);

        // Thread 1: generator — dedicated rayon pool (num_workers/4 threads)
        // so it never competes with the evaluator's global pool.
        let num_gen_threads = (num_workers / 4).max(8);
        s.spawn(move || {
            run_generator(vocab, max_complexity, deadline, candidate_tx, num_gen_threads);
            // candidate_tx dropped here → evaluator threads see channel closed
        });

        // Thread 2: evaluator pool — rayon workers drain candidate_rx.
        s.spawn(|| {
            run_evaluator(candidate_rx, result_tx, training, metric);
            // result_tx dropped here → aggregator sees channel closed
        });

        // Main scope thread: aggregate results and render live progress.
        // Only render the ANSI in-place display when stderr is a real terminal;
        // when redirected to a file/pipe, ANSI cursor moves would be garbage, so
        // we skip the live display entirely (the final table still prints).
        let live_display = std::io::stderr().is_terminal();
        let render_interval = Duration::from_millis(500);
        let mut last_render = Instant::now() - render_interval;
        let mut render_lines: usize = 0;
        let mut current_complexity: usize = 0;

        for job in &result_rx {
            eval_count += 1;
            // Track the deepest complexity level reached so far (monotonic).
            let comp = job.expr.complexity();
            if comp > current_complexity {
                current_complexity = comp;
            }
            let heap_changed = aggregator.push(job);
            let _ = heap_changed; // used only for render decisions below

            if live_display && last_render.elapsed() >= render_interval {
                render_lines = render_live(
                    &aggregator,
                    eval_count,
                    current_complexity,
                    start.elapsed(),
                    render_lines,
                    metric,
                );
                last_render = Instant::now();
            }
        }

        // Clear the live display before printing the final table.
        if live_display {
            clear_live(render_lines);
        }
        // scope waits for all spawned threads before returning
    });

    let elapsed   = start.elapsed();
    let timed_out = elapsed >= Duration::from_secs_f64(config.max_time_secs);

    PipelineResult {
        top_entries: aggregator.into_results(),
        stats: SearchStats {
            expressions_evaluated: eval_count,
            elapsed_secs:          elapsed.as_secs_f64(),
            timed_out,
        },
        warnings,
    }
}

// ---------------------------------------------------------------------------
// Live progress display (stderr, ANSI in-place update)
// ---------------------------------------------------------------------------

/// Render the live candidates table to stderr, overwriting the previous render.
/// Returns the number of lines printed so the next call can clear them.
fn render_live(
    aggregator: &Aggregator,
    evaluated:  u64,
    complexity: usize,
    elapsed:    Duration,
    prev_lines: usize,
    metric:     crate::error_metric::ErrorMetric,
) -> usize {
    use std::io::Write;

    // Move cursor up and clear previous render.
    if prev_lines > 0 {
        eprint!("\x1b[{}A\x1b[0J", prev_lines);
    }

    let mut lines = 0;
    let entries = aggregator.peek_results();

    eprintln!("Xcelerator Solver -- searching...");
    lines += 1;

    eprintln!(
        "  Complexity: {}  |  Evaluated: {}  |  Elapsed: {:.1}s  |  Candidates found: {}",
        complexity,
        evaluated,
        elapsed.as_secs_f64(),
        entries.len(),
    );
    lines += 1;
    eprintln!("  {:-<72}", "");
    lines += 1;

    if entries.is_empty() {
        eprintln!("  (no candidates within threshold yet)");
        lines += 1;
    } else {
        eprintln!("  {:>3}  {:<56} {:>10}  {:>8}", "#", "Expression", format!("Train {}", metric.label()), "Cplx");
        lines += 1;
        eprintln!("  {:-<72}", "");
        lines += 1;
        for (i, e) in entries.iter().enumerate() {
            // Truncate long expressions so they never wrap onto a second physical
            // line — wrapping breaks the ANSI cursor-up line count.
            let disp: String = e.display.chars().take(55).collect();
            let disp = if e.display.chars().count() > 55 {
                format!("{disp}…")
            } else {
                disp
            };
            eprintln!(
                "  {:>3}  {:<56} {:>10.4}  {:>8}",
                i + 1, disp, e.train_error, e.complexity
            );
            lines += 1;
        }
    }

    eprintln!("  {:-<72}", "");
    lines += 1;

    std::io::stderr().flush().ok();
    lines
}

/// Clear the live display by overwriting with blank lines.
fn clear_live(lines: usize) {
    use std::io::Write;
    if lines > 0 {
        eprint!("\x1b[{}A\x1b[0J", lines);
        std::io::stderr().flush().ok();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OperatorsConfig, SolverConfig, TermsConfig};
    use crate::csv_loader::DataPoint;
    use crate::vocabulary::Vocabulary;
    use std::path::PathBuf;

    fn make_config(
        vars: &[&str], consts: &[&str], binary: &[&str], unary: &[&str],
        max_complexity: usize, max_time_secs: f64, max_error_pct: f64,
    ) -> SolverConfig {
        SolverConfig {
            training_csv:    PathBuf::from("t.csv"),
            validation_csv:  PathBuf::from("v.csv"),
            target_column:   "y".to_string(),
            max_error_pct,
            max_complexity,
            max_time_secs,
            output_file:     PathBuf::from("out.txt"),
            top_candidates:  None, max_threads: None, precision_digits: None,
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

    fn data(pairs: &[(f64, f64)]) -> Vec<DataPoint> {
        pairs.iter().map(|&(x, y)| DataPoint {
            inputs: [("x".to_string(), x)].into(), output: y,
        }).collect()
    }

    #[test]
    fn finds_linear_solution() {
        // y = 2*x + 1
        let cfg = make_config(&["x"], &["1", "2"], &["add", "multiply"], &[], 5, 30.0, 0.001);
        let headers = vec!["x".to_string()];
        let vocab = Vocabulary::from_config(&cfg, &headers).unwrap();
        let training = data(&[(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0)]);
        let result = run_pipeline(&vocab, &cfg, &training, vec![]);
        assert!(!result.top_entries.is_empty(), "should find a solution");
        let best = &result.top_entries[0];
        assert!(best.train_error < 0.001, "best error: {}", best.train_error);
    }

    #[test]
    fn streaming_evaluates_expressions() {
        let cfg = make_config(&["x"], &["1"], &["add", "multiply"], &["ln"], 3, 30.0, 100.0);
        let headers = vec!["x".to_string()];
        let vocab = Vocabulary::from_config(&cfg, &headers).unwrap();
        let training = data(&[(1.0, 1.0), (2.0, 2.0)]);
        let result = run_pipeline(&vocab, &cfg, &training, vec![]);
        assert!(result.stats.expressions_evaluated > 0, "should evaluate expressions");
    }

    #[test]
    fn timeout_respected() {
        let cfg = make_config(&["x"], &["1","2","3"], &["add","multiply","divide"], &["sqrt","ln"], 9, 0.001, 100.0);
        let headers = vec!["x".to_string()];
        let vocab = Vocabulary::from_config(&cfg, &headers).unwrap();
        let training = data(&[(1.0, 1.0), (2.0, 2.0)]);
        let result = run_pipeline(&vocab, &cfg, &training, vec![]);
        assert!(result.stats.timed_out, "should have timed out");
    }
}
