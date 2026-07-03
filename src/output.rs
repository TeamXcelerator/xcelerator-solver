// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Output module: validation scoring, table/JSON formatting, Tee writer.
//!
//! Results are written to both stdout (immediately) and an output file.
//! Each run is timestamped and prepended to the output file, preserving
//! the full run history with the most recent run at the top.

use crate::aggregator::ResultEntry;
use crate::csv_loader::DataPoint;
use crate::error_metric::{eval_all, ErrorMetric};
use crate::pipeline::SearchStats;
use std::path::Path;

// ---------------------------------------------------------------------------
// Tee writer
// ---------------------------------------------------------------------------

/// Writes to stdout in real-time and buffers content for the output file.
/// Call `finalize(path)` at the end to prepend the buffer to the output file.
pub struct Tee {
    buffer: String,
}

impl Default for Tee {
    fn default() -> Self {
        Self::new()
    }
}

impl Tee {
    pub fn new() -> Self {
        Self { buffer: String::new() }
    }

    pub fn writeln(&mut self, line: &str) {
        println!("{line}");
        self.buffer.push_str(line);
        self.buffer.push('\n');
    }

    /// Prepend this run's output to the output file.
    /// If the file already exists, its prior content is preserved after
    /// the new run block, giving a chronological history (newest on top).
    pub fn finalize(self, path: &Path) -> std::io::Result<()> {
        let existing = if path.exists() {
            std::fs::read_to_string(path).unwrap_or_default()
        } else {
            String::new()
        };
        let sep = if existing.is_empty() { "" } else { "\n" };
        let full = format!("{}{sep}{}", self.buffer, existing);
        std::fs::write(path, full)
    }
}

// ---------------------------------------------------------------------------
// Final entry (after validation)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FinalEntry {
    pub expr_display: String,
    pub train_error:  f64,
    pub val_error:    Option<f64>,
    pub complexity:   usize,
}

// ---------------------------------------------------------------------------
// Validation phase
// ---------------------------------------------------------------------------

/// Run each top-X result through the validation data set using `metric`.
/// Sorts final entries by validation error ascending; entries without a
/// validation score sort last (fallback: sort by train_error).
pub fn run_validation(
    entries:    Vec<ResultEntry>,
    validation: Option<&[DataPoint]>,
    metric:     ErrorMetric,
) -> Vec<FinalEntry> {
    let mut final_entries: Vec<FinalEntry> = entries.into_iter().map(|e| {
        let val_error = match validation {
            None => None,
            Some(val_data) => {
                let actuals: Vec<f64> = val_data.iter().map(|p| p.output).collect();
                eval_all(&e.expr, val_data)
                    .and_then(|pred| metric.compute(&pred, &actuals))
            }
        };
        FinalEntry {
            expr_display: e.display,
            train_error:  e.train_error,
            val_error,
            complexity:   e.complexity,
        }
    }).collect();

    // Sort by val_error ascending; None sorts last (fallback to train_error).
    final_entries.sort_by(|a, b| {
        match (a.val_error, b.val_error) {
            (Some(va), Some(vb)) => va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None)    => a.train_error.partial_cmp(&b.train_error)
                                    .unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    final_entries
}

// ---------------------------------------------------------------------------
// Timestamp
// ---------------------------------------------------------------------------

fn utc_now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let s  = secs % 60;
    let m  = (secs / 60) % 60;
    let h  = (secs / 3600) % 24;
    let d  = secs / 86400;

    let (year, month, day) = unix_days_to_ymd(d);
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Algorithm from Howard Hinnant's date library (public domain).
fn unix_days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z  = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    let y   = yoe + era as u64 * 400;
    let doy = doe - (365*yoe + yoe/4 - yoe/100);
    let mp  = (5*doy + 2) / 153;
    let day = doy - (153*mp + 2)/5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let yr  = if mon <= 2 { y + 1 } else { y };
    (yr, mon, day)
}

// ---------------------------------------------------------------------------
// Table output
// ---------------------------------------------------------------------------

pub fn print_table(
    tee:             &mut Tee,
    entries:         &[FinalEntry],
    stats:           &SearchStats,
    precision_label: &str,
    metric:          ErrorMetric,
    top_n:           usize,
) {
    let ts = utc_now_iso8601();
    let train_hdr = format!("Train {}", metric.label());
    let val_hdr   = format!("Val {}", metric.label());
    tee.writeln(&format!("=== Run: {ts} ==="));
    tee.writeln("Xcelerator Solver -- results");
    tee.writeln(&format!(
        "Precision: {}   Metric: {}   Top candidates: {}   Timeout: {}",
        precision_label, metric.label(), top_n,
        if stats.timed_out { "yes" } else { "no" }
    ));
    tee.writeln(&"-".repeat(80));
    tee.writeln(&format!(
        " {:<5} {:<30} {:<14} {:<12} {:<10}",
        "Rank", "Expression", train_hdr, val_hdr, "Complexity"
    ));
    tee.writeln(&"-".repeat(80));

    for (i, e) in entries.iter().enumerate() {
        let val_str = e.val_error.map_or("N/A".to_string(), |v| format!("{:.6}", v));
        tee.writeln(&format!(
            " {:<5} {:<30} {:<14.6} {:<12} {:<10}",
            i + 1, e.expr_display, e.train_error, val_str, e.complexity
        ));
    }

    tee.writeln(&"-".repeat(80));
    tee.writeln(&format!(
        "Evaluated: {}  |  Elapsed: {:.2}s",
        stats.expressions_evaluated, stats.elapsed_secs
    ));
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

pub fn print_json(
    tee:             &mut Tee,
    entries:         &[FinalEntry],
    stats:           &SearchStats,
    warnings:        &[String],
    precision_label: &str,
    metric:          ErrorMetric,
    top_n:           usize,
) {
    let ts = utc_now_iso8601();

    let results: Vec<serde_json::Value> = entries.iter().enumerate().map(|(i, e)| {
        serde_json::json!({
            "rank":        i + 1,
            "expression":  e.expr_display,
            "train_error": e.train_error,
            "val_error":   e.val_error,
            "complexity":  e.complexity,
        })
    }).collect();

    let obj = serde_json::json!({
        "run_timestamp":  ts,
        "precision_mode": precision_label,
        "error_metric":   metric.label(),
        "top_candidates": top_n,
        "results": results,
        "stats": {
            "expressions_evaluated": stats.expressions_evaluated,
            "elapsed_secs":          stats.elapsed_secs,
            "timed_out":             stats.timed_out,
        },
        "warnings": warnings,
    });

    tee.writeln(&serde_json::to_string_pretty(&obj).unwrap_or_default());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregator::ResultEntry;
    use crate::expr::{BinOp, ConstSource, Expr};

    fn entry(err: f64, display: &str) -> ResultEntry {
        ResultEntry {
            expr:        Expr::Const(err, ConstSource::Literal(format!("{err}"))),
            display:     display.to_string(),
            train_error: err,
            complexity:  1,
        }
    }

    fn data(pairs: &[(f64, f64)]) -> Vec<DataPoint> {
        pairs.iter().map(|&(x, y)| DataPoint {
            inputs: [("x".to_string(), x)].into(), output: y,
        }).collect()
    }

    #[test]
    fn validation_sorts_by_val_error() {
        let two_x = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(2.0, ConstSource::Literal("2".to_string()))),
            Box::new(Expr::Var("x".to_string())),
        );
        let entries = vec![
            ResultEntry { expr: Expr::Var("x".to_string()), display: "x".to_string(),
                          train_error: 1.0, complexity: 1 },
            ResultEntry { expr: two_x, display: "2 * x".to_string(),
                          train_error: 5.0, complexity: 3 },
        ];
        let val_data = data(&[(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]);
        let final_entries = run_validation(entries, Some(&val_data), ErrorMetric::Mape);
        // "2 * x" matches perfectly (val=0), so it should sort first
        assert_eq!(final_entries[0].expr_display, "2 * x");
        assert!(final_entries[0].val_error.unwrap() < 1e-10);
    }

    #[test]
    fn missing_validation_gives_none_val() {
        let entries = vec![entry(1.0, "x")];
        let final_entries = run_validation(entries, None, ErrorMetric::Mape);
        assert_eq!(final_entries[0].val_error, None);
    }

    #[test]
    fn none_val_sorts_last() {
        let entries = vec![entry(0.5, "a"), entry(0.1, "b")];
        let no_val = run_validation(entries, None, ErrorMetric::Mape);
        // Without validation, sort by train_error ascending
        assert_eq!(no_val[0].expr_display, "b");
        assert_eq!(no_val[1].expr_display, "a");
    }

    #[test]
    fn mae_validation_handles_negative() {
        // Negative targets — MAE must produce finite val_error, no panic.
        let entries = vec![entry(0.3, "x")];
        let val_data = data(&[(1.0, -0.5), (2.0, 0.1)]);
        let fe = run_validation(entries, Some(&val_data), ErrorMetric::Mae);
        assert!(fe[0].val_error.unwrap().is_finite());
    }

    #[test]
    fn table_contains_headers() {
        let mut tee = Tee::new();
        let entries = vec![FinalEntry {
            expr_display: "x + 1".to_string(),
            train_error: 0.5, val_error: Some(0.6), complexity: 3,
        }];
        let stats = SearchStats {
            expressions_evaluated: 100,
            elapsed_secs: 1.5, timed_out: false,
        };
        print_table(&mut tee, &entries, &stats, "f64", ErrorMetric::Mape, 20);
        assert!(tee.buffer.contains("Expression"));
        assert!(tee.buffer.contains("x + 1") || tee.buffer.contains("x+1"));
        assert!(tee.buffer.contains("0.500000"));
    }

    #[test]
    fn json_is_valid_json() {
        let mut tee = Tee::new();
        let entries = vec![FinalEntry {
            expr_display: "x".to_string(),
            train_error: 1.0, val_error: None, complexity: 1,
        }];
        let stats = SearchStats {
            expressions_evaluated: 50,
            elapsed_secs: 0.5, timed_out: false,
        };
        print_json(&mut tee, &entries, &stats, &[], "f64", ErrorMetric::Mape, 20);
        let parsed: serde_json::Value = serde_json::from_str(tee.buffer.trim()).unwrap();
        assert!(parsed["results"].is_array());
        assert_eq!(parsed["results"][0]["expression"], "x");
    }

    #[test]
    fn tee_finalize_prepend() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "OLD CONTENT\n").unwrap();

        let mut tee = Tee::new();
        tee.writeln("NEW LINE");
        tee.finalize(tmp.path()).unwrap();

        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(content.starts_with("NEW LINE\n"));
        assert!(content.contains("OLD CONTENT"));
    }
}
