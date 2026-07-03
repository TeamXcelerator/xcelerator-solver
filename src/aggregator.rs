// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Aggregator: bounded top-X heap, error-threshold filter, pinned check.
//!
//! Receives EvalJob results, applies all filters, and keeps the best
//! `top_n` expressions by training error (MAPE/MAE/RMSE — whichever the
//! pipeline chose). Knows nothing about how expressions were generated.

use crate::evaluator::EvalJob;
use crate::expr::Expr;
use crate::pinned::passes_pinned;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Peek entry (non-consuming live display snapshot)
// ---------------------------------------------------------------------------

/// Lightweight snapshot of one heap entry for live progress display.
#[derive(Debug, Clone)]
pub struct PeekEntry {
    pub display:     String,
    pub train_error: f64,
    pub complexity:  usize,
}

// ---------------------------------------------------------------------------
// Result entry (public — used by output.rs)
// ---------------------------------------------------------------------------

/// A candidate expression that has passed all filters.
#[derive(Debug, Clone)]
pub struct ResultEntry {
    pub expr:        Expr,
    pub display:     String,
    pub train_error: f64,
    pub complexity:  usize,
}

// ---------------------------------------------------------------------------
// Internal heap entry
// ---------------------------------------------------------------------------

/// Wrapper for BinaryHeap: natural ordering so that higher error = greater.
/// This makes the heap a max-heap by error — `peek()` yields the WORST
/// (highest-error) entry, which is the one we displace when a better one arrives.
struct RankedEntry {
    train_error: f64,
    expr:        Expr,
    canonical:   String,
}

impl PartialEq  for RankedEntry { fn eq(&self, o: &Self) -> bool { self.train_error == o.train_error } }
impl Eq         for RankedEntry {}
impl PartialOrd for RankedEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for RankedEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural order: higher error = greater = surfaces to heap top (worst).
        self.train_error.partial_cmp(&other.train_error)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Aggregator
// ---------------------------------------------------------------------------

pub struct Aggregator {
    top_n:     usize,
    /// Acceptance threshold in the active metric's units (percent for MAPE,
    /// absolute target units for MAE/RMSE).
    max_error: f64,
    heap:      BinaryHeap<RankedEntry>,
    /// Canonical forms currently in the heap — bounded by top_n.
    /// Prevents duplicate formulas (e.g. x+1 and 1+x) from both appearing.
    seen:      std::collections::HashSet<String>,
    pinned:    Vec<Expr>,
}

impl Aggregator {
    pub fn new(top_n: usize, max_error: f64, pinned: Vec<Expr>) -> Self {
        Self {
            top_n,
            max_error,
            heap: BinaryHeap::new(),
            seen: std::collections::HashSet::new(),
            pinned,
        }
    }

    /// Consider one evaluated expression for the top-N cache.
    /// Returns `true` if the heap was modified (a new entry was added or
    /// the worst incumbent was displaced).
    ///
    /// Discards the entry if:
    /// - `train_error` is NaN (evaluation failed)
    /// - `train_error > max_error` (above threshold)
    /// - Expression does not contain all pinned sub-components
    pub fn push(&mut self, job: EvalJob) -> bool {
        // Guard: discard failures and above-threshold entries.
        if job.train_error.is_nan()           { return false; }
        if job.train_error > self.max_error   { return false; }
        if !passes_pinned(&job.expr, &self.pinned) { return false; }

        // Dedup by canonical form (bounded by top_n — not a global visited set).
        let canonical = job.expr.canonical();
        if self.seen.contains(&canonical) {
            return false;
        }

        let should_push = if self.heap.len() < self.top_n {
            true
        } else {
            // Replace the worst entry only if new entry is strictly better.
            self.heap.peek()
                .is_some_and(|worst| job.train_error < worst.train_error)
        };

        if should_push {
            if self.heap.len() >= self.top_n {
                if let Some(worst) = self.heap.pop() {
                    self.seen.remove(&worst.canonical); // evict worst's key
                }
            }
            self.seen.insert(canonical.clone());
            self.heap.push(RankedEntry {
                train_error: job.train_error,
                expr:        job.expr,
                canonical,
            });
            true  // heap was modified
        } else {
            false
        }
    }

    /// Non-consuming snapshot of the current top-N entries, sorted by
    /// training error ascending. Used for live progress display.
    pub fn peek_results(&self) -> Vec<PeekEntry> {
        let mut entries: Vec<PeekEntry> = self.heap.iter()
            .map(|e| PeekEntry {
                display:     e.expr.display(),
                train_error: e.train_error,
                complexity:  e.expr.complexity(),
            })
            .collect();
        entries.sort_by(|a, b| {
            a.train_error.partial_cmp(&b.train_error)
                .unwrap_or(Ordering::Equal)
        });
        entries
    }

    /// Drain the heap and return all retained entries sorted by training
    /// error ascending (best first), with complexity as tiebreaker.
    pub fn into_results(self) -> Vec<ResultEntry> {
        let mut results: Vec<ResultEntry> = self.heap
            .into_iter()
            .map(|e| ResultEntry {
                display:     e.expr.display(),
                complexity:  e.expr.complexity(),
                train_error: e.train_error,
                expr:        e.expr,
            })
            .collect();

        results.sort_by(|a, b| {
            a.train_error.partial_cmp(&b.train_error)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.complexity.cmp(&b.complexity))
        });

        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::EvalJob;
    use crate::expr::{ConstSource, Expr};

    fn job(err: f64) -> EvalJob {
        EvalJob {
            expr:        Expr::Const(err, ConstSource::Literal(format!("{err}"))),
            train_error: err,
        }
    }

    #[test]
    fn nan_error_discarded() {
        let mut agg = Aggregator::new(5, 10.0, vec![]);
        assert!(!agg.push(job(f64::NAN)), "NaN should return false");
        assert_eq!(agg.into_results().len(), 0);
    }

    #[test]
    fn above_threshold_discarded() {
        let mut agg = Aggregator::new(5, 10.0, vec![]);
        assert!(!agg.push(job(15.0)), "above threshold should return false");
        assert_eq!(agg.into_results().len(), 0);
    }

    #[test]
    fn top_n_cap_respected() {
        let mut agg = Aggregator::new(3, 100.0, vec![]);
        for mape in [1.0, 2.0, 3.0, 4.0, 5.0] {
            agg.push(job(mape));
        }
        let results = agg.into_results();
        assert_eq!(results.len(), 3, "heap should keep only top 3");
    }

    #[test]
    fn push_returns_true_when_heap_changes() {
        let mut agg = Aggregator::new(3, 100.0, vec![]);
        // First 3 pushes all add to the heap
        assert!(agg.push(job(5.0)));
        assert!(agg.push(job(3.0)));
        assert!(agg.push(job(4.0)));
        // Heap full; worse entry should not change heap
        assert!(!agg.push(job(6.0)));
        // Better entry should displace worst
        assert!(agg.push(job(1.0)));
    }

    #[test]
    fn best_entries_retained() {
        let mut agg = Aggregator::new(3, 100.0, vec![]);
        for err in [5.0, 1.0, 3.0, 0.5, 4.0] {
            agg.push(job(err));
        }
        let results = agg.into_results();
        let errs: Vec<f64> = results.iter().map(|r| r.train_error).collect();
        // Best 3 should be 0.5, 1.0, 3.0
        assert!(errs.contains(&0.5));
        assert!(errs.contains(&1.0));
        assert!(errs.contains(&3.0));
        assert!(!errs.contains(&5.0));
    }

    #[test]
    fn results_sorted_ascending() {
        let mut agg = Aggregator::new(5, 100.0, vec![]);
        for err in [3.0, 1.0, 2.0] {
            agg.push(job(err));
        }
        let results = agg.into_results();
        let errs: Vec<f64> = results.iter().map(|r| r.train_error).collect();
        assert_eq!(errs, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn fewer_than_top_n_shows_all() {
        let mut agg = Aggregator::new(20, 100.0, vec![]);
        agg.push(job(1.0));
        agg.push(job(2.0));
        assert_eq!(agg.into_results().len(), 2);
    }

    #[test]
    fn pinned_filter_applied() {
        use crate::expr::{BinOp, Expr};
        let pin = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(2.0, ConstSource::Literal("2".to_string()))),
            Box::new(Expr::Var("x".to_string())),
        );
        let mut agg = Aggregator::new(5, 100.0, vec![pin.clone()]);

        // This expression does NOT contain the pin.
        agg.push(EvalJob { expr: Expr::Var("x".to_string()), train_error: 0.5 });
        assert_eq!(agg.into_results().len(), 0);

        // Now try one that does contain the pin.
        let outer = Expr::Binary(
            BinOp::Add,
            Box::new(pin),
            Box::new(Expr::Const(1.0, ConstSource::Literal("1".to_string()))),
        );
        let mut agg2 = Aggregator::new(5, 100.0, vec![
            Expr::Binary(BinOp::Mul,
                Box::new(Expr::Const(2.0, ConstSource::Literal("2".to_string()))),
                Box::new(Expr::Var("x".to_string())),
            )
        ]);
        agg2.push(EvalJob { expr: outer, train_error: 0.5 });
        assert_eq!(agg2.into_results().len(), 1);
    }
}
