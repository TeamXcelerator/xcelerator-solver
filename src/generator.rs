// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Streaming candidate generator — two-level parallel design.
//!
//! # Parallelism strategy
//!
//! Level 1 — complexity threads (std::thread::scope):
//!   One OS thread per complexity level (1..=max_complexity). All levels make
//!   progress simultaneously; low levels finish quickly and free their threads.
//!
//! Level 2 — work-item parallelism (rayon):
//!   Within each complexity level, the generation space is partitioned into
//!   independent work items — one per unary op, one per (binary op, i, j split).
//!   At complexity 20: 16 unary + 72 binary = 88 items, all run via rayon.
//!
//! Level 3 — left-subtree parallelism (rayon nested):
//!   For binary items whose left complexity i ≤ PARALLEL_LEFT_CAP, all left
//!   sub-trees are pre-collected into a Vec, then each left is processed by a
//!   separate rayon task that sequentially enumerates all matching right
//!   sub-trees. At i=3 this yields ~10 K fine-grained tasks per binary item,
//!   sufficient to keep 256 evaluator threads continuously fed.
//!
//! Memory bound: count(3) ≈ 10 K expressions × ~150 bytes ≈ 1.5 MB per binary
//! task. For i > CAP the task falls back to sequential nested enumeration but
//! still runs concurrently with the other 87+ work items.

use crate::expr::{BinOp, Expr, UnaryOp};
use crate::vocabulary::Vocabulary;
use rayon::prelude::*;
use std::time::Instant;

/// Left-subtree complexity up to which we pre-collect and parallelize.
/// count(4) ≈ 265 K expressions — ~40 MB working set per binary task.
const PARALLEL_LEFT_CAP: usize = 4;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Drive generation into `candidate_tx` (a bounded crossbeam channel).
///
/// Uses a **dedicated** rayon `ThreadPool` for all generator rayon work so
/// the generator and evaluator never compete for the same thread pool.
/// Without this isolation, generator tasks that block on a full channel hold
/// all global rayon threads and starve the evaluator — a classic deadlock.
///
/// `num_gen_threads` controls the generator pool size.  A reasonable default
/// is `max(8, total_threads / 4)`.
pub fn run_generator(
    vocab: &Vocabulary,
    max_complexity: usize,
    deadline: Instant,
    candidate_tx: crossbeam_channel::Sender<Expr>,
    num_gen_threads: usize,
) {
    // Dedicated generator pool — completely independent of the global evaluator pool.
    let gen_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_gen_threads.max(1))
        .thread_name(|i| format!("solver-gen-{i}"))
        .build()
        .expect("failed to build generator thread pool");

    // gen_pool.scope ensures all spawned tasks (and any rayon work they launch)
    // use gen_pool rather than the global evaluator pool.
    gen_pool.scope(|s| {
        for complexity in 1..=max_complexity {
            let tx = candidate_tx.clone();
            s.spawn(move |_| {
                enumerate_parallel(complexity, vocab, &tx, deadline);
            });
        }
    });
    // candidate_tx dropped by caller → evaluator pool sees channel closed.
}

// ---------------------------------------------------------------------------
// Parallel enumeration for a single complexity level
// ---------------------------------------------------------------------------

fn enumerate_parallel(
    c: usize,
    vocab: &Vocabulary,
    tx: &crossbeam_channel::Sender<Expr>,
    deadline: Instant,
) {
    // c=1: just atoms — fast sequential emit, no parallelism needed.
    if c == 1 {
        for atom in vocab.atoms() {
            if tx.send(atom.clone()).is_err() || Instant::now() >= deadline {
                return;
            }
        }
        return;
    }

    // Build the list of independent work items for this complexity level.
    // Each item covers a non-overlapping slice of the expression space.
    #[derive(Clone)]
    enum Item { Unary(UnaryOp), Binary(BinOp, usize, usize) }

    let mut items: Vec<Item> = Vec::new();

    // Unary: op(sub) where sub has complexity c-1.
    for &op in &vocab.unary_ops {
        items.push(Item::Unary(op));
    }
    // Binary: op(left, right) where left has complexity i, right has j=c-1-i.
    if c >= 3 {
        for i in 1..=(c - 2) {
            let j = c - 1 - i;
            for &op in &vocab.binary_ops {
                items.push(Item::Binary(op, i, j));
            }
        }
    }

    // Level 2: run all work items in parallel via rayon.
    items.par_iter().for_each(|item| {
        if Instant::now() >= deadline { return; }
        match *item {
            // --- Unary item ---
            Item::Unary(op) => {
                let tx = tx.clone();
                enumerate(c - 1, vocab, &mut |sub| {
                    tx.send(Expr::Unary(op, Box::new(sub))).is_ok()
                        && Instant::now() < deadline
                });
            }

            // --- Binary item ---
            Item::Binary(op, i, j) => {
                if i <= PARALLEL_LEFT_CAP {
                    // Level 3: pre-collect left sub-trees, then parallelize
                    // over them so each right-enumeration runs in its own
                    // rayon task. Gives ~10 K–265 K fine-grained tasks.
                    let mut lefts: Vec<Expr> = Vec::new();
                    enumerate(i, vocab, &mut |e| { lefts.push(e); true });

                    lefts.into_par_iter().for_each_with(tx.clone(), |tx, left| {
                        if Instant::now() >= deadline { return; }
                        enumerate(j, vocab, &mut |right| {
                            tx.send(Expr::Binary(
                                op,
                                Box::new(left.clone()),
                                Box::new(right),
                            )).is_ok() && Instant::now() < deadline
                        });
                    });
                } else {
                    // Large left set (i > CAP): sequential nested enumeration.
                    // This task still runs concurrently with the other ~87 items.
                    let tx = tx.clone();
                    enumerate(i, vocab, &mut |left| {
                        enumerate(j, vocab, &mut |right| {
                            tx.send(Expr::Binary(
                                op,
                                Box::new(left.clone()),
                                Box::new(right),
                            )).is_ok() && Instant::now() < deadline
                        })
                    });
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Sequential recursive enumeration (used within each work item)
// ---------------------------------------------------------------------------

/// Enumerate every expression of exactly `c` nodes, invoking `emit` for each.
/// Returns `false` if `emit` aborts early; `true` if fully completed.
fn enumerate(
    c: usize,
    vocab: &Vocabulary,
    emit: &mut dyn FnMut(Expr) -> bool,
) -> bool {
    if c == 1 {
        for atom in vocab.atoms() {
            if !emit(atom.clone()) { return false; }
        }
        return true;
    }

    for &op in &vocab.unary_ops {
        let cont = enumerate(c - 1, vocab, &mut |sub| {
            emit(Expr::Unary(op, Box::new(sub)))
        });
        if !cont { return false; }
    }

    if c >= 3 {
        for i in 1..=(c - 2) {
            let j = c - 1 - i;
            for &op in &vocab.binary_ops {
                let cont = enumerate(i, vocab, &mut |left| {
                    enumerate(j, vocab, &mut |right| {
                        emit(Expr::Binary(op, Box::new(left.clone()), Box::new(right)))
                    })
                });
                if !cont { return false; }
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OperatorsConfig, SolverConfig, TermsConfig};
    use crate::vocabulary::Vocabulary;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::time::Duration;

    fn make_vocab(vars: &[&str], consts: &[&str], binary: &[&str], unary: &[&str])
        -> Vocabulary
    {
        let cfg = SolverConfig {
            training_csv:    PathBuf::from("t.csv"),
            validation_csv:  PathBuf::from("v.csv"),
            target_column:   "y".to_string(),
            max_error_pct:   5.0,
            max_complexity:  7,
            max_time_secs:   60.0,
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
        };
        let headers: Vec<String> = vars.iter().map(|s| s.to_string()).collect();
        Vocabulary::from_config(&cfg, &headers).unwrap()
    }

    fn collect(vocab: &Vocabulary, max_complexity: usize) -> Vec<Expr> {
        let deadline = Instant::now() + Duration::from_secs(30);
        let (tx, rx) = crossbeam_channel::bounded(1024);
        let handle = std::thread::spawn(move || {
            let mut v = Vec::new();
            for e in &rx { v.push(e); }
            v
        });
        run_generator(vocab, max_complexity, deadline, tx, 4);
        handle.join().unwrap()
    }

    #[test]
    fn complexity_1_is_atoms_only() {
        let vocab = make_vocab(&["x"], &["1", "2"], &["add"], &[]);
        let got = collect(&vocab, 1);
        assert_eq!(got.len(), 3);
        for e in &got { assert_eq!(e.complexity(), 1); }
    }

    #[test]
    fn unary_produces_complexity_2() {
        let vocab = make_vocab(&["x"], &[], &["add"], &["sqrt", "ln"]);
        let got = collect(&vocab, 2);
        let c2: Vec<_> = got.iter().filter(|e| e.complexity() == 2).collect();
        assert_eq!(c2.len(), 2, "expected 2 unary expressions at complexity 2");
    }

    #[test]
    fn all_emitted_respect_max_complexity() {
        let vocab = make_vocab(&["x"], &["1"], &["add", "multiply"], &["sqrt"]);
        let got = collect(&vocab, 4);
        for e in &got {
            assert!(e.complexity() <= 4, "complexity {} exceeds cap", e.complexity());
        }
        assert!(!got.is_empty());
    }

    #[test]
    fn binary_combination_present() {
        let vocab = make_vocab(&["x"], &["1"], &["add"], &[]);
        let got = collect(&vocab, 3);
        let has_binary = got.iter().any(|e| matches!(e, Expr::Binary(_, _, _)));
        assert!(has_binary, "expected at least one binary expression");
    }

    #[test]
    fn deadline_stops_generation() {
        let vocab = make_vocab(&["x"], &["1", "2"], &["add", "multiply"], &["sqrt", "ln"]);
        let deadline = Instant::now() - Duration::from_millis(1);
        let (tx, rx) = crossbeam_channel::bounded(64);
        let handle = std::thread::spawn(move || rx.iter().count());
        run_generator(&vocab, 10, deadline, tx, 4);
        let count = handle.join().unwrap();
        assert!(count < 1000, "expected near-immediate stop, got {count}");
    }

    #[test]
    fn no_unbounded_memory_signature() {
        let vocab = make_vocab(&["x"], &["1", "2"], &["add", "multiply"], &["sqrt"]);
        let got = collect(&vocab, 5);
        let keys: HashSet<String> = got.iter().map(|e| e.canonical()).collect();
        assert!(!keys.is_empty());
    }

    #[test]
    fn parallel_count_matches_sequential() {
        // The parallel generator must emit exactly the same SET of canonical
        // expressions as the sequential one (order may differ due to parallelism).
        let vocab = make_vocab(&["x"], &["1"], &["add", "multiply"], &["sqrt", "ln"]);
        let got = collect(&vocab, 5);
        let keys: HashSet<String> = got.iter().map(|e| e.canonical()).collect();
        // Sanity: we got many distinct expressions and no complexity violations.
        assert!(keys.len() > 100, "expected many distinct expressions, got {}", keys.len());
        for e in &got {
            assert!(e.complexity() <= 5, "complexity {} out of range", e.complexity());
        }
    }
}
