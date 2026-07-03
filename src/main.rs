// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Xcelerator Solver — CLI entry point.

use clap::Parser;
use std::path::PathBuf;
use xcelerator_solver::{
    config::SolverConfig,
    csv_loader::load_csv,
    output::{print_json, print_table, run_validation, Tee},
    pipeline::run_pipeline,
    vocabulary::Vocabulary,
};

#[derive(Parser, Debug)]
#[command(
    name = "xcelerator-solver",
    about = "Deterministic symbolic regression engine.\nPass a TOML config file as the only argument.",
    version
)]
struct Cli {
    /// Path to the TOML configuration file
    config: PathBuf,

    /// Emit results as JSON instead of a formatted table
    #[arg(long)]
    json: bool,
}

/// Enable ANSI/virtual-terminal processing on the Windows stderr handle so the
/// live in-place progress display (cursor-up + clear) is honored by the console.
/// On Linux/macOS/WSL terminals ANSI is always interpreted, so this is a no-op.
#[cfg(windows)]
fn enable_ansi_support() {
    // kernel32 is linked by default on Windows; std-only FFI, no extra deps.
    extern "system" {
        fn GetStdHandle(n_std_handle: u32) -> isize;
        fn GetConsoleMode(handle: isize, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: isize, mode: u32) -> i32;
    }
    const STD_ERROR_HANDLE: u32 = 0xFFFF_FFF4; // (DWORD)(-12)
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    const INVALID_HANDLE_VALUE: isize = -1;
    unsafe {
        let h = GetStdHandle(STD_ERROR_HANDLE);
        if h != 0 && h != INVALID_HANDLE_VALUE {
            let mut mode = 0u32;
            // GetConsoleMode fails for non-console handles (e.g. redirected to a
            // file/pipe); in that case we leave it alone and the TTY guard in the
            // pipeline suppresses the ANSI render entirely.
            if GetConsoleMode(h, &mut mode) != 0 {
                SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }
        }
    }
}

#[cfg(not(windows))]
fn enable_ansi_support() {}

fn main() {
    enable_ansi_support();
    let cli = Cli::parse();

    // --- Load and validate config ---
    let config = match SolverConfig::load(&cli.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading config '{}': {}", cli.config.display(), e);
            std::process::exit(1);
        }
    };

    // --- Open output file early (create or truncate) for startup warnings ---
    let output_path = config.output_file.clone();
    let mut tee = Tee::new();

    // --- Load training CSV ---
    let var_names = config.terms.variables.clone();
    let (training, train_warns) = match load_csv(
        &config.training_csv,
        &var_names,
        &config.target_column,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Training CSV error: {}", e);
            std::process::exit(1);
        }
    };

    let mut warnings: Vec<String> = train_warns;

    // --- Load validation CSV (optional) ---
    let validation = match load_csv(
        &config.validation_csv,
        &var_names,
        &config.target_column,
    ) {
        Ok((data, warns)) => {
            warnings.extend(warns);
            Some(data)
        }
        Err(e) => {
            warnings.push(format!("Validation CSV unavailable ({}); val_mape will show N/A", e));
            None
        }
    };

    // --- Build vocabulary ---
    let csv_headers = var_names.clone();
    let vocab = match Vocabulary::from_config(&config, &csv_headers) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Vocabulary error: {}", e);
            std::process::exit(1);
        }
    };

    // --- Write startup warnings ---
    for w in &warnings {
        tee.writeln(&format!("WARNING: {w}"));
    }

    // --- Run the search pipeline ---
    let result = run_pipeline(&vocab, &config, &training, warnings.clone());

    // --- Validation scoring ---
    let metric = config.effective_metric();
    let val_slice = validation.as_deref();
    let final_entries = run_validation(result.top_entries, val_slice, metric);

    // --- Format output ---
    let precision_label = config.precision_label();
    let top_n = config.effective_top_candidates();

    if cli.json {
        print_json(&mut tee, &final_entries, &result.stats,
                   &result.warnings, &precision_label, metric, top_n);
    } else {
        print_table(&mut tee, &final_entries, &result.stats,
                    &precision_label, metric, top_n);
    }

    // --- Write output file (prepend) ---
    if let Err(e) = tee.finalize(&output_path) {
        eprintln!("Warning: could not write output file '{}': {}", output_path.display(), e);
    }
}
