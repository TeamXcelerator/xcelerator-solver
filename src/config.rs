// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! TOML configuration loading and validation for the solver.
//!
//! All solver parameters live in a single TOML file whose path is passed as the
//! sole CLI argument. The [`SolverConfig`] struct maps directly to that file via
//! `serde`. Calling `SolverConfig::load(path)` reads, deserializes, and validates
//! the config, returning a descriptive error if any field is missing or invalid.
//!
//! # Field summary
//! - `training_csv` / `validation_csv` — paths to CSV data files
//! - `target_column` — name of the column to predict
//! - `max_error_pct` — acceptance threshold (in MAPE % or absolute units for MAE/RMSE)
//! - `max_complexity` — maximum expression tree node count
//! - `max_time_secs` — wall-clock timeout for the search
//! - `output_file` — results written here AND to console
//! - `top_candidates` — how many top training candidates to validate (default 20)
//! - `max_threads` — rayon thread cap (default: all cores)
//! - `precision_digits` — HP decimal digits; 0 or absent = f64 mode
//! - `error_metric` — `"mape"` (default), `"mae"`, or `"rmse"`
//! - `pinned_terms` — required sub-expression patterns (function-call notation)
//! - `[terms]` — `variables`, `constants`, `composite`
//! - `[operators]` — `binary`, `unary`

use serde::Deserialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TermsConfig {
    #[serde(default)]
    pub variables: Vec<String>,
    #[serde(default)]
    pub constants: Vec<String>,
    /// Optional composite expressions treated as atomic building blocks.
    /// Same function-call notation as pinned_terms: op_name(a, b) / op_name(a).
    /// These are OPTIONAL seeds — the solver can use them but is not required to.
    /// Example: ["multiply(2, Pi)", "divide(Pi, 2)"]
    #[serde(default)]
    pub composite: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OperatorsConfig {
    #[serde(default)]
    pub binary: Vec<String>,
    #[serde(default)]
    pub unary: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SolverConfig {
    pub training_csv: PathBuf,
    pub validation_csv: PathBuf,
    pub target_column: String,
    pub max_error_pct: f64,
    pub max_complexity: usize,
    pub max_time_secs: f64,
    pub output_file: PathBuf,

    /// How many top-training-MAPE candidates to retain and validate.
    /// Absent or 0 → default of 20.
    pub top_candidates: Option<u32>,

    /// Cap on rayon thread pool size. Absent or 0 → all available cores.
    pub max_threads: Option<u32>,

    /// HP arithmetic precision in decimal digits.
    /// Absent or 0 → standard f64 mode.
    /// Requires `--features hp` build when > 0.
    pub precision_digits: Option<u32>,

    /// Required sub-expression patterns (function-call notation).
    /// Every accepted candidate must contain all of these as sub-trees.
    #[serde(default)]
    pub pinned_terms: Option<Vec<String>>,

    /// Error metric: "mape" (default), "mae", or "rmse".
    /// Use "mae"/"rmse" when the target can be near-zero or negative
    /// (MAPE explodes/breaks in those regimes).
    #[serde(default)]
    pub error_metric: Option<String>,

    pub terms: TermsConfig,
    pub operators: OperatorsConfig,

    // Phase 2: pub expression_log: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl SolverConfig {
    /// Load and validate a TOML config file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            format!("Cannot read config file '{}': {}", path.display(), e)
        })?;
        let cfg: SolverConfig = toml::from_str(&content).map_err(|e| {
            format!("Config parse error in '{}': {}", path.display(), e)
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate all field values. Called automatically by `load`.
    pub fn validate(&self) -> Result<(), String> {
        // error_metric: must parse if present
        let is_mape = match &self.error_metric {
            None => true,
            Some(s) => {
                let m = crate::error_metric::ErrorMetric::from_name(s)?;
                m == crate::error_metric::ErrorMetric::Mape
            }
        };

        // max_error_pct: for MAPE must be in (0, 100]; for MAE/RMSE it is an
        // absolute threshold in target units, so only require it to be > 0.
        if self.max_error_pct <= 0.0 {
            return Err(format!(
                "max_error_pct must be > 0, got {}",
                self.max_error_pct
            ));
        }
        if is_mape && self.max_error_pct > 100.0 {
            return Err(format!(
                "max_error_pct must be in (0, 100] for MAPE metric, got {}",
                self.max_error_pct
            ));
        }

        // max_complexity: at least 1
        if self.max_complexity < 1 {
            return Err("max_complexity must be >= 1".to_string());
        }

        // max_time_secs: must be positive
        if self.max_time_secs <= 0.0 {
            return Err(format!(
                "max_time_secs must be > 0, got {}",
                self.max_time_secs
            ));
        }

        // top_candidates: if present, must be >= 1
        if let Some(n) = self.top_candidates {
            if n == 0 {
                return Err("top_candidates must be >= 1 (omit to use default of 20)".to_string());
            }
        }

        // precision_digits: if > 0, requires --features hp build
        if self.precision_digits.unwrap_or(0) > 0 && cfg!(not(feature = "hp")) {
            return Err(
                "precision_digits > 0 requires a high-precision build.\n\
                 Rebuild with: cargo build --features hp\n\
                 (Linux/WSL2 required — GMP/MPFR must be installed)"
                    .to_string(),
            );
        }

        // output_file: parent directory must exist (if non-empty)
        if let Some(parent) = self.output_file.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(format!(
                    "output_file parent directory does not exist: '{}'",
                    parent.display()
                ));
            }
        }

        // Must declare at least one term (variable or constant)
        if self.terms.variables.is_empty() && self.terms.constants.is_empty() {
            return Err(
                "[terms] must declare at least one variable or constant".to_string()
            );
        }

        // Must declare at least one binary operator
        if self.operators.binary.is_empty() {
            return Err(
                "[operators] binary must contain at least one operator".to_string()
            );
        }

        Ok(())
    }

    /// Returns `top_candidates` with a default of 20 when absent.
    pub fn effective_top_candidates(&self) -> usize {
        self.top_candidates.map(|n| n as usize).unwrap_or(20)
    }

    /// Resolve the error metric (defaults to MAPE).
    pub fn effective_metric(&self) -> crate::error_metric::ErrorMetric {
        match &self.error_metric {
            None => crate::error_metric::ErrorMetric::Mape,
            Some(s) => crate::error_metric::ErrorMetric::from_name(s)
                .unwrap_or(crate::error_metric::ErrorMetric::Mape),
        }
    }

    /// Returns the effective precision in bits (0 = f64 mode).
    pub fn effective_precision_bits(&self) -> u32 {
        let digits = self.precision_digits.unwrap_or(0);
        if digits == 0 {
            return 0;
        }
        #[cfg(feature = "hp")]
        {
            crate::hp::HpConfig::for_decimal_digits(digits).precision_bits
        }
        #[cfg(not(feature = "hp"))]
        {
            0 // unreachable after validate(), but needed for compilation
        }
    }

    /// Human-readable precision label for output headers.
    pub fn precision_label(&self) -> String {
        let digits = self.precision_digits.unwrap_or(0);
        if digits == 0 {
            "f64".to_string()
        } else {
            format!("HP-{}", digits)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid config — parent of output_file is "" (current dir, always exists).
    const VALID_TOML: &str = r#"
training_csv   = "train.csv"
validation_csv = "val.csv"
target_column  = "y"
max_error_pct  = 5.0
max_complexity = 7
max_time_secs  = 60.0
output_file    = "out.txt"

[terms]
variables = ["x"]
constants = ["1", "2"]

[operators]
binary = ["add", "multiply"]
unary  = []
"#;

    fn parse(toml: &str) -> Result<SolverConfig, String> {
        toml::from_str::<SolverConfig>(toml)
            .map_err(|e| e.to_string())
            .and_then(|cfg| cfg.validate().map(|_| cfg))
    }

    #[test]
    fn valid_toml_loads() {
        let cfg = parse(VALID_TOML).expect("valid config should load");
        assert_eq!(cfg.target_column, "y");
        assert_eq!(cfg.max_error_pct, 5.0);
        assert_eq!(cfg.max_complexity, 7);
    }

    #[test]
    fn default_top_candidates_is_20() {
        let cfg = parse(VALID_TOML).unwrap();
        assert_eq!(cfg.effective_top_candidates(), 20);
    }

    #[test]
    fn explicit_top_candidates_respected() {
        let toml = VALID_TOML.replace(
            "output_file    = \"out.txt\"",
            "output_file    = \"out.txt\"\ntop_candidates = 5",
        );
        let cfg = parse(&toml).unwrap();
        assert_eq!(cfg.effective_top_candidates(), 5);
    }

    #[test]
    fn missing_required_field_errors() {
        // Remove target_column
        let bad = VALID_TOML.replace("target_column  = \"y\"\n", "");
        assert!(parse(&bad).is_err());
    }

    #[test]
    fn max_error_pct_zero_rejected() {
        let bad = VALID_TOML.replace("max_error_pct  = 5.0", "max_error_pct  = 0.0");
        let err = parse(&bad).unwrap_err();
        assert!(err.contains("max_error_pct"), "msg: {err}");
    }

    #[test]
    fn max_error_pct_over_100_rejected() {
        let bad = VALID_TOML.replace("max_error_pct  = 5.0", "max_error_pct  = 101.0");
        assert!(parse(&bad).is_err());
    }

    #[test]
    fn max_complexity_zero_rejected() {
        let bad = VALID_TOML.replace("max_complexity = 7", "max_complexity = 0");
        assert!(parse(&bad).is_err());
    }

    #[test]
    fn max_time_zero_rejected() {
        let bad = VALID_TOML.replace("max_time_secs  = 60.0", "max_time_secs  = 0.0");
        assert!(parse(&bad).is_err());
    }

    #[test]
    fn no_terms_rejected() {
        let bad = VALID_TOML
            .replace("variables = [\"x\"]", "variables = []")
            .replace("constants = [\"1\", \"2\"]", "constants = []");
        assert!(parse(&bad).is_err());
    }

    #[test]
    fn no_binary_ops_rejected() {
        let bad = VALID_TOML.replace(
            "binary = [\"add\", \"multiply\"]",
            "binary = []",
        );
        assert!(parse(&bad).is_err());
    }

    #[test]
    fn output_file_bad_parent_rejected() {
        let bad = VALID_TOML.replace(
            "output_file    = \"out.txt\"",
            "output_file    = \"nonexistent_dir_xyz/out.txt\"",
        );
        assert!(parse(&bad).is_err());
    }

    #[test]
    fn top_candidates_zero_rejected() {
        let toml = VALID_TOML.replace(
            "output_file    = \"out.txt\"",
            "output_file    = \"out.txt\"\ntop_candidates = 0",
        );
        assert!(parse(&toml).is_err());
    }

    #[test]
    #[cfg(not(feature = "hp"))]
    fn precision_digits_without_hp_feature_rejected() {
        let toml = VALID_TOML.replace(
            "output_file    = \"out.txt\"",
            "output_file    = \"out.txt\"\nprecision_digits = 50",
        );
        let err = parse(&toml).unwrap_err();
        assert!(err.contains("--features hp"), "msg: {err}");
    }

    #[test]
    fn precision_label_f64() {
        let cfg = parse(VALID_TOML).unwrap();
        assert_eq!(cfg.precision_label(), "f64");
    }
}
