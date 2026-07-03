// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! CSV parsing into DataPoints (f64 and HP variants).
//!
//! Both f64 and HP paths share the same column-resolution logic;
//! only the per-cell numeric conversion differs.

use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// f64 DataPoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Input variable values for this row, keyed by column header name.
    pub inputs: HashMap<String, f64>,
    /// Target output value for this row.
    pub output: f64,
}

// ---------------------------------------------------------------------------
// f64 loader
// ---------------------------------------------------------------------------

/// Load a CSV file into `DataPoint` values.
///
/// - First row must be a header row.
/// - All columns in `variable_names` plus `target_column` must be present.
/// - Rows where any required column cannot be parsed as `f64` are skipped
///   with a warning appended to the returned `Vec<String>`.
/// - Returns `Err` if fewer than 2 valid rows are loaded.
pub fn load_csv(
    path: &Path,
    variable_names: &[String],
    target_column: &str,
) -> Result<(Vec<DataPoint>, Vec<String>), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open CSV '{}': {}", path.display(), e))?;
    load_from_reader(file, variable_names, target_column)
}

/// Inner implementation shared between the file and in-memory (test) paths.
pub fn load_from_reader<R: std::io::Read>(
    reader: R,
    variable_names: &[String],
    target_column: &str,
) -> Result<(Vec<DataPoint>, Vec<String>), String> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(reader);

    // Build column index map from header row.
    let headers: Vec<String> = csv_reader
        .headers()
        .map_err(|e| format!("CSV header read error: {e}"))?
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Resolve target column index.
    let target_idx = headers
        .iter()
        .position(|h| h == target_column)
        .ok_or_else(|| format!("Target column '{}' not found in CSV headers", target_column))?;

    // Resolve each variable column index.
    let var_indices: Vec<(String, usize)> = variable_names
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|h| h == name)
                .map(|idx| (name.clone(), idx))
                .ok_or_else(|| format!("Variable column '{}' not found in CSV headers", name))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut data: Vec<DataPoint> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    for (row_num, result) in csv_reader.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                warnings.push(format!("Row {}: CSV parse error — {e}", row_num + 2));
                continue;
            }
        };

        // Parse output (target) column.
        let output_str = record.get(target_idx).unwrap_or("").trim();
        let output = match output_str.parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                warnings.push(format!(
                    "Row {}: column '{}' value '{output_str}' is not a valid number — skipped",
                    row_num + 2, target_column
                ));
                continue;
            }
        };

        // Parse all variable columns.
        let mut inputs = HashMap::new();
        let mut skip = false;
        for (name, idx) in &var_indices {
            let cell = record.get(*idx).unwrap_or("").trim();
            match cell.parse::<f64>() {
                Ok(v) => {
                    inputs.insert(name.clone(), v);
                }
                Err(_) => {
                    warnings.push(format!(
                        "Row {}: column '{name}' value '{cell}' is not a valid number — skipped",
                        row_num + 2
                    ));
                    skip = true;
                    break;
                }
            }
        }

        if !skip {
            data.push(DataPoint { inputs, output });
        }
    }

    if data.len() < 2 {
        return Err(format!(
            "CSV must have at least 2 valid data rows; found {}",
            data.len()
        ));
    }

    Ok((data, warnings))
}

// ---------------------------------------------------------------------------
// HP DataPoint and loader
// ---------------------------------------------------------------------------

#[cfg(feature = "hp")]
pub mod hp_loader {
    use std::collections::HashMap;
    use std::path::Path;

    #[derive(Debug)]
    pub struct HpDataPoint {
        pub inputs: HashMap<String, rug::Float>,
        pub output: rug::Float,
    }

    /// Load a CSV file into `HpDataPoint` values at the given MPFR precision.
    ///
    /// Numeric cells are parsed as exact strings via `rug::Float::parse` with
    /// NO intermediate f64 conversion — preserving full source precision.
    pub fn load_csv_hp(
        path: &Path,
        variable_names: &[String],
        target_column: &str,
        prec: u32,
    ) -> Result<(Vec<HpDataPoint>, Vec<String>), String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open CSV '{}': {}", path.display(), e))?;
        load_hp_from_reader(file, variable_names, target_column, prec)
    }

    pub fn load_hp_from_reader<R: std::io::Read>(
        reader: R,
        variable_names: &[String],
        target_column: &str,
        prec: u32,
    ) -> Result<(Vec<HpDataPoint>, Vec<String>), String> {
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(reader);

        let headers: Vec<String> = csv_reader
            .headers()
            .map_err(|e| format!("CSV header read error: {e}"))?
            .iter()
            .map(|s| s.to_string())
            .collect();

        let target_idx = headers
            .iter()
            .position(|h| h == target_column)
            .ok_or_else(|| format!("Target column '{}' not found", target_column))?;

        let var_indices: Vec<(String, usize)> = variable_names
            .iter()
            .map(|name| {
                headers
                    .iter()
                    .position(|h| h == name)
                    .map(|idx| (name.clone(), idx))
                    .ok_or_else(|| format!("Variable column '{}' not found", name))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut data: Vec<HpDataPoint> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        for (row_num, result) in csv_reader.records().enumerate() {
            let record = match result {
                Ok(r) => r,
                Err(e) => {
                    warnings.push(format!("Row {}: {e}", row_num + 2));
                    continue;
                }
            };

            let output_str = record.get(target_idx).unwrap_or("").trim();
            let output = parse_hp_cell(output_str, prec).ok_or_else(|| {
                format!("Row {}: target '{}' is not valid", row_num + 2, output_str)
            });
            let output = match output {
                Ok(v) => v,
                Err(msg) => {
                    warnings.push(format!("{msg} — skipped"));
                    continue;
                }
            };

            let mut inputs = HashMap::new();
            let mut skip = false;
            for (name, idx) in &var_indices {
                let cell = record.get(*idx).unwrap_or("").trim();
                match parse_hp_cell(cell, prec) {
                    Some(v) => {
                        inputs.insert(name.clone(), v);
                    }
                    None => {
                        warnings.push(format!(
                            "Row {}: '{name}' = '{cell}' is not valid — skipped",
                            row_num + 2
                        ));
                        skip = true;
                        break;
                    }
                }
            }

            if !skip {
                data.push(HpDataPoint { inputs, output });
            }
        }

        if data.len() < 2 {
            return Err(format!(
                "CSV must have at least 2 valid data rows; found {}",
                data.len()
            ));
        }

        Ok((data, warnings))
    }

    fn parse_hp_cell(s: &str, prec: u32) -> Option<rug::Float> {
        let parsed = rug::Float::parse(s).ok()?;
        Some(rug::Float::with_val(prec, parsed))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn load(csv: &str, vars: &[&str], target: &str)
        -> Result<(Vec<DataPoint>, Vec<String>), String>
    {
        let vars: Vec<String> = vars.iter().map(|s| s.to_string()).collect();
        load_from_reader(std::io::Cursor::new(csv), &vars, target)
    }

    const SIMPLE_CSV: &str = "\
x,y,output
1.0,2.0,3.0
4.0,5.0,9.0
7.0,8.0,15.0
";

    #[test]
    fn valid_three_rows() {
        let (data, warnings) = load(SIMPLE_CSV, &["x", "y"], "output").unwrap();
        assert_eq!(data.len(), 3);
        assert!(warnings.is_empty());
        assert_eq!(data[0].inputs["x"], 1.0);
        assert_eq!(data[0].inputs["y"], 2.0);
        assert_eq!(data[0].output, 3.0);
        assert_eq!(data[2].output, 15.0);
    }

    #[test]
    fn bad_row_skipped_with_warning() {
        let csv = "\
x,output
1.0,3.0
abc,5.0
4.0,9.0
";
        let (data, warnings) = load(csv, &["x"], "output").unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("abc"), "warn: {}", warnings[0]);
        assert!(warnings[0].contains("skipped"));
    }

    #[test]
    fn missing_variable_column_errors() {
        let err = load(SIMPLE_CSV, &["x", "z"], "output").unwrap_err();
        assert!(err.contains("'z'"), "err: {err}");
        assert!(err.contains("not found"), "err: {err}");
    }

    #[test]
    fn missing_target_column_errors() {
        let err = load(SIMPLE_CSV, &["x"], "missing").unwrap_err();
        assert!(err.contains("'missing'"), "err: {err}");
    }

    #[test]
    fn fewer_than_two_valid_rows_errors() {
        let csv = "\
x,output
1.0,3.0
abc,bad
";
        let err = load(csv, &["x"], "output").unwrap_err();
        assert!(err.contains("at least 2"), "err: {err}");
    }

    #[test]
    fn single_variable_single_target() {
        let csv = "\
x,y
10.0,20.0
30.0,40.0
";
        let (data, _) = load(csv, &["x"], "y").unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[1].output, 40.0);
    }

    #[test]
    fn target_not_in_variables() {
        // target_column "output" is correctly excluded from inputs
        let (data, _) = load(SIMPLE_CSV, &["x"], "output").unwrap();
        assert!(!data[0].inputs.contains_key("output"));
    }
}
