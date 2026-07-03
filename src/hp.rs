// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! HP precision configuration (requires --features hp).
//!
//! Mirrors the HighPrecConfig pattern from xcelerator-toolkit:
//!   precision_bits = ceil(digits × 3.322) + 16 guard bits

use rug::{float::Constant, Float};

/// log₂(10) ≈ 3.322 — matching the toolkit constant exactly.
pub const DIGITS_TO_BITS: f64 = 3.322;
/// Guard bits added beyond the strict digits-to-bits conversion.
pub const GUARD_BITS: u32 = 16;

/// HP working-precision configuration.
#[derive(Debug, Clone)]
pub struct HpConfig {
    /// MPFR working precision in bits.
    /// Total decimal digits ≈ precision_bits / 3.322.
    pub precision_bits: u32,
}

impl HpConfig {
    /// Construct from a target decimal digit count.
    /// Formula: ceil(digits × 3.322) + 16 guard bits — identical to the toolkit.
    pub fn for_decimal_digits(digits: u32) -> Self {
        let bits = (digits as f64 * DIGITS_TO_BITS).ceil() as u32 + GUARD_BITS;
        HpConfig { precision_bits: bits }
    }

    /// Convenience: zero-valued HP float at this precision.
    pub fn zero(&self) -> Float {
        Float::new(self.precision_bits)
    }
}

/// Compute a named mathematical constant at the given MPFR precision.
///
/// Named constants are computed via MPFR built-ins — NOT promoted from f64
/// literals, which would lose all digits beyond the 15th.
///
/// Returns `None` if `name` is not a recognised named constant.
/// Numeric literals ("1.5", "2", etc.) are handled by the vocabulary layer,
/// not here.
pub fn hp_constant(name: &str, prec: u32) -> Option<Float> {
    match name {
        "Pi" => Some(Float::with_val(prec, Constant::Pi)),

        "e" | "E" => {
            // exp(1) at full MPFR precision — no f64 shortcut.
            let one = Float::with_val(prec, 1u32);
            Some(one.exp())
        }

        "Tau" => {
            // 2π
            let pi = Float::with_val(prec, Constant::Pi);
            let two = Float::with_val(prec, 2u32);
            Some(pi * two)
        }

        "Phi" => {
            // Golden ratio: (1 + √5) / 2
            let five  = Float::with_val(prec, 5u32);
            let sqrt5 = five.sqrt();
            let one   = Float::with_val(prec, 1u32);
            let two   = Float::with_val(prec, 2u32);
            Some((sqrt5 + one) / two)
        }

        "gamma" => {
            // Euler-Mascheroni constant via MPFR built-in.
            Some(Float::with_val(prec, Constant::Euler))
        }

        "Catalan" => {
            // Catalan's constant G ≈ 0.9159655941772190 via MPFR built-in.
            Some(Float::with_val(prec, Constant::Catalan))
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_to_bits_200() {
        // 200 × 3.322 = 664.4 → ceil = 665 → +16 guard = 681
        // Must match toolkit test value exactly.
        let cfg = HpConfig::for_decimal_digits(200);
        assert_eq!(cfg.precision_bits, 681);
    }

    #[test]
    fn digits_to_bits_50() {
        // 50 × 3.322 = 166.1 → ceil = 167 → +16 = 183
        let cfg = HpConfig::for_decimal_digits(50);
        assert_eq!(cfg.precision_bits, 183);
    }

    #[test]
    fn pi_constant_correct() {
        let prec = HpConfig::for_decimal_digits(200).precision_bits;
        let computed = hp_constant("Pi", prec).unwrap();
        let reference = Float::with_val(prec, Constant::Pi);
        // Both use the same MPFR built-in — must be bit-identical.
        assert_eq!(computed, reference);
    }

    #[test]
    fn e_constant_correct() {
        let prec = HpConfig::for_decimal_digits(50).precision_bits;
        let e_computed = hp_constant("e", prec).unwrap();
        let e_upper = hp_constant("E", prec).unwrap();
        // Both spellings give the same value.
        assert_eq!(e_computed, e_upper);
        // Check first ~15 digits match the known f64 value.
        let e_f64: f64 = e_computed.to_f64();
        assert!((e_f64 - std::f64::consts::E).abs() < 1e-14);
    }

    #[test]
    fn tau_is_two_pi() {
        let prec = HpConfig::for_decimal_digits(50).precision_bits;
        let tau  = hp_constant("Tau", prec).unwrap();
        let pi   = Float::with_val(prec, Constant::Pi);
        let two  = Float::with_val(prec, 2u32);
        let expected = pi * two;
        assert_eq!(tau, expected);
    }

    #[test]
    fn phi_golden_ratio() {
        let prec = HpConfig::for_decimal_digits(50).precision_bits;
        let phi = hp_constant("Phi", prec).unwrap();
        let phi_f64 = phi.to_f64();
        assert!((phi_f64 - 1.618033988749895_f64).abs() < 1e-14);
    }

    #[test]
    fn unknown_constant_returns_none() {
        assert!(hp_constant("Gamma", 64).is_none());
        assert!(hp_constant("2.5", 64).is_none());
        assert!(hp_constant("", 64).is_none());
    }

    #[test]
    fn catalan_constant_hp() {
        let prec = HpConfig::for_decimal_digits(50).precision_bits;
        let c = hp_constant("Catalan", prec).unwrap();
        // First ~15 digits of G ≈ 0.9159655941772190
        let c_f64 = c.to_f64();
        assert!((c_f64 - 0.9159655941772190_f64).abs() < 1e-14);
    }

    #[test]
    fn euler_mascheroni_hp() {
        let prec = HpConfig::for_decimal_digits(50).precision_bits;
        let g = hp_constant("gamma", prec).unwrap();
        let g_f64 = g.to_f64();
        assert!((g_f64 - 0.5772156649015329_f64).abs() < 1e-14);
    }
}
