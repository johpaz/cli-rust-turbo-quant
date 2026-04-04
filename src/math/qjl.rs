//! Quantized Johnson-Lindenstrauss (QJL) Transform.
//!
//! The second stage of TurboQuant: after MSE-optimal quantization introduces
//! bias in inner product estimation, the QJL transform provides an unbiased
//! correction using exactly 1 bit per dimension.
//!
//! Algorithm:
//! 1. Compute residual: e = x_original - x_quantized
//! 2. Project: y = R * e, where R is a random ±1 matrix
//! 3. Quantize to signs: b = sign(y) ∈ {+1, -1}^m
//! 4. Store exactly 1 bit per dimension
//!
//! The key property: E[b] preserves the inner product structure in expectation,
//! providing an unbiased correction without requiring any scaling metadata.
//!
//! Paper reference: Section on "Unbiased Inner Product Quantization via QJL"

use rand::{Rng, SeedableRng};

/// Represents a QJL-transformed residual vector.
#[derive(Debug, Clone)]
pub struct QjlTransform {
    /// 1-bit signs (+1 or -1), stored as bytes for efficiency
    pub signs: Vec<i8>,
    /// Random seed used for the projection matrix
    pub seed: u64,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension (number of random projections)
    pub output_dim: usize,
}

impl QjlTransform {
    /// Reconstructs the approximate residual from the 1-bit representation.
    /// Note: this is an approximation; the full reconstruction requires
    /// knowing the original quantized vector and adding this correction.
    pub fn approximate_residual(&self) -> Vec<f32> {
        // The scaling factor is sqrt(input_dim / output_dim) in expectation
        let scale = (self.input_dim as f32 / self.output_dim as f32).sqrt();
        self.signs.iter().map(|&s| s as f32 * scale).collect()
    }

    /// Packs the signs into a compact bit representation.
    pub fn pack_bits(&self) -> Vec<u8> {
        pack_sign_bits(&self.signs)
    }
}

/// Applies the Quantized Johnson-Lindenstrauss transform.
///
/// # Arguments
/// * `residual` - The quantization residual vector (x_original - x_quantized)
/// * `output_dim` - Number of random projections (typically = input_dim for 1 bit/dim)
/// * `seed` - Random seed for the projection matrix
///
/// # Returns
/// QjlTransform containing 1-bit signs
pub fn qjl_transform(residual: &[f32], output_dim: usize, seed: u64) -> QjlTransform {
    let input_dim = residual.len();

    // Generate random projection matrix R ∈ ℝ^{output_dim × input_dim}
    // Each entry is ±1 with equal probability (simpler than Gaussian for quantization)
    let _rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut signs = Vec::with_capacity(output_dim);

    for i in 0..output_dim {
        // Compute the i-th projection: y_i = R_i · residual
        let mut projection: f32 = 0.0;
        let row_seed = seed.wrapping_add(i as u64 * 2654435761); // golden ratio hash
        let mut row_rng = rand::rngs::StdRng::seed_from_u64(row_seed);

        for j in 0..input_dim {
            let r_ij = if row_rng.random::<bool>() { 1.0 } else { -1.0 };
            projection += r_ij * residual[j];
        }

        // Normalize by sqrt(input_dim) for JL lemma
        projection /= (input_dim as f32).sqrt();

        // Quantize to sign: +1 or -1 (1 bit)
        let sign = if projection >= 0.0 { 1 } else { -1 };
        signs.push(sign);
    }

    QjlTransform {
        signs,
        seed,
        input_dim,
        output_dim,
    }
}

/// Applies QJL transform with optimized fast projection.
///
/// Uses Fast Walsh-Hadamard Transform for the random projection,
/// reducing complexity from O(d²) to O(d log d).
///
/// # Arguments
/// * `residual` - The quantization residual vector
/// * `seed` - Random seed
///
/// # Returns
/// QjlTransform with 1-bit signs (output_dim = input_dim)
pub fn qjl_transform_fast(residual: &[f32], seed: u64) -> QjlTransform {
    let dim = residual.len();
    if dim == 0 {
        return QjlTransform {
            signs: vec![],
            seed,
            input_dim: 0,
            output_dim: 0,
        };
    }

    // Pad to power of 2 for FWHT
    let padded_dim = dim.next_power_of_two();
    let mut padded = residual.to_vec();
    padded.resize(padded_dim, 0.0);

    // Random sign flips before FWHT (equivalent to random diagonal)
    crate::math::rotate::random_sign_flips(&mut padded, seed);

    // Apply FWHT for fast random projection
    crate::math::rotate::fwht(&mut padded);

    // Normalize and extract signs
    let _scale = (padded_dim as f32).sqrt();
    let signs: Vec<i8> = padded[..dim]
        .iter()
        .map(|v| if *v >= 0.0 { 1 } else { -1 })
        .collect();

    QjlTransform {
        signs,
        seed,
        input_dim: dim,
        output_dim: dim,
    }
}

/// Corrects an inner product estimate using QJL signs.
///
/// Given two vectors x, y and their QJL signs sx, sy:
///   corrected_dot = <x_q, y_q> + (d/m) * <sx, sy> / d
/// where x_q, y_q are the quantized versions, and the second term
/// is the unbiased correction from the residuals.
///
/// # Arguments
/// * `base_dot` - Inner product of quantized vectors
/// * `signs_x` - QJL signs of residual for x
/// * `signs_y` - QJL signs of residual for y
/// * `dim` - Original dimension
///
/// # Returns
/// Corrected inner product estimate
pub fn correct_inner_product(
    base_dot: f32,
    signs_x: &[i8],
    signs_y: &[i8],
    dim: usize,
) -> f32 {
    assert_eq!(signs_x.len(), signs_y.len(), "Sign vectors must have same length");

    let m = signs_x.len();
    if m == 0 || dim == 0 {
        return base_dot;
    }

    // Compute <sx, sy> / m
    let sign_dot: f32 = signs_x.iter()
        .zip(signs_y.iter())
        .map(|(&a, &b)| (a * b) as f32)
        .sum::<f32>() / m as f32;

    // Correction term scaled by d/m
    let correction = (dim as f32 / m as f32) * sign_dot / (dim as f32).sqrt();

    base_dot + correction
}

/// Packs sign bits into a compact byte array.
pub fn pack_sign_bits(signs: &[i8]) -> Vec<u8> {
    let n = signs.len();
    let mut result = Vec::with_capacity((n + 7) / 8);
    let mut current_byte: u8 = 0;
    let mut bit_offset: usize = 0;

    for &sign in signs {
        let bit = if sign > 0 { 1 } else { 0 };
        current_byte |= bit << bit_offset;
        bit_offset += 1;

        if bit_offset == 8 {
            result.push(current_byte);
            current_byte = 0;
            bit_offset = 0;
        }
    }

    if bit_offset > 0 {
        result.push(current_byte);
    }

    result
}

/// Unpacks sign bits from a compact byte array.
pub fn unpack_sign_bits(packed: &[u8], num_signs: usize) -> Vec<i8> {
    let mut signs = Vec::with_capacity(num_signs);

    for i in 0..num_signs {
        let byte_idx = i / 8;
        let bit_offset = i % 8;
        let bit = if byte_idx < packed.len() {
            (packed[byte_idx] >> bit_offset) & 1
        } else {
            0
        };
        signs.push(if bit == 1 { 1 } else { -1 });
    }

    signs
}

/// Two-stage TurboQuant: MSE quantization + QJL correction.
///
/// # Arguments
/// * `x` - Original vector
/// * `quantizer` - MSE-optimal scalar quantizer
/// * `total_bits` - Total bits per dimension (includes 1 bit for QJL)
///
/// # Returns
/// (quantized_vector, qjl_transform)
pub fn turboquant_two_stage(
    x: &[f32],
    quantizer: &crate::math::quantizer::ScalarQuantizer,
    _total_bits: f32,
) -> (Vec<f32>, QjlTransform) {
    // Stage 1: MSE-optimal quantization
    // Note: quantizer uses (total_bits - 1.0) since QJL takes 1 bit
    let quantized = quantizer.quantize_slice(x);

    // Compute residual
    let residual: Vec<f32> = x.iter()
        .zip(quantized.iter())
        .map(|(&orig, &quant)| orig - quant)
        .collect();

    // Stage 2: QJL transform on residual (1 bit per dimension)
    let qjl = qjl_transform_fast(&residual, crate::math::rotate::fresh_seed());

    (quantized, qjl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qjl_transform_produces_signs() {
        let residual = vec![0.1, -0.3, 0.5, -0.2];
        let qjl = qjl_transform(&residual, 4, 42);

        assert_eq!(qjl.signs.len(), 4);
        for sign in &qjl.signs {
            assert!(*sign == 1 || *sign == -1);
        }
    }

    #[test]
    fn test_pack_unpack_signs_roundtrip() {
        let signs = vec![1, -1, 1, 1, -1, -1, 1, -1, 1];
        let packed = pack_sign_bits(&signs);
        let unpacked = unpack_sign_bits(&packed, signs.len());

        assert_eq!(unpacked, signs);
    }

    #[test]
    fn test_correct_inner_product_basic() {
        let base_dot = 0.5;
        let signs_x = vec![1, -1, 1, 1];
        let signs_y = vec![1, -1, 1, 1];
        let corrected = correct_inner_product(base_dot, &signs_x, &signs_y, 4);

        // Same signs should increase the dot product
        assert!(corrected >= base_dot);
    }

    #[test]
    fn test_qjl_fast_deterministic_with_seed() {
        let residual = vec![0.1, -0.3, 0.5, -0.2, 0.7, -0.1, 0.3, -0.4];
        let qjl1 = qjl_transform_fast(&residual, 123);
        let qjl2 = qjl_transform_fast(&residual, 123);

        assert_eq!(qjl1.signs, qjl2.signs);
    }
}
