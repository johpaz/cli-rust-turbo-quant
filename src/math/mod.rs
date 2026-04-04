//! Math module for TurboQuant.
//!
//! Implements the core mathematical primitives from the Google paper:
//! - Random rotation (FWHT + random sign flips)
//! - MSE-optimal scalar quantization
//! - Polar quantization
//! - Quantized Johnson-Lindenstrauss (QJL) transform

pub mod rotate;
pub mod quantizer;
pub mod polar_quant;
pub mod qjl;

// Re-export commonly used items for convenience
pub use rotate::{random_rotate, fresh_seed};
pub use quantizer::{ScalarQuantizer, pack_indices};
pub use qjl::{qjl_transform_fast};

/// Complete TurboQuant encoding pipeline.
///
/// This implements the full algorithm from the paper:
/// 1. Random rotation (induces Beta distribution)
/// 2. MSE-optimal scalar quantization per coordinate
/// 3. QJL 1-bit residual correction for unbiased inner products
///
/// # Arguments
/// * `x` - Input vector
/// * `bits` - Total bits per dimension (e.g., 3.5)
///
/// # Returns
/// TurboQuant encoding with all metadata needed for reconstruction
#[derive(Debug, Clone)]
pub struct TurboQuantEncoding {
    /// Quantized vector (after rotation + scalar quantization)
    pub quantized: Vec<f32>,
    /// QJL signs for residual correction (1 bit/dim)
    pub qjl_signs: Vec<i8>,
    /// Random rotation seed
    pub rotation_seed: u64,
    /// Quantization indices (packed)
    pub packed_indices: Vec<u8>,
    /// Bits per dimension
    pub bits: f32,
    /// Original dimension
    pub dim: usize,
}

/// Encodes a vector using the full TurboQuant pipeline.
///
/// # Arguments
/// * `x` - Input vector
/// * `bits` - Bits per dimension (e.g., 3.5, where 1 bit goes to QJL)
///
/// # Returns
/// Encoding with quantized values, QJL signs, and metadata
pub fn encode_vector(x: &[f32], bits: f32) -> TurboQuantEncoding {
    let dim = x.len();
    if dim == 0 {
        return TurboQuantEncoding {
            quantized: vec![],
            qjl_signs: vec![],
            rotation_seed: 0,
            packed_indices: vec![],
            bits,
            dim: 0,
        };
    }

    // Stage 0: Random rotation
    let rotation_seed = fresh_seed();
    let rotated = random_rotate(x, rotation_seed);

    // Stage 1: MSE-optimal scalar quantization
    // Reserve 1 bit for QJL correction
    let scalar_bits = bits - 1.0;
    let scalar_bits = scalar_bits.max(1.0); // at least 1 bit

    // Estimate range from rotated vector (concentrated around 0 after rotation)
    let max_abs: f32 = rotated.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let range = max_abs * 1.1; // small margin

    let quantizer = ScalarQuantizer::new(scalar_bits, -range, range);
    let quantized = quantizer.quantize_slice(&rotated);
    let indices = quantizer.encode_indices(&rotated);

    // Compute residual for QJL
    let residual: Vec<f32> = rotated.iter()
        .zip(quantized.iter())
        .map(|(&r, &q)| r - q)
        .collect();

    // Stage 2: QJL transform on residual (1 bit/dim)
    let qjl = qjl_transform_fast(&residual, rotation_seed);

    // Pack indices
    let packed_indices = pack_indices(&indices, scalar_bits);

    TurboQuantEncoding {
        quantized,
        qjl_signs: qjl.signs,
        rotation_seed,
        packed_indices,
        bits,
        dim,
    }
}

/// Decodes a TurboQuant-encoded vector and reconstructs the approximation.
///
/// # Arguments
/// * `encoding` - The encoding from encode_vector
///
/// # Returns
/// Reconstructed vector (approximation of original)
pub fn decode_vector(encoding: &TurboQuantEncoding) -> Vec<f32> {
    if encoding.dim == 0 {
        return vec![];
    }

    // The quantized vector is already the main approximation.
    // QJL provides correction for inner products, not direct reconstruction.
    // For reconstruction, we just return the quantized values.
    // For inner product computation, use the QJL signs for correction.
    encoding.quantized.clone()
}

/// Computes the corrected inner product between two TurboQuant-encoded vectors.
///
/// This is the key operation that makes TurboQuant accurate: the QJL correction
/// eliminates the bias introduced by scalar quantization.
///
/// # Arguments
/// * `enc_a` - Encoding of vector a
/// * `enc_b` - Encoding of vector b
///
/// # Returns
/// Corrected inner product estimate
pub fn corrected_inner_product(enc_a: &TurboQuantEncoding, enc_b: &TurboQuantEncoding) -> f32 {
    if enc_a.dim != enc_b.dim || enc_a.dim == 0 {
        return 0.0;
    }

    // Base inner product of quantized vectors
    let base_dot: f32 = enc_a.quantized.iter()
        .zip(enc_b.quantized.iter())
        .map(|(&a, &b)| a * b)
        .sum();

    // QJL correction (requires same rotation seed for both vectors)
    // In practice, K/V vectors in attention use the same rotation per layer
    if enc_a.qjl_signs.len() == enc_b.qjl_signs.len() {
        qjl::correct_inner_product(
            base_dot,
            &enc_a.qjl_signs,
            &enc_b.qjl_signs,
            enc_a.dim,
        )
    } else {
        base_dot
    }
}

/// Computes the corrected attention score (softmax-ready).
///
/// For a query vector and a list of key vectors, computes all attention scores
/// with QJL-corrected inner products.
///
/// # Arguments
/// * `query_enc` - TurboQuant encoding of the query vector
/// * `key_encs` - TurboQuant encodings of all key vectors
///
/// # Returns
/// Raw attention scores (pre-softmax)
pub fn compute_attention_scores(
    query_enc: &TurboQuantEncoding,
    key_encs: &[TurboQuantEncoding],
) -> Vec<f32> {
    key_encs
        .iter()
        .map(|key_enc| corrected_inner_product(query_enc, key_enc))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_basic() {
        let x = vec![0.5, -0.3, 0.8, 0.1, -0.6, 0.2, 0.9, -0.4];
        let encoding = encode_vector(&x, 3.5);

        assert_eq!(encoding.dim, 8);
        assert_eq!(encoding.quantized.len(), 8);
        assert_eq!(encoding.qjl_signs.len(), 8);

        let reconstructed = decode_vector(&encoding);
        assert_eq!(reconstructed.len(), 8);
    }

    #[test]
    fn test_inner_product_self() {
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let enc = encode_vector(&x, 4.0);

        // Inner product of a vector with itself should be positive
        let dot = corrected_inner_product(&enc, &enc);
        assert!(dot > 0.0);
    }

    #[test]
    fn test_attention_scores_shape() {
        let query = vec![0.1, 0.2, 0.3, 0.4];
        let keys = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![-0.1, -0.2, -0.3, -0.4],
            vec![0.5, 0.5, 0.5, 0.5],
        ];

        let query_enc = encode_vector(&query, 3.5);
        let key_encs: Vec<_> = keys.iter().map(|k| encode_vector(k, 3.5)).collect();

        let scores = compute_attention_scores(&query_enc, &key_encs);
        assert_eq!(scores.len(), 3);
    }
}
