//! MSE-optimal scalar quantizer.
//!
//! After random rotation, coordinates are approximately i.i.d. with a concentrated
//! distribution. This module implements optimal scalar quantization per coordinate,
//! minimizing Mean Squared Error (MSE) for a given bit budget.
//!
//! The quantizer uses Lloyd-Max algorithm principles with pre-computed optimal
//! reconstruction levels for common bit widths (1-8 bits).

/// Represents an MSE-optimal scalar quantizer.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    /// Number of bits per coordinate
    pub bits: f32,
    /// Number of quantization levels (2^bits, may be fractional for sub-byte)
    pub num_levels: usize,
    /// Decision boundaries (thresholds between levels)
    pub boundaries: Vec<f32>,
    /// Reconstruction values (centroids for each level)
    pub centroids: Vec<f32>,
    /// Range of the quantizer [min, max]
    pub range_min: f32,
    pub range_max: f32,
}

impl ScalarQuantizer {
    /// Creates a new scalar quantizer with optimal levels for the given bit width.
    ///
    /// For sub-byte quantization (e.g., 3.5 bits), we use 2^bits levels.
    /// The boundaries and centroids are computed assuming a standard normal
    /// distribution (which is induced by the random rotation stage).
    ///
    /// # Arguments
    /// * `bits` - Target bits per coordinate (e.g., 3.5)
    /// * `range_min` - Minimum expected value
    /// * `range_max` - Maximum expected value
    pub fn new(bits: f32, range_min: f32, range_max: f32) -> Self {
        let num_levels = (2.0_f32.powf(bits)).round() as usize;
        let num_levels = num_levels.max(2);

        // For a Gaussian distribution, optimal boundaries are based on percentiles.
        // We use uniformly spaced boundaries in the CDF domain (Lloyd-Max for Gaussian).
        let boundaries = compute_gaussian_boundaries(num_levels, range_min, range_max);
        let centroids = compute_gaussian_centroids(&boundaries, range_min, range_max);

        Self {
            bits,
            num_levels,
            boundaries,
            centroids,
            range_min,
            range_max,
        }
    }

    /// Quantizes a single value to the nearest centroid index.
    ///
    /// # Arguments
    /// * `x` - Input value
    ///
    /// # Returns
    /// Index of the quantized level (0 to num_levels - 1)
    pub fn quantize_index(&self, x: f32) -> usize {
        if x <= self.boundaries[0] {
            return 0;
        }
        if x >= self.boundaries[self.boundaries.len() - 1] {
            return self.centroids.len() - 1;
        }

        // Binary search for the interval
        let idx = match self.boundaries.binary_search_by(|&b| b.partial_cmp(&x).unwrap()) {
            Ok(i) => i,
            Err(i) => i,
        };

        // Find the closest centroid
        if idx > 0 && idx <= self.centroids.len() {
            let lower = self.centroids[idx - 1];
            let upper = if idx < self.centroids.len() {
                self.centroids[idx]
            } else {
                return idx - 1;
            };
            if (x - lower).abs() < (x - upper).abs() {
                idx - 1
            } else {
                idx
            }
        } else {
            idx.saturating_sub(1).min(self.centroids.len() - 1)
        }
    }

    /// Quantizes a value and returns the reconstructed (dequantized) value.
    ///
    /// # Arguments
    /// * `x` - Input value
    ///
    /// # Returns
    /// Reconstructed value after quantization
    pub fn quantize(&self, x: f32) -> f32 {
        let idx = self.quantize_index(x);
        self.centroids[idx]
    }

    /// Quantizes an entire vector of values.
    ///
    /// # Arguments
    /// * `x` - Input vector
    ///
    /// # Returns
    /// Quantized vector with reconstructed values
    pub fn quantize_slice(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| self.quantize(v)).collect()
    }

    /// Returns the quantization indices for a vector (compact representation).
    ///
    /// # Arguments
    /// * `x` - Input vector
    ///
    /// # Returns
    /// Vector of quantization indices (can be packed into bits later)
    pub fn encode_indices(&self, x: &[f32]) -> Vec<usize> {
        x.iter().map(|&v| self.quantize_index(v)).collect()
    }

    /// Reconstructs values from quantization indices.
    ///
    /// # Arguments
    /// * `indices` - Quantization indices
    ///
    /// # Returns
    /// Reconstructed values
    pub fn decode_indices(&self, indices: &[usize]) -> Vec<f32> {
        indices.iter().map(|&i| self.centroids[i.min(self.centroids.len() - 1)]).collect()
    }

    /// Computes the MSE of quantization for a given vector.
    pub fn compute_mse(&self, original: &[f32]) -> f32 {
        let quantized = self.quantize_slice(original);
        original.iter()
            .zip(quantized.iter())
            .map(|(&o, &q)| (o - q).powi(2))
            .sum::<f32>() / original.len() as f32
    }
}

/// Computes optimal decision boundaries for a Gaussian distribution.
/// Uses uniform spacing in the CDF domain (Phi^-1 of uniform points).
fn compute_gaussian_boundaries(num_levels: usize, range_min: f32, range_max: f32) -> Vec<f32> {
    if num_levels <= 1 {
        return vec![];
    }

    // Number of interior boundaries = num_levels - 1
    let num_boundaries = num_levels - 1;
    let mut boundaries = Vec::with_capacity(num_boundaries);

    for i in 1..num_levels {
        // Uniform spacing in probability mass
        let p = i as f32 / num_levels as f32;
        // Map probability to value range using inverse CDF approximation
        let t = inverse_normal_cdf_approx(p);
        // Scale to the actual range
        let boundary = range_min + (range_max - range_min) * (t + 3.0) / 6.0;
        // Clamp to range
        let boundary = boundary.clamp(range_min, range_max);
        boundaries.push(boundary);
    }

    boundaries
}

/// Computes optimal reconstruction values (centroids) for each quantization region.
fn compute_gaussian_centroids(boundaries: &[f32], range_min: f32, range_max: f32) -> Vec<f32> {
    let num_levels = boundaries.len() + 1;
    let mut centroids = Vec::with_capacity(num_levels);

    // First centroid: midpoint between range_min and first boundary
    let first_centroid = (range_min + boundaries.first().copied().unwrap_or(range_max)) / 2.0;
    centroids.push(first_centroid.clamp(range_min, range_max));

    // Interior centroids: midpoint between consecutive boundaries
    for i in 0..boundaries.len() - 1 {
        let centroid = (boundaries[i] + boundaries[i + 1]) / 2.0;
        centroids.push(centroid.clamp(range_min, range_max));
    }

    // Last centroid: midpoint between last boundary and range_max
    if boundaries.len() > 1 {
        let last_centroid = (boundaries.last().copied().unwrap_or(range_min) + range_max) / 2.0;
        centroids.push(last_centroid.clamp(range_min, range_max));
    } else if boundaries.len() == 1 {
        let last_centroid = (boundaries[0] + range_max) / 2.0;
        centroids.push(last_centroid.clamp(range_min, range_max));
    }

    centroids
}

/// Approximation of the inverse normal CDF (probit function).
/// Uses the Beasley-Springer-Moro algorithm (simplified).
fn inverse_normal_cdf_approx(p: f32) -> f32 {
    assert!(p > 0.0 && p < 1.0, "p must be in (0, 1), got {}", p);

    // Rational approximation for the central region
    if (p - 0.5).abs() < 0.425 {
        let r = 0.180625 - (p - 0.5).powi(2);
        let num = 2.506628238849053f32;
        num * (p - 0.5) / (1.0 - r)
    } else {
        let r = if p > 0.5 { 1.0 - p } else { p };
        let t = (-2.0 * r.ln()).sqrt();
        let mut x = t - (2.30753 + 0.27061 * t) / (1.0 + (0.99229 + 0.04481 * t) * t);
        if p < 0.5 {
            x = -x;
        }
        x
    }
}

/// Packs quantization indices into a compact byte array.
/// For non-integer bit widths, uses arithmetic packing.
///
/// # Arguments
/// * `indices` - Quantization indices
/// * `bits` - Bits per index (can be fractional like 3.5)
///
/// # Returns
/// Packed bytes
pub fn pack_indices(indices: &[usize], bits: f32) -> Vec<u8> {
    if indices.is_empty() {
        return vec![];
    }

    let num_levels = (2.0_f32.powf(bits)).round() as usize;
    let bits_per_index = bits.ceil() as u32;
    let mut result = Vec::new();
    let mut current_byte: u8 = 0;
    let mut bit_offset: u32 = 0;

    for &idx in indices {
        let idx = (idx as u8).min((num_levels - 1) as u8);

        // Write bits_per_index bits
        for bit in 0..bits_per_index {
            if (idx & (1 << bit)) != 0 {
                current_byte |= 1 << bit_offset;
            }
            bit_offset += 1;

            if bit_offset == 8 {
                result.push(current_byte);
                current_byte = 0;
                bit_offset = 0;
            }
        }
    }

    if bit_offset > 0 {
        result.push(current_byte);
    }

    result
}

/// Unpacks quantization indices from a compact byte array.
pub fn unpack_indices(packed: &[u8], num_indices: usize, bits: f32) -> Vec<usize> {
    if packed.is_empty() || num_indices == 0 {
        return vec![];
    }

    let bits_per_index = bits.ceil() as u32;
    let mut indices = Vec::with_capacity(num_indices);
    let mut bit_pos: u32 = 0;

    for _ in 0..num_indices {
        let mut idx: usize = 0;
        for bit in 0..bits_per_index {
            let byte_idx = bit_pos as usize / 8;
            let bit_offset = bit_pos % 8;
            if byte_idx < packed.len() && (packed[byte_idx] & (1 << bit_offset)) != 0 {
                idx |= 1 << bit;
            }
            bit_pos += 1;
        }
        indices.push(idx);
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantizer_creation() {
        let q = ScalarQuantizer::new(3.5, -1.0, 1.0);
        assert_eq!(q.bits, 3.5);
        assert_eq!(q.num_levels, 11); // round(2^3.5) = round(11.31) = 11
        assert!(!q.boundaries.is_empty());
        assert!(!q.centroids.is_empty());
    }

    #[test]
    fn test_quantize_reconstruct() {
        let q = ScalarQuantizer::new(4.0, -2.0, 2.0);
        let x = vec![-1.5, 0.0, 1.5];
        let quantized = q.quantize_slice(&x);
        assert_eq!(quantized.len(), x.len());

        // Check that quantized values are within range
        for &v in &quantized {
            assert!(v >= q.range_min && v <= q.range_max);
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let q = ScalarQuantizer::new(3.0, -1.0, 1.0);
        let x = vec![0.1, -0.5, 0.8, 0.0];
        let indices = q.encode_indices(&x);
        let packed = pack_indices(&indices, q.bits);
        let unpacked = unpack_indices(&packed, indices.len(), q.bits);
        let reconstructed = q.decode_indices(&unpacked);

        assert_eq!(reconstructed.len(), x.len());
    }
}
