//! Random rotation module based on the TurboQuant paper.
//!
//! Implements Fast Walsh-Hadamard Transform (FWHT) combined with random sign flips
//! to induce a concentrated Beta distribution on coordinates post-rotation.
//! This is the first stage of the TurboQuant pipeline.
//!
//! Paper: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! Algorithm: x' = (1/sqrt(d)) * H * D * x
//!   where H = Hadamard matrix, D = random diagonal ±1 matrix

use rand::{Rng, SeedableRng};

/// Applies random sign flips (diagonal random rotation) to a vector.
/// This is the D matrix in H*D*x, where D = diag(±1) with equal probability.
///
/// # Arguments
/// * `x` - Input vector (mutable, modified in-place)
/// * `seed` - Random seed for reproducibility
pub fn random_sign_flips(x: &mut [f32], seed: u64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for val in x.iter_mut() {
        if rng.random::<bool>() {
            *val = -*val;
        }
    }
}

/// Fast Walsh-Hadamard Transform (in-place, unnormalized).
///
/// Computes H * x where H is the Hadamard matrix of order len(x).
/// Requires that len(x) is a power of 2. If not, the vector is zero-padded
/// internally and truncated after transformation.
///
/// # Algorithm (recursive butterfly factorization):
///   H_1 = [1]
///   H_{2n} = [H_n  H_n]
///            [H_n -H_n]
///
/// # Arguments
/// * `x` - Input vector (modified in-place). Length must be a power of 2.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FWHT requires vector length to be a power of 2, got {}", n);
    assert!(n > 0, "FWHT requires non-empty vector");

    // Iterative Cooley-Tukey style FWHT
    let mut step = 1;
    while step < n {
        let half = step;
        step <<= 1;
        for i in (0..n).step_by(step) {
            for j in 0..half {
                let u = x[i + j];
                let v = x[i + half + j];
                x[i + j] = u + v;
                x[i + half + j] = u - v;
            }
        }
    }

    // Normalize: divide by sqrt(n)
    let norm = (n as f32).sqrt();
    for val in x.iter_mut() {
        *val /= norm;
    }
}

/// Full random rotation: x' = (1/sqrt(d)) * H * D * x
///
/// This combines random sign flips with FWHT to produce a rotated vector
/// whose coordinates follow an approximately Gaussian/Beta distribution,
/// which is crucial for the scalar quantization stage.
///
/// # Arguments
/// * `x` - Input vector
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Rotated vector (new allocation). Length is padded to next power of 2 if needed.
pub fn random_rotate(x: &[f32], seed: u64) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Pad to next power of 2
    let padded_len = n.next_power_of_two();
    let mut buf = x.to_vec();
    buf.resize(padded_len, 0.0);

    // Step 1: Random sign flips (D matrix)
    random_sign_flips(&mut buf, seed);

    // Step 2: FWHT (H matrix)
    fwht(&mut buf);

    // Truncate back to original length
    buf.truncate(n);
    buf
}

/// Generates a random rotation seed from system entropy.
pub fn fresh_seed() -> u64 {
    let mut rng = rand::rng();
    rng.random::<u64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_power_of_2() {
        let mut x = vec![1.0, 0.0, 0.0, 0.0];
        fwht(&mut x);
        // H_4 * [1,0,0,0] = [0.5, 0.5, 0.5, 0.5] (normalized)
        for val in &x {
            assert!((val - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_random_sign_flips_preserves_magnitude() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let original_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        
        let mut y = x.clone();
        random_sign_flips(&mut y, 42);
        let new_norm: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        
        assert!((original_norm - new_norm).abs() < 1e-6);
    }

    #[test]
    fn test_random_rotate_output_length() {
        let x = vec![1.0, 2.0, 3.0];
        let y = random_rotate(&x, 123);
        assert_eq!(y.len(), x.len());
    }

    #[test]
    #[should_panic(expected = "FWHT requires vector length to be a power of 2")]
    fn test_fwht_non_power_of_2_panics() {
        let mut x = vec![1.0, 2.0, 3.0];
        fwht(&mut x);
    }
}
