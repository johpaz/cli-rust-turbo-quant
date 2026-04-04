//! Polar Quantization as described in the TurboQuant paper.
//!
//! Converts vectors from Cartesian coordinates to polar coordinates (radius + angles),
//! exploiting the fact that high-dimensional angles exhibit a concentrated distribution
//! that aligns with a fixed geometric grid. This eliminates the need for per-block
//! scaling constants.
//!
//! The key insight: in high dimensions, after random rotation, the direction of vectors
//! concentrates on a sphere, making angular quantization very efficient.
//!
//! Paper reference: Section on PolarQuant - "Cartesian-to-polar coordinate conversion
//! with recursive pairwise grouping"

use std::f32::consts::PI;

/// Represents a polar-quantized vector.
#[derive(Debug, Clone)]
pub struct PolarQuantized {
    /// Radius (magnitude/signal strength)
    pub radius: f32,
    /// Quantized angles (in radians, one per dimension-1)
    pub angles: Vec<f32>,
    /// Original dimension
    pub dim: usize,
    /// Bit allocation for radius
    pub radius_bits: f32,
    /// Bit allocation for angles (per angle)
    pub angle_bits: f32,
}

impl PolarQuantized {
    /// Reconstructs the Cartesian vector from polar representation.
    pub fn reconstruct(&self) -> Vec<f32> {
        polar_to_cartesian(&self.radius, &self.angles)
    }
}

/// Converts Cartesian coordinates to polar coordinates.
///
/// For an n-dimensional vector, produces 1 radius and (n-1) angles.
/// Uses the generalized spherical coordinate transformation:
///   r = sqrt(sum(x_i^2))
///   phi_1 = atan2(sqrt(x_2^2 + ... + x_n^2), x_1)
///   phi_2 = atan2(sqrt(x_3^2 + ... + x_n^2), x_2)
///   ...
///   phi_{n-1} = atan2(x_n, x_{n-1})
///
/// # Arguments
/// * `x` - Input vector in Cartesian coordinates
///
/// # Returns
/// (radius, angles) where angles has length (dim - 1)
pub fn cartesian_to_polar(x: &[f32]) -> (f32, Vec<f32>) {
    let n = x.len();
    if n == 0 {
        return (0.0, vec![]);
    }
    if n == 1 {
        return (x[0].abs(), vec![]);
    }

    // Compute radius
    let radius: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    if radius < 1e-10 {
        return (0.0, vec![0.0; n - 1]);
    }

    // Compute angles recursively
    let mut angles = Vec::with_capacity(n - 1);
    compute_angles(x, &mut angles);

    (radius, angles)
}

/// Helper: recursively computes generalized spherical angles.
fn compute_angles(x: &[f32], angles: &mut Vec<f32>) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    if n == 2 {
        // 2D case: single angle
        let angle = x[1].atan2(x[0]);
        angles.push(angle);
        return;
    }

    // Compute the first angle: atan2(norm(x[1:]), x[0])
    let tail_norm: f32 = x[1..].iter().map(|v| v * v).sum::<f32>().sqrt();
    let angle = tail_norm.atan2(x[0]);
    angles.push(angle);

    // Recurse on the tail, scaled by sin(angle)
    let sin_angle = angle.sin();
    if sin_angle.abs() > 1e-10 {
        let scaled_tail: Vec<f32> = x[1..].iter().map(|v| v / sin_angle).collect();
        compute_angles(&scaled_tail, angles);
    } else {
        // Degenerate case: all remaining angles are 0
        angles.resize(angles.len() + n - 2, 0.0);
    }
}

/// Converts polar coordinates back to Cartesian.
///
/// # Arguments
/// * `radius` - Magnitude
/// * `angles` - Generalized spherical angles
///
/// # Returns
/// Cartesian vector
pub fn polar_to_cartesian(radius: &f32, angles: &[f32]) -> Vec<f32> {
    let n = angles.len() + 1;
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![*radius];
    }

    let mut result = vec![0.0; n];

    // Build up using spherical coordinate formulas
    // x_1 = r * cos(phi_1)
    // x_2 = r * sin(phi_1) * cos(phi_2)
    // x_3 = r * sin(phi_1) * sin(phi_2) * cos(phi_3)
    // ...
    // x_n = r * sin(phi_1) * sin(phi_2) * ... * sin(phi_{n-1})

    let mut cumulative_sin: f32 = 1.0;

    for i in 0..n - 1 {
        result[i] = radius * cumulative_sin * angles[i].cos();
        cumulative_sin *= angles[i].sin();
    }
    result[n - 1] = radius * cumulative_sin;

    result
}

/// Quantizes polar coordinates using a fixed geometric grid.
///
/// The key insight from TurboQuant: angles in high dimensions concentrate
/// around specific values, so we can use a fixed grid rather than
/// data-dependent boundaries.
///
/// # Arguments
/// * `radius` - Original radius
/// * `angles` - Original angles
/// * `radius_bits` - Bits allocated for radius quantization
/// * `angle_bits` - Bits allocated per angle
/// * `radius_range` - Expected [min, max] for radius
///
/// # Returns
/// PolarQuantized with quantized angles
pub fn quantize_polar(
    radius: f32,
    angles: &[f32],
    radius_bits: f32,
    angle_bits: f32,
    radius_range: (f32, f32),
) -> PolarQuantized {
    // Quantize radius with uniform quantizer
    let num_radius_levels = (2.0_f32.powf(radius_bits)).round() as usize;
    let (r_min, r_max) = radius_range;
    let quantized_radius = uniform_quantize_scalar(radius, r_min, r_max, num_radius_levels);

    // Quantize angles with fixed geometric grid
    // The grid is based on the concentration of angles in high dimensions
    let quantized_angles: Vec<f32> = angles
        .iter()
        .map(|&angle| quantize_angle_fixed_grid(angle, angle_bits))
        .collect();

    PolarQuantized {
        radius: quantized_radius,
        angles: quantized_angles,
        dim: angles.len() + 1,
        radius_bits,
        angle_bits,
    }
}

/// Uniform scalar quantization for a single value.
fn uniform_quantize_scalar(x: f32, min: f32, max: f32, num_levels: usize) -> f32 {
    if num_levels <= 1 {
        return (min + max) / 2.0;
    }
    let step = (max - min) / (num_levels - 1) as f32;
    let level = ((x - min) / step).round() as usize;
    let level = level.min(num_levels - 1);
    min + level as f32 * step
}

/// Quantizes an angle using a fixed geometric grid.
///
/// Angles are in [0, PI] after the atan2 transformation.
/// The fixed grid uses uniformly spaced points in this range.
fn quantize_angle_fixed_grid(angle: f32, bits: f32) -> f32 {
    let num_levels = (2.0_f32.powf(bits)).round() as usize;
    if num_levels <= 1 {
        return PI / 2.0; // midpoint
    }

    // Map angle from [0, PI] to [0, num_levels - 1]
    let normalized = (angle / PI) * (num_levels - 1) as f32;
    let quantized = normalized.round() as usize;
    let quantized = quantized.min(num_levels - 1);

    // Map back to [0, PI]
    (quantized as f32 / (num_levels - 1) as f32) * PI
}

/// Full polar quantization pipeline: Cartesian → Polar → Quantize → Reconstruct.
///
/// # Arguments
/// * `x` - Input vector
/// * `total_bits` - Total bit budget (split between radius and angles)
///
/// # Returns
/// (quantized_vector, polar_representation)
pub fn full_polar_quantize(x: &[f32], total_bits: f32) -> (Vec<f32>, PolarQuantized) {
    let n = x.len();
    if n == 0 {
        return (vec![], PolarQuantized {
            radius: 0.0,
            angles: vec![],
            dim: 0,
            radius_bits: 0.0,
            angle_bits: 0.0,
        });
    }

    // Convert to polar
    let (radius, angles) = cartesian_to_polar(x);

    // Split bits: 1 bit for radius (it's a single value), rest for angles
    let radius_bits = 1.0_f32.min(total_bits);
    let remaining_bits = total_bits - radius_bits;
    let angle_bits = if n > 1 {
        remaining_bits / (n - 1) as f32
    } else {
        0.0
    };

    // Quantize
    let radius_range = compute_expected_radius_range(n);
    let polar = quantize_polar(radius, &angles, radius_bits, angle_bits, radius_range);

    // Reconstruct
    let reconstructed = polar.reconstruct();

    (reconstructed, polar)
}

/// Computes the expected radius range for an n-dimensional vector
/// with approximately unit Gaussian coordinates.
/// E[||X||^2] = n for standard normal, so E[||X||] ≈ sqrt(n)
fn compute_expected_radius_range(dim: usize) -> (f32, f32) {
    if dim == 0 {
        return (0.0, 1.0);
    }
    let expected = (dim as f32).sqrt();
    // Concentration: radius is tightly concentrated around sqrt(n)
    let std_dev = 0.5; // approximate
    ((expected - 3.0 * std_dev).max(0.0), (expected + 3.0 * std_dev).max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesian_to_polar_2d() {
        let x = vec![1.0, 0.0];
        let (radius, angles) = cartesian_to_polar(&x);
        assert!((radius - 1.0).abs() < 1e-6);
        assert_eq!(angles.len(), 1);
        assert!(angles[0].abs() < 1e-6); // angle should be ~0
    }

    #[test]
    fn test_polar_reconstruction_roundtrip() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (radius, angles) = cartesian_to_polar(&x);
        let reconstructed = polar_to_cartesian(&radius, &angles);

        assert_eq!(reconstructed.len(), x.len());
        for (a, b) in x.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_polar_quantize_produces_valid_output() {
        let x = vec![0.5, -0.3, 0.8, 0.1];
        let (reconstructed, polar) = full_polar_quantize(&x, 3.5);

        assert_eq!(reconstructed.len(), x.len());
        assert_eq!(polar.dim, 4);
        assert!(polar.radius >= 0.0);
    }
}
