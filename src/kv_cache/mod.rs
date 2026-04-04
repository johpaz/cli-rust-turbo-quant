//! KV Cache module for TurboQuant.
//!
//! Implements compressed KV cache storage with on-the-fly attention computation,
//! following the TurboQuant paper's approach of 3.5-bit/channel compression.

pub mod quantized_kv;
pub mod attention;

// Re-exports
pub use quantized_kv::{QuantizedKVCache, MultiHeadQuantizedKVCache};
pub use attention::quantized_attention;

/// TurboQuant Inference session state.
///
/// Manages the quantized KV cache across all layers and heads for a
/// complete inference session.
pub struct TurboQuantInference {
    /// Per-layer multi-head KV caches
    pub layer_caches: Vec<MultiHeadQuantizedKVCache>,
    /// Bits per dimension
    pub bits: f32,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads per layer
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Total tokens processed
    pub tokens_processed: usize,
}

impl TurboQuantInference {
    /// Creates a new inference session.
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads per layer
    /// * `head_dim` - Dimension per head
    /// * `bits` - Bits per dimension (e.g., 3.5)
    /// * `max_seq_len` - Maximum sequence length
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        bits: f32,
        max_seq_len: usize,
    ) -> Self {
        let layer_caches = (0..num_layers)
            .map(|_| MultiHeadQuantizedKVCache::new(num_heads, head_dim, bits, max_seq_len))
            .collect();

        Self {
            layer_caches,
            bits,
            num_layers,
            num_heads,
            head_dim,
            max_seq_len,
            tokens_processed: 0,
        }
    }

    /// Appends new K/V vectors for all layers and heads at a token position.
    ///
    /// # Arguments
    /// * `keys` - Keys per layer per head: [num_layers][num_heads][head_dim]
    /// * `values` - Values per layer per head: [num_layers][num_heads][head_dim]
    /// * `token_id` - Optional token ID
    pub fn append_token(
        &mut self,
        keys: &[Vec<Vec<f32>>],
        values: &[Vec<Vec<f32>>],
        token_id: Option<usize>,
    ) {
        assert_eq!(keys.len(), self.num_layers);
        assert_eq!(values.len(), self.num_layers);

        for (layer_idx, cache) in self.layer_caches.iter_mut().enumerate() {
            cache.append_multi_head(&keys[layer_idx], &values[layer_idx], token_id);
        }

        self.tokens_processed += 1;
    }

    /// Computes total KV cache memory usage in GB.
    pub fn total_kv_cache_memory_gb(&self) -> f32 {
        self.layer_caches.iter().map(|c| c.total_memory_gb()).sum()
    }

    /// Resets the inference state (for new sequences).
    pub fn reset(&mut self) {
        for cache in &mut self.layer_caches {
            cache.clear_all();
        }
        self.tokens_processed = 0;
    }

    /// Returns the current sequence length.
    pub fn seq_len(&self) -> usize {
        if self.layer_caches.is_empty() {
            0
        } else {
            self.layer_caches[0].seq_len()
        }
    }

    /// Prints a memory usage summary.
    pub fn memory_summary(&self) -> String {
        format!(
            "TurboQuant KV Cache:\n\
             ├─ Bits/channel: {:.1}\n\
             ├─ Layers: {}\n\
             ├─ Heads/layer: {}\n\
             ├─ Head dim: {}\n\
             ├─ Sequence length: {}\n\
             ├─ Total KV cache: {:.4} GB\n\
             └─ Compression vs FP16: ~{:.1}x",
            self.bits,
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.seq_len(),
            self.total_kv_cache_memory_gb(),
            16.0 / self.bits, // FP16 = 16 bits
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_session_creation() {
        let session = TurboQuantInference::new(
            32,  // layers
            32,  // heads
            128, // head_dim
            3.5, // bits
            4096, // max_seq_len
        );

        assert_eq!(session.num_layers, 32);
        assert_eq!(session.num_heads, 32);
        assert_eq!(session.head_dim, 128);
        assert_eq!(session.bits, 3.5);
        assert_eq!(session.tokens_processed, 0);
    }

    #[test]
    fn test_inference_append_and_reset() {
        let mut session = TurboQuantInference::new(
            2,   // layers (small for testing)
            4,   // heads
            32,  // head_dim
            3.5, // bits
            1024, // max_seq_len
        );

        // Create dummy K/V
        let keys: Vec<_> = (0..2)
            .map(|l| (0..4).map(|h| vec![h as f32 * 0.1; 32]).collect())
            .collect();
        let values: Vec<_> = (0..2)
            .map(|l| (0..4).map(|h| vec![h as f32 * 0.2; 32]).collect())
            .collect();

        session.append_token(&keys, &values, Some(0));
        assert_eq!(session.seq_len(), 1);
        assert_eq!(session.tokens_processed, 1);

        session.reset();
        assert_eq!(session.seq_len(), 0);
        assert_eq!(session.tokens_processed, 0);
    }

    #[test]
    fn test_memory_summary() {
        let session = TurboQuantInference::new(4, 8, 64, 3.5, 2048);
        let summary = session.memory_summary();

        assert!(summary.contains("3.5"));
        assert!(summary.contains("Layers: 4"));
        assert!(summary.contains("Heads/layer: 8"));
    }
}
