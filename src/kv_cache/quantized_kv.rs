//! Quantized KV Cache implementation.
//!
//! Implements the KV cache compression described in the TurboQuant paper:
//! - K and V vectors are compressed to 3.5 bits/channel
//! - Two-stage encoding: rotation + scalar quant + QJL correction
//! - On-the-fly decompression for attention computation
//! - No need to materialize the full cache in memory

use crate::math::{TurboQuantEncoding, encode_vector};

/// A single quantized KV cache entry (for one token position).
#[derive(Debug, Clone)]
pub struct QuantizedKVEntry {
    /// Quantized Key vector encoding
    pub key_enc: TurboQuantEncoding,
    /// Quantized Value vector (stored similarly)
    pub value_enc: TurboQuantEncoding,
    /// Token ID (for reference/debugging)
    pub token_id: Option<usize>,
}

/// Quantized KV Cache for a single attention head.
///
/// Stores all past token K/V vectors in compressed form.
/// Memory usage: O(seq_len * hidden_dim * bits) instead of O(seq_len * hidden_dim * 16) for FP16.
#[derive(Debug)]
pub struct QuantizedKVCache {
    /// Compressed entries
    pub entries: Vec<QuantizedKVEntry>,
    /// Bits per dimension (e.g., 3.5)
    pub bits: f32,
    /// Hidden dimension (head_dim)
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl QuantizedKVCache {
    /// Creates a new empty quantized KV cache.
    ///
    /// # Arguments
    /// * `bits` - Bits per dimension (e.g., 3.5 for quality-neutral compression)
    /// * `head_dim` - Dimension of each K/V vector (per head)
    /// * `max_seq_len` - Maximum sequence length the cache can hold
    pub fn new(bits: f32, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_seq_len),
            bits,
            head_dim,
            max_seq_len,
        }
    }

    /// Appends a new K/V pair to the cache (encoding it).
    ///
    /// # Arguments
    /// * `key` - Key vector for the new token
    /// * `value` - Value vector for the new token
    /// * `token_id` - Optional token ID for tracking
    pub fn append(&mut self, key: &[f32], value: &[f32], token_id: Option<usize>) {
        assert_eq!(key.len(), self.head_dim, "Key dimension mismatch");
        assert_eq!(value.len(), self.head_dim, "Value dimension mismatch");
        assert!(self.entries.len() < self.max_seq_len, "KV cache full");

        let key_enc = encode_vector(key, self.bits);
        let value_enc = encode_vector(value, self.bits);

        self.entries.push(QuantizedKVEntry {
            key_enc,
            value_enc,
            token_id,
        });
    }

    /// Returns the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Computes the approximate memory usage in bytes.
    ///
    /// Each entry stores:
    /// - Key: bits * head_dim / 8 bytes + QJL signs (1 bit/dim) + metadata
    /// - Value: same
    pub fn memory_bytes(&self) -> usize {
        let entry_bits = self.bits * self.head_dim as f32; // scalar quant bits
        let qjl_bits = self.head_dim as f32; // 1 bit/dim for QJL
        let total_bits_per_vector = entry_bits + qjl_bits;
        let bytes_per_entry = (total_bits_per_vector * 2.0 / 8.0).ceil() as usize; // *2 for K+V
        bytes_per_entry * self.entries.len()
    }

    /// Computes the memory usage in GB for reporting.
    pub fn memory_gb(&self) -> f32 {
        self.memory_bytes() as f32 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Compression ratio compared to FP16 baseline.
    pub fn compression_ratio(&self) -> f32 {
        let fp16_bytes = self.entries.len() * self.head_dim * 2 * 2; // K+V, 2 bytes each (FP16)
        let quant_bytes = self.memory_bytes();
        if quant_bytes == 0 {
            return 1.0;
        }
        fp16_bytes as f32 / quant_bytes as f32
    }

    /// Retrieves the reconstructed Key vector for a given position.
    pub fn get_key(&self, position: usize) -> Option<Vec<f32>> {
        self.entries.get(position).map(|e| crate::math::decode_vector(&e.key_enc))
    }

    /// Retrieves the reconstructed Value vector for a given position.
    pub fn get_value(&self, position: usize) -> Option<Vec<f32>> {
        self.entries.get(position).map(|e| crate::math::decode_vector(&e.value_enc))
    }

    /// Clears the cache (for reuse across sequences).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the QJL-encoded Key reference for attention computation (without full reconstruction).
    pub fn get_key_encoding(&self, position: usize) -> Option<&TurboQuantEncoding> {
        self.entries.get(position).map(|e| &e.key_enc)
    }

    /// Returns the Value vector for a given position (reconstructed).
    pub fn get_value_reconstructed(&self, position: usize) -> Option<Vec<f32>> {
        self.get_value(position)
    }
}

/// Multi-head quantized KV cache (one cache per attention head).
#[derive(Debug)]
pub struct MultiHeadQuantizedKVCache {
    /// Per-head caches
    pub heads: Vec<QuantizedKVCache>,
    /// Number of heads
    pub num_heads: usize,
    /// Bits per dimension
    pub bits: f32,
}

impl MultiHeadQuantizedKVCache {
    /// Creates a new multi-head quantized KV cache.
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `bits` - Bits per dimension
    /// * `max_seq_len` - Maximum sequence length
    pub fn new(num_heads: usize, head_dim: usize, bits: f32, max_seq_len: usize) -> Self {
        Self {
            heads: (0..num_heads)
                .map(|_| QuantizedKVCache::new(bits, head_dim, max_seq_len))
                .collect(),
            num_heads,
            bits,
        }
    }

    /// Appends K/V vectors for all heads at a new token position.
    ///
    /// # Arguments
    /// * `keys` - Key vectors per head: [num_heads][head_dim]
    /// * `values` - Value vectors per head: [num_heads][head_dim]
    /// * `token_id` - Optional token ID
    pub fn append_multi_head(
        &mut self,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        token_id: Option<usize>,
    ) {
        assert_eq!(keys.len(), self.num_heads, "Keys head count mismatch");
        assert_eq!(values.len(), self.num_heads, "Values head count mismatch");

        for (head_idx, cache) in self.heads.iter_mut().enumerate() {
            cache.append(&keys[head_idx], &values[head_idx], token_id);
        }
    }

    /// Total memory usage across all heads in GB.
    pub fn total_memory_gb(&self) -> f32 {
        self.heads.iter().map(|h| h.memory_gb()).sum()
    }

    /// Clears all head caches.
    pub fn clear_all(&mut self) {
        for head in &mut self.heads {
            head.clear();
        }
    }

    /// Total entries (should be the same across all heads).
    pub fn seq_len(&self) -> usize {
        if self.heads.is_empty() {
            0
        } else {
            self.heads[0].len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_basic_operations() {
        let mut cache = QuantizedKVCache::new(3.5, 64, 1024);
        assert!(cache.is_empty());

        let key = vec![0.1; 64];
        let value = vec![0.2; 64];
        cache.append(&key, &value, Some(0));

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_kv_cache_memory_savings() {
        let mut cache = QuantizedKVCache::new(3.5, 128, 4096);

        // Add 100 entries
        for i in 0..100 {
            let key = vec![(i as f32) * 0.01; 128];
            let value = vec![(i as f32) * 0.02; 128];
            cache.append(&key, &value, Some(i));
        }

        let ratio = cache.compression_ratio();
        // At 3.5 bits + 1 bit QJL vs 16 bits FP16, should be > 2x
        assert!(ratio > 2.0, "Expected compression > 2x, got {}", ratio);
    }

    #[test]
    fn test_multi_head_kv_cache() {
        let mut multi = MultiHeadQuantizedKVCache::new(8, 64, 3.5, 2048);

        let keys: Vec<_> = (0..8).map(|h| vec![h as f32 * 0.1; 64]).collect();
        let values: Vec<_> = (0..8).map(|h| vec![h as f32 * 0.2; 64]).collect();
        multi.append_multi_head(&keys, &values, Some(0));

        assert_eq!(multi.seq_len(), 1);
        assert_eq!(multi.heads.len(), 8);
    }
}
