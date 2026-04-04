//! Attention mechanism with QJL-corrected inner products.
//!
//! Implements the attention computation described in the TurboQuant paper:
//! instead of computing <Q, K> in full precision, we use the QJL-corrected
//! inner products from the quantized representations.
//!
//! Key insight: The MSE-optimal scalar quantizer introduces bias in inner
//! product estimation, but the 1-bit QJL residual correction eliminates this
//! bias, producing attention scores that are nearly identical to full precision.

use crate::math::{encode_vector};
use crate::kv_cache::quantized_kv::{QuantizedKVCache, MultiHeadQuantizedKVCache};
use std::f32::consts::E;

/// Computes attention scores between a query and a quantized KV cache.
///
/// Uses the QJL-corrected inner products from the TurboQuant encoding,
/// which ensures near-lossless accuracy compared to full-precision attention.
///
/// # Arguments
/// * `query` - Query vector [head_dim]
/// * `kv_cache` - Quantized KV cache
/// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
///
/// # Returns
/// Attention weights (post-softmax) for all positions in the cache
pub fn quantized_attention(
    query: &[f32],
    kv_cache: &QuantizedKVCache,
    scale: f32,
) -> Vec<f32> {
    let seq_len = kv_cache.len();
    if seq_len == 0 {
        return vec![];
    }

    // Encode the query
    let query_enc = encode_vector(query, kv_cache.bits);

    // Compute raw scores using QJL-corrected inner products
    let mut raw_scores = Vec::with_capacity(seq_len);
    for entry in &kv_cache.entries {
        let score = crate::math::corrected_inner_product(
            &query_enc,
            &entry.key_enc,
        );
        raw_scores.push(score * scale);
    }

    // Softmax
    softmax(&raw_scores)
}

/// Multi-head quantized attention.
///
/// # Arguments
/// * `queries` - Query vectors per head: [num_heads][head_dim]
/// * `kv_cache` - Multi-head quantized KV cache
/// * `scale` - Attention scale factor
///
/// # Returns
/// Attention outputs per head: [num_heads][seq_len]
pub fn multi_head_quantized_attention(
    queries: &[Vec<f32>],
    kv_cache: &MultiHeadQuantizedKVCache,
    scale: f32,
) -> Vec<Vec<f32>> {
    assert_eq!(queries.len(), kv_cache.num_heads, "Query head count mismatch");

    let mut all_attentions = Vec::with_capacity(kv_cache.num_heads);

    for (head_idx, head_cache) in kv_cache.heads.iter().enumerate() {
        let attentions = quantized_attention(&queries[head_idx], head_cache, scale);
        all_attentions.push(attentions);
    }

    all_attentions
}

/// Computes the attention output by weighting Value vectors.
///
/// # Arguments
/// * `query` - Query vector
/// * `kv_cache` - Quantized KV cache
/// * `scale` - Attention scale
///
/// # Returns
/// Attention output vector (weighted sum of Values)
pub fn quantized_attention_with_values(
    query: &[f32],
    kv_cache: &QuantizedKVCache,
    scale: f32,
) -> Vec<f32> {
    let seq_len = kv_cache.len();
    if seq_len == 0 {
        return vec![0.0; kv_cache.head_dim];
    }

    let attentions = quantized_attention(query, kv_cache, scale);

    // Weighted sum of Value vectors
    let head_dim = kv_cache.head_dim;
    let mut output = vec![0.0; head_dim];

    for (pos, &attn_weight) in attentions.iter().enumerate() {
        if let Some(value) = kv_cache.get_value(pos) {
            for (i, &v) in value.iter().enumerate() {
                output[i] += attn_weight * v;
            }
        }
    }

    output
}

/// Flash-style attention with quantized KV cache.
///
/// Processes the cache in blocks to minimize memory usage,
/// only materializing the current block's K/V vectors.
///
/// # Arguments
/// * `query` - Query vector
/// * `kv_cache` - Quantized KV cache
/// * `scale` - Attention scale
/// * `block_size` - Block size for streaming (e.g., 256)
///
/// # Returns
/// (attention_output, log_sum_exp) for numerical stability
pub fn flash_quantized_attention(
    query: &[f32],
    kv_cache: &QuantizedKVCache,
    scale: f32,
    block_size: usize,
) -> (Vec<f32>, f32) {
    let head_dim = kv_cache.head_dim;
    let seq_len = kv_cache.len();

    if seq_len == 0 {
        return (vec![0.0; head_dim], 0.0);
    }

    // Encode query once
    let query_enc = encode_vector(query, kv_cache.bits);

    let mut max_score = f32::NEG_INFINITY;
    let mut log_sum_exp = 0.0;
    let mut output = vec![0.0; head_dim];

    // Process in blocks
    for block_start in (0..seq_len).step_by(block_size) {
        let block_end = (block_start + block_size).min(seq_len);

        // Compute scores for this block
        let mut block_scores = Vec::with_capacity(block_end - block_start);
        for pos in block_start..block_end {
            if let Some(key_enc) = kv_cache.get_key_encoding(pos) {
                let score = crate::math::corrected_inner_product(
                    &query_enc,
                    key_enc,
                ) * scale;
                block_scores.push(score);
            }
        }

        // Online softmax update
        let block_max = block_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let new_max = max_score.max(block_max);

        // Rescale accumulated output and log_sum_exp
        if new_max > max_score {
            let rescale = E.powf(max_score - new_max);
            for o in &mut output {
                *o *= rescale;
            }
            // Correct online softmax: L_new = L_old * exp(m_old - m_new) + sum(exp(s_i - m_new))
            log_sum_exp = log_sum_exp * E.powf(max_score - new_max);
            for &s in &block_scores {
                log_sum_exp += E.powf(s - new_max);
            }
            max_score = new_max;
        } else {
            // Same max, just add block contributions
            for &s in &block_scores {
                log_sum_exp += E.powf(s - max_score);
            }
        }

        // Accumulate this block's contribution
        for (i, &score) in block_scores.iter().enumerate() {
            let pos = block_start + i;
            let weight = E.powf(score - max_score);
            if let Some(value) = kv_cache.get_value(pos) {
                for (j, &v) in value.iter().enumerate() {
                    output[j] += weight * v;
                }
            }
        }
    }

    // Normalize
    if log_sum_exp > 0.0 {
        for o in &mut output {
            *o /= log_sum_exp;
        }
    }

    (output, max_score + log_sum_exp.ln())
}

/// Softmax function with numerical stability.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| E.powf(x - max_val)).collect();
    let sum: f32 = exps.iter().sum();

    exps.iter().map(|&e| e / sum).collect()
}

/// Computes perplexity difference between full-precision and quantized attention.
///
/// This is useful for validating that the quantization doesn't degrade
/// the model's output quality.
///
/// # Arguments
/// * `fp16_scores` - Full-precision attention scores
/// * `quant_scores` - Quantized attention scores
///
/// # Returns
/// KL divergence between the two distributions
pub fn compute_score_divergence(fp16_scores: &[f32], quant_scores: &[f32]) -> f32 {
    assert_eq!(fp16_scores.len(), quant_scores.len());

    let mut kl_div = 0.0;
    for (&p, &q) in fp16_scores.iter().zip(quant_scores.iter()) {
        if p > 1e-10 && q > 1e-10 {
            kl_div += p * (p / q).ln();
        }
    }

    kl_div
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit should have higher probability
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_quantized_attention_empty_cache() {
        let query = vec![0.1; 64];
        let cache = QuantizedKVCache::new(3.5, 64, 1024);
        let result = quantized_attention(&query, &cache, 1.0 / 8.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_quantized_attention_with_entries() {
        let mut cache = QuantizedKVCache::new(3.5, 64, 1024);

        // Add some entries
        for i in 0..5 {
            let key = vec![(i as f32) * 0.1; 64];
            let value = vec![(i as f32) * 0.05; 64];
            cache.append(&key, &value, Some(i));
        }

        let query = vec![0.5; 64];
        let attentions = quantized_attention(&query, &cache, 1.0 / 8.0);

        assert_eq!(attentions.len(), 5);
        let sum: f32 = attentions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_flash_attention_consistency() {
        let mut cache = QuantizedKVCache::new(3.5, 32, 1024);

        for i in 0..20 {
            let key = vec![(i as f32) * 0.05; 32];
            let value = vec![(i as f32) * 0.02; 32];
            cache.append(&key, &value, Some(i));
        }

        let query = vec![0.3; 32];
        let scale = 1.0 / (32.0_f32).sqrt();

        // Standard attention
        let std_output = quantized_attention_with_values(&query, &cache, scale);

        // Flash attention
        let (flash_output, _) = flash_quantized_attention(&query, &cache, scale, 8);

        // Should be approximately equal (flash attention has small numerical differences)
        for (a, b) in std_output.iter().zip(flash_output.iter()) {
            assert!((a - b).abs() < 0.2, "Mismatch: {} vs {}", a, b);
        }
    }
}
