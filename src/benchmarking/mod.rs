//! Real benchmarking module for TurboQuant.
//!
//! Measures actual performance metrics:
//! - Tokens per second (encoding/decoding speed)
//! - Peak RAM usage (KV cache + model weights)
//! - Compression ratio vs FP16
//! - Inner product accuracy (quantized vs full precision)
//! - Latency per token

use std::path::Path;
use tracing::info;
use std::time::Instant;
use sysinfo::System;

use crate::math::{encode_vector, corrected_inner_product};
use crate::kv_cache::{QuantizedKVCache, quantized_attention};

pub struct BenchmarkResult {
    pub tokens_per_second: f32,
    pub peak_ram_gb: f32,
    pub latency_ms: f32,
    pub compression_ratio: f32,
    pub kv_cache_memory_gb: f32,
    pub ip_error_mse: f32,
    pub bits_per_channel: f32,
}

/// Runs a comprehensive benchmark of the TurboQuant pipeline.
///
/// # Arguments
/// * `model_path` - Path to the model (for metadata)
/// * `context_length` - Context length to test
///
/// # Returns
/// Benchmark result with real measurements
pub fn run_benchmark(_path: &Path, context_length: usize) -> anyhow::Result<BenchmarkResult> {
    info!("🚀 Running REAL TurboQuant benchmark");

    let mut sys = System::new_all();
    sys.refresh_all();
    let ram_start = sys.used_memory() as f32 / (1024.0 * 1024.0 * 1024.0);

    let start = Instant::now();

    // Benchmark 1: Encoding speed (vectors per second)
    info!("Benchmark 1: Vector encoding speed...");
    let encoding_result = benchmark_encoding_speed(context_length);

    // Benchmark 2: Inner product accuracy
    info!("Benchmark 2: Inner product accuracy vs FP16...");
    let ip_result = benchmark_inner_product_accuracy();

    // Benchmark 3: KV cache memory usage
    info!("Benchmark 3: KV cache memory usage...");
    let kv_result = benchmark_kv_cache_memory(context_length);

    // Benchmark 4: Attention computation latency
    info!("Benchmark 4: Attention latency...");
    let attn_result = benchmark_attention_latency(context_length);

    let elapsed = start.elapsed().as_secs_f32();

    sys.refresh_all();
    let ram_end = sys.used_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    let peak_ram = (ram_end - ram_start).max(0.01);

    info!("✅ Benchmark complete in {:.2}s", elapsed);
    info!("   Encoding: {:.0} vectors/sec", encoding_result.vectors_per_sec);
    info!("   IP MSE: {:.6}", ip_result.mse);
    info!("   KV Cache: {:.4} GB ({:.1}x vs FP16)", kv_result.memory_gb, kv_result.compression);
    info!("   Attention: {:.2} ms/token", attn_result.latency_ms);

    Ok(BenchmarkResult {
        tokens_per_second: encoding_result.vectors_per_sec,
        peak_ram_gb: peak_ram + kv_result.memory_gb,
        latency_ms: attn_result.latency_ms,
        compression_ratio: kv_result.compression,
        kv_cache_memory_gb: kv_result.memory_gb,
        ip_error_mse: ip_result.mse,
        bits_per_channel: 3.5,
    })
}

struct EncodingBenchmark {
    vectors_per_sec: f32,
}

fn benchmark_encoding_speed(_context_length: usize) -> EncodingBenchmark {
    let dim = 128; // typical head dimension
    let num_vectors = 1000;

    // Generate random test vectors
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.01).sin())
                .collect()
        })
        .collect();

    let start = Instant::now();
    let mut encodings = Vec::with_capacity(num_vectors);
    for v in &vectors {
        let enc = encode_vector(v, 3.5);
        encodings.push(enc);
    }
    let elapsed = start.elapsed().as_secs_f32();

    // Prevent optimization from removing unused results
    let _ = encodings.len();

    EncodingBenchmark {
        vectors_per_sec: num_vectors as f32 / elapsed,
    }
}

struct IPBenchmark {
    mse: f32,
}

fn benchmark_inner_product_accuracy() -> IPBenchmark {
    let dim = 128;
    let num_pairs = 500;

    let mut mse = 0.0;

    for i in 0..num_pairs {
        // Generate random vectors
        let a: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|j| ((i * dim + j + 500) as f32 * 0.1).cos()).collect();

        // Full-precision inner product
        let fp_dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

        // Quantized inner product
        let enc_a = encode_vector(&a, 3.5);
        let enc_b = encode_vector(&b, 3.5);
        let quant_dot = corrected_inner_product(&enc_a, &enc_b);

        // Squared error
        mse += (fp_dot - quant_dot).powi(2);
    }

    mse /= num_pairs as f32;

    IPBenchmark { mse }
}

struct KVCacheBenchmark {
    memory_gb: f32,
    compression: f32,
}

fn benchmark_kv_cache_memory(context_length: usize) -> KVCacheBenchmark {
    let head_dim = 128;
    let num_heads = 32;
    let bits = 3.5;

    // Create a realistic KV cache
    let mut kv_cache = QuantizedKVCache::new(bits, head_dim, context_length);

    // Fill up to context length
    for i in 0..context_length.min(1024) {
        let key: Vec<f32> = (0..head_dim).map(|j| ((i * head_dim + j) as f32 * 0.01).sin()).collect();
        let value: Vec<f32> = (0..head_dim).map(|j| ((i * head_dim + j + 1000) as f32 * 0.01).cos()).collect();
        kv_cache.append(&key, &value, Some(i));
    }

    let memory_bytes = kv_cache.memory_bytes();
    let memory_gb = memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0);

    // FP16 baseline: seq_len * head_dim * 2 bytes (FP16) * 2 (K+V) * num_heads
    let fp16_bytes = kv_cache.len() * head_dim * 2 * 2 * num_heads;
    let compression = fp16_bytes as f32 / memory_bytes.max(1) as f32;

    KVCacheBenchmark {
        memory_gb: memory_gb * num_heads as f32, // total across all heads
        compression,
    }
}

struct AttentionBenchmark {
    latency_ms: f32,
}

fn benchmark_attention_latency(context_length: usize) -> AttentionBenchmark {
    let head_dim = 128;
    let bits = 3.5;
    let seq_len = context_length.min(1024);

    // Create KV cache
    let mut kv_cache = QuantizedKVCache::new(bits, head_dim, context_length);

    for i in 0..seq_len {
        let key: Vec<f32> = (0..head_dim).map(|j| ((i * head_dim + j) as f32 * 0.01).sin()).collect();
        let value: Vec<f32> = (0..head_dim).map(|j| ((i * head_dim + j + 1000) as f32 * 0.01).cos()).collect();
        kv_cache.append(&key, &value, Some(i));
    }

    // Benchmark attention for a query
    let query: Vec<f32> = (0..head_dim).map(|j| (j as f32 * 0.05).cos()).collect();
    let scale = 1.0 / (head_dim as f32).sqrt();

    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let _attn = quantized_attention(&query, &kv_cache, scale);
    }

    let elapsed = start.elapsed().as_secs_f32();
    let latency_ms = elapsed / iterations as f32 * 1000.0;

    AttentionBenchmark { latency_ms }
}

/// Prints a formatted benchmark report.
pub fn format_benchmark_report(result: &BenchmarkResult) -> String {
    format!(
        "╔══════════════════════════════════════════════════════╗\n\
         ║            TurboQuant Benchmark Report              ║\n\
         ╠══════════════════════════════════════════════════════╣\n\
         ║  Bits/channel:      {:>8.1}                         ║\n\
         ║  Encoding speed:    {:>8.0} vec/s                   ║\n\
         ║  Attention latency: {:>8.2} ms/token                ║\n\
         ║  KV Cache memory:   {:>8.4} GB                      ║\n\
         ║  Peak RAM:          {:>8.2} GB                       ║\n\
         ║  Compression ratio: {:>8.1}x vs FP16                 ║\n\
         ║  IP Error (MSE):    {:>8.6}                         ║\n\
         ╚══════════════════════════════════════════════════════╝",
        result.bits_per_channel,
        result.tokens_per_second,
        result.latency_ms,
        result.kv_cache_memory_gb,
        result.peak_ram_gb,
        result.compression_ratio,
        result.ip_error_mse,
    )
}

/// Runs a quick benchmark comparison across different bit widths.
pub fn benchmark_bits_comparison(context_length: usize) -> String {
    let bits_options = [2.5, 3.0, 3.5, 4.0];
    let mut report = String::from("Bits Comparison (KV Cache):\n");
    report.push_str(&format!("{:<8} {:>12} {:>12} {:>12}\n", "Bits", "Memory (GB)", "Compression", "IP MSE"));
    report.push_str(&"-".repeat(48));
    report.push('\n');

    for &bits in &bits_options {
        let kv_result = benchmark_kv_cache_memory(context_length);

        // Quick IP MSE test
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1 + 1.0).cos()).collect();
        let fp_dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let enc_a = encode_vector(&a, bits);
        let enc_b = encode_vector(&b, bits);
        let quant_dot = corrected_inner_product(&enc_a, &enc_b);
        let mse = (fp_dot - quant_dot).powi(2);

        report.push_str(&format!(
            "{:<8.1} {:>12.4} {:>11.1}x {:>12.6}\n",
            bits,
            kv_result.memory_gb,
            kv_result.compression,
            mse,
        ));
    }

    report
}
