# 🚀 TurboQuant (Rust Implementation)

**TurboQuant** is a Rust implementation of the algorithm from the Google paper:
> **"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"**
> Amir Zandieh, Mahdi Daliri, Amir Hadian, Vahab Mirrokni (arXiv:2504.19874)

This library compresses the **KV cache** of LLMs to **2.5–3.5 bits/channel** with near-zero preprocessing, using a mathematically proven near-optimal distortion rate.

---

## 📖 What is TurboQuant?

TurboQuant is a **vector quantization algorithm** (not weight quantization like GPTQ/AWQ). It targets the **KV cache** — the memory bottleneck during autoregressive inference — achieving **4.5–6× compression** with quality neutrality at 3.5 bits.

### Key Properties (from the paper)

| Property | Value |
|----------|-------|
| **KV Cache bits** | 2.5–3.5 bits/channel |
| **Quality at 3.5 bits** | Neutral (no measurable degradation) |
| **Quality at 2.5 bits** | ~1.2% drop on LongBench |
| **Compression ratio** | 4.5–6× vs FP16 KV cache |
| **Indexing time** | ~0.001 seconds (virtually zero) |
| **Calibration needed** | **None** (data-oblivious) |
| **Distortion bound** | ≤ 2.7× theoretical minimum |

### How It Works (Two-Stage Algorithm)

```
Stage 1: MSE-Optimal Scalar Quantization
  ├── Random rotation (FWHT + random signs) → induces Gaussian distribution
  ├── Per-coordinate scalar quantization → minimizes MSE
  └── Result: quantized vector + residual

Stage 2: QJL 1-Bit Residual Correction
  ├── Quantized Johnson-Lindenstrauss transform on residual
  ├── 1 bit per dimension (sign of random projection)
  └── Result: unbiased inner product estimation
```

**Mathematical insight:** MSE-optimal quantization introduces bias in inner product estimation. The 1-bit QJL correction eliminates this bias in expectation, so attention scores remain accurate.

---

## 🏗️ Architecture

```
src/
├── math/                    # Core TurboQuant algorithm
│   ├── rotate.rs            # Random rotation (FWHT + random sign flips)
│   ├── quantizer.rs         # MSE-optimal scalar quantizer
│   ├── polar_quant.rs       # Polar quantization (Cartesian → polar)
│   ├── qjl.rs               # Quantized Johnson-Lindenstrauss transform
│   └── mod.rs               # Pipeline: encode/decode, corrected inner products
├── kv_cache/                # KV cache management
│   ├── quantized_kv.rs      # Compressed KV cache storage
│   ├── attention.rs         # Attention with QJL-corrected scores
│   └── mod.rs               # TurboQuantInference session manager
├── loader/                  # Model loading (GGUF, Safetensors)
├── benchmarking/            # Real performance benchmarks
├── validation/              # Model quality validation
└── ...
```

---

## 🚀 Quick Start

### 1. Build

```bash
cargo build --release
```

### 2. Check Hardware Compatibility

```bash
./target/release/turbo_quant doctor
```

### 3. Quantize a Model

```bash
# TurboQuant is data-oblivious — no calibration needed!
./target/release/turbo_quant quantize --model ./path/to/model --bits 3.5
```

### 4. Run Benchmarks

```bash
./target/release/turbo_quant benchmark --model ./path/to/model --context 4096
```

This measures:
- **Encoding speed** (vectors/sec)
- **Inner product accuracy** (MSE vs FP16)
- **KV cache memory** (actual GB usage)
- **Attention latency** (ms/token)
- **Comparison** across 2.5, 3.0, 3.5, 4.0 bits

### 5. Generate Text (Scaffold)

```bash
./target/release/turbo_quant generate --model ./path/to/model \
    --prompt "Hello, world!" \
    --bits 3.5 \
    --max-tokens 128 \
    --temperature 0.8
```

> **Note:** Full token generation requires deeper Candle integration. The math pipeline (encode, attention, QJL correction) is fully implemented and tested.

---

## 📊 Expected Results (per the Google paper)

### KV Cache Compression

| Bits/Channel | Compression vs FP16 | Quality Impact |
|-------------|---------------------|----------------|
| 3.5 bits | ~4.5× | **Neutral** (no measurable change) |
| 3.0 bits | ~5.3× | Minimal |
| 2.5 bits | ~6.4× | ~1.2% on LongBench |

### Attention Score Accuracy

At 3.5 bits/channel, the QJL-corrected inner products produce attention scores with MSE < 0.001 compared to full-precision FP16. The distortion bound is proven to be ≤ 2.7× the theoretical information-theoretic minimum.

### Memory Savings

For a model with 32 layers, 32 heads, 128-dim heads, and 4096 context:
- **FP16 KV cache:** ~2.0 GB
- **TurboQuant (3.5 bits):** ~0.44 GB
- **Savings:** ~1.56 GB (only for KV cache, not model weights)

---

## 🛠️ Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize a new workspace |
| `quantize` | Quantize a model (data-oblivious, no calibration) |
| `generate` | Generate text with quantized KV cache |
| `benchmark` | Run performance benchmarks |
| `validate` | Check model quality after quantization |
| `doctor` | Check hardware compatibility |
| `calibrate` | ⚠️ Legacy — TurboQuant doesn't need calibration |
| `package` | ⚠️ Legacy — use `quantize` instead |

---

## 🔬 Algorithm Details

### Random Rotation

Input vector `x` is rotated: `x' = (1/√d) · H · D · x`
- `D` = random diagonal matrix (±1 with equal probability)
- `H` = Hadamard matrix (via Fast Walsh-Hadamard Transform)
- **Effect:** Coordinates become approximately i.i.d. Gaussian

### Scalar Quantization

After rotation, each coordinate is quantized independently using an MSE-optimal scalar quantizer. The number of levels is `2^bits` (e.g., 11 levels for 3.5 bits).

### QJL Correction

The residual `e = x_original - x_quantized` is transformed:
1. Random projection: `y = R · e` (R ∈ {±1})
2. Quantize to signs: `b = sign(y)` → **1 bit per dimension**
3. The key property: `E[<b_x, b_y>]` preserves inner products in expectation

### Why This Works

1. **Rotation** makes coordinates nearly independent (concentration of measure)
2. **Scalar quantization** is optimal for independent coordinates
3. **QJL** corrects the bias that scalar quantization introduces in inner products
4. **Result:** Near-optimal distortion rate with zero calibration

---

## 📚 References

- **Paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- **Google Blog:** [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- **Interactive Visualization:** [TurboQuant Animated](https://mesuvash.github.io/blog/2026/turboquant-interactive/)

---

## ⚠️ Important Notes

1. **This is KV cache quantization, not weight quantization.** Model weights remain in their original precision. Only the KV cache (which grows with sequence length) is compressed.

2. **3.5 bits is the sweet spot.** The paper proves this achieves quality neutrality. Going lower (2.5 bits) introduces measurable but small degradation.

3. **No calibration needed.** Unlike GPTQ/AWQ, TurboQuant is data-oblivious. It works online with zero preprocessing.

4. **CPU implementation.** The paper was validated on H100 GPUs. This Rust implementation targets CPUs with AVX2/AVX-512 support. Performance numbers will differ.

5. **Candle integration is partial.** The math pipeline (rotation, quantization, QJL, attention) is fully implemented. Full end-to-end token generation requires additional work to integrate with Candle's transformer pipeline.

---

*Implementation of TurboQuant algorithm by Zandieh et al. (Google Research, 2025)*
