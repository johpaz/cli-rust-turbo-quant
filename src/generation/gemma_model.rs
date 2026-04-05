//! Gemma model implementation for GGUF.
//!
//! Supports both dense and MoE (Mixture of Experts) variants.
//! Gemma 4 E4B is a MoE model with sparse experts routing.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_core::quantized::{QMatMul, QTensor};
use candle_nn::{Embedding, Module, RmsNorm};
use crate::generation::gguf_loader::{GgufModelLoader, ModelConfig};
use tracing::info;

// ─── Rotary Embeddings ──────────────────────────────────────────────

pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(cfg: &ModelConfig, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq = cfg.max_position_embeddings;
        let inv_freq: Vec<f64> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / cfg.rope_theta.powf(i as f64 / dim as f64))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor, index_pos: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, q_seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, q_seq_len)?;
        let sin = self.sin.narrow(0, index_pos, q_seq_len)?;
        Ok((
            apply_rotary_emb(q, &cos, &sin)?,
            apply_rotary_emb(k, &cos, &sin)?,
        ))
    }
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, seq_len, _) = x.dims4()?;
    let cos = cos.narrow(0, 0, seq_len)?;
    let sin = sin.narrow(0, 0, seq_len)?;

    let x1 = x.narrow(D::Minus1, 0, x.dim(D::Minus1)? / 2)?;
    let x2 = x.narrow(D::Minus1, x.dim(D::Minus1)? / 2, x.dim(D::Minus1)? / 2)?;
    let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    let rope = (x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
    Ok(rope)
}

// ─── KV Cache (standard Candle) ─────────────────────────────────────

#[derive(Clone)]
pub struct KVCache {
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k_cache: None, v_cache: None }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k_out = match &self.k_cache {
            Some(prev) => Tensor::cat(&[prev, k], 2)?,
            None => k.clone(),
        };
        let v_out = match &self.v_cache {
            Some(prev) => Tensor::cat(&[prev, v], 2)?,
            None => v.clone(),
        };
        self.k_cache = Some(k_out.clone());
        self.v_cache = Some(v_out.clone());
        Ok((k_out, v_out))
    }

    pub fn reset(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

// ─── Attention Layer ────────────────────────────────────────────────

pub struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
    kv_cache: KVCache,
}

impl Attention {
    pub fn from_loader(loader: &mut GgufModelLoader, layer: usize, cfg: &ModelConfig, device: &Device) -> Result<Self> {
        let prefix = format!("blk.{}", layer);

        let q_proj = QMatMul::from_qtensor(loader.get_qtensor(&format!("{}.attn_q.weight", prefix), device)?)?;
        let k_proj = QMatMul::from_qtensor(loader.get_qtensor(&format!("{}.attn_k.weight", prefix), device)?)?;
        let v_proj = QMatMul::from_qtensor(loader.get_qtensor(&format!("{}.attn_v.weight", prefix), device)?)?;
        let o_proj = QMatMul::from_qtensor(loader.get_qtensor(&format!("{}.attn_output.weight", prefix), device)?)?;

        let q_norm = if loader.content.gguf.tensor_infos.contains_key(&format!("{}.attn_q_norm.weight", prefix)) {
            let w = loader.get_tensor(&format!("{}.attn_q_norm.weight", prefix), device)?.to_dtype(DType::F32)?;
            Some(RmsNorm::new(w, cfg.rms_norm_eps))
        } else {
            None
        };

        let k_norm = if loader.content.gguf.tensor_infos.contains_key(&format!("{}.attn_k_norm.weight", prefix)) {
            let w = loader.get_tensor(&format!("{}.attn_k_norm.weight", prefix), device)?.to_dtype(DType::F32)?;
            Some(RmsNorm::new(w, cfg.rms_norm_eps))
        } else {
            None
        };

        let rotary_emb = RotaryEmbedding::new(cfg, device)?;

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            q_norm, k_norm,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary_emb,
            kv_cache: KVCache::new(),
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?.to_dtype(dtype)?;
        let v = self.v_proj.forward(x)?.to_dtype(dtype)?;

        // Reshape: (b, s, h*d) -> (b, h, s, d)
        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // Post QK norm (Gemma specific)
        let q = if let Some(norm) = &self.q_norm { norm.forward(&q)? } else { q };
        let k = if let Some(norm) = &self.k_norm { norm.forward(&k)? } else { k };

        // RoPE
        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k, index_pos)?;

        // KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // GQA: repeat KV
        let n_rep = self.n_head / self.n_kv_head;
        let k = if n_rep > 1 { repeat_kv(&k, n_rep)? } else { k };
        let v = if n_rep > 1 { repeat_kv(&v, n_rep)? } else { v };

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn_out = attn.matmul(&v)?;

        // Reshape back and project
        let attn_out = attn_out.transpose(1, 2)?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
        self.o_proj.forward(&attn_out)
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache.reset();
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let x = x.unsqueeze(2)?;
    let x = x.expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?;
    x.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
}

// ─── MLP (Dense) ────────────────────────────────────────────────────

pub struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Mlp {
    pub fn from_loader(loader: &mut GgufModelLoader, layer: usize, device: &Device) -> Result<Self> {
        let prefix = format!("blk.{}", layer);
        let gate_proj = QMatMul::from_qtensor(loader.get_qtensor(&format!("{}.ffn_gate.weight", prefix), device)?)?;
        let up_proj = QMatMul::from_qtensor(loader.get_qtensor(&format!("{}.ffn_up.weight", prefix), device)?)?;
        let down_proj = QMatMul::from_qtensor(loader.get_qtensor(&format!("{}.ffn_down.weight", prefix), device)?)?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate = silu(&gate)?;
        let x = (gate * up)?;
        self.down_proj.forward(&x)
    }
}

// ─── MoE Layer (for Gemma 4 E4B) ────────────────────────────────────

pub struct MoE {
    gate_inp: QMatMul,
    experts: Vec<Mlp>,
    num_experts_per_tok: usize,
}

impl MoE {
    pub fn from_loader(loader: &mut GgufModelLoader, layer: usize, num_experts: usize, num_used: usize, device: &Device) -> Result<Self> {
        let prefix = format!("blk.{}", layer);
        let gate_inp = QMatMul::from_qtensor(
            loader.get_qtensor(&format!("{}.ffn_gate_inp.weight", prefix), device)?
        )?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            // MoE GGUF naming: blk.{i}.ffn_gate_expt.{e}.weight or blk.{i}.ffn_gate.{e}.weight
            let gate_name = format!("blk.{layer}.ffn_gate_expt.{e}.weight");
            let up_name = format!("blk.{layer}.ffn_up_expt.{e}.weight");
            let down_name = format!("blk.{layer}.ffn_down_expt.{e}.weight");

            // Fallback: try without _expt suffix
            let gate_name = if loader.content.gguf.tensor_infos.contains_key(&gate_name) {
                gate_name
            } else {
                format!("blk.{layer}.ffn_gate.{e}.weight")
            };
            let up_name = if loader.content.gguf.tensor_infos.contains_key(&up_name) {
                up_name
            } else {
                format!("blk.{layer}.ffn_up.{e}.weight")
            };
            let down_name = if loader.content.gguf.tensor_infos.contains_key(&down_name) {
                down_name
            } else {
                format!("blk.{layer}.ffn_down.{e}.weight")
            };

            let gate_proj = QMatMul::from_qtensor(loader.get_qtensor(&gate_name, device)?)?;
            let up_proj = QMatMul::from_qtensor(loader.get_qtensor(&up_name, device)?)?;
            let down_proj = QMatMul::from_qtensor(loader.get_qtensor(&down_name, device)?)?;

            experts.push(Mlp { gate_proj, up_proj, down_proj });
        }

        info!("  MoE layer {}: {} experts, using {} per token", layer, num_experts, num_used);

        Ok(Self { gate_inp, experts, num_experts_per_tok: num_used })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;

        // Router: compute expert scores
        let gate_logits = self.gate_inp.forward(x)?; // (b, s, num_experts)
        let gate_probs = candle_nn::ops::softmax_last_dim(&gate_logits)?;

        // Select top-k experts (manual implementation)
        let gate_probs_vec: Vec<f32> = gate_probs.flatten_all()?.to_vec1()?;
        let num_experts = gate_probs.dim(candle_core::D::Minus1)?;
        
        // For simplicity, use all experts with their probabilities
        // TODO: Implement proper top-k selection
        let topk_indices_data: Vec<u32> = (0..num_experts as u32).collect();
        let topk_probs_data: Vec<f32> = gate_probs_vec.chunks(num_experts)
            .flat_map(|chunk| chunk.to_vec())
            .collect();

        let topk_indices = Tensor::from_vec(topk_indices_data, (b_sz, seq_len, num_experts), x.device())?;
        let topk_probs = Tensor::from_vec(topk_probs_data, (b_sz, seq_len, num_experts), x.device())?;

        // Route: for each expert, compute weighted output
        let mut y = Tensor::zeros((b_sz, seq_len, hidden), x.dtype(), x.device())?;

        for e in 0..self.experts.len() {
            let expert_out = self.experts[e].forward(x)?;
            let expert_weight = topk_probs.i((.., .., e))?;
            let weighted = expert_out.broadcast_mul(&expert_weight.unsqueeze(2)?)?;
            y = (y + weighted)?;
        }

        Ok(y)
    }
}

// ─── Transformer Layer ──────────────────────────────────────────────

pub struct LayerWeights {
    pub attn_norm: RmsNorm,
    pub attention: Attention,
    pub post_attn_norm: Option<RmsNorm>,
    pub ffn_norm: RmsNorm,
    pub mlp: MlpOrMoE,
    pub post_ffn_norm: Option<RmsNorm>,
}

pub enum MlpOrMoE {
    Mlp(Mlp),
    MoE(MoE),
}

impl MlpOrMoE {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            MlpOrMoE::Mlp(mlp) => mlp.forward(x),
            MlpOrMoE::MoE(moe) => moe.forward(x),
        }
    }
}

impl LayerWeights {
    pub fn from_loader(loader: &mut GgufModelLoader, layer: usize, cfg: &ModelConfig, device: &Device) -> Result<Self> {
        let prefix = format!("blk.{}", layer);

        let attn_norm_weight = loader.get_tensor(&format!("{}.attn_norm.weight", prefix), device)?.to_dtype(DType::F32)?;
        let attn_norm = RmsNorm::new(attn_norm_weight, cfg.rms_norm_eps);

        let attention = Attention::from_loader(loader, layer, cfg, device)?;

        let post_attn_norm = if loader.content.gguf.tensor_infos.contains_key(&format!("{}.post_attention_norm.weight", prefix)) {
            let w = loader.get_tensor(&format!("{}.post_attention_norm.weight", prefix), device)?.to_dtype(DType::F32)?;
            Some(RmsNorm::new(w, cfg.rms_norm_eps))
        } else {
            None
        };

        let ffn_norm_weight = loader.get_tensor(&format!("{}.ffn_norm.weight", prefix), device)?.to_dtype(DType::F32)?;
        let ffn_norm = RmsNorm::new(ffn_norm_weight, cfg.rms_norm_eps);

        let mlp = if let (Some(num_experts), Some(num_used)) = (cfg.num_local_experts, cfg.num_experts_per_tok) {
            info!("  Layer {}: creating MoE with {} experts", layer, num_experts);
            MlpOrMoE::MoE(MoE::from_loader(loader, layer, num_experts, num_used, device)?)
        } else {
            MlpOrMoE::Mlp(Mlp::from_loader(loader, layer, device)?)
        };

        let post_ffn_norm = if loader.content.gguf.tensor_infos.contains_key(&format!("{}.post_ffw_norm.weight", prefix)) {
            let w = loader.get_tensor(&format!("{}.post_ffw_norm.weight", prefix), device)?.to_dtype(DType::F32)?;
            Some(RmsNorm::new(w, cfg.rms_norm_eps))
        } else {
            None
        };

        Ok(Self {
            attn_norm,
            attention,
            post_attn_norm,
            ffn_norm,
            mlp,
            post_ffn_norm,
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let residual = x;

        // Attention block
        let x = self.attn_norm.forward(x)?;
        let x = self.attention.forward(&x, index_pos)?;
        let x = if let Some(norm) = &self.post_attn_norm {
            norm.forward(&x)?
        } else {
            x
        };
        let x = (x + residual)?;

        // FFN block
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = if let Some(norm) = &self.post_ffn_norm {
            norm.forward(&x)?
        } else {
            x
        };
        let x = (x + residual)?;

        Ok(x)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────

pub struct GemmaGGUF {
    pub tok_embeddings: Embedding,
    pub config: ModelConfig,
    pub layers: Vec<LayerWeights>,
    pub norm: RmsNorm,
    pub output: QMatMul,
    device: Device,
}

impl GemmaGGUF {
    pub fn from_gguf(path: &str, device: Device) -> anyhow::Result<Self> {
        info!("Loading Gemma model from GGUF: {}", path);
        let mut loader = GgufModelLoader::new(path)?;
        let cfg = loader.content.config.clone();

        // Token embeddings
        info!("Loading token embeddings...");
        let tok_emb_weight = loader.get_tensor("token_embd.weight", &device)?.to_dtype(DType::F32)?;
        info!("Token embeddings loaded: shape={:?}, dtype={:?}", tok_emb_weight.shape(), tok_emb_weight.dtype());
        let tok_embeddings = Embedding::new(tok_emb_weight.clone(), cfg.hidden_size);

        // Output projection - try different naming conventions
        // For tied weights, load the raw QTensor from GGUF (avoids dequantize/quantize cycle)
        let output = if loader.content.gguf.tensor_infos.contains_key("output.weight") {
            info!("Loading output projection from 'output.weight'...");
            let output_qtensor = loader.get_qtensor("output.weight", &device)?;
            QMatMul::from_qtensor(output_qtensor)?
        } else if loader.content.gguf.tensor_infos.contains_key("output_proj.weight") {
            info!("Loading output projection from 'output_proj.weight'...");
            let output_qtensor = loader.get_qtensor("output_proj.weight", &device)?;
            QMatMul::from_qtensor(output_qtensor)?
        } else if loader.content.gguf.tensor_infos.contains_key("token_embd.weight") {
            // Use the original quantized GGUF tensor for tied weights (Q6K)
            info!("Using original GGUF quantized token_embd.weight as tied output projection...");
            let output_qtensor = loader.get_qtensor("token_embd.weight", &device)?;
            QMatMul::from_qtensor(output_qtensor)?
        } else {
            anyhow::bail!("No output projection found and no tied weights available");
        };

        // Final norm - try different naming conventions
        let norm = if loader.content.gguf.tensor_infos.contains_key("output_norm.weight") {
            info!("Loading norm from 'output_norm.weight'...");
            let norm_weight = loader.get_tensor("output_norm.weight", &device)?.to_dtype(DType::F32)?;
            RmsNorm::new(norm_weight, cfg.rms_norm_eps)
        } else if loader.content.gguf.tensor_infos.contains_key("norm.weight") {
            info!("Loading norm from 'norm.weight'...");
            let norm_weight = loader.get_tensor("norm.weight", &device)?.to_dtype(DType::F32)?;
            RmsNorm::new(norm_weight, cfg.rms_norm_eps)
        } else {
            // Create default norm with epsilon
            info!("Creating default RMS norm with eps={}", cfg.rms_norm_eps);
            RmsNorm::new(candle_core::Tensor::ones((cfg.hidden_size,), candle_core::DType::F32, &device)?.to_dtype(candle_core::DType::F32)?, cfg.rms_norm_eps)
        };

        // Layers
        info!("Loading {} transformer layers...", cfg.num_hidden_layers);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = LayerWeights::from_loader(&mut loader, i, &cfg, &device)?;
            layers.push(layer);
        }

        info!("Model loaded: {} layers, {} vocab, hidden={}, heads={}/{}",
            cfg.num_hidden_layers, cfg.vocab_size, cfg.hidden_size,
            cfg.num_attention_heads, cfg.num_key_value_heads);

        Ok(Self {
            tok_embeddings,
            config: cfg,
            layers,
            norm,
            output,
            device,
        })
    }

    /// Forward pass: token IDs → logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs tensor, shape (batch, seq_len)
    /// * `index_pos` - Starting position in KV cache
    pub fn forward(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;

        // Embed and scale (Gemma uses sqrt(hidden_size) scaling)
        let mut x = self.tok_embeddings.forward(input_ids)?;
        let scale = (self.config.hidden_size as f64).sqrt();
        x = (x * scale)?;

        // Transformer layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, index_pos)?;
        }

        // Final norm and output
        x = self.norm.forward(&x)?;

        // Take last token if seq_len > 1 and we're in generation mode
        let x = if seq_len == 1 {
            x
        } else {
            x.i((.., seq_len - 1, ..))?
        };

        // Project to vocab
        let logits = self.output.forward(&x)?;
        Ok(logits)
    }

    pub fn reset_caches(&mut self) {
        for layer in &mut self.layers {
            layer.attention.reset_cache();
        }
    }

    /// Generate text from prompt tokens.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Tokenized prompt
    /// * `max_tokens` - Maximum tokens to generate
    /// * `sampler` - Logits sampler for token selection
    /// * `eos_token_id` - End of sequence token ID
    pub fn generate_text(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        sampler: &mut crate::generation::LogitsSampler,
        eos_token_id: Option<u32>,
    ) -> anyhow::Result<Vec<u32>> {
        use crate::generation::IndexOp;
        
        let mut generated = Vec::new();
        let mut tokens = prompt_tokens.to_vec();

        info!("Generating text: {} prompt tokens, max {} tokens", prompt_tokens.len(), max_tokens);

        // Process prompt in one go if it fits
        if tokens.len() <= self.config.max_position_embeddings {
            let prompt_tensor = Tensor::new(tokens.as_slice(), &self.device)?
                .unsqueeze(0)?;
            
            let logits = self.forward(&prompt_tensor, 0)?;
            let next_token = sampler.sample(&logits.squeeze(0)?)?;
            tokens.push(next_token);
            generated.push(next_token);
        }

        // Continue generating
        while generated.len() < max_tokens {
            if let Some(eos) = eos_token_id {
                if generated.last() == Some(&eos) {
                    generated.pop(); // Remove EOS from output
                    break;
                }
            }

            let last_token = *tokens.last().unwrap();
            let token_tensor = Tensor::new(&[last_token], &self.device)?
                .unsqueeze(0)?;
            
            let pos = tokens.len() - 1;
            let logits = match self.forward(&token_tensor, pos) {
                Ok(l) => l,
                Err(e) => {
                    info!("Generation stopped: {}", e);
                    break;
                }
            };
            
            let next_token = sampler.sample(&logits.squeeze(0)?)?;
            tokens.push(next_token);
            generated.push(next_token);
        }

        info!("Generated {} tokens", generated.len());
        Ok(generated)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────

fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    x * sigmoid
}
