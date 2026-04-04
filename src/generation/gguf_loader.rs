//! GGUF model loader.
//!
//! Loads all tensors from a GGUF file and builds the model configuration.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor, Result};
use candle_core::quantized::QTensor;
use tracing::info;

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub architecture: String,
    /// Gemma 4 MoE specific
    pub num_local_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
}

impl ModelConfig {
    pub fn from_gguf(content: &gguf_file::Content) -> anyhow::Result<Self> {
        // Detect architecture prefix
        let arch = content.metadata.get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_else(|| "gemma".to_string());

        let prefixes = if arch.contains("gemma") {
            vec!["gemma4", "gemma3", "gemma2", "gemma"]
        } else {
            vec![arch.as_str()]
        };

        let md_get = |key: &str| -> anyhow::Result<&gguf_file::Value> {
            for prefix in &prefixes {
                let full_key = format!("{}.{}", prefix, key);
                if let Some(v) = content.metadata.get(&full_key) {
                    return Ok(v);
                }
            }
            anyhow::bail!("Missing metadata key. Tried prefixes: {:?}, key: {}", prefixes, key)
        };

        let hidden_size = md_get("embedding_length")?.to_u32()? as usize;
        let num_attention_heads = md_get("attention.head_count")?.to_u32()? as usize;
        let num_key_value_heads = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let num_hidden_layers = md_get("block_count")?.to_u32()? as usize;
        let intermediate_size = md_get("feed_forward_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f64()?;

        let head_dim = if let Ok(v) = md_get("attention.key_length") {
            v.to_u32()? as usize
        } else {
            hidden_size / num_attention_heads
        };

        let rope_theta = if let Ok(v) = md_get("rope.freq_base") {
            v.to_f64()?
        } else {
            10_000.0
        };

        let vocab_size = if let Ok(v) = md_get("logits") {
            v.to_u32()? as usize
        } else {
            // Try to infer from tensor shapes
            content.tensor_infos.get("token_embd.weight")
                .map(|t| t.shape.dims().first().copied().unwrap_or(0))
                .unwrap_or(256000)
        };

        // MoE specific
        let num_local_experts = md_get("expert_count")
            .ok()
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);

        let num_experts_per_tok = md_get("expert_used_count")
            .ok()
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);

        info!("Model config: arch={} hidden={} heads={}/{} layers={} intermediate={} head_dim={} vocab={} rope_theta={:.0} experts={:?}",
            arch, hidden_size, num_attention_heads, num_key_value_heads, num_hidden_layers,
            intermediate_size, head_dim, vocab_size, rope_theta, num_local_experts);

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            head_dim,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            vocab_size,
            architecture: arch,
            num_local_experts,
            num_experts_per_tok,
        })
    }
}

/// Holds all loaded GGUF content for model building.
pub struct GgufContent {
    pub gguf: gguf_file::Content,
    pub config: ModelConfig,
    pub file: BufReader<File>,
}

impl GgufContent {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        info!("Loading GGUF: {}", path);
        let file = BufReader::new(File::open(path)?);
        let mut seekable = std::fs::File::open(path)?;
        let gguf = gguf_file::Content::read(&mut seekable)?;
        let config = ModelConfig::from_gguf(&gguf)?;

        // Re-open file for tensor reading (we consumed the read)
        let file = BufReader::new(File::open(path)?);

        Ok(Self { gguf, config, file })
    }
}

/// GGUF model loader - provides access to tensors.
pub struct GgufModelLoader {
    pub content: GgufContent,
}

impl GgufModelLoader {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let content = GgufContent::load(path)?;
        Ok(Self { content })
    }

    /// Get a QTensor from the GGUF file.
    pub fn get_qtensor(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        self.content.gguf.tensor(&mut self.content.file, name, device)
    }

    /// Get a dequantized tensor from the GGUF file.
    pub fn get_tensor(&mut self, name: &str, device: &Device) -> Result<Tensor> {
        let qtensor = self.get_qtensor(name, device)?;
        qtensor.dequantize(device)
    }
}
