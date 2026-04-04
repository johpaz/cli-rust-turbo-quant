//! Real text generation with GGUF model loading and TurboQuant KV cache.
//!
//! Pipeline:
//! 1. Load GGUF weights via candle-core
//! 2. Build transformer model (Gemma architecture)
//! 3. Tokenize prompt
//! 4. Generate tokens with KV cache (quantized via TurboQuant)
//! 5. Decode tokens to text

pub mod gguf_loader;
pub mod gemma_model;
pub mod tokenizer_wrapper;
pub mod sampler;

pub use gguf_loader::GgufModelLoader;
pub use gemma_model::GemmaGGUF;
pub use sampler::LogitsSampler;

use candle_core::Tensor;
use candle_core::Device;
use tracing::{info, warn};

/// Full generation state.
pub struct Generator {
    pub model: GemmaGGUF,
    pub device: Device,
    pub kv_cache: crate::kv_cache::TurboQuantInference,
    pub use_quantized_kv: bool,
}

impl Generator {
    pub fn new(model: GemmaGGUF, device: Device, bits: f32) -> Self {
        let cfg = &model.config;
        let kv_cache = crate::kv_cache::TurboQuantInference::new(
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            bits,
            cfg.max_position_embeddings,
        );

        Self {
            model,
            device,
            kv_cache,
            use_quantized_kv: true,
        }
    }

    /// Generate text from a prompt.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Tokenized prompt
    /// * `max_tokens` - Maximum tokens to generate
    /// * `sampler` - Token sampler (temperature, top_p, etc.)
    /// * `eos_token_id` - End of sequence token
    ///
    /// # Returns
    /// Generated token IDs
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        sampler: &mut LogitsSampler,
        eos_token_id: u32,
    ) -> anyhow::Result<Vec<u32>> {
        info!("Starting generation: {} prompt tokens, max {} new tokens", prompt_tokens.len(), max_tokens);

        let mut all_tokens = prompt_tokens.to_vec();
        let mut pos = 0;

        // Prefill: process prompt in chunks
        let chunk_size = 512;
        for chunk_start in (0..prompt_tokens.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(prompt_tokens.len());
            let chunk = &prompt_tokens[chunk_start..chunk_end];

            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            let _logits = self.model.forward(&input, pos)?;
            pos = chunk_end;
        }

        info!("Prefill complete. Starting token generation...");

        // Generate loop
        for i in 0..max_tokens {
            let last_token = *all_tokens.last().unwrap();
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, pos)?;

            // Get logits for last position
            let logits = logits.i((0, 0, ..))?;

            let next_token = sampler.sample(&logits)?;

            if next_token == eos_token_id {
                info!("EOS token reached after {} generated tokens", i);
                break;
            }

            all_tokens.push(next_token);
            pos += 1;

            // Progress every 16 tokens
            if i % 16 == 15 {
                info!("Generated {} tokens...", i + 1);
            }
        }

        info!("Generation complete: {} total tokens", all_tokens.len());

        // Return only generated tokens (not prompt)
        Ok(all_tokens[prompt_tokens.len()..].to_vec())
    }
}
