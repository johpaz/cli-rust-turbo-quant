//! Tokenizer wrapper for HuggingFace tokenizers.

use tokenizers::Tokenizer;
use tracing::info;

pub struct GemmaTokenizer {
    tokenizer: Tokenizer,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

impl GemmaTokenizer {
    /// Load tokenizer from a JSON file.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        info!("Loading tokenizer from: {}", path);
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let eos_token_id = tokenizer.get_vocab_size() as u32 - 1; // default fallback
        let bos_token_id = 2u32; // Gemma default

        // Try to extract from tokenizer config
        if let Some(eos) = tokenizer.get_added_vocab_decoder() {
            // Try common EOS token IDs
            if let Some(id) = tokenizer.token_to_id("<eos>") {
                let _ = id;
            }
        }

        Ok(Self {
            tokenizer,
            eos_token_id,
            bos_token_id,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&x| x as u32).collect())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        let ids: Vec<u64> = token_ids.iter().map(|&x| x as u64).collect();
        self.tokenizer.decode(&ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// Get the BOS token as a string for prefix.
    pub fn bos_token_str(&self) -> String {
        "<bos>".to_string()
    }
}

/// Creates a tokenizer from GGUF metadata (when no separate tokenizer file is available).
/// Gemma models use SentencePiece tokenizers.
pub fn create_gemma_tokenizer(gguf_path: &str) -> anyhow::Result<GemmaTokenizer> {
    // Try to find a tokenizer file near the GGUF
    use std::path::Path;
    let gguf = Path::new(gguf_path);
    let parent = gguf.parent().unwrap_or(Path::new("."));

    // Search for tokenizer files
    let candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
    ];

    for name in &candidates {
        let path = parent.join(name);
        if path.exists() {
            if name.ends_with(".json") {
                return GemmaTokenizer::from_file(path.to_str().unwrap());
            }
        }
    }

    // Try current directory
    for name in &candidates {
        let path = Path::new(name);
        if path.exists() && name.ends_with(".json") {
            return GemmaTokenizer::from_file(name);
        }
    }

    // Fallback: create a minimal tokenizer info
    info!("No tokenizer file found. Will use basic token ID mapping.");
    info!("Please download tokenizer.json for the model and place it next to the GGUF file.");

    anyhow::bail!(
        "No tokenizer file found near '{}'.\n\
         Please download the tokenizer for this model (tokenizer.json) and place it\n\
         next to the GGUF file or in the current directory.",
        gguf_path
    )
}
