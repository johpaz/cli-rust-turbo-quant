//! Token sampling from logits.
//!
//! Supports: argmax (greedy), temperature sampling, top-p (nucleus) sampling.

use candle_core::{DType, Result, Tensor};
use rand::Rng;

pub struct LogitsSampler {
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: u64,
}

impl LogitsSampler {
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        Self {
            temperature,
            top_p,
            seed,
        }
    }

    /// Sample a token ID from logits.
    ///
    /// # Arguments
    /// * `logits` - 1D tensor of logits (vocab_size,)
    ///
    /// # Returns
    /// Token ID as u32
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        match self.temperature {
            None | Some(t) if t <= 0.0 => {
                // Argmax (greedy)
                let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
                Ok(next)
            }
            Some(temp) => {
                // Temperature sampling
                let logits_f64 = logits.to_dtype(DType::F64)?;
                let scaled = (&logits_f64 / temp)?;

                let probs = if let Some(top_p_val) = self.top_p {
                    // Top-p (nucleus) sampling
                    top_p_sample(&scaled, top_p_val, self.seed)?
                } else {
                    // Full softmax sampling
                    softmax(&scaled)?
                };

                // Sample from distribution
                let token_id = sample_from_probs(&probs, self.seed)?;
                self.seed = self.seed.wrapping_add(1);
                Ok(token_id)
            }
        }
    }
}

fn softmax(logits: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax_last_dim(logits)
}

fn top_p_sample(logits: &Tensor, top_p: f64, seed: u64) -> Result<Tensor> {
    let probs = softmax(logits)?;

    // Sort probabilities in descending order
    let sorted_probs = probs.sort(D::Minus1, true)?;
    let cumulative_probs = sorted_probs.cumsum(D::Minus1)?;

    // Create mask for top-p
    let mask = cumulative_probs.ge(top_p)?;

    // Shift mask right by 1 to include the first element above threshold
    let vocab_size = mask.dim(D::Minus1)?;
    if vocab_size <= 1 {
        return Ok(probs);
    }

    // Get indices that pass the mask
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let probs_vec: Vec<f64> = probs.to_vec1()?;

    // Nucleus sampling: keep only tokens until cumulative prob exceeds top_p
    let mut indices_and_probs: Vec<(usize, f64)> = probs_vec.iter().enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indices_and_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0;
    let mut nucleus = Vec::new();
    for (idx, prob) in &indices_and_probs {
        nucleus.push((*idx, *prob));
        cumulative += prob;
        if cumulative >= top_p {
            break;
        }
    }

    // Renormalize
    let total: f64 = nucleus.iter().map(|(_, p)| p).sum();
    let nucleus: Vec<(usize, f64)> = nucleus.iter()
        .map(|(idx, p)| (*idx, p / total))
        .collect();

    // Sample
    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    for (idx, prob) in &nucleus {
        cumulative += prob;
        if r <= cumulative {
            return Tensor::from_vec(vec![*idx as f64], (1,), logits.device())
                .and_then(|t| t.to_dtype(DType::F64));
        }
    }

    // Fallback: return the last nucleus token
    let last_idx = nucleus.last().map(|(idx, _)| *idx).unwrap_or(0);
    Tensor::from_vec(vec![last_idx as f64], (1,), logits.device())
        .and_then(|t| t.to_dtype(DType::F64))
}

fn sample_from_probs(probs: &Tensor, seed: u64) -> Result<u32> {
    let probs_vec: Vec<f64> = probs.to_vec1()?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let r: f64 = rng.random();

    let mut cumulative = 0.0;
    for (i, &p) in probs_vec.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return Ok(i as u32);
        }
    }

    Ok((probs_vec.len() - 1) as u32)
}
