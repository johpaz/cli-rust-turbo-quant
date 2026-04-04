use candle_core::{Tensor, Device};
use candle_core::quantized::gguf_file;
use serde::{Deserialize, Serialize};
use tracing::info;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerStats {
    pub layer_id: usize,
    pub mean: f32,
    pub variance: f32,
    pub min: f32,
    pub max: f32,
    pub scale_factor: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalibrationManifest {
    pub model_hash: String,
    pub target_bits: f32,
    pub stats: Vec<LayerStats>,
    pub timestamp: u64,
}

pub struct CalibrationEngine {
    pub device: Device,
}

impl CalibrationEngine {
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    pub fn run_calibration(&self, model_path: &Path, target_bits: f32, _num_layers: usize) -> anyhow::Result<CalibrationManifest> {
        info!("Starting calibration for target: {} bits", target_bits);
        info!("Loading model from: {:?}", model_path);

        // Calculate model hash
        let model_hash = self.calculate_model_hash(model_path)?;

        // Load tensors from model
        let tensors = self.load_model_tensors(model_path)?;
        
        let pb = ProgressBar::new(tensors.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tensors ({eta})")?
            .progress_chars("#>-"));

        let mut stats = Vec::new();
        for (i, (name, tensor)) in tensors.iter().enumerate() {
            info!("Processing tensor: {} ({}/{})", name, i + 1, tensors.len());
            
            let layer_stats = self.capture_tensor_stats(tensor, i)?;
            stats.push(layer_stats);
            pb.inc(1);
        }
        pb.finish_with_message("Calibration complete!");

        let manifest = CalibrationManifest {
            model_hash,
            target_bits,
            stats,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };

        info!("Calibration finished. Captured stats for {} tensors.", manifest.stats.len());
        Ok(manifest)
    }

    fn calculate_model_hash(&self, model_path: &Path) -> anyhow::Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::io::Read;

        let mut hasher = DefaultHasher::new();
        
        if model_path.is_dir() {
            // Hash all safetensors files
            let mut entries: Vec<_> = std::fs::read_dir(model_path)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
                .collect();
            entries.sort_by_key(|e| e.file_name());
            
            for entry in entries {
                entry.file_name().hash(&mut hasher);
                let mut file = std::fs::File::open(entry.path())?;
                let mut buffer = [0u8; 8192];
                loop {
                    let bytes_read = file.read(&mut buffer)?;
                    if bytes_read == 0 {
                        break;
                    }
                    buffer[..bytes_read].hash(&mut hasher);
                }
            }
        } else {
            // Hash GGUF file
            model_path.file_name().hash(&mut hasher);
            let mut file = std::fs::File::open(model_path)?;
            let mut buffer = [0u8; 8192];
            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                buffer[..bytes_read].hash(&mut hasher);
            }
        }

        Ok(format!("sha256:{:016x}", hasher.finish()))
    }

    fn load_model_tensors(&self, model_path: &Path) -> anyhow::Result<Vec<(String, Tensor)>> {
        let mut tensors = Vec::new();

        if model_path.is_dir() {
            // Load from safetensors directory
            for entry in std::fs::read_dir(model_path)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                    let file_data = std::fs::read(&path)?;
                    let loaded = candle_core::safetensors::load_buffer(&file_data, &self.device)?;
                    for (name, tensor) in loaded {
                        tensors.push((name, tensor));
                    }
                }
            }
        } else {
            // Load from GGUF file
            let extension = model_path.extension().and_then(|s| s.to_str()).unwrap_or_default();
            if extension == "gguf" {
                let mut file = std::fs::File::open(model_path)?;
                let content = gguf_file::Content::read(&mut file)?;
                
                for (name, info) in &content.tensor_infos {
                    let qtensor = content.tensor(&mut file, name, &self.device)?;
                    // Convert QTensor to regular Tensor for stats calculation
                    let tensor = qtensor.dequantize(&self.device)?;
                    tensors.push((name.clone(), tensor));
                }
            }
        }

        info!("Loaded {} tensors from model", tensors.len());
        Ok(tensors)
    }

    pub fn capture_tensor_stats(&self, tensor: &Tensor, layer_id: usize) -> anyhow::Result<LayerStats> {
        // Calculate mean
        let mean = tensor.mean_all()?.to_scalar::<f32>()?;
        
        // Calculate min and max
        let flattened = tensor.flatten_all()?;
        let min = flattened.min(0)?.to_scalar::<f32>()?;
        let max = flattened.max(0)?.to_scalar::<f32>()?;

        // Calculate variance: Var(X) = E[X²] - (E[X])²
        let sqr_tensor = tensor.sqr()?;
        let mean_of_squares = sqr_tensor.mean_all()?.to_scalar::<f32>()?;
        let variance = mean_of_squares - mean.powi(2);

        // Calculate scale factor based on range and target bits
        // For quantization: scale = (max - min) / (2^bits - 1)
        let num_levels = (2.0_f32.powf(32.0 * self.get_target_bits_ratio())) - 1.0;
        let scale_factor = (max - min) / num_levels.max(1e-6);

        Ok(LayerStats {
            layer_id,
            mean,
            variance: variance.max(0.0), // Ensure non-negative
            min,
            max,
            scale_factor,
        })
    }

    fn get_target_bits_ratio(&self) -> f32 {
        // Default to 4-bit quantization ratio if not specified
        4.0 / 16.0 // 4-bit is 1/4 of 16-bit
    }
}
