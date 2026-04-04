use candle_core::Device;
use candle_core::quantized::gguf_file;
use std::path::Path;
use tracing::{info, warn};

pub struct ModelMetadata {
    pub architecture: String,
    pub num_layers: usize,
    pub embedding_dim: usize,
    pub context_window: usize,
    pub parameters_count: u64,
}

pub fn inspect_model(path: &Path) -> anyhow::Result<ModelMetadata> {
    if path.is_dir() {
        return load_safetensors_directory(path);
    }

    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or_default();
    
    match extension {
        "gguf" => load_gguf_metadata(path),
        "safetensors" => load_safetensors_metadata(path),
        _ => anyhow::bail!("Unsupported model format: .{}", extension),
    }
}

fn load_safetensors_directory(path: &Path) -> anyhow::Result<ModelMetadata> {
    info!("Inspecting Safetensors directory: {:?}", path);
    
    // Check for config.json to get architecture
    let config_path = path.join("config.json");
    let (architecture, num_layers, embedding_dim, context_window) = if config_path.exists() {
        let file = std::fs::File::open(config_path)?;
        let json: serde_json::Value = serde_json::from_reader(file)?;
        
        let arch = json["model_type"].as_str().unwrap_or("transformer").to_string();
        let layers = json["num_hidden_layers"].as_u64()
            .or_else(|| json["num_layers"].as_u64())
            .unwrap_or(0) as usize;
        let dim = json["hidden_size"].as_u64().unwrap_or(0) as usize;
        let context = json["max_position_embeddings"].as_u64()
            .or_else(|| json["max_seq_len"].as_u64())
            .unwrap_or(0) as usize;
        
        (arch, layers, dim, context)
    } else {
        ("unknown".to_string(), 0, 0, 0)
    };

    // Sum parameters from all .safetensors files
    let mut parameters_count = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            let file = std::fs::read(p)?;
            let tensors = candle_core::safetensors::load_buffer(&file, &Device::Cpu)?;
            parameters_count += tensors.values().map(|t| t.elem_count() as u64).sum::<u64>();
        }
    }

    validate_architecture(&architecture)?;

    Ok(ModelMetadata {
        architecture,
        num_layers,
        embedding_dim,
        context_window,
        parameters_count,
    })
}

fn load_gguf_metadata(path: &Path) -> anyhow::Result<ModelMetadata> {
    info!("Inspecting GGUF model: {:?}", path);
    let mut file = std::fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;

    // Extract metadata from GGUF
    let architecture = content.metadata.get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());

    let num_layers = content.metadata.get(&format!("{}.block_count", architecture))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

    let embedding_dim = content.metadata.get(&format!("{}.embedding_length", architecture))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

    let context_window = content.metadata.get(&format!("{}.context_length", architecture))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

    let parameters_count = content.tensor_infos.values()
        .map(|info| info.shape.elem_count() as u64)
        .sum();

    validate_architecture(&architecture)?;

    Ok(ModelMetadata {
        architecture,
        num_layers,
        embedding_dim,
        context_window,
        parameters_count,
    })
}

#[allow(dead_code)]
fn load_safetensors_metadata(path: &Path) -> anyhow::Result<ModelMetadata> {
    info!("Inspecting Safetensors model: {:?}", path);
    // Candle can read safetensors too
    let file = std::fs::read(path)?;
    let tensors = candle_core::safetensors::load_buffer(&file, &Device::Cpu)?;
    
    // For safetensors, metadata is often in a separate config.json, 
    // but some info can be inferred from tensor shapes.
    // This is a simplified version.
    
    Ok(ModelMetadata {
        architecture: "transformer".to_string(),
        num_layers: 0, // Needs config.json
        embedding_dim: 0,
        context_window: 0,
        parameters_count: tensors.values().map(|t| t.elem_count() as u64).sum(),
    })
}

fn validate_architecture(arch: &str) -> anyhow::Result<()> {
    match arch {
        "llama" | "mistral" | "phi2" | "gemma" | "gemma2" | "gemma4" | "nemotron" | "mamba" => {
            info!("Architecture {} is supported by TurboQuant pipeline.", arch);
            Ok(())
        }
        _ => {
            warn!("Architecture {} may not be fully compatible with TurboQuant optimization.", arch);
            Ok(()) // Proceed with warning
        }
    }
}
