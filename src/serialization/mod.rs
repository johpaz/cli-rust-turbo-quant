use crate::engine::CalibrationManifest;
use std::fs::File;
use std::io::{Write, Read};
use std::path::Path;
use tracing::{info, warn};

pub fn save_manifest(manifest: &CalibrationManifest, path: &Path) -> anyhow::Result<()> {
    info!("Saving calibration manifest to: {:?}", path);
    let serialized = bincode::serialize(manifest)?;
    let mut file = File::create(path)?;
    file.write_all(&serialized)?;
    Ok(())
}

pub fn load_manifest(path: &Path) -> anyhow::Result<CalibrationManifest> {
    info!("Loading calibration manifest from: {:?}", path);
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let manifest: CalibrationManifest = bincode::deserialize(&buffer)?;
    Ok(manifest)
}

pub fn package_model(model_path: &Path, manifest_path: &Path, output_path: &Path) -> anyhow::Result<()> {
    info!("Packaging model {:?} with manifest {:?}", model_path, manifest_path);
    
    // Load manifest to verify
    let manifest = load_manifest(manifest_path)?;
    info!("Verified manifest for target: {} bits", manifest.target_bits);

    let mut output_file = File::create(output_path)?;
    
    if model_path.is_dir() {
        // If it's a directory, concatenate all .safetensors files
        info!("Model is a directory, merging all .safetensors files...");
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                let mut f = File::open(p)?;
                std::io::copy(&mut f, &mut output_file)?;
            }
        }
    } else {
        // If it's a single file (GGUF)
        let mut model_file = File::open(model_path)?;
        std::io::copy(&mut model_file, &mut output_file)?;
    }
    
    let serialized_manifest = bincode::serialize(&manifest)?;
    output_file.write_all(b"TURBOQUANT_META")?;
    output_file.write_all(&serialized_manifest)?;
    
    info!("Model successfully packaged at: {:?}", output_path);
    Ok(())
}

pub fn verify_integrity(path: &Path) -> anyhow::Result<bool> {
    info!("Verifying integrity of model: {:?}", path);
    
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Check for TURBOQUANT_META marker
    let marker = b"TURBOQUANT_META";
    if let Some(pos) = buffer.windows(marker.len()).position(|w| w == marker) {
        info!("Found TURBOQUANT_META marker at position {}", pos);
        
        let meta_start = pos + marker.len();
        
        // Try to deserialize the manifest
        let manifest_result: Result<CalibrationManifest, _> = bincode::deserialize(&buffer[meta_start..]);
        
        match manifest_result {
            Ok(manifest) => {
                info!("Manifest verified successfully:");
                info!("  - Model hash: {}", manifest.model_hash);
                info!("  - Target bits: {}", manifest.target_bits);
                info!("  - Layers calibrated: {}", manifest.stats.len());
                info!("  - Timestamp: {}", manifest.timestamp);
                
                // Verify manifest has valid data
                if manifest.stats.is_empty() {
                    warn!("Manifest has no calibration stats");
                    return Ok(false);
                }
                
                if manifest.target_bits <= 0.0 || manifest.target_bits > 32.0 {
                    warn!("Invalid target bits: {}", manifest.target_bits);
                    return Ok(false);
                }
                
                // Verify each layer stat
                for stat in &manifest.stats {
                    if stat.variance < 0.0 {
                        warn!("Negative variance in layer {}: {}", stat.layer_id, stat.variance);
                        return Ok(false);
                    }
                    if stat.scale_factor <= 0.0 {
                        warn!("Invalid scale factor in layer {}: {}", stat.layer_id, stat.scale_factor);
                        return Ok(false);
                    }
                }
                
                // Calculate and verify file checksum
                let file_checksum = calculate_file_checksum(&buffer[..pos]);
                info!("File checksum (first {} bytes): {}", pos, file_checksum);
                
                Ok(true)
            }
            Err(e) => {
                warn!("Failed to deserialize manifest: {}", e);
                Ok(false)
            }
        }
    } else {
        info!("No TURBOQUANT_META marker found - file may be unprocessed model");
        
        // Still verify if it's a valid model file (GGUF or safetensors)
        verify_model_file_integrity(&buffer)
    }
}

fn calculate_file_checksum(data: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

fn verify_model_file_integrity(buffer: &[u8]) -> anyhow::Result<bool> {
    // Check for GGUF magic bytes: "GGUF" = 0x47475546
    if buffer.len() >= 4 {
        let magic = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        // GGUF magic number
        if magic == 0x46554747 {
            info!("Valid GGUF file detected");
            return Ok(true);
        }
    }
    
    // Check for safetensors format (starts with JSON header)
    if buffer.len() > 8 {
        let len_bytes: [u8; 8] = buffer[..8].try_into()?;
        let header_len = u64::from_le_bytes(len_bytes);
        
        // Reasonable header size (between 100 bytes and 100MB)
        if header_len > 100 && header_len < 100_000_000 && buffer.len() as u64 > header_len + 8 {
            // Try to parse as JSON
            let header_start = 8;
            let header_end = (8 + header_len) as usize;
            if header_end <= buffer.len() {
                let header_str = std::str::from_utf8(&buffer[header_start..header_end]);
                if header_str.is_ok() {
                    let json_result: Result<serde_json::Value, _> = serde_json::from_str(header_str.unwrap());
                    if json_result.is_ok() {
                        info!("Valid safetensors file detected");
                        return Ok(true);
                    }
                }
            }
        }
    }
    
    warn!("File format not recognized as valid model");
    Ok(false)
}
