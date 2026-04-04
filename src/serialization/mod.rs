use crate::engine::CalibrationManifest;
use std::fs::File;
use std::io::{Write, Read};
use std::path::Path;
use tracing::info;

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

#[allow(dead_code)]
pub fn verify_integrity(path: &Path) -> anyhow::Result<bool> {
    // Check for "TURBOQUANT_META" tag and valid manifest
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    if let Some(pos) = buffer.windows(15).position(|w| w == b"TURBOQUANT_META") {
        let meta_start = pos + 15;
        let manifest: Result<CalibrationManifest, _> = bincode::deserialize(&buffer[meta_start..]);
        return Ok(manifest.is_ok());
    }
    
    Ok(false)
}
