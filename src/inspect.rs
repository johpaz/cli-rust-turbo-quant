use std::fs::File;
use std::io::BufReader;
use candle_core::quantized::gguf_file;

pub fn inspect_gguf(path: &str) -> anyhow::Result<()> {
    let mut file = BufReader::new(File::open(path)?);
    let content = gguf_file::Content::read(&mut file)?;

    println!("=== METADATA ===");
    for (key, val) in &content.metadata {
        println!("  {} = {:?}", key, val);
    }

    println!("\n=== TENSORS (all) ===");
    for (name, info) in &content.tensor_infos {
        println!("  {} | shape: {:?} | dtype: {:?}", name, info.shape, info.ggml_dtype);
    }
    println!("\nTotal tensors: {}", content.tensor_infos.len());
    Ok(())
}
