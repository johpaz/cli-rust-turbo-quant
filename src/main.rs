mod cli;
mod config;
mod logger;
mod loader;
mod engine;
mod serialization;
mod validation;
mod benchmarking;
mod doctor;
mod math;
mod kv_cache;
mod generation;
mod inspect;

use clap::Parser;
use cli::{Cli, Commands};
use tracing::{info, debug, warn};
use config::AppConfig;

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    
    // Initialize configuration
    let config = AppConfig::default();
    
    // Initialize logger with verbosity from CLI
    let _guard = logger::init_logger(&args.verbosity.to_string())?;
    
    info!("🚀 TurboQuant Starting...");
    debug!("Configuration: {:?}", args);
    debug!("App Config: {:?}", config);

    if args.dry_run {
        warn!("⚠️ Running in DRY-RUN mode. No changes will be written to disk.");
    }

    match args.command {
        Commands::Init { name } => {
            info!("Initializing project: {}", name);
            std::fs::create_dir_all(&args.output)?;
            info!("Project {} initialized at {:?}", name, args.output);
        }
        Commands::Quantize { model, bits, output } => {
            let output_path = output.unwrap_or_else(|| args.output.join(format!("quantized_{:.1}bits.gguf", bits)));
            info!("Quantizing model {:?} to {} bits/channel", model, bits);
            info!("Output: {:?}", output_path);

            // Inspect model - no fallback to mock data
            let metadata = loader::inspect_model(&model)?;

            info!("Model: {} | Layers: {} | Embedding: {} | Context: {} | Params: {:.1}B",
                metadata.architecture, metadata.num_layers, metadata.embedding_dim,
                metadata.context_window, metadata.parameters_count as f64 / 1e9);

            // Create quantized inference session
            let num_heads = metadata.embedding_dim / 128; // typical head_dim = 128
            let head_dim = 128;
            let _session = kv_cache::TurboQuantInference::new(
                metadata.num_layers,
                num_heads,
                head_dim,
                bits,
                metadata.context_window,
            );

            info!("TurboQuant session created: {:.1} bits/channel (data-oblivious, no calibration needed)", bits);
            info!("Memory savings vs FP16: ~{:.1}x", 16.0 / bits);

            if !args.dry_run {
                std::fs::create_dir_all(&args.output)?;
                info!("Quantized model would be saved to: {:?}", output_path);
            }
        }
        Commands::Generate { model, prompt, max_tokens, bits, context, temperature, top_p } => {
            info!("🚀 Generating text with TurboQuant");
            info!("Model: {:?}", model);
            info!("Prompt: \"{}\"", prompt);
            info!("Config: bits={}, max_tokens={}, context={}, temp={:.2}, top_p={:.2}",
                bits, max_tokens, context, temperature, top_p);

            // Detect model type from GGUF
            let metadata = loader::inspect_model(&model)?;
            info!("Model metadata: arch={} layers={} hidden={} context={}",
                metadata.architecture, metadata.num_layers, metadata.embedding_dim, metadata.context_window);

            // Determine device
            let device = candle_core::Device::Cpu;
            info!("Using device: CPU");

            // Load the full GGUF model
            info!("Loading model weights from GGUF...");
            let mut gemma_model = generation::GemmaGGUF::from_gguf(
                model.to_str().unwrap(),
                device.clone(),
            )?;

            info!("Model loaded successfully!");
            info!("TurboQuant KV Cache memory estimate: {:.4} GB", {
                let kv = crate::kv_cache::TurboQuantInference::new(
                    gemma_model.config.num_hidden_layers,
                    gemma_model.config.num_attention_heads,
                    gemma_model.config.head_dim,
                    bits,
                    context,
                );
                kv.total_kv_cache_memory_gb()
            });

            // Load tokenizer
            let tokenizer = match generation::create_gemma_tokenizer(model.to_str().unwrap()) {
                Ok(t) => t,
                Err(e) => {
                    warn!("Tokenizer not found: {}", e);
                    info!("Attempting to use GGUF metadata for basic token mapping...");
                    anyhow::bail!(
                        "A tokenizer.json file is required for text generation.\n\
                         Download it from HuggingFace and place it next to the GGUF file.\n\
                         For Gemma 4, try: https://huggingface.co/google/gemma-4-E4B-it"
                    );
                }
            };

            // Encode prompt
            let prompt_tokens = tokenizer.encode(&prompt, true)?;
            info!("Prompt tokenized: {} tokens", prompt_tokens.len());

            // Create sampler
            let temp = if temperature > 0.0 { Some(temperature as f64) } else { None };
            let top_p = if top_p > 0.0 && top_p < 1.0 { Some(top_p as f64) } else { None };
            let mut sampler = generation::LogitsSampler::new(42, temp, top_p);

            // Generate
            let generated_tokens = gemma_model.generate_text(
                &prompt_tokens,
                max_tokens,
                &mut sampler,
                tokenizer.eos_token_id,
            )?;

            // Decode and print output
            let generated_text = tokenizer.decode(&generated_tokens, true)?;

            println!("\n{}", "═".repeat(60));
            println!("📝 GENERATED TEXT");
            println!("{}", "═".repeat(60));
            println!("{}", prompt);
            print!("{}", generated_text);
            println!("\n{}", "═".repeat(60));
            println!("📊 Stats: {} tokens generated", generated_tokens.len());
        }
        Commands::Calibrate { model, target } => {
            warn!("Calibrate command is legacy. TurboQuant is data-oblivious and does not require calibration.");
            warn!("Use 'quantize' command instead with --bits flag.");

            // Run calibration with real model inspection
            let metadata = loader::inspect_model(&model)?;
            info!("Model Info: Arch: {}, Layers: {}, Embedding: {}, Context: {}, Params: {}",
                metadata.architecture, metadata.num_layers, metadata.embedding_dim, metadata.context_window, metadata.parameters_count);

            let engine = engine::CalibrationEngine::new();
            let manifest = engine.run_calibration(&model, target, metadata.num_layers)?;

            if !args.dry_run {
                std::fs::create_dir_all(&args.output)?;
                let manifest_path = args.output.join("calibration_manifest.bin");
                serialization::save_manifest(&manifest, &manifest_path)?;
            }
        }
        Commands::Package { model, manifest } => {
            let file_stem = model.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
            let output_name = format!("johpaz_{}_turboquant.gguf", file_stem);
            let output_path = args.output.join(output_name);
            
            if !args.dry_run {
                serialization::package_model(&model, &manifest, &output_path)?;
            }
        }
        Commands::Validate { model } => {
            let report = validation::validate_model(&model)?;
            info!("Validation Report: Perplexity: {}, Integrity: {}, Delta Accuracy: {:.4}", 
                report.perplexity, report.integrity, report.accuracy_delta);
        }
        Commands::Benchmark { model, context } => {
            let res = benchmarking::run_benchmark(&model, context)?;
            let report = benchmarking::format_benchmark_report(&res);
            info!("\n{}", report);

            // Also show bits comparison
            info!("\n{}", benchmarking::benchmark_bits_comparison(context));
        }
        Commands::Doctor => {
            doctor::run_doctor()?;
        }
    }

    info!("✅ Operation completed successfully.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = config::AppConfig::default();
        assert_eq!(config.precision_target, 3.5);
    }

    #[test]
    fn test_verbosity_enum() {
        use cli::Verbosity;
        assert_eq!(Verbosity::Info.to_string(), "info");
    }
}
