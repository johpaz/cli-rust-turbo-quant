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

            // Inspect model
            let metadata = match loader::inspect_model(&model) {
                Ok(m) => m,
                Err(e) if args.dry_run => {
                    warn!("Model inspection failed: {}. Using mock metadata for dry-run.", e);
                    loader::ModelMetadata {
                        architecture: "mock-transformer".to_string(),
                        num_layers: 32,
                        embedding_dim: 4096,
                        context_window: 4096,
                        parameters_count: 7_000_000_000,
                    }
                }
                Err(e) => return Err(e),
            };

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
            info!("Generating text with TurboQuant KV cache at {:.1} bits/channel", bits);
            info!("Prompt: \"{}\"", prompt);
            info!("Max tokens: {}, Context: {}, Temp: {:.2}, Top-p: {:.2}",
                max_tokens, context, temperature, top_p);

            // Inspect model
            let metadata = match loader::inspect_model(&model) {
                Ok(m) => m,
                Err(e) if args.dry_run => {
                    warn!("Model inspection failed: {}. Using mock metadata for dry-run.", e);
                    loader::ModelMetadata {
                        architecture: "mock-transformer".to_string(),
                        num_layers: 32,
                        embedding_dim: 4096,
                        context_window: context,
                        parameters_count: 7_000_000_000,
                    }
                }
                Err(e) => return Err(e),
            };

            let num_heads = metadata.embedding_dim / 128;
            let head_dim = 128;
            let session = kv_cache::TurboQuantInference::new(
                metadata.num_layers,
                num_heads,
                head_dim,
                bits,
                context,
            );

            info!("KV Cache memory estimate: {:.4} GB (vs {:.4} GB for FP16)",
                session.total_kv_cache_memory_gb(),
                session.total_kv_cache_memory_gb() * 16.0 / bits);

            // Note: Full inference requires model loading + forward pass integration with Candle
            // This is a scaffold for the complete pipeline
            if !prompt.is_empty() {
                info!("Tokenizing prompt...");
                // Tokenization and generation loop would go here
                // The math module handles the quantized attention internally
                info!("Generation scaffolding complete. Full Candle integration needed for token generation.");
            }
        }
        Commands::Calibrate { model, target } => {
            warn!("Calibrate command is legacy. TurboQuant is data-oblivious and does not require calibration.");
            warn!("Use 'quantize' command instead with --bits flag.");

            // Still run for backward compatibility
            let metadata = match loader::inspect_model(&model) {
                Ok(m) => m,
                Err(e) if args.dry_run => {
                    warn!("Model inspection failed: {}. Using mock metadata for dry-run.", e);
                    loader::ModelMetadata {
                        architecture: "mock-llama".to_string(),
                        num_layers: 32,
                        embedding_dim: 4096,
                        context_window: 2048,
                        parameters_count: 7_000_000_000,
                    }
                }
                Err(e) => return Err(e),
            };
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
