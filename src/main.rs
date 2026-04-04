mod cli;
mod config;
mod logger;
mod loader;
mod engine;
mod serialization;
mod validation;
mod benchmarking;
mod doctor;

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
        Commands::Calibrate { model, target } => {
            // Inspect model first
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
            info!("Benchmark: {:.2} tokens/s, Peak RAM: {:.2} GB, Latency: {:.2} ms, Compression: {:.2}x", 
                res.tokens_per_second, res.peak_ram_gb, res.latency_ms, res.compression_ratio);
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
