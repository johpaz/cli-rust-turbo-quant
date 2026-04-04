use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "turbo-quant")]
#[command(version, about = "TurboQuant: High-performance quantization and calibration for LLMs", long_about = None)]
#[command(args_conflicts_with_subcommands = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Verbosity level
    #[arg(short, long, default_value = "info", global = true)]
    pub verbosity: Verbosity,

    /// Global output directory
    #[arg(short, long, value_name = "DIR", default_value = "./output", global = true)]
    pub output: PathBuf,

    /// Enable dry-run mode (no disk writes)
    #[arg(long, default_value_t = false, global = true)]
    pub dry_run: bool,

    /// Max RAM limit in GB
    #[arg(long, default_value_t = 8, global = true)]
    pub ram_limit: usize,

    /// Number of threads for parallel processing
    #[arg(long, default_value_t = 0, global = true)]
    pub threads: usize,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum Verbosity {
    Silent,
    Info,
    Debug,
    Trace,
}

impl ToString for Verbosity {
    fn to_string(&self) -> String {
        match self {
            Verbosity::Silent => "silent".to_string(),
            Verbosity::Info => "info".to_string(),
            Verbosity::Debug => "debug".to_string(),
            Verbosity::Trace => "trace".to_string(),
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Initialize a new TurboQuant project/workspace
    Init {
        /// Project name
        #[arg(short, long)]
        name: String,
    },
    /// Calibrate a model for quantization
    Calibrate {
        /// Path to the model (GGUF/Safetensors)
        #[arg(short, long, value_parser = validate_path_exists)]
        model: PathBuf,

        /// Quantization target (bits)
        #[arg(short, long, default_value_t = 3.5, value_parser = validate_bits_range)]
        target: f32,
    },
    /// Package a calibrated model
    Package {
        /// Path to the model
        #[arg(short, long, value_parser = validate_path_exists)]
        model: PathBuf,

        /// Path to the calibration manifest
        #[arg(short = 'f', long, value_parser = validate_path_exists)]
        manifest: PathBuf,
    },
    /// Validate a model's performance and integrity
    Validate {
        /// Path to the model
        #[arg(short, long, value_parser = validate_path_exists)]
        model: PathBuf,
    },
    /// Benchmark throughput and RAM usage
    Benchmark {
        /// Path to the model
        #[arg(short, long, value_parser = validate_path_exists)]
        model: PathBuf,

        /// Context length
        #[arg(short, long, default_value_t = 4096)]
        context: usize,
    },
    /// Analyze host environment and suggest optimizations
    Doctor,
}

fn validate_path_exists(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("File does not exist: {}", s))
    }
}

fn validate_bits_range(s: &str) -> Result<f32, String> {
    let val: f32 = s.parse().map_err(|_| format!("`{}` is not a valid number", s))?;
    if val >= 2.0 && val <= 8.0 {
        Ok(val)
    } else {
        Err(format!("Quantization target must be between 2.0 and 8.0 bits (given: {})", val))
    }
}
