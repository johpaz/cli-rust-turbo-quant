use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub default_output_dir: PathBuf,
    pub precision_target: f32,
    pub context_limit: usize,
    pub hardware_flags: HardwareFlags,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HardwareFlags {
    pub use_avx2: bool,
    pub use_avx512: bool,
    pub alignment: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            default_output_dir: PathBuf::from("./output"),
            precision_target: 3.5,
            context_limit: 4096,
            hardware_flags: HardwareFlags {
                use_avx2: true,
                use_avx512: false,
                alignment: 64,
            },
        }
    }
}
