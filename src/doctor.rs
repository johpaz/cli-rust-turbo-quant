use tracing::{info, warn};
use sysinfo::{System};

pub fn run_doctor() -> anyhow::Result<()> {
    info!("🔍 Analyzing system compatibility...");
    
    let sys = System::new_all();

    // Check CPU features
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 = is_x86_feature_detected!("avx512f");
        
        info!("CPU: AVX2 support: {}", if has_avx2 { "✅" } else { "❌" });
        info!("CPU: AVX512 support: {}", if has_avx512 { "✅" } else { "❌" });
        
        if !has_avx2 {
            warn!("Performance will be significantly limited without AVX2.");
        }
    }

    // Check RAM
    let total_memory = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let free_memory = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    
    info!("RAM: Total: {:.2} GB, Available: {:.2} GB", total_memory, free_memory);
    
    if total_memory < 8.0 {
        warn!("Limited RAM detected. Consider processing in small blocks.");
    }

    // Optimization suggestions
    info!("💡 Optimization suggestions:");
    info!("- Use --threads {} for optimal parallelization", num_cpus::get());
    info!("- Ensure models are stored on NVMe SSD for faster loading.");
    info!("- Current recommended RAM limit: {:.0} GB", total_memory * 0.75);

    Ok(())
}
