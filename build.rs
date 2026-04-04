fn main() {
    // Detect architecture and enable specific features
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            println!("cargo:rustc-cfg=feature=\"avx2\"");
        }
        if is_x86_feature_detected!("avx512f") {
            println!("cargo:rustc-cfg=feature=\"avx512\"");
        }
    }

    // Optimization flags for production
    if std::env::var("PROFILE").unwrap_or_default() == "release" {
        println!("cargo:rustc-link-arg=-Wl,--strip-all");
    }

    // Re-run if build script changes
    println!("cargo:rerun-if-changed=build.rs");
}
