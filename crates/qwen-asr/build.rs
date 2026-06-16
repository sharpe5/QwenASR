fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Check if blas feature is enabled via CARGO_FEATURE_BLAS env var
    if std::env::var("CARGO_FEATURE_BLAS").is_ok() {
        match target_os.as_str() {
            "macos" => {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            "linux" => {
                println!("cargo:rustc-link-lib=openblas");
            }
            _ => {
                // No BLAS available, will use fallback matmul
            }
        }
    }

    // Apple Neural Engine offload: compile the Objective-C CoreML shim and link
    // the CoreML + Foundation frameworks. macOS-only, behind the `mac-ane` feature.
    if std::env::var("CARGO_FEATURE_MAC_ANE").is_ok() && target_os == "macos" {
        println!("cargo:rerun-if-changed=ane/ane_shim.m");
        cc::Build::new()
            .file("ane/ane_shim.m")
            .flag("-fobjc-arc")
            .flag("-fmodules")
            .compile("qwen_ane_shim");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
