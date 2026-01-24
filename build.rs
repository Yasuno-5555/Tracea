fn main() {
    println!("cargo:rerun-if-changed=src/kernels/gpu/templates.cu");
    println!("cargo:rerun-if-changed=src/kernels/gpu/templates.h");

    // Check if CUDA is available
    let cuda_available = std::process::Command::new("nvcc")
        .arg("--version")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    let cpp_feature = std::env::var("CARGO_FEATURE_CPP").is_ok();
    if cuda_available && cpp_feature {
        println!("cargo:info=CUDA found, compiling templates...");
        
        // Detect GPU architecture (default to sm_86 for RTX 3070)
        let arch = std::env::var("TRACEA_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());

        cc::Build::new()
            .cuda(true)
            .file("src/kernels/gpu/templates.cu")
            .flag("-O3")
            .flag("--use_fast_math")
            .flag("-gencode")
            .flag(&format!("arch=compute_{},code={}", &arch[3..], arch))
            .include("src/kernels/gpu")
            .compile("tracea_kernels");

        println!("cargo:rustc-link-lib=static=tracea_kernels");
        println!("cargo:rustc-link-lib=cudart");
    } else {
        println!("cargo:warning=CUDA (nvcc) not found. Skipping template compilation.");
    }
}
