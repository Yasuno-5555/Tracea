fn main() {
    println!("cargo:rerun-if-changed=src/kernels/gpu/templates.cu");
    println!("cargo:rerun-if-changed=src/kernels/gpu/templates.h");

    // Check if CUDA is available
    let cuda_available = std::process::Command::new("nvcc")
        .arg("--version")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if cuda_available {
        println!("cargo:info=CUDA found, compiling templates...");
        
        // Detect GPU architecture (default to sm_86 for RTX 3070)
        let arch = std::env::var("TRACEA_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());

        // Path to cl.exe (Host compiler for nvcc on Windows)
        let host_compiler = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\bin\\Hostx64\\x64\\cl.exe";

        cc::Build::new()
            .cuda(true)
            .file("src/kernels/gpu/templates.cu")
            .flag("-O3")
            .flag("--use_fast_math")
            .flag("-gencode")
            .flag(&format!("arch=compute_{},code={}", &arch[3..], arch))
            .flag(&format!("-ccbin={}", host_compiler))
            .include("src/kernels/gpu")
            .compile("tracea_kernels");

        println!("cargo:rustc-link-lib=static=tracea_kernels");
        println!("cargo:rustc-link-lib=cudart");
    } else {
        println!("cargo:warning=CUDA (nvcc) not found. Skipping template compilation.");
    }
}
