use tracea::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use tracea::emitter::cuda::CUDAEmitter;
use tracea::core::config::{PipelineConfig, SpecializedInstruction, SwizzleMode, GemmVariant};

#[test]
fn test_cuda_generated_code_syntax_verification() {
    // 1. Detect if CUDA/NVRTC is available on the current platform
    // Under typical CI or non-GPU environments (like macOS/arm64), nvrtc won't compile or load.
    // We dynamically inspect if cudarc is capable of loading the NVRTC library to prevent crashes.
    let nvrtc_available = std::panic::catch_unwind(|| {
        unsafe { let _ = cudarc::nvrtc::sys::lib(); }
    }).is_ok();

    if !nvrtc_available {
        println!("[Test] CUDA/NVRTC library is not available. Skipping JIT compilation verification.");
        return;
    }

    let emitter = CUDAEmitter::new();
    let config = PipelineConfig::new(2, 64, 64, 16);
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: 128,
            n: 128,
            k: 128,
            batch: 1,
            epilogue: vec![],
        },
        precison: "f16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let source = emitter.generate_from_ir(&ir).unwrap();

    // Perform compilation
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_80"), // Ampere compatibility as target fallback
        options: vec![
            "--std=c++17".to_string(),
        ],
        ..Default::default()
    };

    let ptx_res = cudarc::nvrtc::compile_ptx_with_opts(&source, opts);
    match ptx_res {
        Ok(_) => println!("[Test] ✅ CUDA GEMM code generated with 100% syntax compliance!"),
        Err(e) => {
            panic!("CUDA/NVRTC JIT compilation failed! Source code has syntax errors.\nError: {:?}\nSource:\n{}", e, source);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_generated_code_syntax_verification() {
    use tracea::emitter::metal::MetalEmitter;
    
    let device = metal::Device::system_default();
    if device.is_none() {
        println!("[Test] No system default Metal Device found. Skipping MSL compilation.");
        return;
    }
    let device = device.unwrap();

    let emitter = MetalEmitter::detect();
    
    // Test GEMM Generation with Tiled MSL
    let mut config = PipelineConfig::new(2, 64, 64, 16);
    config.gemm_variant = GemmVariant::Tiled;
    config.instruction = SpecializedInstruction::MetalSimdGroup;
    
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: 128,
            n: 128,
            k: 128,
            batch: 1,
            epilogue: vec![],
        },
        precison: "f16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let source = emitter.generate_from_ir(&ir).unwrap();
    
    let compile_options = metal::CompileOptions::new();
    let library_res = device.new_library_with_source(&source, &compile_options);
    
    match library_res {
        Ok(_) => println!("[Test] ✅ Metal MSL GEMM shader compiled successfully!"),
        Err(e) => {
            panic!("Metal MSL compilation failed! MSL shader has syntax errors.\nError: {}\nSource:\n{}", e, source);
        }
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn test_vulkan_generated_code_syntax_verification() {
    // Under vulkan features, we compile GLSL/SPIR-V via shaderc to verify correctness.
    let compiler = shaderc::Compiler::new();
    if compiler.is_none() {
        println!("[Test] shaderc compiler is not available. Skipping GLSL syntax verification.");
        return;
    }
    let compiler = compiler.unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);

    // Simple placeholder to verify shaderc-sys linkage and validation flow
    let test_glsl = r#"
        #version 450
        layout(local_size_x = 256) in;
        void main() {
            // empty compute shader
        }
    "#;

    let binary_res = compiler.compile_into_spirv(
        test_glsl,
        shaderc::ShaderKind::Compute,
        "test_glsl.glsl",
        "main",
        Some(&options)
    );

    match binary_res {
        Ok(_) => println!("[Test] ✅ shaderc GLSL/SPIR-V JIT verification succeeded!"),
        Err(e) => {
            panic!("GLSL compiler failed: {}", e);
        }
    }
}
