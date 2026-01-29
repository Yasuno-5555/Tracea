use tracea::core::tuning::{TunableKernel, SearchSpace};
use tracea::optimizer::{ProblemDescriptor, AutoTuner, OptimizationGoal, GPUInfo};
use tracea::core::backend::Device;
use tracea::runtime::manager::DeviceBackend;
use tracea::MicroBenchmark;
use std::sync::Arc;
use tracea::kernels::gemm::cuda_gemm::{CudaGemmAdapter, CudaGemmProblem};

fn main() {
    println!("--- Tracea GPU Template Dispatch Verification ---");

    let runtime = tracea::runtime::manager::RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Runtime init failed");
    
    // Problem size: 1024x1024x1024
    let m = 1024;
    let n = 1024;
    let k = 1024;
    let problem = CudaGemmProblem { m, n, k };

    let adapter = CudaGemmAdapter::new(runtime.clone(), problem);

    // 1. Test Configuration WITHOUT Template (should fall back to NVRTC)
    println!("\n[Test 1] Configuration WITHOUT template (64x64x32, 2 stages)...");
    let mut cfg_fallback = tracea::PipelineConfig::new(2, 64, 64, 32);
    cfg_fallback.force_num_warps = Some(4);
    
    if let Some(gflops) = adapter.benchmark(&cfg_fallback) {
        println!("  -> Fallback (NVRTC) GFLOPS: {:.2}", gflops);
    } else {
        println!("  -> [ERROR] Fallback benchmark failed!");
    }

    // 2. Test Configuration WITH Template (2 stages, Swizzle None, 128x128x32)
    println!("\n[Test 2] Configuration WITH template (128x128x32, 2 stages, Swizzle None)...");
    let mut cfg_template = tracea::PipelineConfig::new(2, 128, 128, 32);
    cfg_template.force_num_warps = Some(8);
    cfg_template.swizzle_mode = tracea::core::config::SwizzleMode::None;
    
    if let Some(gflops) = adapter.benchmark(&cfg_template) {
        println!("  -> Template (Pre-compiled) GFLOPS: {:.2}", gflops);
    } else {
        println!("  -> [ERROR] Template benchmark failed!");
    }

    // 3. Test Configuration WITH Template (3 stages, Swizzle Xor4, 128x128x32)
    println!("\n[Test 3] Configuration WITH template (128x128x32, 3 stages, Swizzle Xor4)...");
    let mut cfg_template2 = tracea::PipelineConfig::new(3, 128, 128, 32);
    cfg_template2.force_num_warps = Some(8);
    cfg_template2.swizzle_mode = tracea::core::config::SwizzleMode::Xor4;
    
    if let Some(gflops) = adapter.benchmark(&cfg_template2) {
        println!("  -> Template (Pre-compiled, Swizzle) GFLOPS: {:.2}", gflops);
    } else {
        println!("  -> [ERROR] Template benchmark failed!");
    }
}
