use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::optimizer::benchmark::{NVRTCBenchmark, MicroBenchmark, BenchmarkResult};
use tracea::core::config::{PipelineConfig, SpecializedInstruction, MagicNumberStrategy};
use std::sync::Arc;
use half::f16;
use rand::Rng;

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).unwrap();
    
    // Choose dimensions
    let m = 2048; // Large enough to amortize setup
    let n = 2048;
    let k = 2048;
    
    println!("Benchmarking GEMM {}x{}x{}", m, n, k);
    let benchmark = NVRTCBenchmark::new(runtime.clone(), m, n, k);
    
    // Manual "Hero" Config for GEMM
    // 128x128x32 tile is standard for Tensor Cores
    let mut config = PipelineConfig::new(2, 128, 128, 32);
    config.instruction = SpecializedInstruction::CudaMMA;
    config.force_num_warps = Some(8); // 8 warps = 256 threads (standard for 128x128)
    
    println!("Testing Config: {:?}", config);
    
    if !benchmark.validate_config(&config) {
        println!("❌ Config Failed Validation!");
        // return;
    }
    
    // Measure
    let result = benchmark.measure(&config);
    println!("Performance: {:.2} TFLOPS", result.tflops);
    println!("Latency: {:.3} ms", result.latency_ms);
}
