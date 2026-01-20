use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::optimizer::benchmark::{NVRTCBenchmark, MicroBenchmark, BenchmarkResult};
use tracea::optimizer::{AutoTuner, OptimizationGoal, GPUInfo};
use std::sync::Arc;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Tracea Phase 4: Meta Tuner Validation (GEMM)             ║");
    println!("║     Target: Validate robust scoring and policy derivation    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize Runtime
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda))
        .expect("Failed to initialize CUDA runtime");
    runtime.doctor.diagnose_environment();
    println!();

    let m = 2048;
    let n = 2048;
    let k = 2048;

    println!("🔬 Benchmarking GEMM [M={}, N={}, K={}]", m, n, k);

    // Create benchmark
    let benchmark = NVRTCBenchmark::new(runtime.clone(), m, n, k);
    
    // Use AutoTuner to find optimal config
    let mut tuner = AutoTuner::new(GPUInfo::rtx3070()).with_runtime(runtime.clone());
    
    // Note: AutoTuner::optimize uses default "Sniper" policy internally now (fixed in mod.rs)
    let config = tuner.optimize(
        &benchmark, 
        40, // 40 iterations to test stability
        OptimizationGoal::MaximizeTFLOPS,
        vec![] // No epilogue
    );
    
    println!("👉 Best Config Found: {:?}", config);

    // Measure final performance logic
    let result = benchmark.measure(&config);
    println!("------------------------------------------------------------");
    println!("🏆 Final Result:");
    println!("   TFLOPS:      {:.2}", result.tflops);
    println!("   Mean TFLOPS: {:.2}", result.mean_tflops);
    println!("   StdDev:      {:.4}", result.std_dev);
    println!("   Latency:     {:.3} ms", result.latency_ms);
    println!("------------------------------------------------------------");

    if result.mean_tflops > 20.0 {
        println!("✅ GEMM > 20 TFLOPS. Meta Tuner is working correctly.");
    } else {
        println!("⚠️ GEMM performance low. Meta Tuner or Kernel issue.");
    }
}
