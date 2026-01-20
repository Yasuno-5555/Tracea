use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::optimizer::benchmark::{FlashAttentionProblem, FlashAttentionBenchmark, MicroBenchmark, BenchmarkResult};
use tracea::core::config::{PipelineConfig, SpecializedInstruction, LayoutPolicy, SwizzleMode};
use std::sync::Arc;
use std::io::Write;
use tracea::half::prelude::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Tracea Phase 6: FA2 Optimization Verification            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda))
        .expect("Failed to init runtime");
    runtime.doctor.diagnose_environment();

    // Problem: Standard FA2    // Problem Definition (S=1024 for saturation)
    let problem = FlashAttentionProblem::new(64, 8, 1024, 64, true); 
// Bidirectional
    let benchmark = FlashAttentionBenchmark::new(runtime.clone(), problem.clone());

    // Config: "Hero" Candidate (M=128, N=64, Stages=2, Warps=4)
    // NOTE: FA2 M=128 means processing 128 queries at a time.
    // N=64 means loading 64 Keys at a time.
    let mut config = PipelineConfig::new(2, 128, 64, 64); // K_Tile param is reused?
    // In FA2 Emitter: mt=m_tile, nt=n_tile (KT). 
    // Wait, check emitter: 
    // let mt = config.m_tile;
    // let nt = config.n_tile; // "This acts as KT"
    // So M=128 (Query Block), N=64 (Key/Value Block).
    
    config.instruction = SpecializedInstruction::CudaMMA;
    config.layout_policy = Some(LayoutPolicy::RowMajor);
    config.force_num_warps = Some(4); // 1 Producer + 3 Consumers? Or 4? GEMM uses Warps=8 for high perf?
    // User suggestion: Start simple. Stages=2.
    // I'll try Warps=4 first.

    println!("Testing Config: {:?}", config);
    
    // Validate
    if !benchmark.validate_config(&config) {
        println!("❌ Config Failed Validation! Check Logs (PTX error?)");
        return;
    }
    println!("✅ Config Validated. Measuring...");

    let res = benchmark.measure(&config);
    println!("----------------------------------------------------------------");
    println!("Result: {:.2} TFLOPS | Latency: {:.3} ms", res.tflops, res.latency_ms);
    println!("----------------------------------------------------------------");

    // Verification (Sanity Check)
    let size = problem.s * problem.d * problem.b * problem.h;
    let mut h_out = vec![0u16; size];
    runtime.copy_from_device(benchmark.d_o, &mut h_out).expect("Copy Output Failed");
    
    println!("Sanity Check (First 10 elements of Output):");
    for i in 0..10.min(size) {
        let val_f16 = half::f16::from_bits(h_out[i]);
        print!("{} ", val_f16);
    }
    println!();
    
    let center = size / 2;
    println!("Center Check (Around index {}):", center);
    for i in center..center+10.min(size) {
        let val_f16 = half::f16::from_bits(h_out[i]);
        print!("{} ", val_f16);
    }
    println!();
    
    // Warps=9 (1 Prod + 8 Cons for MT=128)
    config.force_num_warps = Some(9);
    println!("Testing Warps=9 Config (MT=128 requires 8 consumers): {:?}", config);
    if benchmark.validate_config(&config) {
        let res = benchmark.measure(&config);
        println!("Result: {:.2} TFLOPS | Latency: {:.3} ms", res.tflops, res.latency_ms);
    } else {
        println!("❌ Warps=9 Failed Validation.");
    }

    // Option A: Occupancy Optimization (Tile=64x64, Warps=5)
    // Goal: 2 blocks/SM, >13 TFLOPS
    {
        let mut config_a = PipelineConfig::new(2, 64, 64, 64);
        config_a.instruction = SpecializedInstruction::CudaMMA;
        config_a.layout_policy = Some(LayoutPolicy::RowMajor);
        // M=64 requires 4 Consumers (64/16=4). Total Warps = 1 + 4 = 5.
        config_a.force_num_warps = Some(5);
        
        println!("Testing Option A (Tile=64x64, Warps=5): {:?}", config_a);
        if benchmark.validate_config(&config_a) {
             let res = benchmark.measure(&config_a);
             println!("Result (Option A): {:.2} TFLOPS | Latency: {:.3} ms", res.tflops, res.latency_ms);
        } else {
             println!("❌ Option A Failed Validation.");
        }
    }

    // Option B: Aggressive Occupancy (Tile=64x32, Warps=5)
    // Goal: 2 blocks/SM guaranteed (Smem ~36KB), Target 15+ TFLOPS
    {
        let mut config_b = PipelineConfig::new(2, 64, 32, 64);
        config_b.instruction = SpecializedInstruction::CudaMMA;
        config_b.layout_policy = Some(LayoutPolicy::RowMajor);
        config_b.swizzle_mode = SwizzleMode::Xor8; // Enable 128B Swizzle
        // M=64 requires 4 Consumers. Total 5 Warps.
        config_b.force_num_warps = Some(5);
        
        println!("Testing Option B (Tile=64x32, Warps=5): {:?}", config_b);
        if benchmark.validate_config(&config_b) {
             let res = benchmark.measure(&config_b);
             println!("Result (Option B): {:.2} TFLOPS | Latency: {:.3} ms", res.tflops, res.latency_ms);
        } else {
             println!("❌ Option B Failed Validation.");
        }
    }
}
