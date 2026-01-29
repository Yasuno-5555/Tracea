use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::optimizer::benchmark::{Conv2dProblem, ConvConfig, Conv2dBenchmark, NVRTCConvBenchmark};
use tracea::core::config::{PipelineConfig, MagicNumberStrategy, SpecializedInstruction};
use std::sync::Arc;

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Init failed");
    
    // B=64 Problem
    let problem = Conv2dProblem {
        name: "ResNet50-Conv3x3-64-B32".to_string(),
        batch: 32,
        h_in: 56, w_in: 56, c_in: 64, c_out: 64,
        kernel_h: 3, kernel_w: 3, stride: 1, pad: 1, 
    };
    
    println!("Testing Problem: {}", problem.name);
    let benchmark = NVRTCConvBenchmark::new(runtime.clone(), problem.clone());
    
    // Candidates to test
    let candidates = vec![
        // 1. The "Base Verified" (from B=32)
        // 64x64x16, 5 Warps (1P+4C)
        (64, 64, 16, 5, "Baseline (1P+4C)"),

        // 2. The "Heavy Hitter" (K=32)
        (128, 64, 32, 9, "Heavy Hitter (K32)"),
        // 2b. "Heavy Hitter Lite" (K=16)
        (128, 64, 16, 9, "Heavy Hitter (K16)"),

        // 3. The "Sniper" (K=32)
        (128, 128, 32, 17, "Sniper (K32)"),
        // 3b. "Sniper Lite" (K=16)
        (128, 128, 16, 17, "Sniper (K16)"),
    ];

    println!("┌───────────────────────┬───────────┬──────────────┐");
    println!("│ Configuration         │ TFLOPS    │ Latency (ms) │");
    println!("├───────────────────────┼───────────┼──────────────┤");

    for (m, n, k, warps, label) in candidates {
        let mut config = ConvConfig::default_for_problem(&problem);
        config.base.num_stages = 2; // Keep simple
        config.base.m_tile = m;
        config.base.n_tile = n;
        config.base.k_tile = k;
        config.base.instruction = SpecializedInstruction::CudaMMA;
        config.base.force_num_warps = Some(warps); 

        if !benchmark.validate_config(&config) {
             println!("│ {:<21} │   FAIL    │     N/A      │", label);
             continue;
        }

        let result = benchmark.measure(&config);
        println!("│ {:<21} │ {:9.2} │ {:12.3} │", label, result.tflops, result.latency_ms);
    }
    println!("└───────────────────────┴───────────┴──────────────┘");
}
