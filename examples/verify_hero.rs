use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::optimizer::benchmark::{Conv2dProblem, ConvConfig, Conv2dBenchmark, NVRTCConvBenchmark};
use tracea::core::config::{PipelineConfig, MagicNumberStrategy, SpecializedInstruction};
use std::io::Write;
use std::sync::Arc;

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Init failed");
    
    // B=32 Problem (ResNet50 Conv3x3)
    let batches = vec![64];
    
    println!("| Batch | TFLOPS | Latency (ms) | Config |");
    println!("|-------|--------|--------------|--------|");

    for &b in &batches {
        let problem = Conv2dProblem {
            name: format!("ResNet50-Conv3x3-64-B{}", b),
            batch: b,
            h_in: 56, w_in: 56, c_in: 64, c_out: 64,
            kernel_h: 3, kernel_w: 3, stride: 1, pad: 1, 
        };
        
        let benchmark = NVRTCConvBenchmark::new(runtime.clone(), problem.clone());
        
        // Define Candidates
        let mut candidates = Vec::new();

        // 1. Baseline (M=64, Warps=5)
        let mut c1 = ConvConfig::default_for_problem(&problem);
        c1.base.m_tile = 64; c1.base.n_tile = 64; c1.base.k_tile = 16;
        c1.base.instruction = SpecializedInstruction::CudaMMA;
        c1.base.force_num_warps = Some(5);
        candidates.push(("M64-W5", c1));

        // 2. Target A (M=128, Warps=9, K=16)
        let mut c2 = ConvConfig::default_for_problem(&problem);
        c2.base.m_tile = 128; c2.base.n_tile = 64; c2.base.k_tile = 16;
        c2.base.instruction = SpecializedInstruction::CudaMMA;
        c2.base.force_num_warps = Some(9);
        candidates.push(("M128-W9-K16", c2));

        // 3. Target B (M=128, Warps=5, K=16)
        let mut c3 = ConvConfig::default_for_problem(&problem);
        c3.base.m_tile = 128; c3.base.n_tile = 64; c3.base.k_tile = 16;
        c3.base.instruction = SpecializedInstruction::CudaMMA;
        c3.base.force_num_warps = Some(5);
        candidates.push(("M128-W5-K16", c3));

        // 4. Target C (M=128, K=32, Warps=9)
        let mut c4 = ConvConfig::default_for_problem(&problem);
        c4.base.m_tile = 128; c4.base.n_tile = 64; c4.base.k_tile = 32;
        c4.base.instruction = SpecializedInstruction::CudaMMA;
        c4.base.force_num_warps = Some(9);
        candidates.push(("M128-W9-K32", c4));
        
        // 5. Target D (M=128, K=64, Warps=9) - Failed (3.4T)
        // let mut c5 = ConvConfig::default_for_problem(&problem);...

        let mut copt1 = ConvConfig::default_for_problem(&problem);
        copt1.base.m_tile = 128; copt1.base.n_tile = 64; copt1.base.k_tile = 32;
        copt1.base.num_stages = 3;
        copt1.base.instruction = SpecializedInstruction::CudaMMA;
        copt1.base.force_num_warps = Some(9);
        candidates.push(("M128-K32-S3", copt1));

        let mut copt2 = ConvConfig::default_for_problem(&problem);
        copt2.base.m_tile = 128; copt2.base.n_tile = 64; copt2.base.k_tile = 32;
        copt2.base.num_stages = 4;
        copt2.base.instruction = SpecializedInstruction::CudaMMA;
        copt2.base.force_num_warps = Some(9);
        candidates.push(("M128-K32-S4", copt2));
        
        // Rescue K64 with Stages
        let mut copt3 = ConvConfig::default_for_problem(&problem);
        copt3.base.m_tile = 128; copt3.base.n_tile = 64; copt3.base.k_tile = 64;
        copt3.base.num_stages = 3;
        copt3.base.instruction = SpecializedInstruction::CudaMMA;
        copt3.base.force_num_warps = Some(9);
        candidates.push(("M128-K64-S3", copt3));
        
        for (label, config) in candidates {
            if !benchmark.validate_config(&config) {
                 println!("| {:<5} | FAIL   | N/A          | {:<12} |", b, label);
                 std::io::stdout().flush().unwrap();
                 continue;
            }

            let result = benchmark.measure(&config);
            println!("| {:<5} | {:<6.2} | {:<12.4} | {:<12} |", 
                b, result.tflops, result.latency_ms, label);
            std::io::stdout().flush().unwrap();
        }
    }
}
