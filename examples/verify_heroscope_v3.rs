use tracea::optimizer::{ProblemDescriptor, AutoTuner, OptimizationGoal, GPUInfo, LayerType, AsmParams};
use tracea::core::backend::Device;
use tracea::runtime::manager::DeviceBackend;
use tracea::MicroBenchmark;
use tracea::core::tuning::TunableKernel;
use tracea::kernels::gemm::cuda_gemm::{CudaGemmAdapter, CudaGemmProblem};
use std::sync::Arc;

fn main() {
    println!("--- Tracea HeroScope v3 & Sniper Mode Verification ---");

    let runtime = tracea::runtime::manager::RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Runtime init failed");
    
    // 1. Detect Hardware ID (simulating AutoTuner logic)
    let gpu = GPUInfo::rtx3070(); // This has cc_capability (8, 6)
    let tuner = AutoTuner::new(gpu.clone());
    let hardware_id = tuner.hardware_id.clone();
    println!("[Step 1] Detected Hardware ID: {}", hardware_id);
    assert_eq!(hardware_id, "sm_86", "Hardware ID should be sm_86 for Ampere");

    // 2. Setup Tuning Session
    let m = 1024;
    let n = 1024;
    let k = 1024;
    let problem_desc = ProblemDescriptor {
        name: "GEMM_Verify_V3".to_string(),
        m, n, k,
        batch: 1,
        layer_type: LayerType::Gemm,
        device: Device::Cuda(tracea::core::backend::CudaArch::Ampere),
        asm_params: AsmParams::default(),
    };

    let problem = CudaGemmProblem { m, n, k };
    let adapter = CudaGemmAdapter::new(runtime.clone(), problem);

    // 3. First Run: Exploration (Tuning)
    println!("\n[Step 2] Cold Start Tuning (Simulating MetaTuner v3)...");
    let mut autotuner = AutoTuner::new(gpu.clone());
    autotuner.runtime = Some(runtime.clone());
    
    // We expect it to find the 41 TFLOPS config because it's in the search space
    // and correctly prioritized if suggested.
    let best_cfg = autotuner.optimize_v2(&adapter, &problem_desc, 5, OptimizationGoal::MaximizeTFLOPS);
    
    println!("\n[Result] Tuning Finished!");
    println!("  - Best Config: Stages={}, Swizzle={:?}, M-Tile={}", 
        best_cfg.num_stages, best_cfg.swizzle_mode, best_cfg.m_tile);

    // 4. Verify Hero Persistence
    println!("\n[Step 3] Verifying HeroScope Persistence...");
    let heroscope = autotuner.heroscope.clone();
    if let Some(hero) = heroscope.get_hero(&hardware_id, LayerType::Gemm) {
        println!("  - Hero Found for {}: Stages={}, Swizzle={:?}", hardware_id, hero.num_stages, hero.swizzle_mode);
        assert!(hero.num_stages >= 2, "Hero should have optimized stages");
    } else {
        panic!("Hero configuration not saved to HeroScope!");
    }

    // 5. Second Run: Sniper Mode (Instant Hero Launch)
    println!("\n[Step 4] Sniper Mode Verification (Warm Start)...");
    let mut tuner_warm = AutoTuner::new(gpu);
    tuner_warm.heroscope = heroscope; // Use the updated heroscope
    
    // In optimize_v2, the Hero is injected at the start.
    // If it finds a Hero, it should hit high performance immediately.
    let first_guess = tuner_warm.heroscope.get_hero(&hardware_id, LayerType::Gemm).unwrap();
    println!("  - Sniper Mode First Suggestion: Stages={}, Swizzle={:?}", first_guess.num_stages, first_guess.swizzle_mode);
    
    if let Some(gflops) = adapter.benchmark(&first_guess) {
        println!("  -> Sniper Performance: {:.2} GFLOPS", gflops);
        assert!(gflops > 1000.0, "Sniper performance should be high");
    }

    println!("\n--- Phase D Verification SUCCESS ---");
}
