use tracea::runtime::manager::RuntimeManager;
use tracea::optimizer::{AutoTuner, HardwareProfile, ProblemDescriptor, OptimizationGoal};
use tracea::optimizer::benchmark::NVRTCBenchmark;
use tracea::core::backend::Device;

fn main() {
    let runtime = RuntimeManager::new();
    let m = 2048; let n = 2048; let k = 2048;

    let mut hp = HardwareProfile::apple_m1();
    hp.shared_memory_per_block = 32000;

    let mut tuner = AutoTuner::new(hp.clone()).with_runtime(runtime.clone());

    let benchmark = NVRTCBenchmark::new(runtime.clone(), m as u32, n as u32, k as u32);
    let problem = ProblemDescriptor::new_gemm(m, n, k)
        .with_device(Device::Metal);

    println!("[AutoTuner] optimize_v2 for Metal GEMM {}x{}x{}...", m, n, k);
    let best = tuner.optimize_v2(&benchmark, &problem, 5, OptimizationGoal::MaximizeTFLOPS);

    println!("\n=== Best Config ===");
    println!("  Tile: {}x{}x{}", best.m_tile, best.n_tile, best.k_tile);
    println!("  Stages: {}", best.num_stages);
    println!("  Warps: {:?}", best.force_num_warps);
    println!("  Double Buffer: {}", best.double_buffer);
    println!("  Instruction: {:?}", best.instruction);
    println!("  Gemm Variant: {:?}", best.gemm_variant);
}
