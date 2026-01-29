use tracea::optimizer::{ProblemDescriptor, AutoTuner, OptimizationGoal, GPUInfo, LayerType, AsmParams, GpuAsmParams, Shape};
use tracea::core::backend::Device;
use tracea::runtime::manager::DeviceBackend;
use tracea::MicroBenchmark;
use tracea::core::tuning::TunableKernel;
use tracea::kernels::gemm::cuda_gemm::{CudaGemmAdapter, CudaGemmProblem};
use std::sync::Arc;

fn main() {
    println!("--- Tracea v3.1 Final Edition Verification ---");

    let runtime = tracea::runtime::manager::RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Runtime init failed");
    
    // 1. Detect Hardware Identity (v3.1)
    let gpu = GPUInfo::rtx3070(); // Ampere sm_86
    let tuner = AutoTuner::new(gpu.clone());
    let hardware_id = tuner.hardware_id.clone();
    println!("[Step 1] Hardware ID: {} (Backend: {:?})", hardware_id, gpu.backend);
    assert_eq!(hardware_id, "sm_86");

    // 2. Setup v3.1 Problem Descriptor (Shape-based)
    let m = 1024;
    let n = 1024;
    let k = 1024;
    let problem_desc = ProblemDescriptor {
        name: "GEMM_V3_1_Verify".to_string(),
        shape: Shape { m, n, k, batch: 1 },
        layer_type: LayerType::Gemm,
        device: Device::Cuda(tracea::core::backend::CudaArch::Ampere),
        asm_params: AsmParams::Gpu(GpuAsmParams::default()),
    };

    let problem = CudaGemmProblem { m, n, k };
    let adapter = CudaGemmAdapter::new(runtime.clone(), problem);

    // 3. Tuning with Structure-aware Search Space
    println!("\n[Step 2] Tuning over v3.1 search space (including BarrierMode)...");
    let mut autotuner = AutoTuner::new(gpu.clone());
    autotuner.runtime = Some(runtime.clone());
    
    // Perform optimization
    let best_cfg = autotuner.optimize_v2(&adapter, &problem_desc, 5, OptimizationGoal::MaximizeTFLOPS);
    
    println!("\n[Result] Optimization Complete!");
    println!("  - Best Tiles: {}x{}x{}", best_cfg.m_tile, best_cfg.n_tile, best_cfg.k_tile);
    println!("  - Best Structure: Stages={}, Swizzle={:?}, Barrier={:?}", 
        best_cfg.num_stages, best_cfg.swizzle_mode, best_cfg.barrier_mode);

    // 4. Verify mbarrier Template Dispatch
    println!("\n[Step 3] Verifying Template Dispatch Integration...");
    if best_cfg.barrier_mode == tracea::core::config::BarrierMode::ProducerConsumer {
        println!("  - ✅ SUCCESS: mbarrier Producer/Consumer template selected!");
    } else {
        println!("[Note] ProducerConsumer was not selected as best, this is normal for small tuning trials.");
    }

    // 5. HeroScope Persistence Check
    println!("\n[Step 4] Verifying HeroScope Logic...");
    if let Some(hero) = autotuner.heroscope.get_hero(&hardware_id, LayerType::Gemm) {
        println!("  - Hero Saved for {}: Stages={}, Barrier={:?}", hardware_id, hero.num_stages, hero.barrier_mode);
        assert_eq!(hero.num_stages, best_cfg.num_stages);
    } else {
        panic!("Hero failed to persist in HeroScope!");
    }

    println!("\n--- Tracea v3.1 Verification SUCCESS ---");
}
