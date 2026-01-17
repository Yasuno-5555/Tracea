use tracea::GemmOp;

fn main() {
    println!("Tracea GPU Framework - M1 Genesis Phase");
    
    let op = GemmOp::new(1024, 1024, 1024);
    println!("Logical Operation: {:?}", op);

    let gpu = tracea::GPUInfo::a100();
    let tuner = tracea::AutoTuner::new(gpu.clone());
    
    let config_ok = tracea::PipelineConfig::new(4, 128, 128, 16);
    let config_huge = tracea::PipelineConfig::new(10, 256, 256, 32);

    println!("Checking Config OK: {:?}", tuner.is_feasible(&config_ok).is_ok());
    println!("Checking Config Huge: {:?}", tuner.is_feasible(&config_huge).is_ok());

    // Swizzle Simulation
    let simulator = tracea::semantic::swizzle::BankConflictSimulator::new(32, 4);
    let threads = vec![(0, 0), (0, 1), (0, 2), (0, 3)]; // Simplified thread mapping
    let has_conflict_none = simulator.has_conflicts(&threads, 128, tracea::semantic::swizzle::SwizzleMode::None);
    println!("Conflicts (None): {}", has_conflict_none);

    // Bayesian Optimization Demo
    println!("\nStarting Bayesian Optimization (10 iterations)...");
    let mut tuner = tracea::AutoTuner::new(gpu.clone());
    let benchmark = tracea::optimizer::benchmark::SimulatedBenchmark { m: 1024, n: 1024, k: 1024 };
    
    let best_config = tuner.optimize(&benchmark, 10);
    println!("Optimization Complete!");
    println!("Best Config Found: {:?}", best_config);

    // M3 demo: Add ReLU Fusion
    let mut best_config = best_config;
    best_config.epilogue.push(tracea::semantic::fusion::EpilogueOp::ReLU);
    println!("\nFused Config (Matmul + ReLU): {:?}", best_config);

    // Generate kernel for best config (AMD)
    let hip_emitter = tracea::HIPEmitter::new();
    let best_kernel_hip = hip_emitter.generate_pipelined_gemm(best_config.clone());
    println!("\nGenerated Optimal AMD HIP Kernel Skeleton (Partial):");
    println!("{}", &best_kernel_hip[..300]);

    // Intel SYCL Demo
    let sycl_emitter = tracea::SYCLEmitter::new();
    let best_kernel_sycl = sycl_emitter.generate_pipelined_gemm(best_config);
    println!("\nGenerated Optimal Intel SYCL Kernel (Full with ReLU Fusion):");
    println!("{}", best_kernel_sycl);
}
