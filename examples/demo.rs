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



    // Bayesian Optimization Demo
    println!("\nStarting Bayesian Optimization (10 iterations)...");
    let mut tuner = tracea::AutoTuner::new(gpu.clone());
    let benchmark = tracea::SimulatedBenchmark { m: 1024, n: 1024, k: 1024 };
    
    let best_config = tuner.optimize(&benchmark, 10, tracea::optimizer::OptimizationGoal::MaximizeTFLOPS);
    println!("Optimization Complete!");
    println!("Best Config Found: {:?}", best_config);

    // M3 demo: Add ReLU Fusion
    let mut best_config = best_config;
    best_config.epilogue.push(tracea::core::op::EpilogueOp::ReLU);
    println!("\nFused Config (Matmul + ReLU): {:?}", best_config);
}

