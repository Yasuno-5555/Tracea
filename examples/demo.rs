use tracea::{GemmOp, PipelineConfig};
use tracea::optimizer::{AutoTuner, HardwareProfile, ProblemDescriptor, OptimizationGoal};
use tracea::optimizer::benchmark::SimulatedBenchmark;
use tracea::core::op::EpilogueOp;

fn main() {
    println!("Tracea GPU Framework - M1 Genesis Phase");

    let op = GemmOp::new(1024, 1024, 1024);
    println!("Logical Operation: {:?}", op);

    let gpu = HardwareProfile::a100();
    let problem = ProblemDescriptor::new_gemm(1024, 1024, 1024);

    let config_ok = PipelineConfig::new(4, 128, 128, 16);
    let config_huge = PipelineConfig::new(10, 256, 256, 32);

    println!("Checking Config OK: {:?}", gpu.check_feasibility(&config_ok, &problem).is_ok());
    println!("Checking Config Huge: {:?}", gpu.check_feasibility(&config_huge, &problem).is_ok());



    // Bayesian Optimization Demo
    println!("\nStarting Bayesian Optimization (10 iterations)...");
    let mut tuner = AutoTuner::new(gpu.clone());
    let benchmark = SimulatedBenchmark { m: 1024, n: 1024, k: 1024 };

    let best_config = tuner.optimize(&benchmark, 10, OptimizationGoal::MaximizeTFLOPS, vec![]);
    println!("Optimization Complete!");
    println!("Best Config Found: {:?}", best_config);

    // M3 demo: Add ReLU Fusion
    let mut best_config = best_config;
    best_config.epilogue.push(EpilogueOp::ReLU);
    println!("\nFused Config (Matmul + ReLU): {:?}", best_config);
}
