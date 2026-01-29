use std::path::Path;
use std::env;
use std::fs::File;
use std::io::Write;
use tracea::runtime::{RuntimeManager, DeviceBackend};
use tracea::optimizer::{AutoTuner, OptimizationGoal, HardwareProfile};
use chrono::Local;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: tuner <subcommand> [args]");
        eprintln!("Subcommands:");
        eprintln!("  gemm <m> <n> <k> [--out <file>]");
        return Ok(());
    }

    let subcommand = &args[1];
    
    match subcommand.as_str() {
        "gemm" => run_gemm(&args[2..])?,
        _ => eprintln!("Unknown subcommand: {}", subcommand),
    }

    Ok(())
}

fn run_gemm(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 3 {
        eprintln!("Usage: tuner gemm <m> <n> <k> [--out <file>]");
        return Ok(());
    }

    let m: usize = args[0].parse()?;
    let n: usize = args[1].parse()?;
    let k: usize = args[2].parse()?;

    let mut output_file = "bench_results.json".to_string();
    if args.len() >= 5 && args[3] == "--out" {
        output_file = args[4].clone();
    }

    println!("Starting Tuner for GEMM {}x{}x{}", m, n, k);

    // Init Runtime
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda))
        .map_err(|e| format!("Runtime init failed: {}", e))?;

    // Use NVRTCBenchmark which implements MicroBenchmark trait required by AutoTuner
    let benchmark = tracea::optimizer::benchmark::NVRTCBenchmark::new(
        runtime.clone(), 
        m as u32, n as u32, k as u32
    );

    // Run AutoTuner
    let gpu_info = tracea::optimizer::HardwareProfile::rtx3070(); 
    let mut tuner = AutoTuner::new(gpu_info).with_runtime(runtime.clone());
    
    let best_config = tuner.optimize(
        &benchmark,
        50, 
        OptimizationGoal::MaximizeTFLOPS,
        vec![] 
    );

    // Collect Results
    println!("Tuning Complete. Best Scores: {:.2} TFLOPS", 0.0); // need to extract score from observations

    let observations = &tuner.gp.observations;
    
    // Find best manually to print
    let best_obs = observations.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    if let Some(best) = best_obs {
        println!("Best Configuration: {:?}", best.config);
        println!("Performance: {:.2} TFLOPS", best.score);
    }
    
    // Serialize Records
    let timestamp = Local::now().to_rfc3339();
    
    // Prepare JSON structure
    use serde_json::json;
    let json_output = json!({
        "timestamp": timestamp,
        "kernel": "gemm",
        "problem": { "m": m, "n": n, "k": k },
        "best_performance": best_obs.map(|o| o.score).unwrap_or(0.0),
        "results": observations.iter().map(|obs| {
            json!({
                "config": obs.config,
                "tflops": obs.score,
                "latency_ms": 0.0 // latency not in Observation struct? Wait.
            })
        }).collect::<Vec<_>>()
    });

    let path = Path::new(&output_file);
    let mut file = File::create(&path)?;
    file.write_all(serde_json::to_string_pretty(&json_output)?.as_bytes())?;
    
    println!("Results saved to {}", output_file);
    Ok(())
}
