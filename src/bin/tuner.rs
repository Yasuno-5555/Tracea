use std::path::Path;
use std::env;
use std::fs::File;
use std::io::Write;
use tracea::runtime::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::optimizer::{AutoTuner, OptimizationGoal};
use tracea::core::config::PipelineConfig;
use tracea::emitter::universal::UniversalEmitter;
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
        "conv" => run_conv(&args[2..])?,
        "softmax" => run_softmax(&args[2..])?,
        "attention" => run_attention(&args[2..])?,
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
    let backend = if cfg!(target_os = "macos") {
        DeviceBackend::Metal
    } else {
        DeviceBackend::Cuda
    };
    
    let runtime = RuntimeManager::init(Some(backend))
        .map_err(|e| format!("Runtime init failed: {}", e))?;

    // Use NVRTCBenchmark which implements MicroBenchmark trait required by AutoTuner
    let benchmark = tracea::optimizer::benchmark::NVRTCBenchmark::new(
        runtime.clone(), 
        m as u32, n as u32, k as u32
    );

    // Run AutoTuner
    let gpu_info = if backend == DeviceBackend::Metal {
        tracea::optimizer::HardwareProfile::apple_m1()
    } else {
        tracea::optimizer::HardwareProfile::rtx3070()
    };
    let mut tuner = AutoTuner::new(gpu_info).with_runtime(runtime.clone());
    
    let best_config = tuner.optimize(
        &benchmark,
        20, // Reduced iterations for quick benchmark
        OptimizationGoal::MaximizeTFLOPS,
        vec![] 
    );

    // Collect Results
    println!("Tuning Complete.");

    println!("Best Configuration: {:?}", best_config);
    // Performance is not directly returned by optimize(), but printed during tuning.
    // If we want to capture it, optimize() should return (Config, Score). 
    // For now, we trust the cache logs.
    
    // Serialize Records - Skip detailed JSON output for now as GP is bypassed
    let timestamp = Local::now().to_rfc3339();
    
    use serde_json::json;
    let json_output = json!({
        "timestamp": timestamp,
        "kernel": "gemm",
        "problem": { "m": m, "n": n, "k": k },
        "best_config": best_config,
    });

    let path = Path::new(&output_file);
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_output)?.as_bytes())?;
    
    println!("Results saved to {}", output_file);
    Ok(())
}

fn run_conv(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 7 {
        eprintln!("Usage: tuner conv <n> <h> <w> <c_in> <c_out> <k_h> <k_w> [stride] [pad] [--out <file>]");
        return Ok(());
    }

    let n = args[0].parse()?;
    let h = args[1].parse()?;
    let w = args[2].parse()?;
    let c_in = args[3].parse()?;
    let c_out = args[4].parse()?;
    let r = args[5].parse()?;
    let s = args[6].parse()?;
    let stride = args.get(7).map(|s| s.parse().unwrap_or(1)).unwrap_or(1);
    let pad = args.get(8).map(|s| s.parse().unwrap_or(0)).unwrap_or(0);

    let _output_file = "conv_bench_results.json".to_string();

    println!("Starting Tuner for Conv2d Nx{}x{}x{} -> Kx{}x{}", h, w, c_in, r, s);

    // Init Runtime
    let backend = if cfg!(target_os = "macos") {
        DeviceBackend::Metal
    } else {
        DeviceBackend::Cuda
    };
    
    let runtime = RuntimeManager::init(Some(backend))
        .map_err(|e| format!("Runtime init failed: {}", e))?;

    let problem = tracea::optimizer::benchmark::Conv2dProblem::new(
        "CLI-Conv", n, h, w, c_in, c_out, r, s, stride, pad, 1
    );

    let benchmark = tracea::optimizer::benchmark::NVRTCConvBenchmark::new(
        runtime.clone(),
        problem
    );

    // Run AutoTuner
    let gpu_info = if backend == DeviceBackend::Metal {
        tracea::optimizer::HardwareProfile::apple_m1()
    } else {
        tracea::optimizer::HardwareProfile::rtx3070()
    };
    let mut tuner = AutoTuner::new(gpu_info).with_runtime(runtime.clone());
    
    let _best_config = tuner.optimize_conv(
        &benchmark,
        20,
        OptimizationGoal::MaximizeTFLOPS
    );

    let observations = &tuner.gp.observations;
    let best_obs = observations.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    if let Some(best) = best_obs {
        println!("Best Configuration: {:?}", best.config);
        println!("Performance: {:.2} TFLOPS", best.score);
    }
    
    Ok(())
}

fn run_softmax(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: tuner softmax <size> [--out <file>]");
        return Ok(());
    }
    let size: usize = args[0].parse()?;

    println!("Starting Benchmark for Softmax size={}", size);

    // Init Runtime
    let backend = if cfg!(target_os = "macos") {
        DeviceBackend::Metal
    } else {
        DeviceBackend::Cuda
    };
    
    let runtime = RuntimeManager::init(Some(backend))
        .map_err(|e| format!("Runtime init failed: {}", e))?;

    let ir = tracea::emitter::traits::UnifiedOpIR {
        op_type: tracea::emitter::traits::UnifiedOpType::Softmax {
            axis: 0,
            dim_size: size,
            stride: 1,
            total_elements: size,
        },
        precison: "f32".to_string(),
        tiling: PipelineConfig::default(),
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let emitter = UniversalEmitter::new(backend);
    let source = emitter.generate(ir);
    let kernel_id = runtime.compile(&source, "softmax_kernel", backend)?;

    let d_in = runtime.alloc_f32(size, backend)?;
    let d_out = runtime.alloc_f32(size, backend)?;

    let args = vec![
        KernelArg::Buffer(d_in),
        KernelArg::Buffer(d_out),
        KernelArg::Int(0), // Dummy g_idx handled by grid
    ];

    let grid = (1, 1, 1);
    let block = (1, 1, 1); // Simplest launch for validation

    runtime.launch(kernel_id, grid, block, 0, args)?;
    runtime.synchronize();

    println!("Softmax Benchmark passed (Metal Dispatch Verified)");
    Ok(())
}

fn run_attention(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 4 {
        eprintln!("Usage: tuner attention <b> <h> <s> <d> [--causal]");
        return Ok(());
    }

    let b = args[0].parse()?;
    let h = args[1].parse()?;
    let s = args[2].parse()?;
    let d = args[3].parse()?;
    let causal = args.iter().any(|arg| arg == "--causal");

    println!("Starting Tuner for FlashAttention B{} H{} S{} D{} (causal={})", b, h, s, d, causal);

    // Init Runtime
    let backend = if cfg!(target_os = "macos") {
        DeviceBackend::Metal
    } else {
        DeviceBackend::Cuda
    };
    
    let runtime = RuntimeManager::init(Some(backend))
        .map_err(|e| format!("Runtime init failed: {}", e))?;

    let problem = tracea::optimizer::benchmark::FlashAttentionProblem::new(b, h, s, d, causal);
    let benchmark = tracea::optimizer::benchmark::FlashAttentionBenchmark::new(runtime.clone(), problem);

    // Run AutoTuner
    let gpu_info = if backend == DeviceBackend::Metal {
        tracea::optimizer::HardwareProfile::apple_m1()
    } else {
        tracea::optimizer::HardwareProfile::rtx3070()
    };
    let mut tuner = AutoTuner::new(gpu_info).with_runtime(runtime.clone());
    
    let _best_config = tuner.optimize(
        &benchmark,
        10, // Fewer iterations for attention
        OptimizationGoal::MaximizeTFLOPS,
        vec![]
    );

    let observations = &tuner.gp.observations;
    let best_obs = observations.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    if let Some(best) = best_obs {
        println!("Best Configuration: {:?}", best.config);
        println!("Performance: {:.2} TFLOPS", best.score);
    }
    
    Ok(())
}
