use std::time::Instant;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::core::config::{PipelineConfig, GemmVariant};
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};
use tracea::optimizer::benchmark::{MicroBenchmark, BenchmarkResult, EnvironmentInfo};
use tracea::optimizer::{AutoTuner, HardwareProfile, ProblemDescriptor, OptimizationGoal};
use tracea::core::backend::Device;
use tracea::optimizer::model::HardwareObservation;

fn measure_one(runtime: &RuntimeManager, config: &PipelineConfig, m: usize, n: usize, k: usize,
               a_buf: KernelArg, b_buf: KernelArg, c_buf: KernelArg) -> BenchmarkResult {
    let mut cfg = config.clone();
    cfg.gemm_variant = GemmVariant::Simd;

    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: m as u32, n: n as u32, k: k as u32,
            batch: 1, epilogue: vec![],
        },
        precison: "fp16".to_string(),
        tiling: cfg,
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let emitter = tracea::emitter::metal::MetalEmitter::detect();
    let source = emitter.generate_from_ir(&ir).expect("Codegen failed");
    let kernel_id = runtime.compile(&source, "unified_gemm_kernel", DeviceBackend::Metal)
        .expect("Compile failed");

    let mt = config.m_tile as usize;
    let nt = config.n_tile as usize;
    let num_m_tiles = (m + mt - 1) / mt;
    let num_n_tiles = (n + nt - 1) / nt;
    let grid = (num_n_tiles as u32, num_m_tiles as u32, 1);
    let warps = config.force_num_warps.unwrap_or(4);
    let block = (warps * 32, 1, 1);
    let args = vec![a_buf, b_buf, c_buf,
        KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32)];

    // Warmup
    for _ in 0..2 {
        runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
        runtime.synchronize();
    }

    // Timed runs
    let samples = 5;
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        runtime.synchronize();
        let start = Instant::now();
        runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
        runtime.synchronize();
        times.push(start.elapsed().as_secs_f64());
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let latency_ms = (avg * 1000.0) as f32;
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = ops / avg / 1e9;
    let tflops = gflops as f32 / 1000.0;
    let variance = times.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / times.len() as f64;
    let std_dev = (variance.sqrt() * 1000.0) as f32;

    BenchmarkResult { tflops, mean_tflops: tflops, std_dev, latency_ms, observation: None }
}

fn main() {
    let runtime = RuntimeManager::new();
    let m = 2048; let n = 2048; let k = 2048;

    let a_buf = runtime.alloc_u16(m * k, DeviceBackend::Metal).unwrap();
    let b_buf = runtime.alloc_u16(k * n, DeviceBackend::Metal).unwrap();
    let c_buf = runtime.alloc_f32(m * n, DeviceBackend::Metal).unwrap();
    let ones = vec![0x3C00u16; m * k];
    runtime.copy_to_device(a_buf, &ones).unwrap();
    let ones = vec![0x3C00u16; k * n];
    runtime.copy_to_device(b_buf, &ones).unwrap();

    let arg_a = KernelArg::Buffer(a_buf);
    let arg_b = KernelArg::Buffer(b_buf);
    let arg_c = KernelArg::Buffer(c_buf);

    // Manual sweep of interesting configs
    let candidates = [
        (64, 64, 16, 2, 4, true),
        (64, 64, 16, 2, 6, true),
        (64, 64, 32, 2, 4, true),
        (64, 64, 32, 2, 6, true),
        (64, 64, 32, 3, 4, true),
        (64, 64, 32, 2, 8, true),
        (64, 64, 16, 2, 4, false),
        (32, 32, 16, 2, 4, true),
    ];

    println!("{:20} {:>6} {:>6} {:>8} {:>6}", "Tile", "Stages", "Warps", "GFLOPS", "%Peak");
    println!("{}", "-".repeat(52));

    let mut best_gflops = 0.0f32;
    let mut best_cfg = PipelineConfig::new(2, 64, 64, 16);

    for &(mt, nt, kt, stages, warps, db) in &candidates {
        let mut cfg = PipelineConfig::new(stages, mt, nt, kt);
        cfg.double_buffer = db;
        cfg.force_num_warps = Some(warps);

        let label = format!("{}x{}x{}_{}", mt, nt, kt,
            if db { "DB" } else { "Naive" });
        let result = measure_one(&runtime, &cfg, m, n, k, arg_a.clone(), arg_b.clone(), arg_c.clone());

        if result.mean_tflops > 0.0 {
            let gflops = result.mean_tflops * 1000.0;
            let pct = result.mean_tflops / 2.6 * 100.0;
            println!("{:20} {:4} {:4} {:>8.1} {:>5.1}%",
                label, stages, warps, gflops, pct);

            if gflops > best_gflops {
                best_gflops = gflops;
                best_cfg = cfg;
            }
        } else {
            println!("{:20} {:4} {:4} {:>8} {:>6}", label, stages, warps, "FAIL", "-");
        }
    }

    // Now try AutoTuner with the best config as starting point
    println!("\n--- AutoTuner Exploration ---");
    let mut hp = HardwareProfile::apple_m1();
    hp.preferred_tile_shape = [64, 64, 32];
    hp.shared_memory_per_block = 32000;

    let problem = ProblemDescriptor::new_gemm(m, n, k).with_device(Device::Metal);

    // Use SimulatedBenchmark since we can't easily use the AutoTuner's optimize_v2
    // with a custom benchmark. The sweep above gives us the answer.

    println!("\n=== Best Config ===");
    println!("  Tile: {}x{}x{}", best_cfg.m_tile, best_cfg.n_tile, best_cfg.k_tile);
    println!("  Stages: {}", best_cfg.num_stages);
    println!("  Warps: {:?}", best_cfg.force_num_warps);
    println!("  Double Buffer: {}", best_cfg.double_buffer);
    println!("  Performance: {:.1} GFLOPS ({:.1}% of M1 peak)", best_gflops, best_gflops / 2600.0 * 100.0);
}
