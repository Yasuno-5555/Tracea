use std::time::Instant;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::core::config::{PipelineConfig, GemmVariant};
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};

fn main() {
    let runtime = std::sync::Arc::new(RuntimeManager::new());
    let m = 2048usize; let n = 2048usize; let k = 2048usize;

    let a_buf = runtime.alloc(m * k * 2, DeviceBackend::Metal).unwrap();
    let b_buf = runtime.alloc(k * n * 2, DeviceBackend::Metal).unwrap();
    let c_buf = runtime.alloc(m * n * 4, DeviceBackend::Metal).unwrap();

    // Sweep over tile sizes and warp counts
    let configs = vec![
        // (m_tile, n_tile, k_tile, num_stages, warps, double_buf, label)
        ( 64,  64, 32, 2, 4, false, "64x64x32_Naive"),
        ( 64,  64, 32, 2, 4, true,  "64x64x32_DB4"),
        ( 64,  64, 32, 2, 8, true,  "64x64x32_DB8"),
        ( 64,  64, 32, 2, 6, true,  "64x64x32_DB6"),
        (128,  64, 32, 2, 8, true,  "128x64x32_DB8"),
        (128,  64, 32, 2, 12, true, "128x64x32_DB12"),
        (128, 128, 32, 2, 8, true,  "128x128x32_DB8"),
        (128, 128, 32, 2, 16, true, "128x128x32_DB16"),
        ( 64,  64, 16, 2, 8, true,  "64x64x16_DB8"),
        (128,  64, 64, 2, 8, true,  "128x64x64_DB8"),
        (128, 128, 64, 2, 16, true, "128x128x64_DB16"),
    ];

    println!("{:25} {:>10} {:>8} {:>8}", "Config", "Time(ms)", "GFLOPS", "%Peak");
    println!("{}", "-".repeat(55));

    let num_m_tiles = |mt: u32| (m + mt as usize - 1) / mt as usize;
    let num_n_tiles = |nt: u32| (n + nt as usize - 1) / nt as usize;

    for (mt, nt, kt, stages, warps, double_buf, label) in &configs {
        let mut config = PipelineConfig::new(*stages, *mt, *nt, *kt);
        config.double_buffer = *double_buf;
        config.gemm_variant = GemmVariant::Simd;
        config.force_num_warps = Some(*warps);

        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm {
                m: m as u32, n: n as u32, k: k as u32,
                batch: 1, epilogue: vec![],
            },
            precison: "fp16".to_string(),
            tiling: config,
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };

        let emitter = tracea::emitter::metal::MetalEmitter::detect();
        let source = match emitter.generate_from_ir(&ir) {
            Ok(s) => s,
            Err(e) => {
                println!("{:25} {:>10} {:>8} {:>8}", label, "ERR", "-", "-");
                continue;
            }
        };
        let kernel_id = match runtime.compile(&source, "unified_gemm_kernel", DeviceBackend::Metal) {
            Ok(id) => id,
            Err(e) => {
                println!("{:25} {:>10}", label, e);
                continue;
            }
        };

        let grid = (num_n_tiles(*nt) as u32, num_m_tiles(*mt) as u32, 1);
        let block = (warps * 32, 1, 1);
        let smem = 48 * 1024u32;

        // Warmup
        let wargs = vec![
            KernelArg::Buffer(a_buf), KernelArg::Buffer(b_buf), KernelArg::Buffer(c_buf),
            KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32),
        ];
        for _ in 0..3 {
            if runtime.launch(kernel_id, grid, block, smem, wargs.clone()).is_err() {
                break;
            }
        }

        // Timed runs with explicit synchronization
        let args = vec![
            KernelArg::Buffer(a_buf), KernelArg::Buffer(b_buf), KernelArg::Buffer(c_buf),
            KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32),
        ];

        let samples = 10;
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            runtime.synchronize();
            let start = Instant::now();
            if runtime.launch(kernel_id, grid, block, smem, args.clone()).is_err() {
                break;
            }
            runtime.synchronize();
            times.push(start.elapsed().as_secs_f64());
        }

        if times.is_empty() {
            println!("{:25} {:>10} {:>8} {:>8}", label, "LAUNCH_ERR", "-", "-");
            continue;
        }

        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let ops = 2.0 * (m * n * k) as f64;
        let gflops = (ops / avg / 1e9) as f32;
        let peak_pct = gflops / 26.0 * 100.0; // M1 FP32 peak ~2.6 TFLOPS → 2600 GFLOPS? No, 2.6TF is 2600GF
        // Actually M1 FP32 peak is 2.6 TFLOPS = 2600 GFLOPS

        println!("{:25} {:>8.2}ms {:>8.1} {:>7.1}%",
            label, avg * 1000.0, gflops, gflops / 2600.0 * 100.0);
    }
}
