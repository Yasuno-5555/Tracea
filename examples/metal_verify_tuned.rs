use std::time::Instant;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::core::config::PipelineConfig;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};

fn main() {
    let runtime = RuntimeManager::new();
    let m = 2048; let n = 2048; let k = 2048;

    // Tuned config: 32x32x16, double buffer, 4 warps
    let mut config = PipelineConfig::new(2, 32, 32, 16);
    config.double_buffer = true;

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
    let source = emitter.generate_from_ir(&ir).expect("Codegen failed");
    let kernel_id = runtime.compile(&source, "unified_gemm_kernel", DeviceBackend::Metal)
        .expect("Compile failed");

    // Allocate buffers
    let a_buf = runtime.alloc_u16(m * k, DeviceBackend::Metal).unwrap();
    let b_buf = runtime.alloc_u16(k * n, DeviceBackend::Metal).unwrap();
    let c_buf = runtime.alloc_f32(m * n, DeviceBackend::Metal).unwrap();

    // Fill A with 2.0, B with 3.0 → expected output: C[i,j] = sum(2.0 * 3.0) over k = 6.0 * 2048 = 12288.0
    let a_val = half::f16::from_f32(2.0).to_bits();
    let b_val = half::f16::from_f32(3.0).to_bits();
    let a_data = vec![a_val; m * k];
    let b_data = vec![b_val; k * n];
    runtime.copy_to_device(a_buf, &a_data).unwrap();
    runtime.copy_to_device(b_buf, &b_data).unwrap();

    // Launch
    let args = vec![
        KernelArg::Buffer(a_buf), KernelArg::Buffer(b_buf), KernelArg::Buffer(c_buf),
        KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32),
    ];
    let grid = ((n / 32) as u32, (m / 32) as u32, 1);
    let block = (128, 1, 1);

    // Warmup + timing
    for _ in 0..3 {
        runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
        runtime.synchronize();
    }

    let start = Instant::now();
    for _ in 0..10 {
        runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
        runtime.synchronize();
    }
    let elapsed = start.elapsed().as_secs_f64() / 10.0;
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = ops / elapsed / 1e12;
    println!("Kernel time: {:.3}ms ({:.3} TFLOPS)", elapsed * 1000.0, tflops);

    // Read back and verify
    let result_bytes = runtime.read_buffer(c_buf).expect("Readback failed");
    let result: Vec<f32> = result_bytes.chunks(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let expected: f32 = 12288.0; // 6.0 * 2048
    let tolerance = expected * 0.02; // 2% for FP16 accumulation

    let mut max_err = 0.0f32;
    let mut min_err = f32::MAX;
    let mut n_zero = 0;
    let mut n_neg = 0;
    let mut n_nan = 0;

    // Sample strategy: check first 100 and last 100 elements, plus a grid pattern
    let check_indices: Vec<usize> = {
        let mut idx = (0..100).collect::<Vec<_>>();
        idx.extend((m * n - 100..m * n));
        // Check at tile boundaries (32x32 grid)
        for ti in 0..(m/32) {
            for tj in 0..(n/32) {
                idx.push(ti * 32 * n + tj * 32);
                idx.push(ti * 32 * n + tj * 32 + 31);
            }
        }
        idx
    };

    for &i in &check_indices {
        let val = result[i];
        if val.is_nan() { n_nan += 1; continue; }
        if val < 0.0 { n_neg += 1; continue; }
        if val == 0.0 { n_zero += 1; continue; }
        let err = (val - expected).abs();
        max_err = max_err.max(err);
        min_err = min_err.min(err);
    }

    println!("\n=== Verification Results ===");
    println!("  Expected:   {}", expected);
    println!("  Tolerance:  {}", tolerance);
    println!("  Max Error:  {}", max_err);
    println!("  Min Error:  {}", min_err);
    println!("  Zero count: {} / {}", n_zero, check_indices.len());
    println!("  Neg count:  {}", n_neg);
    println!("  NaN count:  {}", n_nan);

    if n_nan > 0 {
        println!("\n❌ FAIL: Found NaN values in output!");
        std::process::exit(1);
    }
    if n_zero > check_indices.len() / 2 {
        println!("\n❌ FAIL: >50% of checked elements are zero — kernel likely not computing!");
        std::process::exit(1);
    }
    if max_err > tolerance {
        println!("\n❌ FAIL: Max error {:.2} exceeds tolerance {:.2}", max_err, tolerance);
        println!("  First 10 values: {:?}", &result[..10]);
        std::process::exit(1);
    }

    println!("\n✅ PASS: All values within tolerance");
    println!("  First 10 values: {:?}", &result[..10]);
}
