use std::time::Instant;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::core::config::PipelineConfig;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};

fn main() {
    let runtime = RuntimeManager::new();
    let m = 2048u32; let n = 2048u32; let k = 2048u32;
    let mt = 32; let nt = 32; let kt = 16;
    let nw = 4;

    let sz_a = (m * k) as usize;
    let sz_b = (k * n) as usize;
    let sz_c = (m * n) as usize;

    let mut config = PipelineConfig::new(2, mt, nt, kt);
    config.double_buffer = true;
    config.force_num_warps = Some(nw);

    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm { m, n, k, batch: 1, epilogue: vec![] },
        precison: "fp16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let emitter = tracea::emitter::metal::MetalEmitter::detect();
    let source = emitter.generate_from_ir(&ir).expect("Codegen failed");
    let kernel_id = runtime.compile(&source, "unified_gemm_kernel", DeviceBackend::Metal).expect("Compile failed");

    let a_buf = runtime.alloc_u16(sz_a, DeviceBackend::Metal).unwrap();
    let b_buf = runtime.alloc_u16(sz_b, DeviceBackend::Metal).unwrap();
    let c_buf = runtime.alloc_f32(sz_c, DeviceBackend::Metal).unwrap();
    let ones = vec![0x3C00u16; sz_a];
    runtime.copy_to_device(a_buf, &ones).unwrap();
    runtime.copy_to_device(b_buf, &ones).unwrap();

    let args = vec![
        KernelArg::Buffer(a_buf), KernelArg::Buffer(b_buf), KernelArg::Buffer(c_buf),
        KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32),
    ];
    let grid = (n / mt, m / mt, 1);
    let block = (nw * 32, 1, 1);

    for _ in 0..3 {
        runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
        runtime.synchronize();
    }

    // Method A: launch + sync per iteration (manual test style)
    let start = Instant::now();
    for _ in 0..10 {
        runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
        runtime.synchronize();
    }
    let a = start.elapsed().as_secs_f64() / 10.0;
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    println!("Method A (sync each): {:.3}ms = {:.3} TFLOPS", a*1000.0, ops / a / 1e12);

    // Method B: batch launch, single sync (NVRTCBenchmark style)
    let start = Instant::now();
    for _ in 0..10 {
        runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
    }
    runtime.synchronize();
    let b = start.elapsed().as_secs_f64() / 10.0;
    println!("Method B (batch sync): {:.3}ms = {:.3} TFLOPS", b*1000.0, ops / b / 1e12);
}
