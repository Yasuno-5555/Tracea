use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::emitter::universal::UniversalEmitter;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use tracea::core::config::{PipelineConfig, SpecializedInstruction, SwizzleMode, QuantizationMode, LayoutPolicy};

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).unwrap();
    
    // Problem Size (Standard BERT-Base/GPT-2 minimal)
    let b = 1;
    let h = 12;
    let s = 4096;
    let d = 64;
    
    let total_elements_io = b * h * s * d;
    let total_bytes_io = total_elements_io * 2; // f16
    
    println!("Allocating FA2 Buffers (B={}, H={}, S={}, D={})...", b, h, s, d);
    let q_buf = runtime.alloc_u16(total_elements_io, DeviceBackend::Cuda).unwrap();
    let k_buf = runtime.alloc_u16(total_elements_io, DeviceBackend::Cuda).unwrap();
    let v_buf = runtime.alloc_u16(total_elements_io, DeviceBackend::Cuda).unwrap();
    let o_buf = runtime.alloc_u16(total_elements_io, DeviceBackend::Cuda).unwrap();

    // Init Data (Random normal-ish)
    // We can't easily init f16 from Rust without half crate, so we'll init as u16 with bit patterns
    // 0x3C00 is 1.0 in f16.
    let one_f16 = 0x3C00u16; 
    let init_data = vec![one_f16; total_elements_io];
    
    runtime.copy_to_device(q_buf, &init_data).unwrap();
    runtime.copy_to_device(k_buf, &init_data).unwrap();
    runtime.copy_to_device(v_buf, &init_data).unwrap();
    // O initialized to 0
    let zero_data = vec![0u16; total_elements_io];
    runtime.copy_to_device(o_buf, &zero_data).unwrap();

    println!("Generating FA2 Kernel...");
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::FusedAttention { 
            b: b as u32, s: s as u32, d: d as u32, h: h as u32, dh: d as u32, causal: true 
        },
        precison: "f16".to_string(),
        tiling: fa2_config(),
    };

    let source = UniversalEmitter::new(DeviceBackend::Cuda).generate(ir);
    // println!("{}", source); // Debug
    let kid = runtime.compile(&source, "flash_attention_v2_kernel", DeviceBackend::Cuda).unwrap();

    // Launch Params
    // Grid: (S/M_BLOCK, H, B)
    let m_block = 64; // from config
    let grid = ((s as u32 + m_block - 1) / m_block, h as u32, b as u32);
    let block = (128, 1, 1); // 4 warps
    // Smem: Dynamic (calculated by kernel/runtime usually, or passed here)
    // For now passing 0, trusting the kernel to set attribute or runtime to handle it? 
    // Wait, RuntimeManager uses `cuFuncSetAttribute` for MaxDynamicShared. 
    // We should pass enough bytes if we want to be safe, but the kernel `extern __shared__` uses dynamic.
    let smem_bytes = 100000; // 100KB safe bet

    println!("Launching FA2 (Warmup)...");
    let args = vec![
        KernelArg::Buffer(q_buf),
        KernelArg::Buffer(k_buf),
        KernelArg::Buffer(v_buf),
        KernelArg::Buffer(o_buf),
        KernelArg::Int(b as i32),
        KernelArg::Int(h as i32),
        KernelArg::Int(s as i32),
        KernelArg::Int(d as i32),
        KernelArg::Float(1.0 / (d as f32).sqrt()), 
    ];
    
    runtime.launch(kid, grid, block, smem_bytes, args.clone()).unwrap();
    runtime.synchronize();

    println!("Benchmarking (10 iters)...");
    let start = std::time::Instant::now();
    let iters = 10;
    for _ in 0..iters {
        runtime.launch(kid, grid, block, smem_bytes, args.clone()).unwrap();
    }
    runtime.synchronize();
    let duration = start.elapsed();
    
    println!("FA2 Time: {:?}", duration / iters);
    let ops = 4.0 * b as f64 * h as f64 * s as f64 * s as f64 * d as f64;
    let tflops = ops / (duration.as_secs_f64() / iters as f64) / 1e12;
    println!("TFLOPS: {:.2}", tflops);

    // Verify
    let mut o_host = vec![0u16; total_elements_io];
    runtime.copy_from_device(o_buf, &mut o_host).unwrap();
    
    // Check first 10 elements
    println!("Output[:10]: {:?}", &o_host[..10]);
    // Check sum
    let sum: f64 = o_host.iter().map(|&x| x as f64).sum();
    println!("Output Sum (u16 interpret): {}", sum);
    if sum == 0.0 {
        println!("❌ FA2 Verification Failed: Output is all zeros.");
    } else {
        println!("✅ FA2 Verification Passed (Non-zero output).");
    }
}

fn fa2_config() -> PipelineConfig {
    PipelineConfig {
        num_stages: 2, m_tile: 64, n_tile: 64, k_tile: 64, // Standard 64x64 tiles
        instruction: SpecializedInstruction::None,
        swizzle_mode: SwizzleMode::None,
        quantization: QuantizationMode::None,
        layout_policy: None,
        epilogue: vec![],
        force_num_warps: Some(4),
    }
}
