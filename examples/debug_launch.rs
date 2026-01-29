use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::core::config::{PipelineConfig, SpecializedInstruction, SwizzleMode, QuantizationMode};
use tracea::emitter::fa2::FlashAttentionEmitter;
use std::sync::Arc;
use std::fs::File;
use std::io::Write;

fn log(msg: &str) {
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open("launch_status.log").unwrap();
    writeln!(file, "{}", msg).unwrap();
    println!("{}", msg);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = std::fs::remove_file("launch_status.log");
    log("[Debug] Starting Pure Rust FA2 Launch...");

    // 1. Initialize Runtime
    let runtime = RuntimeManager::init(None).unwrap();
    log("[Debug] Runtime Initialized");

    // 2. Generate FA2 Kernel Source
    let config = PipelineConfig {
        num_stages: 2,
        m_tile: 64, n_tile: 64, k_tile: 32,
        instruction: SpecializedInstruction::None,
        swizzle_mode: SwizzleMode::None,
        quantization: QuantizationMode::None,
        epilogue: vec![],
        force_num_warps: Some(4),
        intrinsic_shape: IntrinsicShape::None,
        vectorize_epilogue: true,
        ttg_enabled: false,
        attention_variant: Default::default(),
    };
    let emitter = FlashAttentionEmitter::new(config);
    // H=1, D=64, Causal=false
    let source = emitter.generate_kernel(1, 64, false);
    log(&format!("[Debug] Generated Source (Length: {})", source.len()));
    let _ = std::fs::write("generated.cu", &source);
    
    // 3. Compile
    log("[Debug] Compiling...");
    let kernel_id = runtime.compile(&source, "flash_attention_v2_kernel", DeviceBackend::Cuda)
        .map_err(|e| { log(&format!("Compile Failed: {}", e)); e })?;
    log(&format!("[Debug] Compiled Kernel ID: {}", kernel_id.0));

    // 4. Allocate Buffers
    // S=128 (2 tiles), D=64.
    // Q, K, V, O: S*D*sizeof(half) = 128*64*2 = 16384 bytes
    let size_bytes = 128 * 64 * 2;
    let b_q = runtime.alloc(size_bytes, DeviceBackend::Cuda).map_err(|e| { log(&format!("Alloc Q failed: {}", e)); e })?;
    let b_k = runtime.alloc(size_bytes, DeviceBackend::Cuda).map_err(|e| { log(&format!("Alloc K failed: {}", e)); e })?;
    let b_v = runtime.alloc(size_bytes, DeviceBackend::Cuda).map_err(|e| { log(&format!("Alloc V failed: {}", e)); e })?;
    let b_o = runtime.alloc(size_bytes, DeviceBackend::Cuda).map_err(|e| { log(&format!("Alloc O failed: {}", e)); e })?;
    
    log(&format!("[Debug] Buffers Allocated: Q={:?}, K={:?}, V={:?}, O={:?}", b_q, b_k, b_v, b_o));

    // 5. Prepare Args
    // Signature: (Q, K, V, O, B, H, S, D, scale)
    let args = vec![
        KernelArg::Buffer(b_q),
        KernelArg::Buffer(b_k),
        KernelArg::Buffer(b_v),
        KernelArg::Buffer(b_o),
        KernelArg::Usize(1_usize), // B
        KernelArg::Usize(1_usize), // H
        KernelArg::Usize(128_usize), // S
        KernelArg::Usize(64_usize), // D
        KernelArg::Float(1.0f32), // scale
    ];
    
    log("[Debug] Launching Kernel...");
    log("[Debug] Grid=(1,1,1), Block=(128,1,1), Smem=48000");

    let smem_size = 48000; 

    runtime.launch(kernel_id, (1,1,1), (128,1,1), smem_size, args)
        .map_err(|e| { log(&format!("Launch Failed: {}", e)); e })?;

    log("[Debug] Launch Success!");
    
    Ok(())
}
