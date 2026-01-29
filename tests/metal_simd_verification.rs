use tracea::core::config::{PipelineConfig, AttentionVariant};
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg, BufferId};
use tracea::emitter::universal::UniversalEmitter;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use std::sync::Arc;

#[test]
fn test_metal_simd_qk_verification() {
    #[cfg(not(target_os = "macos"))]
    {
        println!("Skipping Metal test on non-macOS");
        return;
    }
    
    // Explicitly init Metal
    let runtime = RuntimeManager::init(Some(DeviceBackend::Metal)).unwrap();
    let backend = DeviceBackend::Metal;
    
    // Params
    let b: u32 = 1;
    let h: u32 = 2;
    let s: u32 = 128;
    let d: u32 = 64;
    let dh: u32 = 64;
    
    // Alloc u16 as f16 storage
    let num_elements = (b * h * s * dh) as usize;
    let size_bytes = num_elements * 2;
    let q = runtime.alloc_u16(size_bytes, backend).unwrap();
    let k = runtime.alloc_u16(size_bytes, backend).unwrap();
    let v = runtime.alloc_u16(size_bytes, backend).unwrap();
    let o_naive = runtime.alloc_u16(size_bytes, backend).unwrap();
    let o_simd = runtime.alloc_u16(size_bytes, backend).unwrap();
    let o_full = runtime.alloc_u16(size_bytes, backend).unwrap();
    
    // Init Data using 1.0 (0x3C00) and small values
    // 0.1 approx 0x2E66
    let val_init = 0x2E66u16; 
    let init_vec = vec![val_init; num_elements];
    let zero_vec = vec![0u16; num_elements];
    
    runtime.copy_to_device(q, &init_vec).unwrap();
    runtime.copy_to_device(k, &init_vec).unwrap();
    runtime.copy_to_device(v, &init_vec).unwrap();
    runtime.copy_to_device(o_naive, &zero_vec).unwrap();
    runtime.copy_to_device(o_simd, &zero_vec).unwrap();
    runtime.copy_to_device(o_full, &zero_vec).unwrap();
    
    println!("Init Complete. Running Naive...");
    
    // 1. Run Naive
    run_variant(&runtime, q, k, v, o_naive, b, h, s, d, dh, AttentionVariant::Naive);
    
    println!("Naive Launched. Running SimdQK...");
    
    // 2. Run SimdQK
    run_variant(&runtime, q, k, v, o_simd, b, h, s, d, dh, AttentionVariant::SimdQK);
    
    println!("SimdQK Launched. Running SimdFull...");

    // 3. Run SimdFull
    run_variant(&runtime, q, k, v, o_full, b, h, s, d, dh, AttentionVariant::SimdFull);
    
    println!("SimdFull Launched. Synchronizing...");
    
    runtime.synchronize();
    
    println!("Synced. Copying back...");
    
    // 4. Compare
    let mut out_naive = vec![0u16; num_elements];
    let mut out_simd = vec![0u16; num_elements];
    let mut out_full = vec![0u16; num_elements];
    runtime.copy_from_device(o_naive, &mut out_naive).unwrap();
    runtime.copy_from_device(o_simd, &mut out_simd).unwrap();
    runtime.copy_from_device(o_full, &mut out_full).unwrap();
    
    println!("Comparing...");
    
    let mut max_diff_bits_simd = 0;
    let mut max_diff_bits_full = 0;
    
    for i in 0..num_elements {
        let n_val = out_naive[i];
        let s_val = out_simd[i];
        let f_val = out_full[i];
        
        // SimdQK check
        let diff_s = if n_val > s_val { n_val - s_val } else { s_val - n_val };
        if diff_s > max_diff_bits_simd { max_diff_bits_simd = diff_s; }
        
        // SimdFull check
        let diff_f = if n_val > f_val { n_val - f_val } else { f_val - n_val };
        if diff_f > max_diff_bits_full { max_diff_bits_full = diff_f; }
    }
    
    println!("Max Bit Diff Naive vs SimdQK: {}", max_diff_bits_simd);
    println!("Max Bit Diff Naive vs SimdFull: {}", max_diff_bits_full);
    
    assert!(max_diff_bits_simd <= 5, "SimdQK Mismatch! Diff: {}", max_diff_bits_simd);
    assert!(max_diff_bits_full <= 5, "SimdFull Mismatch! Diff: {}", max_diff_bits_full);
    
    let sum_naive: u64 = out_naive.iter().map(|&x| x as u64).sum();
    println!("Naive Sum: {}", sum_naive);
    assert!(sum_naive > 0, "Naive output is all zeros!");
}
fn run_variant(
    runtime: &Arc<RuntimeManager>, q: BufferId, k: BufferId,
    v: BufferId, o: BufferId,
    b: u32, h: u32, s: u32, d: u32, dh: u32, variant: AttentionVariant
) {
    let backend = DeviceBackend::Metal;
    let emitter = UniversalEmitter::new(backend);
    
    let mut config = PipelineConfig::new(2, 64, 64, dh);
    config.attention_variant = variant;
    
    // Note: The Emitter uses HARDCODED mt=16 inside `metal.rs` for attention!
    // But PipelineConfig mt=64 helps calculate Shared Mem?
    // Actually `smem` logic in `metal.rs` is mostly ignored/hardcoded in the String?
    // `metal.rs` attention string has NO dynamic shared mem size param passed to launch? 
    // Wait. `launch` takes `smem` size.
    // In `generate_metal_attention`, we see: `threadgroup half sK[...]`.
    // It's static shared memory (Metal logic).
    // So dynamic smem param is ignored?
    // Correct. Metal kernels define `threadgroup` arrays inside.
    
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::FusedAttention {
            b, s, d, h, dh, causal: false
        },
        precison: "f16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
    };
    
    // Generate Name based on variant to avoid cache collision if content is diff but name same?
    // Compiler hashes source?
    // RuntimeManager `compile` takes name.
    let name = format!("flash_attn_{:?}", variant);
    
    let source = emitter.generate(ir);
    let kernel = runtime.compile(&source, "flash_attention_v2_kernel", backend).unwrap();
    
    // Params struct
    let scale = 1.0 / (dh as f32).sqrt();
    let mut params = Vec::with_capacity(20);
    params.extend_from_slice(&(b as u32).to_ne_bytes());
    params.extend_from_slice(&(h as u32).to_ne_bytes());
    params.extend_from_slice(&(s as u32).to_ne_bytes());
    params.extend_from_slice(&(dh as u32).to_ne_bytes());
    params.extend_from_slice(&scale.to_ne_bytes());
    
    // Grid matches Kernel hardcoded BLOCK_M=16
    let mt = 16;
    let grid = ( (s + mt - 1) / mt, h, b );
    let block = (32, 1, 1);
    
    runtime.launch(kernel, grid, block, 0, vec![
        KernelArg::Buffer(q), KernelArg::Buffer(k), KernelArg::Buffer(v), KernelArg::Buffer(o),
        KernelArg::Bytes(params)
    ]).unwrap();
}
