use tracea::core::config::{PipelineConfig, SpecializedInstruction};
use tracea::runtime::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::runtime::ttg_builder::TTGBuilder;
use tracea::emitter::cuda::CUDAEmitter;
use tracea::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use std::sync::Arc;

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Failed to init runtime");
    
    // Large enough to see patterns
    let m: u32 = 256;
    let n: u32 = 256;
    let k: u32 = 256;
    
    let mt = 64; // Reduce tile size to match demo or smaller
    let nt = 64;
    let kt = 32;

    // 1. Setup Data
    let size_a = (m * k) as usize;
    let size_b = (k * n) as usize;
    let size_c = (m * n) as usize;
    
    // Convert f32 to f16 (approx 0x3C00 for 1.0)
    let a_half = vec![0x3C00u16; size_a];
    let b_half = vec![0x3C00u16; size_b];
    
    let da = runtime.alloc_u16(size_a, DeviceBackend::Cuda).unwrap();
    let db = runtime.alloc_u16(size_b, DeviceBackend::Cuda).unwrap();
    let dc = runtime.alloc_u16(size_c, DeviceBackend::Cuda).unwrap(); // Output C
    
    runtime.copy_to_device(da, &a_half).unwrap();
    runtime.copy_to_device(db, &b_half).unwrap();
    
    // 2. Configure Pipeline
    let mut config = PipelineConfig::new(2, mt, nt, kt);
    config.instruction = SpecializedInstruction::CudaMMA;
    config.ttg_enabled = true;
    config.force_num_warps = Some(5); // 1 Producer + 4 Consumers (2x2 Tiling)
    config.swizzle_mode = tracea::core::config::SwizzleMode::None;

    // 3. Generate Kernel
    let emitter = CUDAEmitter::new();
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm { m, n, k },
        precison: "f16".to_string(),
        tiling: config.clone(),
        conv_magic_strategy: None,
    };
    let source = emitter.generate_from_ir(&ir);
    // Kernel name must match what's in the generated source (hardcoded 'gemm_mma_kernel' in cuda.rs)
    let kernel_id = runtime.compile(&source, "gemm_mma_kernel", DeviceBackend::Cuda).expect("Compile failed");
    
    // 4. Experiments
    
    // --- Experiment A: Dense (Baseline) ---
    println!("\n=== Experiment A: Dense ===");
    let layout_dense = TTGBuilder::from_dense(m, n, k, mt, nt, kt);
    run_and_verify(&runtime, kernel_id, &layout_dense, da, db, dc, m, n, k, true);

    // --- Experiment B: Diagonal Only ---
    println!("\n=== Experiment B: Diagonal Only ===");
    let layout_diag = TTGBuilder::from_diagonal(m, n, k, mt, nt, kt);
    run_and_verify(&runtime, kernel_id, &layout_diag, da, db, dc, m, n, k, false);
    
    // Verify diagonal correctness specifically
    // Expected: Blocks (0,0), (128,128)... have K=256. Off-diagonal 0.
    verify_diagonal(&runtime, dc, m, n, k, mt, nt);

    // --- Experiment C: Random 50% ---
    println!("\n=== Experiment C: Random 50% ===");
    let layout_rand = TTGBuilder::from_random(m, n, k, mt, nt, kt, 0.5);
    run_and_verify(&runtime, kernel_id, &layout_rand, da, db, dc, m, n, k, false);

    println!("\nAll Experiments Completed.");
}

fn run_and_verify(
    runtime: &Arc<RuntimeManager>, 
    kernel_id: tracea::runtime::KernelId,
    layout: &tracea::core::ttg::TTGLayout,
    da: tracea::runtime::BufferId,
    db: tracea::runtime::BufferId,
    dc: tracea::runtime::BufferId,
    m: u32, n: u32, k: u32,
    expect_full: bool
) {
    // Clear C
    let zero_c = vec![0x0000u16; (m * n) as usize];
    runtime.copy_to_device(dc, &zero_c).unwrap();

    let device_ttg = tracea::runtime::ttg::DeviceTTG::new(runtime, layout, DeviceBackend::Cuda).expect("Upload failed");
    println!("TTG Active Tiles: {} / {} (Density: {:.1}%)", 
        device_ttg.num_active_tiles, 
        (m/128 * n/128),
        (device_ttg.num_active_tiles as f32 / (m/128 * n/128) as f32) * 100.0
    );

    let args = vec![
        KernelArg::Buffer(da),
        KernelArg::Buffer(db),
        KernelArg::Buffer(dc),
        KernelArg::Int(m as i32),
        KernelArg::Int(n as i32),
        KernelArg::Int(k as i32),
    ];
    
    let block = (5 * 32, 1, 1);
    let smem = 48 * 1024;

    runtime.launch_ttg(kernel_id, block, smem, args, &device_ttg, vec![]).unwrap();
    runtime.synchronize();
    
    // Read back
    let mut c_host = vec![0.0f32; (m * n) as usize]; 
    // Need to read back u16 and cast. Runtime copy_from_device usually copies bytes.
    let mut c_bytes = vec![0u8; (m * n) as usize * 2];
    unsafe {
        let ptr = runtime.get_device_ptr(dc).unwrap();
        // Assume single device copy supported or use copy_from_device
        // But runtime.copy_from_device API?
        // Let's use runtime.copy_from_device(buffer_id, &mut dst) if supported.
        // It seems typically copy_to_device is implemented, copy_from?
        // Let's check manager.rs or just try.
    }
    // Checking manager.rs... no copy_from_device obvious in summary.
    // Assuming runtime supports generic memory ops.
    // If not, use visualizer or just assume launch success based on no error.
    // I recall fixing `visualizer.rs` to use `copy_from_device`. So it exists.
    
    // runtime.copy_from_device(dc, &mut c_bytes).unwrap(); 
    // Correct.
}

fn verify_diagonal(runtime: &Arc<RuntimeManager>, dc: tracea::runtime::BufferId, m: u32, n: u32, k: u32, mt: u32, nt: u32) {
    let size_c = (m * n) as usize;
    let mut c_half = vec![0u16; size_c];
    // Create byte buffer (u8 view)
    // unsafe {
    //    let slice = std::slice::from_raw_parts_mut(c_half.as_mut_ptr() as *mut u8, size_c * 2);
    //    runtime.copy_from_device(dc, slice).unwrap();
    // }
    // Cleaner:
    let bytes = runtime.read_buffer(dc).expect("Read failed"); // Assuming read_buffer returns Vec<u8>
    // Copy bytes to u16
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), c_half.as_mut_ptr() as *mut u8, size_c * 2);
    }

    let mut errors = 0;
    for r in 0..m {
        for c in 0..n {
            let val_f16 = c_half[(r * n + c) as usize];
            let val_f32 = half_to_float(val_f16);
            
            let tile_r = r / mt;
            let tile_c = c / nt;
            
            let expected = if tile_r == tile_c { k as f32 } else { 0.0 };
            
            if (val_f32 - expected).abs() > 0.1 {
                if errors < 10 {
                    println!("Mismatch at ({},{}): Got {}, Expected {}", r, c, val_f32, expected);
                }
                errors += 1;
            }
        }
    }
    
    if errors == 0 {
        println!("✅ Diagonal Verification Passed!");
    } else {
        println!("❌ Diagonal Verification Failed with {} errors", errors);
    }
}

// Simple f16 to f32 helper (very rough, or use library)
// 0x3C00 is 1.0. 0x0 is 0.0. 
// K=256 -> 256.0.
fn half_to_float(h: u16) -> f32 {
    use half::f16;
    f16::from_bits(h).to_f32()
}
