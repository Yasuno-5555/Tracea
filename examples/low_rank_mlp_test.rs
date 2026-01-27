use tracea::runtime::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::policy::types::*;
use tracea::emitter::cuda::CUDAEmitter;
use tracea::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use tracea::core::config::PipelineConfig;
use tracea::runtime::ttg_builder::TTGBuilder;
use std::sync::Arc;

fn main() {
    println!("=== Low-Rank MLP Integration Test ===");
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Failed to init runtime");
    
    let m: u32 = 256;
    let n: u32 = 256;
    let k: u32 = 256;
    let r: u32 = 64;

    // 1. Setup Data
    let size_x = (m * k) as usize;
    let size_a = (k * r) as usize;
    let size_b = (r * n) as usize;
    let size_c = (m * n) as usize;

    let x_half = vec![0x3C00u16; size_x]; // 1.0
    let a_half = vec![0x3C00u16; size_a]; 
    let b_half = vec![0x3C00u16; size_b];
    
    let dx = runtime.alloc_u16(size_x, DeviceBackend::Cuda).unwrap();
    let da = runtime.alloc_u16(size_a, DeviceBackend::Cuda).unwrap();
    let db = runtime.alloc_u16(size_b, DeviceBackend::Cuda).unwrap();
    let dc = runtime.alloc_u16(size_c, DeviceBackend::Cuda).unwrap();
    
    runtime.copy_to_device(dx, &x_half).unwrap();
    runtime.copy_to_device(da, &a_half).unwrap();
    runtime.copy_to_device(db, &b_half).unwrap();
    
    // 2. Prepare Kernel
    let mut config = PipelineConfig::new(2, 64, 64, 32);
    config.ttg_enabled = true;
    
    let emitter = CUDAEmitter::new();
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::LowRankMlp { m, n, k, r },
        precison: "f16".to_string(),
        tiling: config.clone(),
        conv_magic_strategy: None,
    };
    
    let source = emitter.generate_from_ir(&ir);
    // println!("Generated Source:\n{}", source);
    
    let kernel_id = runtime.compile(&source, "low_rank_mlp_kernel", DeviceBackend::Cuda).expect("Compile failed");
    
    // 3. Setup TTG
    let layout = TTGBuilder::from_dense(m, n, k, 64, 64, 32); 
    let device_ttg = tracea::runtime::ttg::DeviceTTG::new(&runtime, &layout, DeviceBackend::Cuda).expect("Upload failed");

    // 4. Launch
    let args = vec![
        KernelArg::Buffer(dx),
        KernelArg::Buffer(da),
        KernelArg::Buffer(db),
        KernelArg::Buffer(dc),
        KernelArg::Int(m as i32),
        KernelArg::Int(n as i32),
        KernelArg::Int(k as i32),
    ];
    
    let block = (4 * 32, 1, 1); // 4 warps
    let smem = 48 * 1024; // 48KB

    println!("[Runtime] Launching Low-Rank MLP kernel...");
    runtime.launch_ttg(kernel_id, block, smem, args, &device_ttg, vec![]).expect("Launch failed");
    runtime.synchronize();
    
    // 5. Verify
    let c_bytes = runtime.read_buffer(dc).expect("Read back failed");
    let mut c_half = vec![0u16; size_c];
    unsafe {
        std::ptr::copy_nonoverlapping(c_bytes.as_ptr(), c_half.as_mut_ptr() as *mut u8, size_c * 2);
    }
    
    use half::f16;
    let first_val = f16::from_bits(c_half[0]).to_f32();
    // Rough expectation for all-ones: 1 * K * 1 * R * 1 = K * R? 
    // Wait, (X * A) result is K. Sum of A for one r is K.
    // (T * B) result is r. Sum of B for one n is r.
    // Each element in T is K.
    // Each element in C is T_val * r = K * r.
    let expected = (k * r) as f32;
    println!("Verification: C[0,0] = {} (Expected {})", first_val, expected);
    
    println!("SUCCESS: Low-Rank MLP kernel executed.");
}
