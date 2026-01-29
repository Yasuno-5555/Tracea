use tracea::core::config::{PipelineConfig, SpecializedInstruction};
use tracea::runtime::{RuntimeManager, DeviceBackend, BufferId, KernelArg};
use tracea::runtime::ttg_builder::TTGBuilder;
use tracea::emitter::cuda::CUDAEmitter;
use tracea::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use std::sync::Arc;

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Failed to init runtime");
    
    let m: u32 = 256;
    let n: u32 = 256;
    let k: u32 = 256;
    
    // 1. Setup Data
    let size_a = (m * k) as usize;
    let size_b = (k * n) as usize;
    let size_c = (m * n) as usize;
    
    let a_host = vec![1.0f32; size_a];
    let b_host = vec![1.0f32; size_b];
    
    let da = runtime.alloc_u16(size_a, DeviceBackend::Cuda).unwrap();
    let db = runtime.alloc_u16(size_b, DeviceBackend::Cuda).unwrap();
    let dc = runtime.alloc_u16(size_c, DeviceBackend::Cuda).unwrap();
    
    // Convert f32 to f16 (approx 0x3C00 for 1.0)
    let a_half = vec![0x3C00u16; size_a];
    let b_half = vec![0x3C00u16; size_b];
    
    runtime.copy_to_device(da, &a_half).unwrap();
    runtime.copy_to_device(db, &b_half).unwrap();
    
    // 2. Configure TTG
    let mut config = PipelineConfig::new(2, 64, 64, 32);
    config.instruction = SpecializedInstruction::CudaMMA;
    config.ttg_enabled = true; // <--- The magic flag
    config.force_num_warps = Some(4);

    // 3. Generate Kernel
    let emitter = CUDAEmitter::new();
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm { m, n, k },
        precison: "f16".to_string(),
        tiling: config.clone(),
        conv_magic_strategy: None,
    };
    let source = emitter.generate_from_ir(&ir);
    
    let kernel_id = runtime.compile(&source, "gemm_mma_kernel", DeviceBackend::Cuda).expect("Compile failed");
    
    // 4. Build TTG Layout
    // 4. Build TTG Layout
    let layout = TTGBuilder::from_dense(m, n, k, config.m_tile, config.n_tile, config.k_tile);
    println!("TTG Layout: {} active tiles", layout.num_active_tiles);
    
    // 5. Upload TTG Tables
    /*
    let l1_bytes: Vec<u8> = layout.l1_map.iter().flat_map(|x| x.to_ne_bytes().to_vec()).collect();
    let l2_bytes: Vec<u8> = layout.l2_table.iter().flat_map(|meta| {
         let mut b = Vec::with_capacity(20);
         b.extend_from_slice(&meta.region_m.to_ne_bytes());
         b.extend_from_slice(&meta.region_n.to_ne_bytes());
         b.extend_from_slice(&meta.k_start.to_ne_bytes());
         b.extend_from_slice(&meta.k_end.to_ne_bytes());
         b.extend_from_slice(&meta.role.to_ne_bytes());
         b
    }).collect();

    let d_l1 = runtime.alloc(l1_bytes.len(), DeviceBackend::Cuda).unwrap();
    let d_l2 = runtime.alloc(l2_bytes.len(), DeviceBackend::Cuda).unwrap();
    runtime.copy_to_device(d_l1, &l1_bytes).unwrap();
    runtime.copy_to_device(d_l2, &l2_bytes).unwrap();
    */
    let device_ttg = tracea::runtime::ttg::DeviceTTG::new(&runtime, &layout, DeviceBackend::Cuda).expect("Failed to upload TTG");
    
    // 6. Launch
    let args = vec![
        KernelArg::Buffer(da),
        KernelArg::Buffer(db),
        KernelArg::Buffer(dc),
        KernelArg::Int(m as i32),
        KernelArg::Int(n as i32),
        KernelArg::Int(k as i32),
    ];
    
    // let grid = (layout.num_active_tiles, 1, 1);
    let block = (4 * 32, 1, 1);
    let smem = 48 * 1024; // Approximation
    
    println!("Launching Kernel with TTG (Tiles={})", device_ttg.num_active_tiles);
    // runtime.launch(kernel_id, grid, block, smem, args).unwrap();
    runtime.launch_ttg(kernel_id, block, smem, args, &device_ttg, vec![]).unwrap();
    runtime.synchronize();
    
    println!("Execution Complete.");
}
