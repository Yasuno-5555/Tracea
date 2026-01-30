use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::core::config::PipelineConfig;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};
use std::time::Instant;

fn main() {
    println!("🧪 Accurate GEMM Timing Test");
    
    let runtime = RuntimeManager::new();
    let backend = DeviceBackend::Metal;
    
    let m = 2048usize;
    let n = 2048usize;
    let k = 2048usize;
    
    // Allocate buffers
    let a_buf = runtime.alloc(m * k * 2, backend).unwrap(); // FP16
    let b_buf = runtime.alloc(k * n * 2, backend).unwrap(); // FP16
    let c_buf = runtime.alloc(m * n * 4, backend).unwrap(); // FP32
    
    // Generate kernel
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm { 
            m: m as u32, n: n as u32, k: k as u32, 
            batch: 1, epilogue: vec![] 
        },
        precison: "fp16".to_string(),
        tiling: PipelineConfig {
            m_tile: 64, n_tile: 64, k_tile: 32,
            double_buffer: true,
            ..Default::default()
        },
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };
    
    let emitter = tracea::emitter::metal::MetalEmitter::detect();
    let source = emitter.generate_from_ir(&ir);
    let kernel_id = runtime.compile(&source, "unified_gemm_kernel", backend).unwrap();
    
    // TTG for dispatch (L1/L2 tile map)
    let num_m_tiles = (m + 64 - 1) / 64;
    let num_n_tiles = (n + 64 - 1) / 64;
    let num_tiles = num_m_tiles * num_n_tiles;
    
    // Create simple L1 map (identity)
    let l1_map: Vec<u32> = (0..num_tiles as u32).collect();
    let l1_bytes: Vec<u8> = l1_map.iter().flat_map(|x| x.to_ne_bytes().to_vec()).collect();
    let l1_buf = runtime.alloc(l1_bytes.len(), backend).unwrap();
    runtime.copy_to_device(l1_buf, &l1_bytes).unwrap();
    
    // L2 table (20 bytes per tile: region_m, region_n, k_start, k_end, role)
    let mut l2_bytes = Vec::new();
    for tile_id in 0..num_tiles {
        let region_m = (tile_id / num_n_tiles) as u32;
        let region_n = (tile_id % num_n_tiles) as u32;
        l2_bytes.extend_from_slice(&region_m.to_ne_bytes());
        l2_bytes.extend_from_slice(&region_n.to_ne_bytes());
        l2_bytes.extend_from_slice(&0u32.to_ne_bytes()); // k_start
        l2_bytes.extend_from_slice(&(k as u32).to_ne_bytes()); // k_end
        l2_bytes.extend_from_slice(&0u32.to_ne_bytes()); // role
    }
    let l2_buf = runtime.alloc(l2_bytes.len(), backend).unwrap();
    runtime.copy_to_device(l2_buf, &l2_bytes).unwrap();
    
    let args = vec![
        KernelArg::Buffer(a_buf),
        KernelArg::Buffer(b_buf),
        KernelArg::Buffer(c_buf),
        KernelArg::Int(m as i32),  // M - must be u32, using Int which is i32
        KernelArg::Int(n as i32),  // N
        KernelArg::Int(k as i32),  // K
    ];
    
    let grid = (num_n_tiles as u32, num_m_tiles as u32, 1);
    let block = (128, 1, 1);
    let smem = 48 * 1024;
    
    // Warmup
    println!("[Warmup] Running 3 iterations...");
    for _ in 0..3 {
        runtime.launch(kernel_id, grid, block, smem as u32, args.clone()).unwrap();
    }
    
    // Benchmark
    println!("[Benchmark] Running 10 iterations...");
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        runtime.launch(kernel_id, grid, block, smem as u32, args.clone()).unwrap();
    }
    let elapsed = start.elapsed();
    
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let tflops = flops / (avg_ms / 1000.0) / 1e12;
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("GEMM {}x{}x{} Benchmark Results", m, n, k);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Average Latency: {:.3} ms", avg_ms);
    println!("Performance:     {:.2} TFLOPS", tflops);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Sanity check
    if tflops > 15.0 {
        println!("⚠️  WARNING: Performance over 15 TFLOPS on M1/M2 is suspicious!");
        println!("    M1 Max FP32: ~10 TFLOPS, M3 Max: ~28 TFLOPS");
        println!("    Please verify synchronization is working correctly.");
    } else {
        println!("✅ Performance looks realistic for Apple Silicon.");
    }
}
