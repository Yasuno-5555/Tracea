use criterion::{criterion_group, criterion_main, Criterion};
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, BufferId, KernelArg};
use tracea::core::config::PipelineConfig;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};
use std::sync::Arc;

fn bench_gemm_execute(c: &mut Criterion) {
    #[cfg(target_os = "macos")]
    {
        let runtime = Arc::new(RuntimeManager::new());
        let m = 2048usize; let n = 2048usize; let k = 2048usize;
        
        let a_buf = runtime.alloc(m * k * 2, DeviceBackend::Metal).unwrap();
        let b_buf = runtime.alloc(k * n * 2, DeviceBackend::Metal).unwrap();
        let c_buf = runtime.alloc(m * n * 4, DeviceBackend::Metal).unwrap();

        // Generate kernel with double-buffer GEMM
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
        };
        
        let emitter = tracea::emitter::metal::MetalEmitter::detect();
        let source = emitter.generate_from_ir(&ir);
        let kernel_id = runtime.compile(&source, "unified_gemm_kernel", DeviceBackend::Metal).unwrap();

        // Pre-allocate TTG buffers (L1/L2 tile maps)
        let num_m_tiles = (m + 64 - 1) / 64;
        let num_n_tiles = (n + 64 - 1) / 64;
        let num_tiles = num_m_tiles * num_n_tiles;
        
        let l1_map: Vec<u32> = (0..num_tiles as u32).collect();
        let l1_bytes: Vec<u8> = l1_map.iter().flat_map(|x| x.to_ne_bytes().to_vec()).collect();
        let l1_buf = runtime.alloc(l1_bytes.len(), DeviceBackend::Metal).unwrap();
        runtime.copy_to_device(l1_buf, &l1_bytes).unwrap();
        
        let mut l2_bytes = Vec::new();
        for tile_id in 0..num_tiles {
            let region_m = (tile_id / num_n_tiles) as u32;
            let region_n = (tile_id % num_n_tiles) as u32;
            l2_bytes.extend_from_slice(&region_m.to_ne_bytes());
            l2_bytes.extend_from_slice(&region_n.to_ne_bytes());
            l2_bytes.extend_from_slice(&0u32.to_ne_bytes());
            l2_bytes.extend_from_slice(&(k as u32).to_ne_bytes());
            l2_bytes.extend_from_slice(&0u32.to_ne_bytes());
        }
        let l2_buf = runtime.alloc(l2_bytes.len(), DeviceBackend::Metal).unwrap();
        runtime.copy_to_device(l2_buf, &l2_bytes).unwrap();

        let args = vec![
            KernelArg::Buffer(a_buf),
            KernelArg::Buffer(b_buf),
            KernelArg::Buffer(c_buf),
            KernelArg::Int(m as i32),
            KernelArg::Int(n as i32),
            KernelArg::Int(k as i32),
        ];
        
        let grid = (num_n_tiles as u32, num_m_tiles as u32, 1);
        let block = (128, 1, 1);
        let smem = 48 * 1024u32;

        // Warmup
        for _ in 0..3 {
            runtime.launch(kernel_id, grid, block, smem, args.clone()).unwrap();
        }

        let mut group = c.benchmark_group("metal_gemm");
        group.sample_size(20); // Reduce samples since each takes ~40ms
        
        group.bench_function("gemm_2048_sync", |bencher| {
            bencher.iter(|| {
                // launch() already includes wait_until_completed()
                runtime.launch(kernel_id, grid, block, smem, args.clone()).unwrap();
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_gemm_execute);
criterion_main!(benches);
