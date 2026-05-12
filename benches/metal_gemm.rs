use criterion::{criterion_group, criterion_main, Criterion};
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::core::config::PipelineConfig;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};

fn bench_gemm_execute(c: &mut Criterion) {
    #[cfg(target_os = "macos")]
    {
        let runtime = RuntimeManager::new();
        let m = 2048usize; let n = 2048usize; let k = 2048usize;

        let a_buf = runtime.alloc_u16(m * k, DeviceBackend::Metal).unwrap();
        let b_buf = runtime.alloc_u16(k * n, DeviceBackend::Metal).unwrap();
        let c_buf = runtime.alloc_f32(m * n, DeviceBackend::Metal).unwrap();

        let mut config = PipelineConfig::new(2, 32, 16, 16);
        config.force_num_warps = Some(2);
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
        let kernel_id = runtime.compile(&source, "unified_gemm_kernel", DeviceBackend::Metal).unwrap();

        let args = vec![
            KernelArg::Buffer(a_buf), KernelArg::Buffer(b_buf), KernelArg::Buffer(c_buf),
            KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32),
        ];

        let grid = ((n / 16) as u32, (m / 32) as u32, 1);
        let block = (64, 1, 1);

        // Warmup with sync
        for _ in 0..3 {
            runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
            runtime.synchronize();
        }

        let mut group = c.benchmark_group("metal_gemm");
        group.sample_size(10);

        group.bench_function("gemm_2048_metal", |bencher| {
            bencher.iter(|| {
                runtime.launch(kernel_id, grid, block, 0, args.clone()).unwrap();
                runtime.synchronize();
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_gemm_execute);
criterion_main!(benches);
