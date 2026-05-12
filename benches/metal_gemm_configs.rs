use criterion::{criterion_group, criterion_main, Criterion};
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::core::config::PipelineConfig;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};

static CONFIGS: &[(u32, u32, u32, u32, bool, &str)] = &[
    (32, 16, 16, 2, false, "32x16x16_w2_sb"),
    (32, 16, 16, 2, true,  "32x16x16_w2_db"),
    (32, 32, 16, 4, false, "32x32x16_w4_sb"),
    (32, 32, 16, 4, true,  "32x32x16_w4_db"),
    (64, 32, 16, 4, false, "64x32x16_w4_sb"),
    (64, 32, 16, 4, true,  "64x32x16_w4_db"),
    (64, 64, 16, 8, false, "64x64x16_w8_sb"),
    (64, 64, 16, 8, true,  "64x64x16_w8_db"),
    (64, 64, 32, 8, false, "64x64x32_w8_sb"),
    (64, 64, 32, 8, true,  "64x64x32_w8_db"),
    (64, 32, 32, 4, false, "64x32x32_w4_sb"),
    (64, 32, 32, 4, true,  "64x32x32_w4_db"),
    (32, 64, 32, 8, false, "32x64x32_w8_sb"),
    (32, 64, 32, 8, true,  "32x64x32_w8_db"),
    (128, 64, 16, 8, false, "128x64x16_w8_sb"),
    (128, 64, 16, 8, true,  "128x64x16_w8_db"),
    (64, 128, 16, 16, false, "64x128x16_w16_sb"),
    (64, 128, 16, 16, true,  "64x128x16_w16_db"),
];

fn bench_gemm_configs(c: &mut Criterion) {
    #[cfg(target_os = "macos")]
    {
        let runtime = RuntimeManager::new();
        let m = 2048usize; let n = 2048usize; let k = 2048usize;

        let a_buf = runtime.alloc_u16(m * k, DeviceBackend::Metal).unwrap();
        let b_buf = runtime.alloc_u16(k * n, DeviceBackend::Metal).unwrap();
        let c_buf = runtime.alloc_f32(m * n, DeviceBackend::Metal).unwrap();

        // Initialize input data to avoid zero-data GPU power throttling
        let a_data: Vec<u16> = (0..m*k).map(|i| ((i % 256) as u16).max(1)).collect();
        let b_data: Vec<u16> = (0..k*n).map(|i| ((i * 3) % 256).max(1) as u16).collect();
        runtime.copy_to_device(a_buf, &a_data).unwrap();
        runtime.copy_to_device(b_buf, &b_data).unwrap();

        let emitter = tracea::emitter::metal::MetalEmitter::detect();

        let mut group = c.benchmark_group("metal_gemm_configs");
        group.sample_size(10);

        for &(mt, nt, kt, nw, db, label) in CONFIGS {
            let mut config = PipelineConfig::new(2, mt, nt, kt);
            config.force_num_warps = Some(nw);
            config.double_buffer = db;
            config.instruction = tracea::core::config::SpecializedInstruction::MetalSimdGroup;
            config.gemm_variant = tracea::core::config::GemmVariant::Simd;

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

            let source = match emitter.generate_from_ir(&ir) {
                Ok(s) => s,
                Err(e) => { eprintln!("  [SKIP] {} codegen failed: {:?}", label, e); continue; }
            };

            let kid = match runtime.compile(&source, "unified_gemm_kernel", DeviceBackend::Metal) {
                Ok(id) => id,
                Err(_) => { eprintln!("  [SKIP] {} compile failed", label); continue; }
            };

            let args = vec![
                KernelArg::Buffer(a_buf), KernelArg::Buffer(b_buf), KernelArg::Buffer(c_buf),
                KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32),
            ];

            let grid = ((n / nt as usize) as u32, (m / mt as usize) as u32, 1);
            let block = (nw * 32, 1, 1);

            // warmup
            for _ in 0..5 {
                runtime.launch(kid, grid, block, 0, args.clone()).unwrap();
                runtime.synchronize();
            }

            group.bench_function(label, |bencher| {
                bencher.iter(|| {
                    runtime.launch(kid, grid, block, 0, args.clone()).unwrap();
                    runtime.synchronize();
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_gemm_configs);
criterion_main!(benches);
