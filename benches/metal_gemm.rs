use criterion::{criterion_group, criterion_main, Criterion};
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, BufferId, KernelArg};
use tracea::policy::types::{OperatorTopology, GraphTopology, TopologyKind};
use std::collections::HashMap;
use std::sync::Arc;

fn bench_gemm_execute(c: &mut Criterion) {
    #[cfg(target_os = "macos")]
    {
        let runtime = Arc::new(RuntimeManager::new());
        let m = 2048; let n = 2048; let k = 2048;
        
        let a_buf = runtime.alloc(m * k * 2, DeviceBackend::Metal).unwrap();
        let b_buf = runtime.alloc(k * n * 2, DeviceBackend::Metal).unwrap();

        let mut input_buffers = HashMap::<u64, BufferId>::new();
        input_buffers.insert(100, a_buf);
        input_buffers.insert(101, b_buf);

        let graph = GraphTopology {
            operators: vec![
                OperatorTopology::Elementwise { op_id: 100, name: "a_in".into(), kind: "Identity".into() },
                OperatorTopology::Elementwise { op_id: 101, name: "b_in".into(), kind: "Identity".into() },
                OperatorTopology::Gemm {
                    op_id: 1, name: "gemm".into(),
                    m: m as u32, n: n as u32, k: k as u32, kind: TopologyKind::Dense,
                }
            ],
            dependencies: vec![
                (100, 1), (101, 1)
            ],
        };

        let _ = runtime.execute_graph(&graph, &input_buffers, DeviceBackend::Metal).unwrap();

        let device = tracea::core::device::DeviceProfile::from_backend(DeviceBackend::Metal);
        let mut engine = tracea::policy::standard::StandardPolicyEngine::new();
        let ctx = tracea::policy::types::GraphContext { device: &device, graph: &graph };
        let decision = tracea::policy::scheduler::StandardScheduler::schedule(&mut engine, &ctx);
        
        let tp = decision.tile_policies.iter().find(|p| p.operator_id() == 1).unwrap().clone();
        let ep = decision.exec_policies.iter().find(|p| p.operator_id == 1).unwrap().clone();
        let op = graph.operators.iter().find(|o| o.op_id() == 1).unwrap().clone();

        let output_buf = runtime.alloc(m * n * 4, DeviceBackend::Metal).unwrap(); // fp32
        
        // Compile once
        let ir = tracea::emitter::traits::UnifiedOpIR {
            op_type: tracea::emitter::traits::UnifiedOpType::Gemm { m: m as u32, n: n as u32, k: k as u32 },
            precison: "fp16".to_string(),
            tiling: tracea::core::config::PipelineConfig {
                 m_tile: 32, n_tile: 32, k_tile: 32,
                 gemm_variant: tracea::core::config::GemmVariant::Tiled,
                 ..Default::default()
            },
            conv_magic_strategy: None,
        };
        use tracea::emitter::traits::Emitter;
        let emitter = tracea::emitter::metal::MetalEmitter::detect();
        let source = emitter.generate_from_ir(&ir);
        let kernel_id = runtime.compile(&source, "gemm_tiled_kernel", DeviceBackend::Metal).unwrap();

        let mut group = c.benchmark_group("metal_gemm_hot");
        
        let args = vec![
            KernelArg::Buffer(a_buf),
            KernelArg::Buffer(b_buf),
            KernelArg::Buffer(output_buf),
        ];

        group.bench_function("launch_hot_gemm", |bencher| {
            bencher.iter(|| {
                runtime.launch_with_policy(
                    kernel_id.clone(),
                    args.clone(),
                    &op,
                    &tp,
                    &ep,
                    vec![],
                    DeviceBackend::Metal
                ).unwrap();
                runtime.synchronize();
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_gemm_execute);
criterion_main!(benches);
