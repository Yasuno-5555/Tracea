use criterion::{criterion_group, criterion_main, Criterion};
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, BufferId, KernelArg};
use tracea::policy::types::{OperatorTopology, GraphTopology};
use std::collections::HashMap;
use std::sync::Arc;

fn bench_attention_execute(c: &mut Criterion) {
    #[cfg(target_os = "macos")]
    {
        let runtime = Arc::new(RuntimeManager::new());
        let b = 1; let h = 8; let s = 2048; let d = 64;
        
        // 1. Pre-allocate inputs
        let q_buf = runtime.alloc(b * h * s * d * 2, DeviceBackend::Metal).unwrap();
        let k_buf = runtime.alloc(b * h * s * d * 2, DeviceBackend::Metal).unwrap();
        let v_buf = runtime.alloc(b * h * s * d * 2, DeviceBackend::Metal).unwrap();

        let mut input_buffers = HashMap::<u64, BufferId>::new();
        input_buffers.insert(100, q_buf);
        input_buffers.insert(101, k_buf);
        input_buffers.insert(102, v_buf);

        let graph = GraphTopology {
            operators: vec![
                OperatorTopology::Elementwise { op_id: 100, name: "q_in".into(), kind: "Identity".into() },
                OperatorTopology::Elementwise { op_id: 101, name: "k_in".into(), kind: "Identity".into() },
                OperatorTopology::Elementwise { op_id: 102, name: "v_in".into(), kind: "Identity".into() },
                OperatorTopology::Attention {
                    op_id: 1, name: "attn".into(),
                    b: b as u32, s: s as u32, h: h as u32, d: d as u32,
                }
            ],
            dependencies: vec![
                (100, 1), (101, 1), (102, 1)
            ],
        };

        // 2. Warm up and get necessary objects to bypass full orchestration in hot loop
        // We still use the objects derived from the policy engine
        let _ = runtime.execute_graph(&graph, &input_buffers, DeviceBackend::Metal).unwrap();
        
        // Let's get the KernelID and Policy for direct launch (to measure RAW performance)
        let device = tracea::core::device::DeviceProfile::from_backend(DeviceBackend::Metal);
        let mut engine = tracea::policy::standard::StandardPolicyEngine::new();
        let ctx = tracea::policy::types::GraphContext { device: &device, graph: &graph };
        let decision = tracea::policy::scheduler::StandardScheduler::schedule(&mut engine, &ctx);
        
        let tp = decision.tile_policies.iter().find(|p| p.operator_id() == 1).unwrap().clone();
        let ep = decision.exec_policies.iter().find(|p| p.operator_id == 1).unwrap().clone();
        let op = graph.operators.iter().find(|o| o.op_id() == 1).unwrap().clone();

        // Get Output Buffer from execute_graph result if we want it, or just use a fresh one for bench
        let output_buf = runtime.alloc(b * h * s * d * 2, DeviceBackend::Metal).unwrap();
        
        // Compile once
        let source = "/* Source handled by manager */"; // We'll let manager handle it
        // Actually, we need to generate source to get ID
        let ir = tracea::emitter::traits::UnifiedOpIR {
            op_type: tracea::emitter::traits::UnifiedOpType::FusedAttention { 
                b: b as u32, s: s as u32, d: d as u32, h: h as u32, dh: d as u32, causal: false 
            },
            precison: "fp16".to_string(),
            tiling: tracea::core::config::PipelineConfig {
                 m_tile: 64, n_tile: 64, k_tile: 64,
                 attention_variant: tracea::core::config::AttentionVariant::FlashV2,
                 ..Default::default()
            },
            conv_magic_strategy: None,
        };
        use tracea::emitter::traits::Emitter;
        let emitter = tracea::emitter::metal::MetalEmitter::detect();
        let source = emitter.generate_from_ir(&ir);
        let kernel_id = runtime.compile(&source, "flash_attention_v2_kernel", DeviceBackend::Metal).unwrap();

        let mut group = c.benchmark_group("metal_attention_hot");
        
        let args = vec![
            KernelArg::Buffer(q_buf),
            KernelArg::Buffer(k_buf),
            KernelArg::Buffer(v_buf),
            KernelArg::Buffer(output_buf),
            KernelArg::Float(1.0 / (d as f32).sqrt()),
        ];

        group.bench_function("launch_hot_attn", |bencher| {
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

criterion_group!(benches, bench_attention_execute);
criterion_main!(benches);
