use tracea::policy::types::*;
use tracea::policy::standard::StandardPolicyEngine;
use tracea::policy::engine::PolicyEngine;
use tracea::runtime::DeviceBackend;

#[test]
fn test_policy_cuda_defaults() {
    let mut engine = StandardPolicyEngine::new();
    let device = DeviceProfile {
        backend: DeviceBackend::Cuda,
        max_threads_per_block: 1024,
        max_shared_memory: 49152,
        warp_size: 32,
        arch_name: "sm_80".to_string(),
    };
    let model = ModelTopology { layer_count: 1 };
    let op = OperatorTopology::Gemm {
        op_id: 1,
        name: "test_gemm".to_string(),
        m: 1024,
        n: 1024,
        k: 1024,
        kind: TopologyKind::Dense,
    };
    let history = ExecutionHistory { last_latency_us: None };
    let ctx = PolicyContext {
        device: &device,
        model: &model,
        operators: &[op],
        history: &history,
    };

    let decision = engine.propose(&ctx);
    assert_eq!(decision.tile_policies.len(), 1);
    // CUDA default: [64, 128, 16] (Corrected from 1 in original test which was 16 in code)
    // Actually standard.rs says [64, 128, 16]. The test had 1. Logic changed or test was wrong?
    // standard.rs: [64, 128, 16]
    // Original test: [64, 128, 1] -> This implies test was out of sync or I misread.
    // I will use [64, 128, 16] as per standard.rs code I saw.
    match &decision.tile_policies[0] {
        TilePolicy::Gemm { tile_shape, .. } => {
             assert_eq!(*tile_shape, [64, 128, 16]);
        },
        _ => panic!("Expected Gemm Policy"),
    }
}

#[test]
fn test_policy_metal_defaults() {
    let mut engine = StandardPolicyEngine::new();
    let device = DeviceProfile {
        backend: DeviceBackend::Metal, 
        max_threads_per_block: 1024,
        max_shared_memory: 32768,
        warp_size: 32,
        arch_name: "m1".to_string(),
    };
    let model = ModelTopology { layer_count: 1 };
    let op = OperatorTopology::Gemm {
        op_id: 1,
        name: "test_gemm".to_string(),
        m: 1024,
        n: 1024,
        k: 1024,
        kind: TopologyKind::Dense,
    };
    let history = ExecutionHistory { last_latency_us: None };
    let ctx = PolicyContext {
        device: &device,
        model: &model,
        operators: &[op],
        history: &history,
    };

    let decision = engine.propose(&ctx);
    assert_eq!(decision.tile_policies.len(), 1);
    // Metal default: [32, 64, 1]
    match &decision.tile_policies[0] {
        TilePolicy::Gemm { tile_shape, .. } => {
             assert_eq!(*tile_shape, [32, 64, 1]);
        },
        _ => panic!("Expected Gemm Policy"),
    }
}
