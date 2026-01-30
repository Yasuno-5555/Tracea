use tracea::policy::types::*;
use tracea::policy::standard::StandardPolicyEngine;
use tracea::policy::engine::PolicyEngine;
use tracea::core::device::BackendType;

#[test]
fn test_policy_cuda_defaults() {
    let mut engine = StandardPolicyEngine::new();
    let device = DeviceProfile {
        backend: BackendType::Cuda,
        name: "sm_80".to_string(),
        max_threads_per_block: 1024,
        local_memory_size: 49152,
        simd_width: 32,
        has_tensor_cores: true,
        has_fp16_storage: true,
        texture_alignment: 512,
    };
    let model = ModelTopology { layer_count: 1 };
    let op = OperatorTopology::Gemm {
        op_id: 1,
        name: "test_gemm".to_string(),
        m: 1024,
        n: 1024,
        k: 1024,
        batch: 1,
        kind: TopologyKind::Dense,
        epilogue: vec![],
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
    
    // CUDA default with has_tensor_cores: [64, 128, 16]
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
        backend: BackendType::Metal, 
        name: "m1".to_string(),
        max_threads_per_block: 1024,
        local_memory_size: 32768,
        simd_width: 32,
        has_tensor_cores: true,
        has_fp16_storage: true,
        texture_alignment: 256,
    };
    let model = ModelTopology { layer_count: 1 };
    let op = OperatorTopology::Gemm {
        op_id: 1,
        name: "test_gemm".to_string(),
        m: 1024,
        n: 1024,
        k: 1024,
        batch: 1,
        kind: TopologyKind::Dense,
        epilogue: vec![],
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
    
    // Metal default with has_tensor_cores && local_mem >= 48k? No, M1 has 32k in this mock.
    // Standard logic says:
    // [64, 128, 16] if TC && local_mem >= 48k
    // [32, 64, 1] if simd_width == 32 (Metal default)
    match &decision.tile_policies[0] {
        TilePolicy::Gemm { tile_shape, .. } => {
             assert_eq!(*tile_shape, [32, 64, 1]);
        },
        _ => panic!("Expected Gemm Policy"),
    }
}
