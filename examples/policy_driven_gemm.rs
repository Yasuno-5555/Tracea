use tracea::core::config::{PipelineConfig, SpecializedInstruction};
use tracea::runtime::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::policy::types::*;
use tracea::policy::standard::StandardPolicyEngine;
use tracea::policy::engine::PolicyEngine;
use tracea::emitter::cuda::CUDAEmitter;
use tracea::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use std::sync::Arc;

fn main() {
    println!("=== Policy-Driven GEMM Experiment ===");
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Failed to init runtime");
    
    // 1. Setup Policy Context
    let device_profile = DeviceProfile {
        backend: DeviceBackend::Cuda,
        max_threads_per_block: 1024,
        max_shared_memory: 49152,
        warp_size: 32,
        arch_name: "sm_86".to_string(),
    };
    let model_topology = ModelTopology { layer_count: 1 };
    let operator = OperatorTopology {
        op_id: 1,
        name: "matmul_1".to_string(),
        op_type: "Gemm".to_string(),
        m: 256, n: 256, k: 256,
        kind: TopologyKind::Dense,
    };
    let history = ExecutionHistory { last_latency_us: None };
    
    let ctx = PolicyContext {
        device: &device_profile,
        model: &model_topology,
        operators: &[operator.clone()],
        history: &history,
    };
    
    // 2. Get Decision from Policy Engine
    let mut engine = StandardPolicyEngine::new();
    let decision = engine.propose(&ctx);
    let t_policy = &decision.tile_policies[0];
    let e_policy = &decision.exec_policies[0];
    
    println!("Policy Decision for '{}':", operator.name);
    println!("  - Tiling: {:?} | Shape: {:?}", t_policy.tiling_kind, t_policy.tile_shape);
    println!("  - Execution Order: {:?}", e_policy.execution_order);
    println!("  - Preferred Block Dim: {:?}", e_policy.backend_hint.preferred_block_dim);
    
    // 3. Prepare Kernel (Automated configuration derivation)
    let mut config = PipelineConfig::new(2, t_policy.tile_shape[0], t_policy.tile_shape[1], t_policy.tile_shape[2]);
    config.instruction = SpecializedInstruction::CudaMMA;
    config.ttg_enabled = true;
    config.force_num_warps = Some(e_policy.backend_hint.preferred_block_dim.0 / 32); 
    
    let emitter = CUDAEmitter::new();
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm { m: operator.m, n: operator.n, k: operator.k },
        precison: "f16".to_string(),
        tiling: config.clone(),
        conv_magic_strategy: None,
    };
    let source = emitter.generate_from_ir(&ir);
    let kernel_id = runtime.compile(&source, "gemm_mma_kernel", DeviceBackend::Cuda).expect("Compile failed");
    
    // 4. Setup Data
    let size_a = (operator.m * operator.k) as usize;
    let size_b = (operator.k * operator.n) as usize;
    let size_c = (operator.m * operator.n) as usize;
    let a_half = vec![0x3C00u16; size_a]; // 1.0
    let b_half = vec![0x3C00u16; size_b];
    
    let da = runtime.alloc_u16(size_a, DeviceBackend::Cuda).unwrap();
    let db = runtime.alloc_u16(size_b, DeviceBackend::Cuda).unwrap();
    let dc = runtime.alloc_u16(size_c, DeviceBackend::Cuda).unwrap();
    runtime.copy_to_device(da, &a_half).unwrap();
    runtime.copy_to_device(db, &b_half).unwrap();
    
    // 5. Automated Launch using Policy Integration
    let args = vec![
        KernelArg::Buffer(da),
        KernelArg::Buffer(db),
        KernelArg::Buffer(dc),
        KernelArg::Int(operator.m as i32),
        KernelArg::Int(operator.n as i32),
        KernelArg::Int(operator.k as i32),
    ];
    
    println!("[Runtime] Launching with automated Policy Integration...");
    runtime.launch_with_policy(
        kernel_id,
        args,
        &operator,
        t_policy,
        e_policy,
        vec![],
        DeviceBackend::Cuda
    ).expect("Policy driven launch failed");
    
    runtime.synchronize();
    
    // Verification
    let c_bytes = runtime.read_buffer(dc).expect("Read back failed");
    let mut c_half = vec![0u16; size_c];
    unsafe {
        std::ptr::copy_nonoverlapping(c_bytes.as_ptr(), c_half.as_mut_ptr() as *mut u8, size_c * 2);
    }
    
    // Check one value (0,0) should be 256.0
    use half::f16;
    let val = f16::from_bits(c_half[0]).to_f32();
    println!("Verification: C[0,0] = {} (Expected {})", val, operator.k as f32);
    
    if (val - operator.k as f32).abs() < 1.0 {
        println!("✅ Phase C Integration SUCCESS!");
    } else {
        println!("❌ Phase C Integration FAILED (Value mismatch)");
    }
}
