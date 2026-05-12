use tracea::core::config::{PipelineConfig, SpecializedInstruction};
use tracea::runtime::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::policy::types::*;
use tracea::policy::standard::StandardPolicyEngine;
use tracea::policy::engine::PolicyEngine;
use tracea::emitter::cuda::CUDAEmitter;
use tracea::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType, EmissionError};
use tracea::optimizer::HardwareProfile;
use std::sync::Arc;

fn main() {
    println!("=== Policy-Driven GEMM Experiment ===");
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Failed to init runtime");

    // 1. Setup Policy Context
    let device_profile = HardwareProfile::rtx3070().to_device_profile();
    let model_topology = ModelTopology { layer_count: 1 };
    let operator = OperatorTopology::Gemm {
        op_id: 1,
        name: "matmul_1".to_string(),
        m: 256,
        n: 256,
        k: 256,
        batch: 1,
        kind: TopologyKind::Dense,
        epilogue: vec![],
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
    if decision.tile_policies.is_empty() || decision.exec_policies.is_empty() {
        println!("Policy engine returned no candidates; using defaults.");
        return;
    }
    let t_policy = &decision.tile_policies[0];
    let e_policy = &decision.exec_policies[0];

    if let OperatorTopology::Gemm { name, m, n, k, .. } = &operator {
        println!("Policy Decision for '{}':", name);
        println!("  - Tiling: {:?}", t_policy);
        println!("  - Execution Order: {:?}", e_policy.execution_order);
        println!("  - Preferred Block Dim: {:?}", e_policy.backend_hint.preferred_block_dim);
    }

    // 3. Prepare Kernel
    let mut config = PipelineConfig::new(2, 64, 64, 32);
    config.instruction = SpecializedInstruction::CudaMMA;
    config.ttg_enabled = true;
    let (block_x, _, _) = e_policy.backend_hint.preferred_block_dim;
    config.force_num_warps = Some(block_x / 32);

    let m: u32 = 256;
    let n: u32 = 256;
    let k: u32 = 256;

    let emitter = CUDAEmitter::new();
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm { m, n, k, batch: 1, epilogue: vec![] },
        precison: "f16".to_string(),
        tiling: config.clone(),
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };
    let source = emitter.generate_from_ir(&ir).expect("Codegen failed");
    let kernel_id = runtime.compile(&source, "gemm_mma_kernel", DeviceBackend::Cuda).expect("Compile failed");

    // 4. Setup Data
    let size_a = (m * k) as usize;
    let size_b = (k * n) as usize;
    let size_c = (m * n) as usize;
    let a_half = vec![0x3C00u16; size_a]; // 1.0
    let b_half = vec![0x3C00u16; size_b];

    let da = runtime.alloc_u16(size_a, DeviceBackend::Cuda).unwrap();
    let db = runtime.alloc_u16(size_b, DeviceBackend::Cuda).unwrap();
    let dc = runtime.alloc_u16(size_c, DeviceBackend::Cuda).unwrap();
    runtime.copy_to_device(da, &a_half).unwrap();
    runtime.copy_to_device(db, &b_half).unwrap();

    // 5. Launch
    let args = vec![
        KernelArg::Buffer(da),
        KernelArg::Buffer(db),
        KernelArg::Buffer(dc),
        KernelArg::Int(m as i32),
        KernelArg::Int(n as i32),
        KernelArg::Int(k as i32),
    ];

    println!("[Runtime] Launching with automated Policy Integration...");
    runtime.launch(
        kernel_id,
        (4, 1, 1),
        (256, 1, 1),
        0,
        args,
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
    println!("Verification: C[0,0] = {} (Expected {})", val, k as f32);

    if (val - k as f32).abs() < 1.0 {
        println!("✅ Phase C Integration SUCCESS!");
    } else {
        println!("❌ Phase C Integration FAILED (Value mismatch)");
    }
}
