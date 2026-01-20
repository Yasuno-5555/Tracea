use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use crate::semantic::transition::SyncRequirement;

pub struct MetalEmitter {
    pub device_name: String,
    pub max_threadgroup_memory: usize,
}

impl MetalEmitter {
    pub fn detect() -> Self {
        // Placeholder for Metal discovery
        // On non-macOS, this will just be a dummy
        Self {
            device_name: "Apple M-Series (Simulated)".to_string(),
            max_threadgroup_memory: 32768,
        }
    }

    pub fn generate_gemm(&self, config: crate::PipelineConfig) -> String {
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();
        
        format!(r#"
{primitives}

kernel void gemm_metal_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {{
    // Threadgroup Memory (Shared)
    threadgroup half sA[{mt} * {kt}];
    threadgroup half sB[{kt} * {nt}];

    // simdgroup_matrix is usually 8x8 or 16x16
    // We assume 8x8 for compatibility across early M1/M2
    simdgroup_float8x8 acc;
    #pragma unroll
    for(int i=0; i<1; ++i) acc = simdgroup_float8x8(0.0f);

    for (uint k_step = 0; k_step < K; k_step += {kt}) {{
        // Load data into threadgroup memory
        // (Simplified parallel load)
        uint t_idx = tid.y * 32 + tid.x;
        for (uint i = t_idx; i < {mt} * {kt}; i += 32*4) {{
             uint r = i / {kt}; uint c = i % {kt};
             if (bid.y * {mt} + r < M && k_step + c < K)
                 sA[i] = A[(bid.y * {mt} + r) * K + (k_step + c)];
             else sA[i] = 0;
        }}
        for (uint i = t_idx; i < {kt} * {nt}; i += 32*4) {{
             uint r = i / {nt}; uint c = i % {nt};
             if (k_step + r < K && bid.x * {nt} + c < N)
                 sB[i] = B[(k_step + r) * N + (bid.x * {nt} + c)];
             else sB[i] = 0;
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Matrix Multiply-Accumulate block
        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;
        
        // Simdgroup distribution: Assume 4 simdgroups (128 threads)
        // Each simdgroup handles a 16x16 or 8x8 sub-tile
        uint sg_r = (simd_id / 2) * 8;
        uint sg_c = (simd_id % 2) * 8;

        for (uint ki = 0; ki < {kt}; ki += 8) {{
            simdgroup_load(ma, &sA[sg_r * {kt} + ki], {kt});
            simdgroup_load(mb, &sB[ki * {nt} + sg_c], {nt});
            simdgroup_multiply_accumulate(acc, ma, mb, acc);
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Epilogue: Store results
    uint sg_r = (simd_id / 2) * 8;
    uint sg_c = (simd_id % 2) * 8;
    simdgroup_store(acc, (device float*)&C[(bid.y * {mt} + sg_r) * N + (bid.x * {nt} + sg_c)], N);
}}
"#, mt=mt, nt=nt, kt=kt, primitives=primitives)
    }
}

impl Emitter for MetalEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::Barrier => "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string(),
            _ => String::new(),
        }
    }

    fn generate_from_ir(&self, ir: &UnifiedOpIR) -> String {
        match ir.op_type {
            UnifiedOpType::Gemm { .. } => self.generate_gemm(ir.tiling.clone()),
            UnifiedOpType::FusedAttention { .. } => {
                "// Metal FA2 not yet implemented in Unified Emitter\n".to_string()
            }
            UnifiedOpType::Elementwise { .. } => {
                 panic!("Elementwise Ops should be handled by UniversalEmitter for now.");
            }
            UnifiedOpType::Conv2d { .. } => {
                panic!("Conv2d Ops should be handled by UniversalEmitter.");
            }
        }
    }
}
