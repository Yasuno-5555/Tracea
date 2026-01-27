use super::registry::BackendKind;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendCapabilities {
    pub backend: BackendKind,

    // Common
    pub max_shared_mem: u32,        // CUDA: smem/block, ROCm: LDS, Metal: threadgroup mem, CPU: L1/L2
    pub warp_or_wavefront: u32,     // CUDA: 32, ROCm: 32/64, Metal: simdgroup size, CPU: 1
    pub has_tensor_core_like: bool, // CUDA: Tensor Core, ROCm: Matrix Core, Metal: simdgroup_matrix

    // GPU specific
    pub arch_code: u32,             // sm_xx, gfx_xx, metal_family_xx
    pub driver_or_runtime_version: u32,

    // CPU specific
    pub simd_width_bits: u32,       // 128/256/512
    pub core_count: u32,

    // Real-time Metrics (Phase 4)
    pub current_occupancy: Option<f32>,
    pub register_pressure: Option<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TraceaCapabilities {
    pub env_id: [u8; 32],
    pub backends: Vec<BackendCapabilities>,
}

impl TraceaCapabilities {
    /// Helper to find a specific backend capability
    pub fn get_backend(&self, kind: BackendKind) -> Option<&BackendCapabilities> {
        self.backends.iter().find(|b| b.backend == kind)
    }
}

