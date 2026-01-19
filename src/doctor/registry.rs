use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cuda,
    Rocm,
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub id: String,
    pub backend: BackendKind, // Changed from DeviceBackend
    pub compile_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelVariant {
    pub id: &'static str,
    pub backend: BackendKind,
    pub hard_requirements: Vec<Requirement>,
    pub soft_preferences: Vec<Preference>,
    pub priority: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Requirement {
    BackendIs(BackendKind),
    SmAtLeast(u32), // e.g., 80 for sm_80
    HasTensorCoreLike,
    MaxSharedMemAtLeast(u32),
    WarpOrWavefrontIs(u32),
    SimdWidthAtLeast(u32),
    Precision(super::engine::PrecisionPolicy),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Preference {
    PreferTensorCoreLike,
    PreferLargerSharedMem,
    PreferLowerRegUsage,
}

pub fn get_variants_for(kernel_id: &str) -> Vec<KernelVariant> {
     // Stub implementation of registry lookup
     if kernel_id == "gemm_kernel" {
         vec![
             KernelVariant {
                 id: "gemm_cuda_v1",
                 backend: BackendKind::Cuda,
                 hard_requirements: vec![Requirement::BackendIs(BackendKind::Cuda)],
                 soft_preferences: vec![Preference::PreferTensorCoreLike],
                 priority: 10,
             }
         ]
     } else {
         vec![]
     }
}

pub fn get_variants_for_id(vid: &str) -> Vec<Variant> {
    // Stub implementation
    vec![
        Variant {
            id: vid.to_string(),
            backend: BackendKind::Cuda,
            compile_strategy: "JIT".to_string(),
        }
    ]
}
