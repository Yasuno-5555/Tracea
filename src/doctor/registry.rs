#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Requirement {
    BackendIs(BackendKind),
    MaxSharedMemAtLeast(u32),
    WarpOrWavefrontIs(u32),
    HasTensorCoreLike,
    SimdWidthAtLeast(u32),
    Precision(super::engine::PrecisionPolicy),
    // Legacy support for existing variants if needed, or mapped to new ones
    SmAtLeast(u32), 
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Preference {
    PreferTensorCoreLike,
    PreferLargerSharedMem,
    // Legacy
    PreferLowerRegUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BackendKind {
    Cuda,
    Rocm,
    Metal,
    Cpu,
}

#[derive(Debug, Clone)]
pub struct KernelVariant {
    pub id: &'static str,
    pub kernel_id: &'static str,
    pub priority: u32,

    pub hard_requirements: &'static [Requirement],
    pub soft_preferences: &'static [Preference],

    pub backend: BackendKind,
}

pub fn get_variants_for(kernel_id: &str) -> Vec<&'static KernelVariant> {
    REGISTRY
        .iter()
        .filter(|v| v.kernel_id == kernel_id)
        .collect()
}

pub fn get_variants_for_id(variant_id: &str) -> Vec<&'static KernelVariant> {
    REGISTRY
        .iter()
        .filter(|v| v.id == variant_id)
        .collect()
}

static REGISTRY: &[KernelVariant] = &[
    // --- CUDA Variants ---
    KernelVariant {
        id: "fa2_cuda_v2",
        kernel_id: "flash_attention_v2_kernel",
        priority: 100,
        hard_requirements: &[
            Requirement::BackendIs(BackendKind::Cuda),
            Requirement::SmAtLeast(80),
            Requirement::HasTensorCoreLike,
        ],
        soft_preferences: &[Preference::PreferTensorCoreLike],
        backend: BackendKind::Cuda,
    },
    KernelVariant {
        id: "gemm_cuda_v2",
        kernel_id: "gemm_kernel",
        priority: 100,
        hard_requirements: &[
            Requirement::BackendIs(BackendKind::Cuda),
        ],
        soft_preferences: &[],
        backend: BackendKind::Cuda,
    },
    
    // --- ROCm Variants ---
    KernelVariant {
        id: "gemm_rocm_matrix_core",
        kernel_id: "gemm",
        priority: 100,
        hard_requirements: &[
            Requirement::BackendIs(BackendKind::Rocm),
            Requirement::HasTensorCoreLike,
        ],
        soft_preferences: &[Preference::PreferTensorCoreLike],
        backend: BackendKind::Rocm,
    },
    KernelVariant {
        id: "fa2_rocm_matrix_core",
        kernel_id: "flash_attention_2",
        priority: 10,
        hard_requirements: &[
            Requirement::BackendIs(BackendKind::Rocm),
            Requirement::HasTensorCoreLike,
            Requirement::MaxSharedMemAtLeast(65536),
        ],
        soft_preferences: &[Preference::PreferTensorCoreLike],
        backend: BackendKind::Rocm,
    },
    // ... (existing variants) ...
    KernelVariant {
        id: "gemm_rocm_standard",
        kernel_id: "gemm",
        priority: 10,
        hard_requirements: &[Requirement::BackendIs(BackendKind::Rocm)],
        soft_preferences: &[],
        backend: BackendKind::Rocm,
    },

    // --- Metal Variants ---
    KernelVariant {
        id: "gemm_metal_simdgroup",
        kernel_id: "gemm",
        priority: 100,
        hard_requirements: &[
            Requirement::BackendIs(BackendKind::Metal),
            Requirement::HasTensorCoreLike,
        ],
        soft_preferences: &[Preference::PreferTensorCoreLike],
        backend: BackendKind::Metal,
    },
    KernelVariant {
        id: "fa2_metal_simdgroup",
        kernel_id: "flash_attention_2",
        priority: 10,
        hard_requirements: &[
            Requirement::BackendIs(BackendKind::Metal),
            Requirement::HasTensorCoreLike,
            Requirement::MaxSharedMemAtLeast(32768),
        ],
        soft_preferences: &[Preference::PreferTensorCoreLike],
        backend: BackendKind::Metal,
    },
    KernelVariant {
        id: "gemm_metal_standard",
        kernel_id: "gemm",
        priority: 10,
        hard_requirements: &[Requirement::BackendIs(BackendKind::Metal)],
        soft_preferences: &[],
        backend: BackendKind::Metal,
    },

    // --- CPU Variants ---
    KernelVariant {
        id: "fa2_cpu_simd",
        kernel_id: "flash_attention_2",
        priority: 90,
        hard_requirements: &[
            Requirement::BackendIs(BackendKind::Cpu),
            Requirement::SimdWidthAtLeast(256),
        ],
        soft_preferences: &[],
        backend: BackendKind::Cpu,
    },
    KernelVariant {
        id: "fa2_cpu_naive",
        kernel_id: "flash_attention_2",
        priority: 100,
        hard_requirements: &[Requirement::BackendIs(BackendKind::Cpu)],
        soft_preferences: &[],
        backend: BackendKind::Cpu,
    },
    KernelVariant {
        id: "gemm_cpu_simd",
        kernel_id: "gemm",
        priority: 30,
        hard_requirements: &[
            Requirement::BackendIs(BackendKind::Cpu),
            Requirement::SimdWidthAtLeast(256),
        ],
        soft_preferences: &[],
        backend: BackendKind::Cpu,
    },
    KernelVariant {
        id: "gemm_cpu_scalar",
        kernel_id: "gemm",
        priority: 40,
        hard_requirements: &[Requirement::BackendIs(BackendKind::Cpu)],
        soft_preferences: &[],
        backend: BackendKind::Cpu,
    },

    // --- Legacy / Generic Fallback ---
    KernelVariant {
        id: "softmax_safe",
        kernel_id: "softmax",
        priority: 1,
        hard_requirements: &[],
        soft_preferences: &[],
        backend: BackendKind::Cpu, // Fallback to safe CPU impl usually
    },
];
