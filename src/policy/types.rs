use crate::core::ttg::LogicalID;
use crate::runtime::DeviceBackend;
use serde::{Deserialize, Serialize};

// ===============================
// Core Context & Decision Types
// ===============================

#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub backend: DeviceBackend,
    pub max_threads_per_block: u32,
    pub max_shared_memory: u32,
    pub warp_size: u32,
    pub arch_name: String,
}

#[derive(Debug, Clone)]
pub struct ModelTopology {
    // Placeholder for model graph info
    pub layer_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyKind {
    Dense,
    LowRank { r: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorTopology {
    pub op_id: u64,
    pub name: String,
    pub op_type: String, // "Gemm", "Attention", etc.
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub kind: TopologyKind,
}

#[derive(Debug, Clone)]
pub struct ExecutionHistory {
    // Placeholder for runtime stats
    pub last_latency_us: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct PolicyContext<'a> {
    pub device: &'a DeviceProfile,
    pub model: &'a ModelTopology,
    pub operators: &'a [OperatorTopology],
    pub history: &'a ExecutionHistory,
}

#[derive(Debug, Clone)]
pub struct GlobalPolicyHints {
    pub prefer_fusion: bool,
    pub debug_flags: u32,
}

#[derive(Debug, Clone)]
pub struct PolicyDecision {
    pub tile_policies: Vec<TilePolicy>,       // per-operator
    pub exec_policies: Vec<ExecPolicy>,       // per-operator
    pub global_hints: GlobalPolicyHints,
}

// ===============================
// Tile Policy
// ===============================

#[derive(Debug, Clone)]
pub enum TilingKind {
    Dense,
    Windowed { window: usize },
    BlockSparse { block_m: usize, block_n: usize },
    LocalGlobal { window: usize, num_global: usize },
    LowRank { r: u32, tile_m: u32, tile_n: u32 },
}

#[derive(Debug, Clone)]
pub enum ActivityPattern {
    AllActive,
    DiagonalOnly,
    RandomDrop { keep_ratio: f32 },
    LowRank { r: u32 },
    Custom, // Closure not serializable, flag for custom handling
}

#[derive(Debug, Clone)]
pub struct TilePolicy {
    pub operator_id: u64,
    pub tile_shape: [u32; 3], // [M, N, K]
    pub tiling_kind: TilingKind,
    pub activity_pattern: ActivityPattern,
}

// ===============================
// Execution Policy
// ===============================

#[derive(Debug, Clone)]
pub enum ExecutionOrder {
    RowMajor,
    ColMajor,
    DiagonalWavefront,
    Custom(Vec<LogicalID>), 
}

#[derive(Debug, Clone)]
pub enum KernelKind {
    Gemm,
    Attention,
    LowRankMlp,
    MoE,
    Generic,
}

#[derive(Debug, Clone)]
pub struct KernelBindingPolicy {
    pub kernel_kind: KernelKind,
    pub fuse_with: Vec<u64>, // OperatorIds
}

#[derive(Debug, Clone)]
pub struct BackendExecHint {
    pub preferred_block_dim: (u32, u32, u32),
    pub max_registers_per_thread: Option<u32>,
    pub use_async_copy: bool,
}

#[derive(Debug, Clone)]
pub struct ExecPolicy {
    pub operator_id: u64,
    pub execution_order: ExecutionOrder,
    pub kernel_binding: KernelBindingPolicy,
    pub backend_hint: BackendExecHint,
}
