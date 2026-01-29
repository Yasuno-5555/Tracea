use crate::core::ttg::LogicalID;
// use crate::runtime::DeviceBackend; // Removed conflict
use serde::{Deserialize, Serialize};

// ===============================
// Core Context & Decision Types
// ===============================

pub use crate::core::device::{DeviceProfile, BackendType as DeviceBackend};

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
pub enum OperatorTopology {
    Gemm {
        op_id: u64,
        name: String,
        m: u32,
        n: u32,
        k: u32,
        kind: TopologyKind,
    },
    Attention {
        op_id: u64,
        name: String,
        b: u32, 
        s: u32, 
        h: u32, 
        d: u32,
    },
    // For Fusion Testing
    Conv2d {
        op_id: u64,
        name: String,
        n: u32, c: u32, h: u32, w: u32,
        k: u32,
    },
    Relu {
        op_id: u64,
        name: String,
    },
    Elementwise {
        op_id: u64,
        name: String,
        kind: String, // "Add", "Mul"
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTopology {
    pub operators: Vec<OperatorTopology>,
    pub dependencies: Vec<(u64, u64)>, // (producer, consumer)
}

impl OperatorTopology {
    pub fn op_id(&self) -> u64 {
        match self {
            OperatorTopology::Gemm { op_id, .. } => *op_id,
            OperatorTopology::Attention { op_id, .. } => *op_id,
            OperatorTopology::Conv2d { op_id, .. } => *op_id,
            OperatorTopology::Relu { op_id, .. } => *op_id,
            OperatorTopology::Elementwise { op_id, .. } => *op_id,
        }
    }
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
pub struct GraphContext<'a> {
    pub device: &'a DeviceProfile,
    pub graph: &'a GraphTopology,
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
pub enum TilePolicy {
    Gemm {
        operator_id: u64,
        tile_shape: [u32; 3], // [M, N, K]
        tiling_kind: TilingKind,
        activity_pattern: ActivityPattern,
        variant: crate::core::config::GemmVariant,
    },
    Attention {
        operator_id: u64,
        qk_tile: (u32, u32), // (M_Tile, N_Tile) e.g. (64, 64)
        v_tile: (u32, u32),  // (M_Tile, K_Tile) e.g. (64, 32)
        variant: crate::core::config::AttentionVariant,
    },
    Conv {
        operator_id: u64,
    },
    Elementwise {
        operator_id: u64,
    },
}

impl TilePolicy {
    pub fn operator_id(&self) -> u64 {
        match self {
            TilePolicy::Gemm { operator_id, .. } => *operator_id,
            TilePolicy::Attention { operator_id, .. } => *operator_id,
            TilePolicy::Conv { operator_id, .. } => *operator_id,
            TilePolicy::Elementwise { operator_id, .. } => *operator_id,
        }
    }
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

#[derive(Debug, Clone, Default)]
pub struct MemoryAliasPolicy {
    pub output_offset: Option<usize>, // Offset in shared memory pool
    pub workspace_offset: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ExecPolicy {
    pub operator_id: u64,
    pub execution_order: ExecutionOrder,
    pub kernel_binding: KernelBindingPolicy,
    pub backend_hint: BackendExecHint,
    pub memory_alias_hint: MemoryAliasPolicy,
}
