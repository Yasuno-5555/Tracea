use crate::core::ttg::LogicalID;
// use crate::runtime::DeviceBackend; // Removed conflict
use serde::{Deserialize, Serialize};

// ===============================
// Core Context & Decision Types
// ===============================

pub use crate::core::device::{DeviceProfile, BackendType as DeviceBackend};
use std::hash::{Hash, Hasher};

fn hash_f32<H: Hasher>(x: &f32, state: &mut H) {
    x.to_bits().hash(state);
}

#[derive(Debug, Clone)]
pub struct ModelTopology {
    // Placeholder for model graph info
    pub layer_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
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
        batch: u32,
        kind: TopologyKind,
        epilogue: Vec<crate::core::op::EpilogueOp>,
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
        r: u32, s: u32, stride: u32, padding: u32,
        epilogue: Vec<crate::core::op::EpilogueOp>, // Added for fusion
    },
    Relu {
        op_id: u64,
        name: String,
        n: usize,
    },
    Elementwise {
        op_id: u64,
        name: String,
        kind: String, // "Add", "Mul"
        n: usize,
    },
    Input {
        op_id: u64,
        name: String,
    },
    Softmax {
        op_id: u64,
        name: String,
        axis: i32,
    },
    BatchNorm {
        op_id: u64,
        name: String,
        n: usize, c: usize, h: usize, w: usize,
        epsilon: f32, momentum: f32,
    },
    GlobalAveragePool {
        op_id: u64,
        name: String,
        n: usize, c: usize, h: usize, w: usize,
    },
    Linear {
        op_id: u64,
        name: String,
        batch: usize,
        m: usize, n: usize, k: usize, // M=1 for typical inference
        epilogue: Vec<crate::core::op::EpilogueOp>, // Added for fusion
    },
}

impl Hash for OperatorTopology {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            OperatorTopology::Gemm { op_id, name, m, n, k, batch, kind, epilogue } => {
                op_id.hash(state); name.hash(state); m.hash(state); n.hash(state); k.hash(state); batch.hash(state); kind.hash(state); epilogue.hash(state);
            },
            OperatorTopology::Attention { op_id, name, b, s, h, d } => {
                op_id.hash(state); name.hash(state); b.hash(state); s.hash(state); h.hash(state); d.hash(state);
            },
            OperatorTopology::Conv2d { op_id, name, n, c, h, w, k, r, s, stride, padding, epilogue } => {
                op_id.hash(state); name.hash(state); n.hash(state); c.hash(state); h.hash(state); w.hash(state); k.hash(state); r.hash(state); s.hash(state); stride.hash(state); padding.hash(state); epilogue.hash(state);
            },
            OperatorTopology::Relu { op_id, name, n } => {
                op_id.hash(state); name.hash(state); n.hash(state);
            },
            OperatorTopology::Elementwise { op_id, name, kind, n } => {
                op_id.hash(state); name.hash(state); kind.hash(state); n.hash(state);
            },
            OperatorTopology::Input { op_id, name } => {
                op_id.hash(state); name.hash(state);
            },
            OperatorTopology::Softmax { op_id, name, axis } => {
                op_id.hash(state); name.hash(state); axis.hash(state);
            },
            OperatorTopology::BatchNorm { op_id, name, n, c, h, w, epsilon, momentum } => {
                op_id.hash(state); name.hash(state); n.hash(state); c.hash(state); h.hash(state); w.hash(state);
                hash_f32(epsilon, state); hash_f32(momentum, state);
            },
            OperatorTopology::GlobalAveragePool { op_id, name, n, c, h, w } => {
                op_id.hash(state); name.hash(state); n.hash(state); c.hash(state); h.hash(state); w.hash(state);
            },
            OperatorTopology::Linear { op_id, name, batch, m, n, k, epilogue } => {
                op_id.hash(state); name.hash(state); batch.hash(state); m.hash(state); n.hash(state); k.hash(state); epilogue.hash(state);
            },
        }
    }
}

// Implement PartialEq manually for OperatorTopology to handle floats if needed, 
// or rely on a wrapper. For now, we assume simple equality is sufficient if we hash properly.
// However, strictly speaking, f32 doesn't implement Eq. We need Eq for HashMap keys.
impl PartialEq for OperatorTopology {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (OperatorTopology::Gemm { op_id: a, name: b, m: c, n: d, k: e, batch: f, kind: g, epilogue: h },
             OperatorTopology::Gemm { op_id: i, name: j, m: k, n: l, k: m, batch: n, kind: o, epilogue: p }) => {
                 a == i && b == j && c == k && d == l && e == m && f == n && g == o && h == p
            },
            (OperatorTopology::Attention { op_id: a, name: b, b: c, s: d, h: e, d: f },
             OperatorTopology::Attention { op_id: g, name: h, b: i, s: j, h: k, d: l }) => {
                 a == g && b == h && c == i && d == j && e == k && f == l
            },
            (OperatorTopology::Conv2d { op_id: a, name: b, n: c, c: d, h: e, w: f, k: g, r: h, s: i, stride: j, padding: k, epilogue: l },
             OperatorTopology::Conv2d { op_id: m, name: n, n: o, c: p, h: q, w: r, k: s, r: t, s: u, stride: v, padding: w, epilogue: x }) => {
                 a == m && b == n && c == o && d == p && e == q && f == r && g == s && h == t && i == u && j == v && k == w && l == x
            },
            (OperatorTopology::Relu { op_id: a, name: b, n: c },
             OperatorTopology::Relu { op_id: d, name: e, n: f }) => {
                 a == d && b == e && c == f
            },
            (OperatorTopology::Elementwise { op_id: a, name: b, kind: c, n: d },
             OperatorTopology::Elementwise { op_id: e, name: f, kind: g, n: h }) => {
                 a == e && b == f && c == g && d == h
            },
            (OperatorTopology::Input { op_id: a, name: b },
             OperatorTopology::Input { op_id: c, name: d }) => {
                 a == c && b == d
            },
            (OperatorTopology::Softmax { op_id: a, name: b, axis: c },
             OperatorTopology::Softmax { op_id: d, name: e, axis: f }) => {
                 a == d && b == e && c == f
            },
            (OperatorTopology::BatchNorm { op_id: a, name: b, n: c, c: d, h: e, w: f, epsilon: g, momentum: h },
             OperatorTopology::BatchNorm { op_id: i, name: j, n: k, c: l, h: m, w: n, epsilon: o, momentum: p }) => {
                 a == i && b == j && c == k && d == l && e == m && f == n && g.to_bits() == o.to_bits() && h.to_bits() == p.to_bits()
            },
            (OperatorTopology::GlobalAveragePool { op_id: a, name: b, n: c, c: d, h: e, w: f },
             OperatorTopology::GlobalAveragePool { op_id: g, name: h, n: i, c: j, h: k, w: l }) => {
                 a == g && b == h && c == i && d == j && e == k && f == l
            },
            (OperatorTopology::Linear { op_id: a, name: b, batch: c, m: d, n: e, k: f, epilogue: g },
             OperatorTopology::Linear { op_id: h, name: i, batch: j, m: k, n: l, k: m, epilogue: n }) => {
                 a == h && b == i && c == j && d == k && e == l && f == m && g == n
            },
            _ => false,
        }
    }
}
impl Eq for OperatorTopology {}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
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
            OperatorTopology::Input { op_id, .. } => *op_id,
            OperatorTopology::Softmax { op_id, .. } => *op_id,
            OperatorTopology::BatchNorm { op_id, .. } => *op_id,
            OperatorTopology::GlobalAveragePool { op_id, .. } => *op_id,
            OperatorTopology::Linear { op_id, .. } => *op_id,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            OperatorTopology::Gemm { name, .. } => name,
            OperatorTopology::Attention { name, .. } => name,
            OperatorTopology::Conv2d { name, .. } => name,
            OperatorTopology::Relu { name, .. } => name,
            OperatorTopology::Elementwise { name, .. } => name,
            OperatorTopology::Input { name, .. } => name,
            OperatorTopology::Softmax { name, .. } => name,
            OperatorTopology::BatchNorm { name, .. } => name,
            OperatorTopology::GlobalAveragePool { name, .. } => name,
            OperatorTopology::Linear { name, .. } => name,
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
        tile_m: u32,          // Output tile height
        tile_n: u32,          // Output tile width
        tile_c: u32,          // Input channel tile
        use_simd: bool,       // Use simd_matrix_multiply
        use_double_buffer: bool,
    },
    Elementwise {
        operator_id: u64,
    },
    BatchNorm {
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
            TilePolicy::BatchNorm { operator_id, .. } => *operator_id,
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
