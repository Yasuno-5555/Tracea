use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FragmentType {
    Scalar,
    Vector4,
    /// Tensor Core 16x16 Tile (Standard Ampere/A100 fragment)
    TC16x16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FragmentRole {
    OperandA,
    OperandB,
    Accumulator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fragment {
    pub frag_type: FragmentType,
    pub role: FragmentRole,
    pub precision: String, // e.g., "f16", "f32"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FragmentOp {
    /// Load from SMEM via ldmatrix
    LoadTC { 
        is_x4: bool,
        transposed: bool,
    },
    /// Tensor Core FMA via mma.sync
    MMA {
        m: u32,
        n: u32,
        k: u32,
    },
    /// Scalar FMA for CUDA Cores
    FMA,
}
