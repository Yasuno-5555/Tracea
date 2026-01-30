// src/core/manifold.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Tensor identifier in the manifold space
pub type TensorId = u32;

/// A dimension in the N-dimensional computation space
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Dimension {
    pub id: String,
    pub range_min: i64,
    pub range_max: i64,
}

/// N-dimensional iteration space (Computation Domain)
/// Contains range definitions for each axis.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IterationSpace {
    pub dimensions: Vec<Dimension>,
    pub constraints: Vec<crate::core::polyhedral::Constraint>,
}

/// Affine representation of index calculations.
/// Mapping from Iteration Space -> Tensor Index Space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineMatrix {
    /// Coefficients for each dimension in IterationSpace
    pub coeffs: Vec<i64>,
    pub constant: i64,
}

/// Defines how a tensor is accessed within the iteration space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessMap {
    pub tensor_id: TensorId,
    /// One expression per tensor dimension
    pub index_expressions: Vec<AffineMatrix>,
}

/// Atomic mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathOp {
    FMA,    // Floating point multiply-accumulate
    Max,    // Used for ReLU/Softmax
    Add,
    Mul,
    Exp,
    Div,
    Custom(String),
}

/// The "DNA" of a computation.
/// Describes WHAT is being calculated without specifying HOW (loop order, hardware).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeAtom {
    pub name: String,
    pub domain: IterationSpace,
    pub reads: Vec<AccessMap>,
    pub write: AccessMap,
    pub op: MathOp,
}

impl ComputeAtom {
    pub fn get_dim_by_id(&self, id: &str) -> Option<&Dimension> {
        self.domain.dimensions.iter().find(|d| d.id == id)
    }

    pub fn get_dim_index(&self, id: &str) -> Option<usize> {
        self.domain.dimensions.iter().position(|d| d.id == id)
    }

    /// Conversion helper: Standard Conv2d (NHWC) to ComputeAtom
    pub fn from_conv2d(
        n: u32, ic: u32, ih: u32, iw: u32, oc: u32,
        r: u32, s: u32, stride: u32, pad: u32, dilation: u32,
    ) -> Self {
        let oh = (ih + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
        let ow = (iw + 2 * pad - dilation * (s - 1) - 1) / stride + 1;

        let dims = vec![
            Dimension { id: "n".into(), range_min: 0, range_max: n as i64 },
            Dimension { id: "oh".into(), range_min: 0, range_max: oh as i64 },
            Dimension { id: "ow".into(), range_min: 0, range_max: ow as i64 },
            Dimension { id: "oc".into(), range_min: 0, range_max: oc as i64 },
            Dimension { id: "r".into(), range_min: 0, range_max: r as i64 },
            Dimension { id: "s".into(), range_min: 0, range_max: s as i64 },
            Dimension { id: "ic".into(), range_min: 0, range_max: ic as i64 },
        ];

        // Access Maps: 
        // 1. Input [n, oh*stride - pad + r*dilation, ow*stride - pad + s*dilation, ic]
        let input_map = AccessMap {
            tensor_id: 0, // Input
            index_expressions: vec![
                AffineMatrix { coeffs: vec![1, 0, 0, 0, 0, 0, 0], constant: 0 }, // n
                AffineMatrix { coeffs: vec![0, stride as i64, 0, 0, dilation as i64, 0, 0], constant: -(pad as i64) }, // hi
                AffineMatrix { coeffs: vec![0, 0, stride as i64, 0, 0, dilation as i64, 0], constant: -(pad as i64) }, // wi
                AffineMatrix { coeffs: vec![0, 0, 0, 0, 0, 0, 1], constant: 0 }, // ic
            ],
        };

        // 2. Weight [oc, r, s, ic]
        let weight_map = AccessMap {
            tensor_id: 1, // Weight
            index_expressions: vec![
                AffineMatrix { coeffs: vec![0, 0, 0, 1, 0, 0, 0], constant: 0 }, // oc
                AffineMatrix { coeffs: vec![0, 0, 0, 0, 1, 0, 0], constant: 0 }, // r
                AffineMatrix { coeffs: vec![0, 0, 0, 0, 0, 1, 0], constant: 0 }, // s
                AffineMatrix { coeffs: vec![0, 0, 0, 0, 0, 0, 1], constant: 0 }, // ic
            ],
        };

        // 3. Output [n, oh, ow, oc]
        let output_map = AccessMap {
            tensor_id: 2, // Output
            index_expressions: vec![
                AffineMatrix { coeffs: vec![1, 0, 0, 0, 0, 0, 0], constant: 0 }, // n
                AffineMatrix { coeffs: vec![0, 1, 0, 0, 0, 0, 0], constant: 0 }, // oh
                AffineMatrix { coeffs: vec![0, 0, 1, 0, 0, 0, 0], constant: 0 }, // ow
                AffineMatrix { coeffs: vec![0, 0, 0, 1, 0, 0, 0], constant: 0 }, // oc
            ],
        };

        ComputeAtom {
            name: "conv2d_fma".into(),
            domain: IterationSpace { dimensions: dims, constraints: Vec::new() },
            reads: vec![input_map, weight_map],
            write: output_map,
            op: MathOp::FMA,
        }
    }

    /// Conversion helper: Standard GEMM to ComputeAtom
    pub fn from_gemm(m: u32, n: u32, k: u32, batch: u32) -> Self {
        let dims = vec![
            Dimension { id: "b".into(), range_min: 0, range_max: batch as i64 },
            Dimension { id: "m".into(), range_min: 0, range_max: m as i64 },
            Dimension { id: "n".into(), range_min: 0, range_max: n as i64 },
            Dimension { id: "k_idx".into(), range_min: 0, range_max: k as i64 },
        ];

        // Access Maps:
        // 1. A [b, m, k_idx]
        let a_map = AccessMap {
            tensor_id: 0,
            index_expressions: vec![
                AffineMatrix { coeffs: vec![1, 0, 0, 0], constant: 0 }, // b
                AffineMatrix { coeffs: vec![0, 1, 0, 0], constant: 0 }, // m
                AffineMatrix { coeffs: vec![0, 0, 0, 1], constant: 0 }, // k_idx
            ],
        };

        // 2. B [b, k_idx, n]
        let b_map = AccessMap {
            tensor_id: 1,
            index_expressions: vec![
                AffineMatrix { coeffs: vec![1, 0, 0, 0], constant: 0 }, // b
                AffineMatrix { coeffs: vec![0, 0, 0, 1], constant: 0 }, // k_idx
                AffineMatrix { coeffs: vec![0, 0, 1, 0], constant: 0 }, // n
            ],
        };

        // 3. C [b, m, n]
        let c_map = AccessMap {
            tensor_id: 2,
            index_expressions: vec![
                AffineMatrix { coeffs: vec![1, 0, 0, 0], constant: 0 }, // b
                AffineMatrix { coeffs: vec![0, 1, 0, 0], constant: 0 }, // m
                AffineMatrix { coeffs: vec![0, 0, 1, 0], constant: 0 }, // n
            ],
        };

        ComputeAtom {
            name: "gemm_fma".into(),
            domain: IterationSpace { dimensions: dims, constraints: Vec::new() },
            reads: vec![a_map, b_map],
            write: c_map,
            op: MathOp::FMA,
        }
    }
}
