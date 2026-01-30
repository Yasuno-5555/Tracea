// src/core/manifold.rs

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimension {
    pub id: String,
    pub range_min: i64,
    pub range_max: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationSpace {
    pub dimensions: Vec<Dimension>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineMatrix {
    pub coeffs: Vec<i64>,
    pub constant: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessMap {
    pub tensor_id: u32,
    pub index_expressions: Vec<AffineMatrix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeAtom {
    pub name: String,
    pub domain: IterationSpace,
    pub reads: Vec<AccessMap>,
    pub write: AccessMap,
}

impl ComputeAtom {
    pub fn get_dim_by_id(&self, id: &str) -> Option<&Dimension> {
        self.domain.dimensions.iter().find(|d| d.id == id)
    }

    pub fn from_conv2d(
        n: usize, c: usize, h: usize, w: usize, k: usize,
        r: usize, s: usize, stride: usize, pad: usize, dilation: usize,
    ) -> Self {
        let oh = (h + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
        let ow = (w + 2 * pad - dilation * (s - 1) - 1) / stride + 1;

        let dims = vec![
            Dimension { id: "n".to_string(), range_min: 0, range_max: n as i64 },
            Dimension { id: "oc".to_string(), range_min: 0, range_max: k as i64 },
            Dimension { id: "oh".to_string(), range_min: 0, range_max: oh as i64 },
            Dimension { id: "ow".to_string(), range_min: 0, range_max: ow as i64 },
            Dimension { id: "ic".to_string(), range_min: 0, range_max: c as i64 },
            Dimension { id: "r".to_string(), range_min: 0, range_max: r as i64 },
            Dimension { id: "s".to_string(), range_min: 0, range_max: s as i64 },
        ];

        ComputeAtom {
            name: format!("conv2d_{}x{}x{}x{}", n, k, oh, ow),
            domain: IterationSpace { dimensions: dims },
            reads: vec![
                AccessMap { // Input
                    tensor_id: 0,
                    index_expressions: vec![
                        AffineMatrix { coeffs: vec![1, 0, 0, 0, 0, 0, 0], constant: 0 }, // n
                        AffineMatrix { coeffs: vec![0, 0, 0, 0, 1, 0, 0], constant: 0 }, // ic
                        AffineMatrix { coeffs: vec![0, 0, 1, 0, 0, 1, 0], constant: -(pad as i64) }, // oh*stride + r - pad
                        AffineMatrix { coeffs: vec![0, 0, 0, 1, 0, 0, 1], constant: -(pad as i64) }, // ow*stride + s - pad
                    ],
                },
                AccessMap { // Weight
                    tensor_id: 1,
                    index_expressions: vec![
                        AffineMatrix { coeffs: vec![0, 1, 0, 0, 0, 0, 0], constant: 0 }, // oc
                        AffineMatrix { coeffs: vec![0, 0, 0, 0, 1, 0, 0], constant: 0 }, // ic
                        AffineMatrix { coeffs: vec![0, 0, 0, 0, 0, 1, 0], constant: 0 }, // r
                        AffineMatrix { coeffs: vec![0, 0, 0, 0, 0, 0, 1], constant: 0 }, // s
                    ],
                },
            ],
            write: AccessMap { // Output
                tensor_id: 2,
                index_expressions: vec![
                    AffineMatrix { coeffs: vec![1, 0, 0, 0, 0, 0, 0], constant: 0 }, // n
                    AffineMatrix { coeffs: vec![0, 1, 0, 0, 0, 0, 0], constant: 0 }, // oc
                    AffineMatrix { coeffs: vec![0, 0, 1, 0, 0, 0, 0], constant: 0 }, // oh
                    AffineMatrix { coeffs: vec![0, 0, 0, 1, 0, 0, 0], constant: 0 }, // ow
                ],
            },
        }
    }
}
