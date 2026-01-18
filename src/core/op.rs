use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct Dim(pub u32);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Scalar(pub f32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmOp {
    pub m: Dim,
    pub n: Dim,
    pub k: Dim,
    pub alpha: Scalar,
    pub beta: Scalar,
}

impl GemmOp {
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            m: Dim(m),
            n: Dim(n),
            k: Dim(k),
            alpha: Scalar(1.0),
            beta: Scalar(0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EpilogueOp {
    None,
    BiasAdd { bias_ptr: usize },
    ReLU,
    Gelu,
}

#[derive(Debug, Clone, Serialize, Deserialize)] // Cannot derive PartialEq, Eq, Hash due to Scalar(f32) in GemmOp
pub struct FusedGemmOp {
    pub base: GemmOp,
    pub epilogue: Vec<EpilogueOp>,
}

impl FusedGemmOp {
    pub fn new(base: GemmOp) -> Self {
        Self {
            base,
            epilogue: Vec::new(),
        }
    }


    pub fn fuse(&mut self, op: EpilogueOp) {
        self.epilogue.push(op);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LinearOp {
    pub in_features: Dim,
    pub out_features: Dim,
    pub bias: bool,
    pub activation: Vec<EpilogueOp>, // Supports multiple epilogues (e.g. Bias + Gelu)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AttentionOp {
    pub embed_dim: Dim,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub causal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NNOp {
    Linear(LinearOp),
    Attention(AttentionOp),
    // Future: ConvOp
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SoftmaxOp {
    pub axis: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FusedAttentionOp {
    pub b: u32,
    pub s: u32,
    pub d: u32,
    pub h: u32,
    pub dh: u32,
    pub causal: bool,
    pub scale_inv_sqrt_d: bool,
}
