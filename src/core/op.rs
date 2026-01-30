use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum DimExpr {
    Static(u32),
    Symbol(String),
}

impl From<u32> for DimExpr {
    fn from(v: u32) -> Self {
        DimExpr::Static(v)
    }
}

impl std::fmt::Display for DimExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimExpr::Static(v) => write!(f, "{}", v),
            DimExpr::Symbol(s) => write!(f, "{}", s),
        }
    }
}

impl DimExpr {
    pub fn as_static(&self) -> Option<u32> {
        match self {
            DimExpr::Static(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Scalar(pub f32);

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for Scalar {}
impl std::hash::Hash for Scalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct GemmOp {
    pub m: DimExpr,
    pub n: DimExpr,
    pub k: DimExpr,
    pub alpha: Scalar,
    pub beta: Scalar,
}

impl GemmOp {
    pub fn new<M: Into<DimExpr>, N: Into<DimExpr>, K: Into<DimExpr>>(m: M, n: N, k: K) -> Self {
        Self {
            m: m.into(),
            n: n.into(),
            k: k.into(),
            alpha: Scalar(1.0),
            beta: Scalar(0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpilogueOp {
    None,
    BiasAdd { bias_ptr: usize },
    ReLU,
    Gelu,
    /// SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
    SiLU,
    /// Residual connection: x + residual
    ResidualAdd { residual_ptr: usize },
    /// Combined Bias + SiLU: (x + bias) / (1 + exp(-(x + bias)))
    BiasAddSiLU { bias_ptr: usize },
    /// BatchNorm: (x - mean) * rsqrt(var + eps) * gamma + beta
    BatchNorm {
        op_id: u64,
        gamma_id: u64,
        beta_id: u64,
        mean_id: u64,
        var_id: u64,
        epsilon: f32,
    },
}

impl PartialEq for EpilogueOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (EpilogueOp::None, EpilogueOp::None) => true,
            (EpilogueOp::BiasAdd { bias_ptr: a }, EpilogueOp::BiasAdd { bias_ptr: b }) => a == b,
            (EpilogueOp::ReLU, EpilogueOp::ReLU) => true,
            (EpilogueOp::Gelu, EpilogueOp::Gelu) => true,
            (EpilogueOp::SiLU, EpilogueOp::SiLU) => true,
            (EpilogueOp::ResidualAdd { residual_ptr: a }, EpilogueOp::ResidualAdd { residual_ptr: b }) => a == b,
            (EpilogueOp::BiasAddSiLU { bias_ptr: a }, EpilogueOp::BiasAddSiLU { bias_ptr: b }) => a == b,
            (EpilogueOp::BatchNorm { op_id: a, gamma_id: b, beta_id: c, mean_id: d, var_id: e, epsilon: f },
             EpilogueOp::BatchNorm { op_id: g, gamma_id: h, beta_id: i, mean_id: j, var_id: k, epsilon: l }) => {
                 a == g && b == h && c == i && d == j && e == k && f.to_bits() == l.to_bits()
            },
            _ => false,
        }
    }
}
impl Eq for EpilogueOp {}
impl std::hash::Hash for EpilogueOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            EpilogueOp::None | EpilogueOp::ReLU | EpilogueOp::Gelu | EpilogueOp::SiLU => {},
            EpilogueOp::BiasAdd { bias_ptr } | EpilogueOp::ResidualAdd { residual_ptr: bias_ptr } | EpilogueOp::BiasAddSiLU { bias_ptr } => {
                bias_ptr.hash(state);
            },
            EpilogueOp::BatchNorm { op_id, gamma_id, beta_id, mean_id, var_id, epsilon } => {
                op_id.hash(state); gamma_id.hash(state); beta_id.hash(state); mean_id.hash(state); var_id.hash(state);
                epsilon.to_bits().hash(state);
            }
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
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

    pub batch_size: DimExpr,
    pub in_features: DimExpr,
    pub out_features: DimExpr,
    pub bias: bool,
    pub activation: Vec<EpilogueOp>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AttentionOp {
    pub embed_dim: DimExpr,
    pub num_heads: DimExpr,
    pub head_dim: DimExpr,
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
    pub b: DimExpr,
    pub s: DimExpr,
    pub d: DimExpr,
    pub h: DimExpr,
    pub dh: DimExpr,
    pub causal: bool,
    pub scale_inv_sqrt_d: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ElementwiseType {
    Add,
    Mul,
    Relu,
    Gelu,
    Sigmoid,
    Tanh,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ElementwiseOp {
    pub op_type: ElementwiseType,
    pub n: usize,
}
