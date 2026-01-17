use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpilogueOp {
    None,
    BiasAdd { bias_ptr: usize },
    ReLU,
    Gelu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
