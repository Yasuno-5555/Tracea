use crate::core::op::GemmOp;
use serde::{Serialize, Deserialize};

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
