use crate::semantic::transition::SyncRequirement;
use crate::core::op::EpilogueOp;
use crate::semantic::fragment::{FragmentOp, Fragment};

pub trait Emitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String;
    fn emit_epilogue(&self, _ops: &[EpilogueOp], _acc_name: &str, _global_n: &str) -> String {
        "".to_string()
    }
    fn emit_fragment_op(&self, _op: FragmentOp, _frags: &[Fragment]) -> String {
        "".to_string()
    }
    fn generate_from_ir(&self, ir: &UnifiedOpIR) -> String;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackendTarget {
    CudaMMA,
    RocmMFMA,
    MetalSIMD,
}

#[derive(Debug, Clone)]
pub struct UnifiedOpIR {
    pub op_type: UnifiedOpType,
    pub precison: String,
    pub tiling: crate::PipelineConfig,
}

#[derive(Debug, Clone)]
pub enum UnifiedOpType {
    Gemm {
        m: u32,
        n: u32,
        k: u32,
    },
    FusedAttention {
        b: u32,
        s: u32,
        d: u32,
        h: u32,
        dh: u32,
        causal: bool,
    },
}
