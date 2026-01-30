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
    pub conv_magic_strategy: Option<crate::core::config::MagicNumberStrategy>,
    pub polyhedral_strategy: Option<crate::core::polyhedral::TilingStrategy>,
}

#[derive(Debug, Clone)]
pub enum UnifiedOpType {
    Gemm {
        m: u32,
        n: u32,
        k: u32,
        batch: u32,
        epilogue: Vec<crate::core::op::EpilogueOp>,
    },
    FusedAttention {
        b: u32,
        s: u32,
        d: u32,
        h: u32,
        dh: u32,
        causal: bool,
    },
    Elementwise {
        op_type: crate::core::op::ElementwiseType,
        n: usize,
    },
    Conv2d {
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        k: usize,
        r: usize,
        s: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        layout: crate::core::config::LayoutPolicy,
        epilogue: Vec<crate::core::op::EpilogueOp>,
    },
    /// Unified Matrix-Multiply-Accumulate (Tensor Core / Cooperative Matrix)
    MatrixCore {
        m: u32,
        n: u32,
        k: u32,
    },
    /// ConvTranspose2d (Deconvolution) for VAE Decoder
    /// Constraints: groups=1, dilation=1, FP32 only (v3.1)
    ConvTranspose2d {
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        k: usize,
        r: usize,
        s: usize,
        stride: usize,
        pad: usize,
        output_padding: usize,
        layout: crate::core::config::LayoutPolicy,
    },
    LowRankMlp {
        m: u32,
        n: u32,
        k: u32,
        r: u32,
    },
    Softmax {
        axis: i32,
        dim_size: usize,
        stride: usize,
        total_elements: usize,
    },
    BatchNorm {
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        epsilon: f32,
        momentum: f32,
    },
    GlobalAveragePool {
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    },
    Linear {
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        epilogue: Vec<crate::core::op::EpilogueOp>,
    },
}
