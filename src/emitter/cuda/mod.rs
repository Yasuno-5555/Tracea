use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType, EmissionError};
use crate::semantic::transition::SyncRequirement;

pub mod gemm;
pub mod attention;
pub mod conv;
pub mod low_rank;
pub mod nn;

pub use gemm::generate_gemm;
pub use attention::generate_attention;
pub use conv::generate_conv;
pub use low_rank::generate_low_rank_mlp;
pub use nn::{generate_softmax, generate_batchnorm, generate_global_avg_pool, generate_linear, generate_matrix_core};

pub struct CUDAEmitter {}

impl CUDAEmitter {
    pub fn new() -> Self {
        Self {}
    }

    pub fn emit(&self, ir: &UnifiedOpIR) -> Result<String, EmissionError> {
        self.generate_from_ir(ir)
    }
}

impl Emitter for CUDAEmitter {
    fn emit_sync(&mut self, _req: SyncRequirement) -> String {
        "__syncthreads();\n".to_string()
    }

    fn generate_from_ir(&self, ir: &UnifiedOpIR) -> Result<String, EmissionError> {
        match &ir.op_type {
            UnifiedOpType::FusedAttention { h, dh, causal, .. } => {
                attention::generate_attention(*h, *dh, *causal, ir)
            }
            UnifiedOpType::Gemm { m, n, k, .. } => {
                self.generate_gemm(*m, *n, *k, &ir.tiling)
            }
            UnifiedOpType::Elementwise { .. } => {
                Err(EmissionError::UnsupportedOpType {
                    reason: "Elementwise Ops should be handled by UniversalEmitter.".to_string(),
                })
            }
            UnifiedOpType::Conv2d { .. } => {
                conv::generate_conv(ir)
            }
            UnifiedOpType::MatrixCore { m, n, k } => {
                nn::generate_matrix_core(*m, *n, *k)
            }
            UnifiedOpType::ConvTranspose2d { .. } => {
                crate::emitter::conv_transpose::generate_conv_transpose_res(ir)
            }
            UnifiedOpType::LowRankMlp { .. } => {
                low_rank::generate_low_rank_mlp(ir)
            }
            UnifiedOpType::Softmax { dim_size, total_elements, .. } => {
                nn::generate_softmax(*dim_size, *total_elements)
            }
            UnifiedOpType::BatchNorm { n, c, h, w, epsilon, .. } => {
                nn::generate_batchnorm(*n, *c, *h, *w, *epsilon)
            }
            UnifiedOpType::GlobalAveragePool { n, c, h, w } => {
                nn::generate_global_avg_pool(*n, *c, *h, *w)
            }
            UnifiedOpType::Linear { batch, m, n, k, .. } => {
                nn::generate_linear(*batch, *m, *n, *k)
            }
        }
    }
}

impl CUDAEmitter {
    pub fn generate_gemm(&self, m: u32, n: u32, k: u32, config: &crate::PipelineConfig) -> Result<String, EmissionError> {
        gemm::generate_gemm(m, n, k, config)
    }
}
