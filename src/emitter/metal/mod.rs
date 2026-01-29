use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use crate::semantic::transition::SyncRequirement;

pub mod conv;
pub mod attention;
pub mod gemm;
pub mod nn;
pub mod utils;

pub use conv::generate_metal_conv;
pub use attention::generate_metal_attention;
pub use gemm::generate_metal_gemm;
pub use utils::generate_metal_softmax;
pub use nn::{generate_batchnorm, generate_global_avg_pool, generate_linear};

pub struct MetalEmitter {
    pub device_name: String,
    pub max_threadgroup_memory: usize,
}

impl MetalEmitter {
    pub fn detect() -> Self {
        Self {
            device_name: "Apple M-Series (Simulated)".to_string(),
            max_threadgroup_memory: 32768,
        }
    }
}

impl Emitter for MetalEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::Barrier => "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string(),
            _ => String::new(),
        }
    }

    fn generate_from_ir(&self, ir: &UnifiedOpIR) -> String {
        match &ir.op_type {
            UnifiedOpType::Gemm { .. } => {
                gemm::generate_metal_gemm(ir)
            },
            UnifiedOpType::FusedAttention { .. } => {
                attention::generate_metal_attention(ir)
            }
            UnifiedOpType::Elementwise { .. } => {
                crate::emitter::elementwise::generate_elementwise(ir, crate::runtime::manager::DeviceBackend::Metal)
            }
            UnifiedOpType::Conv2d { .. } => {
                conv::generate_metal_conv(ir)
            }
            UnifiedOpType::ConvTranspose2d { .. } => {
                "// Metal ConvTranspose2d not yet implemented - fallback to CPU\n".to_string()
            }
            UnifiedOpType::MatrixCore { .. } => {
                panic!("MatrixCore Ops not supported on Metal yet.");
            }
            UnifiedOpType::Softmax { axis: _, dim_size, total_elements, .. } => {
                utils::generate_metal_softmax(*dim_size, 0, *total_elements)
            }
            UnifiedOpType::BatchNorm { n, c, h, w, epsilon, momentum: _ } => {
                nn::generate_batchnorm(*n, *c, *h, *w, *epsilon)
            }
            UnifiedOpType::GlobalAveragePool { n, c, h, w } => {
                nn::generate_global_avg_pool(*n, *c, *h, *w)
            }
            UnifiedOpType::Linear { batch, m, n, k, .. } => {
                nn::generate_linear(*batch, *m, *n, *k)
            }
            UnifiedOpType::LowRankMlp { .. } => {
                panic!("LowRankMlp not supported on Metal yet.");
            }
        }
    }
}
