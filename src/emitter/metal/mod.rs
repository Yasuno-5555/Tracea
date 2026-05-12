use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType, EmissionError};
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

    fn generate_from_ir(&self, ir: &UnifiedOpIR) -> Result<String, EmissionError> {
        match &ir.op_type {
            UnifiedOpType::Gemm { .. } => {
                Ok(gemm::generate_metal_gemm(ir))
            },
            UnifiedOpType::FusedAttention { .. } => {
                Ok(attention::generate_metal_attention(ir))
            }
            UnifiedOpType::Elementwise { .. } => {
                Ok(crate::emitter::elementwise::generate_elementwise(ir, crate::runtime::manager::DeviceBackend::Metal))
            }
            UnifiedOpType::Conv2d { .. } => {
                Ok(conv::generate_metal_conv(ir))
            }
            UnifiedOpType::ConvTranspose2d { .. } => {
                Err(EmissionError::UnsupportedOpType {
                    reason: "Metal ConvTranspose2d not yet implemented".to_string(),
                })
            }
            UnifiedOpType::MatrixCore { .. } => {
                Err(EmissionError::UnsupportedOpType {
                    reason: "MatrixCore Ops not supported on Metal yet.".to_string(),
                })
            }
            UnifiedOpType::Softmax { axis: _, dim_size, total_elements, .. } => {
                Ok(utils::generate_metal_softmax(*dim_size, 0, *total_elements))
            }
            UnifiedOpType::BatchNorm { n, c, h, w, epsilon, momentum: _ } => {
                Ok(nn::generate_batchnorm(*n, *c, *h, *w, *epsilon))
            }
            UnifiedOpType::GlobalAveragePool { n, c, h, w } => {
                Ok(nn::generate_global_avg_pool(*n, *c, *h, *w))
            }
            UnifiedOpType::Linear { batch, m, n, k, .. } => {
                Ok(nn::generate_linear(*batch, *m, *n, *k))
            }
            UnifiedOpType::LowRankMlp { .. } => {
                Err(EmissionError::UnsupportedOpType {
                    reason: "LowRankMlp not supported on Metal yet.".to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{PipelineConfig, GemmVariant, AttentionVariant, LayoutPolicy};
    use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};

    #[test]
    fn test_metal_gemm_generation() {
        let emitter = MetalEmitter::detect();
        
        // 1. Single Buffer
        let mut config_single = PipelineConfig::new(1, 64, 64, 32);
        config_single.double_buffer = false;
        config_single.gemm_variant = GemmVariant::Naive;
        let ir_single = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm {
                m: 256,
                n: 256,
                k: 256,
                batch: 1,
                epilogue: vec![],
            },
            precison: "f16".to_string(),
            tiling: config_single,
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        let msl_single = emitter.generate_from_ir(&ir_single).unwrap();
        assert!(msl_single.contains("Single Buffer GEMM"));
        assert!(msl_single.contains("simdgroup_float8x8 acc"));
        assert!(msl_single.contains("threadgroup_barrier"));

        // 2. Double Buffer
        let mut config_double = PipelineConfig::new(2, 64, 64, 32);
        config_double.double_buffer = true;
        config_double.gemm_variant = GemmVariant::Naive;
        let ir_double = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm {
                m: 256,
                n: 256,
                k: 256,
                batch: 1,
                epilogue: vec![],
            },
            precison: "f16".to_string(),
            tiling: config_double,
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        let msl_double = emitter.generate_from_ir(&ir_double).unwrap();
        assert!(msl_double.contains("Double Buffer GEMM"));
        assert!(msl_double.contains("threadgroup half sA[2]"));
        assert!(msl_double.contains("threadgroup_barrier"));

        // 3. Tiled
        let mut config_tiled = PipelineConfig::new(2, 64, 64, 32);
        config_tiled.gemm_variant = GemmVariant::Tiled;
        let ir_tiled = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm {
                m: 256,
                n: 256,
                k: 256,
                batch: 1,
                epilogue: vec![],
            },
            precison: "f16".to_string(),
            tiling: config_tiled,
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        let msl_tiled = emitter.generate_from_ir(&ir_tiled).unwrap();
        assert!(msl_tiled.contains("gemm_tiled_kernel"));
        assert!(msl_tiled.contains("threadgroup half sA[2]"));
    }

    #[test]
    fn test_metal_attention_generation() {
        let emitter = MetalEmitter::detect();
        
        let variants = vec![
            AttentionVariant::Naive,
            AttentionVariant::SimdQK,
            AttentionVariant::SimdFull,
            AttentionVariant::FlashV2,
        ];

        for variant in variants {
            let mut config = PipelineConfig::new(2, 64, 64, 32);
            config.attention_variant = variant;
            let ir = UnifiedOpIR {
                op_type: UnifiedOpType::FusedAttention {
                    b: 2,
                    h: 4,
                    s: 128,
                    d: 64,
                    dh: 64,
                    causal: false,
                },
                precison: "f16".to_string(),
                tiling: config,
                conv_magic_strategy: None,
                polyhedral_strategy: None,
            };
            
            let msl = emitter.generate_from_ir(&ir).unwrap();
            assert!(msl.contains("flash_attention_v2_kernel") || msl.contains("fap_kernel"));
            assert!(msl.contains("threadgroup half sK") || msl.contains("sK_T"));
        }
    }

    #[test]
    fn test_metal_conv_generation() {
        let emitter = MetalEmitter::detect();
        
        // Single Buffer Conv
        let mut config = PipelineConfig::new(1, 32, 32, 16);
        config.double_buffer = false;
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Conv2d {
                n: 1,
                h: 14,
                w: 14,
                c: 32,
                k: 64,
                r: 3,
                s: 3,
                stride: 1,
                pad: 1,
                dilation: 1,
                layout: LayoutPolicy::NHWC,
                epilogue: vec![],
            },
            precison: "f16".to_string(),
            tiling: config,
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        
        let msl = emitter.generate_from_ir(&ir).unwrap();
        assert!(msl.contains("conv2d_implicit_gemm"));
        assert!(msl.contains("simdgroup_float8x8"));
        assert!(msl.contains("threadgroup_barrier"));
    }

    // Compilation check on macOS using the actual metal API if possible
    #[cfg(all(target_os = "macos", feature = "metal_compile_test"))]
    #[test]
    fn test_metal_actual_compilation() {
        use objc::rc::autoreleasepool;
        
        autoreleasepool(|| {
            let emitter = MetalEmitter::detect();
            let mut config = PipelineConfig::new(1, 64, 64, 32);
            config.gemm_variant = GemmVariant::Tiled;
            let ir = UnifiedOpIR {
                op_type: UnifiedOpType::Gemm {
                    m: 128,
                    n: 128,
                    k: 128,
                    batch: 1,
                    epilogue: vec![],
                },
                precison: "f16".to_string(),
                tiling: config,
                conv_magic_strategy: None,
                polyhedral_strategy: None,
            };
            
            let msl_source = emitter.generate_from_ir(&ir).unwrap();
            
            // Try to load Metal Device and compile the source code!
            if let Some(device) = metal::Device::system_default() {
                let options = metal::CompileOptions::new();
                let library = device.new_library_with_source(&msl_source, &options);
                assert!(library.is_ok(), "Metal compilation failed: {:?}", library.err());
            }
        });
    }
}
