use serde::{Serialize, Deserialize};
use crate::PipelineConfig;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Layout {
    NCHW,
    NHWC,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Fa2Variant {
    Causal,
    NonCausal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    Gemm,
    Conv2d(Layout),
    FlashAttention(Fa2Variant),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProblemDescriptor {
    pub layer_type: LayerType,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    
    // Optional fields for specific layers can be handled via methods or extra fields
    // For now, minimal set to support the logic
    pub name: String,
}

impl ProblemDescriptor {
    pub fn new_gemm(m: usize, n: usize, k: usize) -> Self {
        Self {
            layer_type: LayerType::Gemm,
            m, n, k,
            batch: 1,
            name: format!("GEMM_M{}_N{}_K{}", m, n, k),
        }
    }
    
    pub fn new_conv2d(batch: usize, h: usize, w: usize, c: usize, k: usize, layout: Layout) -> Self {
         // Approximation for M, N, K mapping in Conv2d
         // M = H*W, N = K (filters), K = C * KernelSize (simplified)
         // This is just a descriptor, exact GEMM mapping logic resides in Conv2dProblem usually
         Self {
            layer_type: LayerType::Conv2d(layout),
            m: batch * h * w,
            n: k,
            k: c, 
            batch,
            name: format!("Conv2d_B{}_{:?}", batch, layout),
        }
    }

    pub fn new_fa2(batch: usize, seq_len: usize, head_dim: usize, variant: Fa2Variant) -> Self {
        Self {
             layer_type: LayerType::FlashAttention(variant),
             m: batch * seq_len,
             n: head_dim, // Approximation
             k: head_dim,
             batch,
             name: format!("FA2_B{}_S{}_D{}_{:?}", batch, seq_len, head_dim, variant),
        }
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum ArchHint {
    Any,
    NvidiaAmpere,
    NvidiaAda,
    NvidiaHopper,
    // Future expansion
}

#[derive(Debug, Clone, PartialEq)]
pub enum HeroScope {
    Exact, // Exact match for this problem
    Arch,  // Good for this architecture
    Layer, // Good for this layer type generally
}

#[derive(Debug, Clone)]
pub struct HeroConfig {
    pub config: PipelineConfig,
    pub note: &'static str,
    pub arch_hint: ArchHint,
    pub scope: HeroScope,
}
