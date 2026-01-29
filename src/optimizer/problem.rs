use serde::{Serialize, Deserialize};
use crate::core::config::{PipelineConfig, BarrierMode};
use crate::core::backend::Device;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum Layout {
    #[default]
    NCHW,
    NHWC,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum Fa2Variant {
    Causal,
    #[default]
    NonCausal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct Shape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CpuAsmParams {
    pub k_unroll: usize,
    pub prefetch_distance: usize,
    pub micro_m: usize,
}

impl Default for CpuAsmParams {
    fn default() -> Self {
        Self {
            k_unroll: 1,
            prefetch_distance: 0,
            micro_m: 6,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct GpuAsmParams {
    pub pipe_depth: usize,
    pub swizzle_mode: usize,
    pub cp_async_distance: usize,
    pub barrier_mode: BarrierMode,
}

impl Default for GpuAsmParams {
    fn default() -> Self {
        Self {
            pipe_depth: 2,
            swizzle_mode: 0,
            cp_async_distance: 1,
            barrier_mode: BarrierMode::None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AsmParams {
    Cpu(CpuAsmParams),
    Gpu(GpuAsmParams),
}

impl Default for AsmParams {
    fn default() -> Self {
        // Default to GPU for now as it's the primary target
        Self::Gpu(GpuAsmParams::default())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum LayerType {
    #[default]
    Gemm,
    Conv2d(Layout),
    FlashAttention(Fa2Variant),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProblemDescriptor {
    pub device: Device,
    pub layer_type: LayerType,
    pub shape: Shape,
    pub asm_params: AsmParams,
    pub name: String,
}

impl ProblemDescriptor {
    pub fn new_gemm(m: usize, n: usize, k: usize) -> Self {
        Self {
            device: Device::default(),
            layer_type: LayerType::Gemm,
            shape: Shape { m, n, k, batch: 1 },
            asm_params: AsmParams::default(),
            name: format!("GEMM_M{}_N{}_K{}", m, n, k),
        }
    }
    
    pub fn new_conv2d(batch: usize, h: usize, w: usize, c: usize, k: usize, r: usize, s: usize, layout: Layout) -> Self {
         Self {
            device: Device::default(),
            layer_type: LayerType::Conv2d(layout),
            shape: Shape {
                m: batch * h * w,
                n: k,
                k: c * r * s, 
                batch,
            },
            asm_params: AsmParams::default(),
            name: format!("Conv2d_B{}_H{}x{}_C{}_K{}_R{}x{}_{:?}", batch, h, w, c, k, r, s, layout),
        }
    }

    pub fn new_fa2(batch: usize, seq_len: usize, head_dim: usize, variant: Fa2Variant) -> Self {
        Self {
             device: Device::default(),
             layer_type: LayerType::FlashAttention(variant),
             shape: Shape {
                 m: batch * seq_len,
                 n: head_dim,
                 k: head_dim,
                 batch,
             },
             asm_params: AsmParams::default(),
             name: format!("FA2_B{}_S{}_D{}_{:?}", batch, seq_len, head_dim, variant),
        }
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
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
