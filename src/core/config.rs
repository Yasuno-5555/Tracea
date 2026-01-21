use serde::{Serialize, Deserialize};
use crate::core::op::EpilogueOp;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum SwizzleMode {
    #[default]
    None,
    Xor2,
    Xor4,
    Xor8,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum BarrierMode {
    #[default]
    None,
    ProducerConsumer,
}

impl BarrierMode {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::ProducerConsumer => 1.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::ProducerConsumer,
            _ => Self::None,
        }
    }
}

impl SwizzleMode {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Xor2 => 1.0,
            Self::Xor4 => 2.0,
            Self::Xor8 => 3.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::Xor2,
            2 => Self::Xor4,
            3 => Self::Xor8,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum QuantizationMode {
    #[default]
    None,
    Int8,
    Int4,
}

impl QuantizationMode {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Int8 => 1.0,
            Self::Int4 => 2.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::Int8,
            2 => Self::Int4,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum SpecializedInstruction {
    #[default]
    None,
    CudaMMA,
    RocmMFMA,
    MetalSimdGroup,
    Avx2,
    Avx512,
    Neon,
}

impl SpecializedInstruction {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::CudaMMA => 1.0,
            Self::RocmMFMA => 2.0,
            Self::MetalSimdGroup => 3.0,
            Self::Avx2 => 4.0,
            Self::Avx512 => 5.0,
            Self::Neon => 6.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::CudaMMA,
            2 => Self::RocmMFMA,
            3 => Self::MetalSimdGroup,
            4 => Self::Avx2,
            5 => Self::Avx512,
            6 => Self::Neon,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum LayoutPolicy {
    #[default]
    RowMajor,
    ColumnMajor,
    XorSwizzled,
    NHWC,
    NCHW,
}

impl LayoutPolicy {
    /// Generates the C++ expression for calculating the shared memory offset
    pub fn get_offset_expr(&self, row: &str, col: &str, stride: &str) -> String {
        match self {
            LayoutPolicy::RowMajor => format!("(({}) * ({}) + ({}))", row, stride, col),
            LayoutPolicy::ColumnMajor => format!("(({}) * ({}) + ({}))", col, stride, row),
            LayoutPolicy::XorSwizzled => {
                // Canonical XOR swizzling: (row * stride + col) ^ (row % 8) logic is usually embedded in specific access patterns
                // But for basic linear offset, we keep it linear and swizzle the PTR.
                format!("(({}) * ({}) + ({}))", row, stride, col)
            }
            LayoutPolicy::NHWC | LayoutPolicy::NCHW => {
                 // For now, treat as RowMajor linear addressing with appropriate stride passed in
                 format!("(({}) * ({}) + ({}))", row, stride, col)
            }
        }
    }

    /// Generates the C++ expression for a pointer that is swizzled at the PTX level.
    pub fn get_swizzled_ptr_expr(&self, base_ptr: &str, _row: &str, _col: &str) -> String {
        match self {
            LayoutPolicy::XorSwizzled => {
                // Use the smem_swizzle primitive.
                // It expects a shared memory address (32-bit).
                format!("(void*)(uint64_t)smem_swizzle((uint32_t)__cvta_generic_to_shared({}))", base_ptr)
            }
            _ => format!("(void*){}", base_ptr),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PipelineConfig {
    pub num_stages: u32,
    pub m_tile: u32,
    pub n_tile: u32,
    pub k_tile: u32,
    pub instruction: SpecializedInstruction,
    pub swizzle_mode: SwizzleMode,
    pub quantization: QuantizationMode,
    pub layout_policy: Option<LayoutPolicy>,
    pub epilogue: Vec<EpilogueOp>,
    pub force_num_warps: Option<u32>,
    pub micro_m: u32,
    pub micro_n: u32,
    pub k_unroll: u32,
    pub prefetch_distance: u32,
    pub cp_async_distance: u32,
    pub barrier_mode: BarrierMode,
}

impl PipelineConfig {
    pub fn new(num_stages: u32, m_tile: u32, n_tile: u32, k_tile: u32) -> Self {
        Self {
            num_stages,
            m_tile,
            n_tile,
            k_tile,
            instruction: SpecializedInstruction::None,
            swizzle_mode: SwizzleMode::None,
            quantization: QuantizationMode::None,
            layout_policy: Some(LayoutPolicy::RowMajor),
            epilogue: Vec::new(),
            force_num_warps: None,
            micro_m: 1,
            micro_n: 1,
            k_unroll: 1,
            prefetch_distance: 0,
            cp_async_distance: 0,
            barrier_mode: BarrierMode::None,
        }
    }

    pub fn with_warps(mut self, nw: u32) -> Self {
        self.force_num_warps = Some(nw);
        self
    }

    pub fn use_tensor_cores(&self) -> bool {
        self.instruction == SpecializedInstruction::CudaMMA
    }

    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.num_stages as f32,
            self.m_tile as f32,
            self.n_tile as f32,
            self.k_tile as f32,
            self.instruction.to_f32(),
            self.swizzle_mode.to_f32(),
            self.quantization.to_f32(),
            self.micro_m as f32,
            self.micro_n as f32,
            self.k_unroll as f32,
            self.prefetch_distance as f32,
            self.cp_async_distance as f32,
            self.barrier_mode.to_f32(),
        ]
    }

    pub fn from_vector(vec: &[f32]) -> Self {
        Self {
            num_stages: vec[0] as u32,
            m_tile: vec[1] as u32,
            n_tile: vec[2] as u32,
            k_tile: vec[3] as u32,
            instruction: vec.get(4).map(|&v| SpecializedInstruction::from_f32(v)).unwrap_or(SpecializedInstruction::None),
            swizzle_mode: vec.get(5).map(|&v| SwizzleMode::from_f32(v)).unwrap_or(SwizzleMode::None),
            quantization: vec.get(6).map(|&v| QuantizationMode::from_f32(v)).unwrap_or(QuantizationMode::None),
            layout_policy: Some(LayoutPolicy::RowMajor),
            epilogue: Vec::new(),
            force_num_warps: None,
            micro_m: vec.get(7).map(|&v| v as u32).unwrap_or(1),
            micro_n: vec.get(8).map(|&v| v as u32).unwrap_or(1),
            k_unroll: vec.get(9).map(|&v| v as u32).unwrap_or(1),
            prefetch_distance: vec.get(10).map(|&v| v as u32).unwrap_or(0),
            cp_async_distance: vec.get(11).map(|&v| v as u32).unwrap_or(0),
            barrier_mode: vec.get(12).map(|&v| BarrierMode::from_f32(v)).unwrap_or(BarrierMode::None),
        }
    }

    pub fn with_micro_tile(mut self, m: u32, n: u32) -> Self {
        self.micro_m = m;
        self.micro_n = n;
        self
    }
}

/// Magic Number Division Strategy (for address calculation)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MagicNumberStrategy {
    /// Standard magic number: multiply + shift + add. Universal but more instructions.
    Standard,
    /// Simple bit shift. Only valid when divisor is power of 2.
    PowerOfTwo,
    /// 32-bit fast division. Valid for small divisors (<= 65535).
    FastSmall,
}

impl MagicNumberStrategy {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::Standard => 0.0,
            Self::PowerOfTwo => 1.0,
            Self::FastSmall => 2.0,
        }
    }
    
    pub fn select_for(divisor: usize) -> Self {
        if divisor > 0 && divisor.is_power_of_two() {
            Self::PowerOfTwo
        } else if divisor <= 65535 {
            Self::FastSmall
        } else {
            Self::Standard
        }
    }
}
