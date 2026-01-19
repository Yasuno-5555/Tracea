use serde::{Serialize, Deserialize};
use crate::core::op::EpilogueOp;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SwizzleMode {
    None,
    Xor2,
    Xor4,
    Xor8,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum QuantizationMode {
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SpecializedInstruction {
    None,
    CudaMMA,
    RocmMFMA,
    MetalSimdGroup,
}

impl SpecializedInstruction {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::CudaMMA => 1.0,
            Self::RocmMFMA => 2.0,
            Self::MetalSimdGroup => 3.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::CudaMMA,
            2 => Self::RocmMFMA,
            3 => Self::MetalSimdGroup,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub num_stages: u32,
    pub m_tile: u32,
    pub n_tile: u32,
    pub k_tile: u32,
    pub instruction: SpecializedInstruction,
    pub swizzle_mode: SwizzleMode,
    pub quantization: QuantizationMode,
    pub epilogue: Vec<EpilogueOp>,
    pub force_num_warps: Option<u32>, // ROCm: Wavefront Count, Metal: Simdgroup Count
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
            epilogue: Vec::new(),
            force_num_warps: None,
        }
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
            epilogue: Vec::new(),
            force_num_warps: None,
        }
    }
}
