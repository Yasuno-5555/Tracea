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

/// Softmax update granularity for FlashAttention-2 kernels.
/// Controls how often the online softmax statistics (max/sum) are updated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum SoftmaxGranularity {
    /// Update softmax stats after each K/V tile (most accurate, baseline)
    #[default]
    PerTile,
    /// Update softmax stats after every 2 tiles (reduced sync overhead)
    PerTwoTiles,
    /// Update softmax stats for full Br rows at once (reserved for future optimization)
    FullBr,
}

impl SoftmaxGranularity {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::PerTile => 0.0,
            Self::PerTwoTiles => 1.0,
            Self::FullBr => 2.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::PerTwoTiles,
            2 => Self::FullBr,
            _ => Self::PerTile,
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
pub enum AttentionVariant {
    #[default]
    Naive,      // Golden Reference (Tiled Loop, Scalar/Vector fallback)
    SimdQ,      // Simdgroup for Q (Experimental)
    SimdQK,     // Simdgroup for QK, Naive for PV (Step 1)
    SimdFull,   // Simdgroup for QK and PV
    FlashV2,    // Optimized FA2: 64x64 blocks, vectorized loads, async copy
}

impl AttentionVariant {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::Naive => 0.0,
            Self::SimdQ => 1.0,
            Self::SimdQK => 2.0,
            Self::SimdFull => 3.0,
            Self::FlashV2 => 4.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::SimdQ,
            2 => Self::SimdQK,
            3 => Self::SimdFull,
            4 => Self::FlashV2,
            _ => Self::Naive,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum GemmVariant {
    #[default]
    Naive,      // Simple loop-based
    Tiled,      // Shared memory tiling
    Simd,       // Simdgroup/Warp-level Optimization (TensorCore/AMX)
}

impl GemmVariant {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::Naive => 0.0,
            Self::Tiled => 1.0,
            Self::Simd => 2.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::Tiled,
            2 => Self::Simd,
            _ => Self::Naive,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum IntrinsicShape {
    #[default]
    None,
    // CUDA Tensor Core (m16n8k16)
    M16N8K16,
    // CUDA Tensor Core (m16n8k32 for TF32)
    M16N8K32,
    // ROCm MFMA (m32n32k2)
    M32N32K2,
    // ROCm MFMA (m16n16k4)
    M16N16K4,
    // Metal Simdgroup (m16n16k1)
    M16N16K1,
    // Fallback / SIMD
    SimdAvx2,
}

impl IntrinsicShape {
    pub fn k_split(&self) -> u32 {
        match self {
            Self::M16N8K16 => 16,
            Self::M16N8K32 => 32,
            Self::M32N32K2 => 2,
            Self::M16N16K4 => 4,
            Self::M16N16K1 => 1,
            Self::SimdAvx2 => 8, // Approx
            Self::None => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum RegisterStrategy {
    #[default]
    Array,
    Expanded,
}

impl RegisterStrategy {
    pub fn to_f32(self) -> f32 {
        match self {
            Self::Array => 0.0,
            Self::Expanded => 1.0,
        }
    }
    pub fn from_f32(val: f32) -> Self {
        match val as u32 {
            1 => Self::Expanded,
            _ => Self::Array,
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
    pub softmax_granularity: SoftmaxGranularity,
    pub intrinsic_shape: IntrinsicShape,
    pub vectorize_epilogue: bool,
    pub ttg_enabled: bool,
    pub attention_variant: AttentionVariant,
    pub gemm_variant: GemmVariant,
    /// Enable Double Buffering for memory latency hiding
    pub double_buffer: bool,
    pub register_strategy: RegisterStrategy,
    /// Bank conflict avoidance padding (e.g. 8 for shared memory)
    pub bank_conflict_padding: u32,
    /// Number of N-tiles to fuse in a single threadgroup (default: 1).
    /// Fused tiles share A data across multiple N-tiles, reducing global memory reads.
    /// Set by TTG topology when RowAdjacent edges are detected.
    pub fusion_count: u32,
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
            softmax_granularity: SoftmaxGranularity::PerTile,
            intrinsic_shape: IntrinsicShape::None,
            vectorize_epilogue: true,
            ttg_enabled: false,
            attention_variant: AttentionVariant::Naive,
            gemm_variant: GemmVariant::Naive,
            double_buffer: false,
            register_strategy: RegisterStrategy::Array,
            bank_conflict_padding: 8,
            fusion_count: 1,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.m_tile % 16 != 0 {
            return Err(format!("m_tile ({}) must be a multiple of 16", self.m_tile));
        }
        if self.n_tile % 16 != 0 {
            return Err(format!("n_tile ({}) must be a multiple of 16", self.n_tile));
        }
        if self.num_stages < 2 {
            return Err(format!("num_stages ({}) must be at least 2 to prevent pipeline hazards", self.num_stages));
        }
        if self.k_tile < 16 {
            return Err(format!("k_tile ({}) must be at least 16", self.k_tile));
        }
        if let Some(warps) = self.force_num_warps {
            if warps == 0 || warps > 32 {
                return Err(format!("force_num_warps ({}) must be between 1 and 32", warps));
            }
        }
        Ok(())
    }

    pub fn with_warps(mut self, nw: u32) -> Self {
        self.force_num_warps = Some(nw);
        self
    }

    pub fn with_gemm_variant(mut self, v: GemmVariant) -> Self {
        self.gemm_variant = v;
        self
    }

    pub fn use_tensor_cores(&self) -> bool {
        self.instruction == SpecializedInstruction::CudaMMA
    }

    pub fn to_vector(&self) -> Vec<f32> {
        let value = serde_json::to_value(self).unwrap();
        let mut vec = Vec::new();
        
        fn flatten_value(val: &serde_json::Value, vec: &mut Vec<f32>) {
            match val {
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        vec.push(f as f32);
                    }
                }
                serde_json::Value::Bool(b) => {
                    vec.push(if *b { 1.0 } else { 0.0 });
                }
                serde_json::Value::Array(arr) => {
                    for v in arr {
                        flatten_value(v, vec);
                    }
                }
                serde_json::Value::Object(obj) => {
                    let mut sorted_keys: Vec<_> = obj.keys().collect();
                    sorted_keys.sort();
                    for k in sorted_keys {
                        flatten_value(&obj[k], vec);
                    }
                }
                _ => {
                    vec.push(0.0);
                }
            }
        }
        
        flatten_value(&value, &mut vec);
        vec
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_validation() {
        // Valid config
        let valid = PipelineConfig::new(2, 64, 64, 16);
        assert!(valid.validate().is_ok());

        // Invalid m_tile
        let invalid_m = PipelineConfig::new(2, 63, 64, 16);
        assert!(invalid_m.validate().is_err());

        // Invalid n_tile
        let invalid_n = PipelineConfig::new(2, 64, 63, 16);
        assert!(invalid_n.validate().is_err());

        // Invalid stages
        let invalid_stages = PipelineConfig::new(1, 64, 64, 16);
        assert!(invalid_stages.validate().is_err());

        // Invalid k_tile
        let invalid_k = PipelineConfig::new(2, 64, 64, 15);
        assert!(invalid_k.validate().is_err());

        // Invalid force_num_warps
        let mut invalid_warps = PipelineConfig::new(2, 64, 64, 16);
        invalid_warps.force_num_warps = Some(0);
        assert!(invalid_warps.validate().is_err());
        invalid_warps.force_num_warps = Some(33);
        assert!(invalid_warps.validate().is_err());
    }

    #[test]
    fn test_pipeline_config_serde_roundtrip() {
        let mut original = PipelineConfig::new(3, 128, 64, 32);
        original.double_buffer = true;
        original.bank_conflict_padding = 16;
        original.vectorize_epilogue = false;

        // Verify JSON roundtrip
        let json_str = original.to_json().expect("to_json failed");
        let deserialized = PipelineConfig::from_json(&json_str).expect("from_json failed");

        assert_eq!(original, deserialized);
        assert!(deserialized.double_buffer);
        assert_eq!(deserialized.bank_conflict_padding, 16);
        assert!(!deserialized.vectorize_epilogue);

        // Verify vector flattening matches length expectation
        let vec_repr = original.to_vector();
        assert!(!vec_repr.is_empty());
    }
}

