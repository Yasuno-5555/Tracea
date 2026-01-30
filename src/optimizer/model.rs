use crate::core::config::PipelineConfig;
use serde::{Serialize, Deserialize};

/// Layer 2: Virtual ISA Model
/// Represents a kernel launch as a high-level "Instruction"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualInstruction {
    pub opcode: String,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub unroll: u32,
    pub vector_width: u32,
    pub shared_mem_bytes: u32,
}

impl VirtualInstruction {
    pub fn from_config(opcode: &str, config: &PipelineConfig) -> Self {
        Self {
            opcode: opcode.to_string(),
            tile_m: config.m_tile,
            tile_n: config.n_tile,
            tile_k: config.k_tile,
            unroll: 1, // Placeholder for now
            vector_width: 1, // Placeholder for now
            shared_mem_bytes: 0, // Calculated during emission
        }
    }
}

/// Layer 1: Hardware Observation
/// Raw metrics collected from the backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareObservation {
    pub latency_ns: u64,
    pub throughput_tflops: f32,
    
    // Metal/GPU Specific Features
    pub thread_execution_width: u32,
    pub max_threads_per_threadgroup: u32,
    pub static_threadgroup_memory_len: u32,
    pub estimated_occupancy: f32,
}

/// Layer 3: Inference Engine (MetaTuner++)
/// Bridges Virtual ISA -> Hardware Observation
pub struct InferenceEngine {}

impl InferenceEngine {
    pub fn infer_bottleneck(vi: &VirtualInstruction, obs: &HardwareObservation) -> String {
        // 1. Check LDS Pressure
        if obs.static_threadgroup_memory_len > 32768 {
             return "LDS Bound (High shared memory usage)".to_string();
        }

        // 2. Check Occupancy
        if obs.max_threads_per_threadgroup < 512 {
             return "Occupancy Bound (Small threadgroup limit)".to_string();
        }

        // 3. Simple Inference based on Tile/Latency
        if obs.latency_ns > 10_000_000 && vi.tile_m * vi.tile_n < 1024 {
             return "Under-utilization (Tile size too small for latency)".to_string();
        }

        "Likely Compute or Global Memory Bound".to_string()
    }
}
