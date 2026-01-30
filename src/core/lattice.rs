// src/core/lattice.rs

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLevel {
    pub name: String,
    pub size_bytes: usize,
    pub bandwidth_gbps: f32,
    pub latency_cycles: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeUnit {
    pub name: String,
    pub parallelism: u32,
    pub instruction_throughput: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareLattice {
    pub name: String,
    pub memory_hierarchy: Vec<MemoryLevel>,
    pub compute_units: Vec<ComputeUnit>,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_shared_memory: usize,
}

impl HardwareLattice {
    pub fn rtx3070() -> Self {
        Self {
            name: "RTX3070".to_string(),
            memory_hierarchy: vec![
                MemoryLevel { name: "Register".to_string(), size_bytes: 256, bandwidth_gbps: 10000.0, latency_cycles: 1 },
                MemoryLevel { name: "Shared".to_string(), size_bytes: 48 * 1024, bandwidth_gbps: 2000.0, latency_cycles: 20 },
                MemoryLevel { name: "L2".to_string(), size_bytes: 4 * 1024 * 1024, bandwidth_gbps: 800.0, latency_cycles: 100 },
                MemoryLevel { name: "Global".to_string(), size_bytes: 8 * 1024 * 1024 * 1024, bandwidth_gbps: 448.0, latency_cycles: 400 },
            ],
            compute_units: vec![
                ComputeUnit { name: "SM".to_string(), parallelism: 46, instruction_throughput: 160.0 },
            ],
            warp_size: 32,
            max_threads_per_block: 1024,
            max_shared_memory: 48 * 1024,
        }
    }
}
