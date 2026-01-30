// src/core/lattice.rs

use serde::{Serialize, Deserialize};

/// Memory levels in the hardware hierarchy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryHierarchy {
    Global,   // e.g. VRAM, DRAM
    L2,       // L2 Cache (Shared among groups)
    Shared,   // e.g. L1, Shared Memory, Local Memory
    Register, // Local register file
}

/// A node in the hardware resource tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeNode {
    pub name: String,
    /// Parallelism at this level (e.g. [128, 4] for 128 items processed in parallel across 4 subunits)
    pub parallelism: Vec<usize>,
    /// Memory resource available at this node
    pub memory: Option<MemoryHierarchy>,
    /// Sub-components
    pub children: Vec<ComputeNode>,
}

/// The hierarchical model of the execution environment.
/// Abandoning backend-specific names in favor of relative "Lattices".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareLattice {
    pub name: String,
    /// Root of the hierarchy (e.g. Device)
    pub root: ComputeNode,
    /// Profile information for bandwidth between nodes
    pub bandwidth_mbps: Vec<(String, String, f32)>, 
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_shared_memory: usize,
}

impl HardwareLattice {
    pub fn total_parallelism(&self) -> usize {
        fn traverse(node: &ComputeNode) -> usize {
            let base: usize = node.parallelism.iter().product();
            if node.children.is_empty() {
                base
            } else {
                base * node.children.iter().map(|c| traverse(c)).sum::<usize>()
            }
        }
        traverse(&self.root)
    }

    pub fn rtx3070() -> Self {
        // Basic reproduction of the tree structure for RTX3070
        let sm = ComputeNode {
            name: "SM".to_string(),
            parallelism: vec![128],
            memory: Some(MemoryHierarchy::Shared),
            children: vec![],
        };
        HardwareLattice {
            name: "RTX3070".to_string(),
            root: ComputeNode {
                name: "Device".to_string(),
                parallelism: vec![46],
                memory: Some(MemoryHierarchy::Global),
                children: vec![sm],
            },
            bandwidth_mbps: vec![],
            warp_size: 32,
            max_threads_per_block: 1024,
            max_shared_memory: 48 * 1024,
        }
    }
}
