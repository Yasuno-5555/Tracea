// src/core/mapper.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareLevel(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingStrategy {
    pub tile_sizes: HashMap<String, usize>,
    pub loop_order: Vec<String>,
    pub spatial_map: HashMap<String, HardwareLevel>,
}

impl MappingStrategy {
    pub fn get_launch_params(&self, atom: &crate::core::manifold::ComputeAtom) -> ((u32, u32, u32), (u32, u32, u32)) {
        let mut grid = (1, 1, 1);
        let mut block = (1, 1, 1);

        for (dim_id, level) in &self.spatial_map {
            if let Some(dim) = atom.get_dim_by_id(dim_id) {
                let range = (dim.range_max - dim.range_min) as u32;
                match level.0.as_str() {
                    "BlockX" => grid.0 = range,
                    "BlockY" => grid.1 = range,
                    "BlockZ" => grid.2 = range,
                    "ThreadX" => block.0 = range,
                    "ThreadY" => block.1 = range,
                    "ThreadZ" => block.2 = range,
                    _ => {}
                }
            }
        }
        (grid, block)
    }
}

/// The "Tracea Intelligence" - finds the isomorphism between Math and Physics.
pub trait EvolutionEngine {
    fn evolve(&self, atom: &crate::core::manifold::ComputeAtom, lattice: &crate::core::lattice::HardwareLattice) -> MappingStrategy;
}
