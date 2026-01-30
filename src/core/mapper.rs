// src/core/mapper.rs

use std::collections::HashMap;
use crate::core::manifold::{Dimension, ComputeAtom};
use crate::core::lattice::HardwareLattice;
use serde::{Serialize, Deserialize};

/// Defines a level in the Hardware Lattice (e.g. "Grid", "Block", "Warp")
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct HardwareLevel(pub String);

/// The projection of a Computation Manifold onto a Hardware Lattice.
/// This is the "Solution" that the Emitter uses to generate code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingStrategy {
    /// Mapping from Manifold Dimension ID -> Tile Size
    pub tile_sizes: HashMap<String, usize>,
    /// Order of computation (innermost to outermost)
    pub loop_order: Vec<String>,
    /// Mapping from Manifold Dimension ID -> Hardware Lattice Level
    pub spatial_map: HashMap<String, HardwareLevel>,
}

impl MappingStrategy {
    pub fn get_launch_params(&self, atom: &crate::core::manifold::ComputeAtom) -> ((u32, u32, u32), (u32, u32, u32)) {
        let mut grid = (1, 1, 1);
        let mut block = (1, 1, 1);

        for (dim_id, level) in &self.spatial_map {
            if let Some(dim) = atom.get_dim_by_id(dim_id) {
                let range = (dim.range_max - dim.range_min) as u32;
                // Basic mapping: if dimension is mapped to a level, its range determines the hardware size
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
    /// Evolve a strategy to map an Atom onto a Lattice
    fn evolve(
        &self,
        atom: &ComputeAtom,
        lattice: &HardwareLattice
    ) -> MappingStrategy;
}

/// Simple baseline engine for standard mappings
pub struct BaselineEngine;

impl EvolutionEngine for BaselineEngine {
    fn evolve(&self, _atom: &ComputeAtom, _lattice: &HardwareLattice) -> MappingStrategy {
        // Placeholder implementation
        MappingStrategy {
            tile_sizes: HashMap::new(),
            loop_order: Vec::new(),
            spatial_map: HashMap::new(),
        }
    }
}
