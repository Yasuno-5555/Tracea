// src/core/evolution.rs

use rand::prelude::*;
use std::collections::HashMap;
use crate::core::mapper::{MappingStrategy, HardwareLevel};

pub trait Mutatable {
    /// Randomly mutate the current state to produce a new candidate
    fn mutate(&self, rng: &mut StdRng) -> Self;
    
    /// Combine genes from two parents (Optional for GA, required for Crossover)
    fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self;
}

impl Mutatable for MappingStrategy {
    fn mutate(&self, rng: &mut StdRng) -> Self {
        let mut child = self.clone();
        
        // Randomly select mutation target
        match rng.gen_range(0..3) {
            0 => {
                // Mutate Tile Sizes
                if !child.tile_sizes.is_empty() {
                    let keys: Vec<String> = child.tile_sizes.keys().cloned().collect();
                    let key = &keys[rng.gen_range(0..keys.len())];
                    let current = child.tile_sizes[key];
                    
                    // Perturb tile size (Power of 2 friendly)
                    let new_val = if rng.gen_bool(0.5) {
                        (current * 2).min(2048)
                    } else {
                        (current / 2).max(1)
                    };
                    child.tile_sizes.insert(key.clone(), new_val);
                }
            },
            1 => {
                // Mutate Loop Order
                if child.loop_order.len() > 1 {
                    let i = rng.gen_range(0..child.loop_order.len());
                    let j = rng.gen_range(0..child.loop_order.len());
                    child.loop_order.swap(i, j);
                }
            },
            2 => {
                // Mutate Spatial Mapping (Reassign a dimension to a different hardware level)
                if !child.spatial_map.is_empty() {
                    let keys: Vec<String> = child.spatial_map.keys().cloned().collect();
                    let key = &keys[rng.gen_range(0..keys.len())];
                    
                    let levels = vec!["BlockX", "BlockY", "ThreadX", "ThreadY", "Warp"];
                    let new_level = HardwareLevel(levels[rng.gen_range(0..levels.len())].to_string());
                    child.spatial_map.insert(key.clone(), new_level);
                }
            },
            _ => unreachable!(),
        }
        
        child
    }

    fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self {
        let mut child = self.clone();
        
        // Mix tile sizes
        for (k, v) in &other.tile_sizes {
            if rng.gen_bool(0.5) {
                child.tile_sizes.insert(k.clone(), *v);
            }
        }
        
        // Mix spatial mapping
        for (k, v) in &other.spatial_map {
            if rng.gen_bool(0.5) {
                child.spatial_map.insert(k.clone(), v.clone());
            }
        }
        
        child
    }
}
