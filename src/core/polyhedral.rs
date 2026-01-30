// src/core/polyhedral.rs

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilingStrategy {
    pub tile_sizes: Vec<usize>,
    pub loop_order: Vec<usize>,
}
