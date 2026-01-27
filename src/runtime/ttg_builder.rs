use crate::core::ttg::{TTGLayout, TileMetadata};

pub struct TTGBuilder;

impl TTGBuilder {
    pub fn new() -> Self {
        Self
    }

    /// Creates a dense TTG layout for testing purposes.
    /// Essentially maps the entire (M/MT, N/NT) grid to linear indices.
    pub fn from_dense(m: u32, n: u32, k: u32, mt: u32, nt: u32, _kt: u32) -> TTGLayout {
        let grid_m = (m + mt - 1) / mt;
        let grid_n = (n + nt - 1) / nt;
        let num_tiles = grid_m * grid_n;

        let mut l1_map = Vec::with_capacity(num_tiles as usize);
        let mut l2_table = Vec::with_capacity(num_tiles as usize);

        for tile_idx in 0..num_tiles {
            // Standard row-major layout for dense
            // Global Tile ID: tile_idx
            // For dense, physical_id == logical_id.
            
            // Map logical ID back to coordinates
            // Assuming row-major grid: idx = m_grid * GRID_N + n_grid
            let tile_m = tile_idx / grid_n;
            let tile_n = tile_idx % grid_n;
            
            // L1: Simply identity mapping for dense
            l1_map.push(tile_idx);

            // L2: Metadata
            let is_boundary = if (tile_m + 1) * mt > m || (tile_n + 1) * nt > n {
               1 
            } else {
               0
            };

            l2_table.push(TileMetadata {
                region_m: tile_m,
                region_n: tile_n,
                k_start: 0,
                k_end: k,
                role: is_boundary,
            });
        }

        TTGLayout {
            l1_map,
            l2_table,
            num_active_tiles: num_tiles,
        }
    }

    /// Creates a sparse TTG layout with only diagonal tiles.
    pub fn from_diagonal(m: u32, n: u32, k: u32, mt: u32, nt: u32, _kt: u32) -> TTGLayout {
        let grid_m = (m + mt - 1) / mt;
        let grid_n = (n + nt - 1) / nt;
        
        let mut l1_map = Vec::new();
        let mut l2_table = Vec::new();
        
        for tm in 0..grid_m {
            for tn in 0..grid_n {
                // Diagonal check
                if tm == tn {
                    l1_map.push(l2_table.len() as u32);
                    let is_boundary = if (tm + 1) * mt > m || (tn + 1) * nt > n { 1 } else { 0 };
                    l2_table.push(TileMetadata {
                        region_m: tm,
                        region_n: tn,
                        k_start: 0,
                        k_end: k,
                        role: is_boundary,
                    });
                }
            }
        }
        
        TTGLayout {
            l1_map,
            l2_table: l2_table.clone(), // Clone to satisfy struct (or we could structure differently)
            num_active_tiles: l2_table.len() as u32,
        }
    }

    /// Creates a sparse TTG layout with random tiles based on density.
    pub fn from_random(m: u32, n: u32, k: u32, mt: u32, nt: u32, _kt: u32, density: f64) -> TTGLayout {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let grid_m = (m + mt - 1) / mt;
        let grid_n = (n + nt - 1) / nt;
        
        let mut l1_map = Vec::new();
        let mut l2_table = Vec::new();
        
        for tm in 0..grid_m {
            for tn in 0..grid_n {
                if rng.gen_bool(density) {
                    l1_map.push(l2_table.len() as u32);
                    let is_boundary = if (tm + 1) * mt > m || (tn + 1) * nt > n { 1 } else { 0 };
                    l2_table.push(TileMetadata {
                        region_m: tm,
                        region_n: tn,
                        k_start: 0,
                        k_end: k,
                        role: is_boundary,
                    });
                }
            }
        }
        
        TTGLayout {
            l1_map,
            l2_table: l2_table.clone(),
            num_active_tiles: l2_table.len() as u32,
        }
    }

    /// Creates a TTG layout driven by a Policy decision.
    pub fn from_policy(op: &crate::policy::types::OperatorTopology, policy: &crate::policy::types::TilePolicy) -> TTGLayout {
        let m = op.m;
        let n = op.n;
        let k = op.k;
        let mt = policy.tile_shape[0];
        let nt = policy.tile_shape[1];
        let kt = policy.tile_shape[2];
        
        match policy.activity_pattern {
            crate::policy::types::ActivityPattern::AllActive => {
                Self::from_dense(m, n, k, mt, nt, kt)
            },
            crate::policy::types::ActivityPattern::DiagonalOnly => {
                Self::from_diagonal(m, n, k, mt, nt, kt)
            },
            crate::policy::types::ActivityPattern::RandomDrop { keep_ratio } => {
                Self::from_random(m, n, k, mt, nt, kt, keep_ratio as f64)
            },
            _ => {
                // Fallback to tiling_kind if activity_pattern doesn't match
                match &policy.tiling_kind {
                    crate::policy::types::TilingKind::LowRank { r, .. } => {
                        Self::from_low_rank(m, n, k, *r, mt, nt, kt)
                    },
                    _ => Self::from_dense(m, n, k, mt, nt, kt)
                }
            }
        }
    }

    /// Creates a 2-hop Low-Rank TTG layout.
    pub fn from_low_rank(m: u32, n: u32, k: u32, _r: u32, mt: u32, nt: u32, kt: u32) -> TTGLayout {
        let mut layout = Self::from_dense(m, n, k, mt, nt, kt);
        
        // Mark as low-rank in role field (bit 2)
        for meta in &mut layout.l2_table {
            meta.role |= 0x2; 
        }

        layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttg_low_rank() {
        let layout = TTGBuilder::from_low_rank(256, 256, 256, 64, 64, 64, 32);
        assert_eq!(layout.num_active_tiles, 16); // (256/64)*(256/64) = 4*4 = 16
        for meta in &layout.l2_table {
            assert!((meta.role & 0x2) != 0); // Check low-rank bit
        }
    }
}
