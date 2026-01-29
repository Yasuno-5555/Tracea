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
            variant: None,
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
            variant: None,
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
            variant: None,
        }
    }

    /// Creates a TTG layout driven by a Policy decision.
    pub fn from_policy(op: &crate::policy::types::OperatorTopology, policy: &crate::policy::types::TilePolicy) -> TTGLayout {
        use crate::policy::types::{OperatorTopology, TilePolicy};
        
        match (op, policy) {
            (OperatorTopology::Gemm { m, n, k, .. }, TilePolicy::Gemm { tile_shape, activity_pattern, tiling_kind, variant, .. }) => {
                let mt = tile_shape[0];
                let nt = tile_shape[1];
                let kt = tile_shape[2];
                
                let mut layout = match activity_pattern {
                     crate::policy::types::ActivityPattern::AllActive => {
                         Self::from_dense(*m, *n, *k, mt, nt, kt)
                     },
                     crate::policy::types::ActivityPattern::DiagonalOnly => {
                         Self::from_diagonal(*m, *n, *k, mt, nt, kt)
                     },
                     crate::policy::types::ActivityPattern::RandomDrop { keep_ratio } => {
                         Self::from_random(*m, *n, *k, mt, nt, kt, *keep_ratio as f64)
                     },
                     _ => {
                         // Fallback to tiling_kind if activity_pattern doesn't match
                         match tiling_kind {
                             crate::policy::types::TilingKind::LowRank { r, .. } => {
                                 Self::from_low_rank(*m, *n, *k, *r, mt, nt, kt)
                             },
                             _ => Self::from_dense(*m, *n, *k, mt, nt, kt)
                         }
                     }
                };
                layout.variant = Some(*variant);
                layout
            },
            (OperatorTopology::Attention { b, s, h, d, .. }, TilePolicy::Attention { qk_tile, .. }) => {
                // Metal Attention Logic
                // Naive Mapping: 1 Tile = 1 Threadgroup processing (BlockM, Head, Batch)
                // Grid: (S/BlockM, H, B)
                let mt = qk_tile.0; // e.g. 64
                // let nt = qk_tile.1; // e.g. 64 (KV Block Size)
                
                let grid_m = (s + mt - 1) / mt;
                let num_tiles = grid_m * h * b;
                
                let mut l1_map = Vec::with_capacity(num_tiles as usize);
                let mut l2_table = Vec::with_capacity(num_tiles as usize);
                
                // Construct linear layout
                for i in 0..num_tiles {
                     l1_map.push(i);
                     
                     // Decode i back to (s_idx, h_idx, b_idx)
                     // i = b * (H*S_grid) + h * S_grid + s_idx
                     let s_grid_dim = grid_m;
                     let h_grid_dim = h;
                     
                     let temp = i;
                     let s_idx = temp % s_grid_dim;
                     let rem = temp / s_grid_dim;
                     let h_idx = rem % h_grid_dim;
                     let b_idx = rem / h_grid_dim;
                     
                     l2_table.push(TileMetadata {
                         region_m: s_idx, 
                         region_n: h_idx, // Encode Head in Region N?
                         k_start: b_idx,  // Encode Batch in K Start?
                         k_end: *d,       // Head Dim in K End?
                         role: 0,
                     });
                }
                
                TTGLayout {
                    l1_map,
                    l2_table,
                    num_active_tiles: num_tiles,
                    variant: None,
                }
            },
            (OperatorTopology::Elementwise { .. } | OperatorTopology::Relu { .. } | OperatorTopology::BatchNorm { .. }, _) => {
                // Parallelize elementwise ops over 1D blocks
                let total = match op {
                    OperatorTopology::BatchNorm { n, c, h, w, .. } => (*n * *c * *h * *w) as u32,
                    OperatorTopology::Relu { n, .. } => *n as u32,
                    OperatorTopology::Elementwise { n, .. } => *n as u32,
                    _ => 1024,
                };
                
                let tile_s = 1024; // 1D Tile
                let grid_s = (total + tile_s - 1) / tile_s;
                
                let mut l1_map = Vec::with_capacity(grid_s as usize);
                let mut l2_table = Vec::with_capacity(grid_s as usize);
                for i in 0..grid_s {
                    l1_map.push(i);
                    l2_table.push(TileMetadata { region_m: i, region_n: 0, k_start: 0, k_end: 0, role: 0 });
                }
                TTGLayout {
                    l1_map, l2_table, num_active_tiles: grid_s, variant: None
                }
            },
            (OperatorTopology::Softmax { .. }, _) => {
                TTGLayout {
                    l1_map: vec![0],
                    l2_table: vec![TileMetadata { region_m: 0, region_n: 0, k_start: 0, k_end: 0, role: 0 }],
                    num_active_tiles: 1,
                    variant: None,
                }
            },
            (OperatorTopology::GlobalAveragePool { n: _, c, h: _, w: _, .. }, _) => {
                // Partition by Channel
                let mut l1_map = Vec::with_capacity(*c);
                let mut l2_table = Vec::with_capacity(*c);
                for i in 0..*c {
                    l1_map.push(i as u32);
                    l2_table.push(TileMetadata { region_m: i as u32, region_n: 0, k_start: 0, k_end: 0, role: 0 });
                }
                TTGLayout {
                    l1_map, l2_table, num_active_tiles: *c as u32, variant: None
                }
            },
            (OperatorTopology::Linear { batch, m, n, k, .. }, _) => {
                // Same as Gemm
                let mt = 32; let nt = 32;
                let m_grid = (*m + mt - 1) / mt;
                let n_grid = (*n + nt - 1) / nt;
                let num_tiles = m_grid * n_grid * *batch;
                let mut l1_map = Vec::with_capacity(num_tiles);
                let mut l2_table = Vec::with_capacity(num_tiles);
                for b in 0..*batch {
                    for i in 0..m_grid {
                        for j in 0..n_grid {
                            l1_map.push((b * m_grid * n_grid + i * n_grid + j) as u32);
                            l2_table.push(TileMetadata {
                                region_m: i as u32,
                                region_n: j as u32,
                                k_start: b as u32,
                                k_end: 0,
                                role: 0,
                            });
                        }
                    }
                }
                TTGLayout {
                    l1_map, l2_table, num_active_tiles: num_tiles as u32, variant: None
                }
            },
            (OperatorTopology::Conv2d { n, k, h, w, r, s, stride, padding, .. }, _) => {
                // Implicit GEMM Mapping: M = N*H_out*W_out, N = K
                let h_out = (*h + 2 * padding - *r) / *stride + 1;
                let w_out = (*w + 2 * padding - *s) / *stride + 1;
                let m = (*n as u32) * (h_out as u32) * (w_out as u32);
                let n_out = *k as u32;
                Self::from_dense(m, n_out, 32, 64, 64, 32)
            },
            _ => panic!("Mismatching Policy and Operator Topology Types!"),
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

    #[test]
    fn test_ttg_attention() {
        use crate::policy::types::{OperatorTopology, TilePolicy};
        
        let op = OperatorTopology::Attention {
            op_id: 1,
            name: "Attention".to_string(),
            b: 2,
            s: 128,
            h: 4,
            d: 64,
        };
        
        let policy = TilePolicy::Attention {
            operator_id: 1,
            qk_tile: (64, 64),
            v_tile: (64, 32),
            variant: crate::core::config::AttentionVariant::Naive,
        };
        
        let layout = TTGBuilder::from_policy(&op, &policy);
        
        // Grid M = 128/64 = 2.
        // H = 4.
        // B = 2.
        // Total Tiles = 2 * 4 * 2 = 16.
        assert_eq!(layout.num_active_tiles, 16);
        assert_eq!(layout.l1_map.len(), 16);
        assert_eq!(layout.l2_table.len(), 16);
        
        // Verify mapping of last tile
        // Linear Index 15.
        // 15 = 1*(4*2) + 3*(2) + 1  => b=1, h=3, s=1
        // Formula used in builder: i = b * (H*S_grid) + h * S_grid + s_idx? 
        // Or reverse? 
        // Builder Code:
        // let s_idx = i % s_grid;
        // let rem = i / s_grid;
        // let h_idx = rem % h;
        // let b_idx = rem / h;
        //
        // 15:
        // s_idx = 15 % 2 = 1.
        // rem = 7.
        // h_idx = 7 % 4 = 3.
        // b_idx = 7 / 4 = 1.
        // Matches: b=1, h=3, s=1.
        
        let last_meta = &layout.l2_table[15];
        assert_eq!(last_meta.region_m, 1); // s_idx
        assert_eq!(last_meta.region_n, 3); // h_idx
        assert_eq!(last_meta.k_start, 1);  // b_idx
    }
}
