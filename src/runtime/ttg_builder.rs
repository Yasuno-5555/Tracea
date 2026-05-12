use crate::core::ttg::{TTGLayout, TileMetadata, TileTopology};
use crate::core::device::DeviceProfile;
use crate::core::config::PipelineConfig;

pub struct TTGBuilder;

impl TTGBuilder {
    pub fn new() -> Self {
        Self
    }

    /// Hardware-aware tile configuration optimizer.
    /// Given a device profile and problem size, returns optimal tile configs
    /// ranked by estimated occupancy/utilization.
    ///
    /// Metal M1 example: 32KB threadgroup memory → 64×64×32 tiles with 8 simdgroups
    /// for high arithmetic intensity vs 32×32's 4 simdgroups.
    pub fn optimize_tile_configs(device: &DeviceProfile, m: u32, n: u32, _k: u32) -> Vec<PipelineConfig> {
        let max_threads = device.max_threads_per_block;
        let max_smem = device.local_memory_size;
        let simd_width = device.simd_width;
        let is_metal = device.backend == crate::core::device::BackendType::Metal;

        // tile candidates to evaluate: (mt, nt, kt)
        // For Metal (32KB), sStore dominates single-buffer smem cost,
        // so double-buffer variants with direct store enable larger tiles.
        let tiles: Vec<(u32, u32, u32)> = match (max_smem, is_metal) {
            // Metal: 32KB threadgroup memory
            _ if max_smem <= 48 * 1024 => vec![
                // Small tiles (always fit, even with sStore)
                (16, 16, 16), (32, 16, 16), (16, 32, 16),
                (32, 32, 16), (32, 32, 32),
                // Medium tiles (fit single buffer)
                (64, 32, 16), (32, 64, 16), (64, 64, 16),
                (64, 32, 32), (32, 64, 32),
                // Large tiles (fit with double buffer + direct store)
                (64, 64, 32),
                // Larger tiles require double buffer + direct store (no sStore)
                (128, 64, 16), (64, 128, 16),
                // Max tile for 32KB (double buffer + direct store only)
                (128, 128, 16),
            ],
            // CUDA: 48KB+ shared memory
            _ => vec![
                (64, 64, 16), (64, 64, 32), (128, 64, 16), (64, 128, 16),
                (128, 128, 16), (128, 64, 32), (64, 128, 32),
            ],
        };

        let mut scored: Vec<(PipelineConfig, f32)> = Vec::new();

        for &(mt, nt, kt) in &tiles {
            if mt > m || nt > n { continue; }
            let sub_tiles = (mt / 8) * (nt / 8);  // total 8x8 sub-tiles in tile
            if sub_tiles == 0 { continue; }

            for &db in &[false, true] {
                // smem = sA + sB (+ sStore for single buffer or epilogue path)
                let buf_mult = if db { 2usize } else { 1 };
                let s_a = buf_mult * (mt as usize * kt as usize) * 2;
                let s_b = buf_mult * (kt as usize * nt as usize) * 2;
                // Double buffer with direct device store: no sStore needed
                // Single buffer: always uses sStore
                // Conservative: assume sStore for single buffer, no sStore for double buffer
                let s_store = if db { 0 } else { mt as usize * nt as usize * 4 };
                let smem_needed = s_a + s_b + s_store;
                if smem_needed > max_smem { continue; }

                // Max safe simdgroups = n_subtiles (8x8 column positions).
                // The current template's sg_base_row/col formula requires
                // exactly n_subtiles SIMD groups for correct coverage.
                let n_subtiles_val = nt / 8;
                let max_sg = (max_threads / simd_width) as u32;
                let usable_sg = n_subtiles_val.min(max_sg).max(1);

                // Work per thread: output elements ÷ total threads
                let total_threads = usable_sg * 32;
                let output_elements = mt * nt;
                let work_per_thread = output_elements as f32 / total_threads as f32;

                // Arithmetic intensity: compute ops ÷ memory bytes
                // ops = 2 * M * N * K (multiply + accumulate per element)
                // bytes = load(M*K + K*N) + store(M*N), each FP16=2B, FP32=4B
                let compute_ops = 2.0f64 * mt as f64 * nt as f64 * kt as f64;
                let load_bytes = (mt as f64 * kt as f64 + kt as f64 * nt as f64) * 2.0
                    * (if db { 2 } else { 1 }) as f64;
                let store_bytes = if db { 0.0 } else { mt as f64 * nt as f64 * 4.0 };
                let total_bytes = load_bytes + store_bytes;
                let arith_intensity = if total_bytes > 0.0 {
                    compute_ops / total_bytes
                } else {
                    compute_ops / load_bytes
                };

                // Normalized metrics [0..1]
                let mem_efficiency = smem_needed as f32 / max_smem as f32;
                let intensity_score = (arith_intensity as f32 / 64.0).min(1.0); // higher kt → better
                let work_efficiency = (work_per_thread / 64.0).min(1.0);         // target ~64 elements/thread

                // Penalize very small work-per-thread (occupancy starvation)
                let work_penalty = if work_per_thread < 8.0 { 0.6 }
                                  else if work_per_thread < 16.0 { 0.85 }
                                  else { 1.0 };

                // Combined score: prioritize arithmetic intensity and good thread utilization
                let score = if is_metal {
                    // Metal: memory bandwidth is the bottleneck, so intensity matters most
                    intensity_score * 0.45 + mem_efficiency * 0.25 + work_efficiency * 0.15 + (work_penalty - 0.85) * 0.15
                } else {
                    intensity_score * 0.4 + mem_efficiency * 0.25 + work_efficiency * 0.2 + (work_penalty - 0.85) * 0.15
                };

                let mut cfg = PipelineConfig::new(2, mt, nt, kt);
                cfg.double_buffer = db;
                if is_metal {
                    cfg.instruction = crate::core::config::SpecializedInstruction::MetalSimdGroup;
                }
                cfg.force_num_warps = Some(usable_sg);
                scored.push((cfg, score));

                // Fusion_count=2 variant (TTG row-adjacent)
                if is_metal && nt <= 32 && nt * 2 <= n {
                    let mut cfg2 = PipelineConfig::new(2, mt, nt, kt);
                    cfg2.double_buffer = db;
                    cfg2.fusion_count = 2;
                    cfg2.instruction = crate::core::config::SpecializedInstruction::MetalSimdGroup;
                    cfg2.force_num_warps = Some(usable_sg);
                    scored.push((cfg2, score * 1.1));
                }
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(cfg, _)| cfg).collect()
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
            let tile_m = tile_idx / grid_n;
            let tile_n = tile_idx % grid_n;
            l1_map.push(tile_idx);
            let is_boundary = if (tile_m + 1) * mt > m || (tile_n + 1) * nt > n { 1 } else { 0 };
            l2_table.push(TileMetadata {
                region_m: tile_m, region_n: tile_n,
                k_start: 0, k_end: k, role: is_boundary,
            });
        }

        // Build topology with K-split support (default: no split)
        let topology = TileTopology::from_grid(grid_m, grid_n, mt, nt, m, n, k);

        TTGLayout { l1_map, l2_table, num_active_tiles: num_tiles, variant: None, topology: Some(topology) }
    }

    /// Creates a dense layout with K-split topology.
    /// Each tile is split into `k_splits` KPipeline stages.
    pub fn from_dense_k_split(m: u32, n: u32, k: u32, mt: u32, nt: u32, _kt: u32, k_splits: u32) -> TTGLayout {
        let grid_m = (m + mt - 1) / mt;
        let grid_n = (n + nt - 1) / nt;
        let kt_per_stage = k / k_splits.max(1);
        let total_nodes = (grid_m * grid_n * k_splits) as usize;

        let mut l1_map = Vec::with_capacity(total_nodes);
        let mut l2_table = Vec::with_capacity(total_nodes);
        let mut node_idx = 0u32;

        for tm in 0..grid_m {
            for tn in 0..grid_n {
                for ks in 0..k_splits {
                    l1_map.push(node_idx);
                    let k_start = ks * kt_per_stage;
                    let k_end = if ks + 1 == k_splits { k } else { (ks + 1) * kt_per_stage };
                    let is_boundary = if (tm + 1) * mt > m || (tn + 1) * nt > n { 1 } else { 0 };
                    l2_table.push(TileMetadata {
                        region_m: tm, region_n: tn, k_start, k_end, role: is_boundary,
                    });
                    node_idx += 1;
                }
            }
        }

        let topology = TileTopology::from_grid_k_split(grid_m, grid_n, mt, nt, m, n, k, k_splits);

        TTGLayout { l1_map, l2_table, num_active_tiles: total_nodes as u32, variant: None, topology: Some(topology) }
    }

    /// Creates a layout from an inclusion mask (non-rectangular/sparse).
    pub fn from_mask<F>(m: u32, n: u32, k: u32, mt: u32, nt: u32, _kt: u32, include_fn: F) -> TTGLayout
    where F: Fn(u32, u32) -> bool {
        let grid_m = (m + mt - 1) / mt;
        let grid_n = (n + nt - 1) / nt;
        let mut node_idx = 0u32;
        let mut l1_map = Vec::new();
        let mut l2_table = Vec::new();
        let mut l2_idx_map = vec![u32::MAX; (grid_m * grid_n) as usize];

        for tm in 0..grid_m {
            for tn in 0..grid_n {
                if !include_fn(tm, tn) { continue; }
                let idx = (tm * grid_n + tn) as usize;
                l2_idx_map[idx] = node_idx;
                l1_map.push(node_idx);
                let _is_boundary = if (tm + 1) * mt > m || (tn + 1) * nt > n { 1 } else { 0 };
                l2_table.push(TileMetadata {
                    region_m: tm, region_n: tn, k_start: 0, k_end: k, role: 0,
                });
                node_idx += 1;
            }
        }

        let topology = TileTopology::from_mask(grid_m, grid_n, mt, nt, m, n, k, include_fn);

        TTGLayout { l1_map, l2_table, num_active_tiles: node_idx, variant: None, topology: Some(topology) }
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
            topology: None,
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
            topology: None,
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
                    variant: None, topology: None,
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
                    l1_map, l2_table, num_active_tiles: grid_s, variant: None, topology: None
                }
            },
            (OperatorTopology::Softmax { .. }, _) => {
                TTGLayout {
                    l1_map: vec![0],
                    l2_table: vec![TileMetadata { region_m: 0, region_n: 0, k_start: 0, k_end: 0, role: 0 }],
                    num_active_tiles: 1,
                    variant: None, topology: None,
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
                    l1_map, l2_table, num_active_tiles: *c as u32, variant: None, topology: None
                }
            },
            (OperatorTopology::Linear { batch, m, n, k: _, .. }, _) => {
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
                    l1_map, l2_table, num_active_tiles: num_tiles as u32, variant: None, topology: None
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

    #[test]
    fn test_ttg_dense_and_diagonal_from_policy() {
        use crate::policy::types::{OperatorTopology, TilePolicy, ActivityPattern, TilingKind, TopologyKind};
        
        let op = OperatorTopology::Gemm {
            op_id: 2,
            name: "GEMM".to_string(),
            m: 128,
            n: 128,
            k: 64,
            batch: 1,
            kind: TopologyKind::Dense,
            epilogue: vec![],
        };
        
        // 1. AllActive (Dense)
        let policy_dense = TilePolicy::Gemm {
            operator_id: 2,
            tile_shape: [64, 64, 32],
            activity_pattern: ActivityPattern::AllActive,
            tiling_kind: TilingKind::Dense,
            variant: crate::core::config::GemmVariant::Tiled,
        };
        
        let layout_dense = TTGBuilder::from_policy(&op, &policy_dense);
        // (128/64) * (128/64) = 2 * 2 = 4 tiles
        assert_eq!(layout_dense.num_active_tiles, 4);
        assert_eq!(layout_dense.l1_map.len(), 4);
        assert_eq!(layout_dense.l2_table.len(), 4);
        
        // 2. DiagonalOnly
        let policy_diag = TilePolicy::Gemm {
            operator_id: 2,
            tile_shape: [64, 64, 32],
            activity_pattern: ActivityPattern::DiagonalOnly,
            tiling_kind: TilingKind::Dense,
            variant: crate::core::config::GemmVariant::Tiled,
        };
        
        let layout_diag = TTGBuilder::from_policy(&op, &policy_diag);
        // Diagonal of 2x2 grid is 2 tiles
        assert_eq!(layout_diag.num_active_tiles, 2);
        assert_eq!(layout_diag.l1_map.len(), 2);
        assert_eq!(layout_diag.l2_table.len(), 2);
        assert_eq!(layout_diag.l2_table[0].region_m, layout_diag.l2_table[0].region_n);
        assert_eq!(layout_diag.l2_table[1].region_m, layout_diag.l2_table[1].region_n);
    }

    #[test]
    #[should_panic(expected = "Mismatching Policy and Operator Topology Types!")]
    fn test_ttg_mismatched_policy_should_panic() {
        use crate::policy::types::{OperatorTopology, TilePolicy, ActivityPattern, TilingKind};
        
        let op = OperatorTopology::Attention {
            op_id: 3,
            name: "Attention".to_string(),
            b: 2,
            s: 128,
            h: 4,
            d: 64,
        };
        
        // Mismatched policy: Gemm policy for Attention op
        let policy = TilePolicy::Gemm {
            operator_id: 3,
            tile_shape: [64, 64, 32],
            activity_pattern: ActivityPattern::AllActive,
            tiling_kind: TilingKind::Dense,
            variant: crate::core::config::GemmVariant::Tiled,
        };
        
        let _ = TTGBuilder::from_policy(&op, &policy);
    }
}
