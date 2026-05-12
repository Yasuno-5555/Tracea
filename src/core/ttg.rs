pub type LogicalID = u32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileMetadata {
    pub region_m: u32,
    pub region_n: u32,
    pub k_start: u32,
    pub k_end: u32,
    pub role: u32,       // 0=Main, 1=Boundary, 2=Skip
}

/// Directed dependency between two tiles.
/// If tile `from` must complete before `to`, this edge encodes that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileEdge {
    pub from: u32,
    pub to: u32,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Data dependency: `to` reads output of `from`
    DataDep,
    /// Adjacent in M dimension (share rows of A) → fusion opportunity
    RowAdjacent,
    /// Adjacent in N dimension (share columns of B) → fusion opportunity
    ColAdjacent,
    /// K-dimension pipeline (same tile, next K step)
    KPipeline,
}

/// Data reuse hint derived from adjacency relationships.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseHint {
    /// No reuse possible
    None,
    /// This tile can reuse A data from `row_source` tile (same row, adjacent column)
    ReuseA { row_source: u32 },
    /// This tile can reuse B data from `col_source` tile (same column, adjacent row)
    ReuseB { col_source: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleHint {
    Normal,
    /// High priority — execute early for cache warm
    WarmCache,
    /// Low priority — defer to batch with neighbors
    DeferFusion,
}

#[derive(Debug, Clone)]
pub struct TileNode {
    pub tile_id: u32,
    pub meta: TileMetadata,
    pub schedule_hint: ScheduleHint,
    pub reuse_hint: ReuseHint,
}

/// Topological representation of tile relationships.
#[derive(Debug, Clone)]
pub struct TileTopology {
    pub nodes: Vec<TileNode>,
    pub edges: Vec<TileEdge>,
    pub grid_m: u32,
    pub grid_n: u32,
    /// K dimension split count (0 = no split)
    pub k_splits: u32,
}

impl TileTopology {
    /// Build topology from a rectangular grid (dense).
    pub fn from_grid(grid_m: u32, grid_n: u32, tile_size_m: u32, tile_size_n: u32,
                     total_m: u32, total_n: u32, k_total: u32) -> Self {
        let mut nodes = Vec::with_capacity((grid_m * grid_n) as usize);
        let mut edges = Vec::new();

        for tm in 0..grid_m {
            for tn in 0..grid_n {
                let tile_id = tm * grid_n + tn;
                let is_boundary = if (tm + 1) * tile_size_m > total_m
                                    || (tn + 1) * tile_size_n > total_n { 1 } else { 0 };
                nodes.push(TileNode {
                    tile_id,
                    meta: TileMetadata {
                        region_m: tm, region_n: tn,
                        k_start: 0, k_end: k_total,
                        role: is_boundary,
                    },
                    schedule_hint: ScheduleHint::Normal,
                    reuse_hint: ReuseHint::None,
                });

                if tn > 0 {
                    edges.push(TileEdge { from: tm * grid_n + (tn - 1), to: tile_id, kind: EdgeKind::RowAdjacent });
                }
                if tm > 0 {
                    edges.push(TileEdge { from: (tm - 1) * grid_n + tn, to: tile_id, kind: EdgeKind::ColAdjacent });
                }
            }
        }

        // Attach reuse hints from edges
        Self { nodes, edges, grid_m, grid_n, k_splits: 0 }.compute_reuse_hints()
    }

    /// Build topology from a grid with K-dimension splitting.
    /// Each logical tile is split into `k_split_count` pipeline stages.
    pub fn from_grid_k_split(grid_m: u32, grid_n: u32, tile_size_m: u32, tile_size_n: u32,
                              total_m: u32, total_n: u32, k_total: u32, k_split_count: u32) -> Self {
        let kt_per_stage = k_total / k_split_count.max(1);
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut tile_counter = 0u32;

        for tm in 0..grid_m {
            for tn in 0..grid_n {
                for ks in 0..k_split_count {
                    let tile_id = tile_counter;
                    tile_counter += 1;
                    let k_start = ks * kt_per_stage;
                    let k_end = if ks + 1 == k_split_count { k_total } else { (ks + 1) * kt_per_stage };
                    let is_boundary = if (tm + 1) * tile_size_m > total_m || (tn + 1) * tile_size_n > total_n { 1 } else { 0 };

                    nodes.push(TileNode {
                        tile_id,
                        meta: TileMetadata { region_m: tm, region_n: tn, k_start, k_end, role: is_boundary },
                        schedule_hint: ScheduleHint::Normal,
                        reuse_hint: ReuseHint::None,
                    });

                    // KPipeline edge: this K-stage depends on the previous
                    if ks > 0 {
                        edges.push(TileEdge { from: tile_id - 1, to: tile_id, kind: EdgeKind::KPipeline });
                    }
                    // Row adjacency for first K-stage of each tile
                    if tn > 0 && ks == 0 {
                        let neighbor = tm * grid_n * k_split_count + (tn - 1) * k_split_count;
                        edges.push(TileEdge { from: neighbor, to: tile_id, kind: EdgeKind::RowAdjacent });
                    }
                    // Col adjacency for first K-stage
                    if tm > 0 && ks == 0 {
                        let neighbor = (tm - 1) * grid_n * k_split_count + tn * k_split_count;
                        edges.push(TileEdge { from: neighbor, to: tile_id, kind: EdgeKind::ColAdjacent });
                    }
                }
            }
        }

        Self { nodes, edges, grid_m, grid_n, k_splits: k_split_count }.compute_reuse_hints()
    }

    /// Build topology from an inclusion mask (non-rectangular).
    /// `include_fn(tm, tn) -> bool` controls which tiles are active.
    /// Only active tiles become nodes; edges connect active neighbors.
    pub fn from_mask<F>(grid_m: u32, grid_n: u32, tile_size_m: u32, tile_size_n: u32,
                         total_m: u32, total_n: u32, k_total: u32,
                         include_fn: F) -> Self
    where F: Fn(u32, u32) -> bool {
        let mut l2_idx = 0u32;
        let mut l2_map = vec![u32::MAX; (grid_m * grid_n) as usize];
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for tm in 0..grid_m {
            for tn in 0..grid_n {
                if !include_fn(tm, tn) { continue; }
                let tile_id = l2_idx;
                l2_map[(tm * grid_n + tn) as usize] = tile_id;
                l2_idx += 1;
                let is_boundary = if (tm + 1) * tile_size_m > total_m || (tn + 1) * tile_size_n > total_n { 1 } else { 0 };
                nodes.push(TileNode {
                    tile_id,
                    meta: TileMetadata { region_m: tm, region_n: tn, k_start: 0, k_end: k_total, role: is_boundary },
                    schedule_hint: ScheduleHint::Normal,
                    reuse_hint: ReuseHint::None,
                });
            }
        }

        // Adjacency edges between active neighbors
        for tm in 0..grid_m {
            for tn in 0..grid_n {
                let this_id = l2_map[(tm * grid_n + tn) as usize];
                if this_id == u32::MAX { continue; }
                if tn > 0 {
                    if let Some(&neighbor) = l2_map.get((tm * grid_n + tn - 1) as usize) {
                        if neighbor != u32::MAX {
                            edges.push(TileEdge { from: neighbor, to: this_id, kind: EdgeKind::RowAdjacent });
                        }
                    }
                }
                if tm > 0 {
                    if let Some(&neighbor) = l2_map.get(((tm - 1) * grid_n + tn) as usize) {
                        if neighbor != u32::MAX {
                            edges.push(TileEdge { from: neighbor, to: this_id, kind: EdgeKind::ColAdjacent });
                        }
                    }
                }
            }
        }

        Self { nodes, edges, grid_m, grid_n, k_splits: 0 }.compute_reuse_hints()
    }

    /// Attach data reuse hints from adjacency edges.
    fn compute_reuse_hints(mut self) -> Self {
        for i in 0..self.nodes.len() {
            let tm = self.nodes[i].meta.region_m;
            let tn = self.nodes[i].meta.region_n;
            // Find row neighbor (same tm, adjacent tn) for A reuse
            let row_src = if tn > 0 { Some(tm * self.grid_n + tn - 1) } else { None };
            // Find col neighbor (adjacent tm, same tn) for B reuse
            let col_src = if tm > 0 { Some((tm - 1) * self.grid_n + tn) } else { None };

            self.nodes[i].reuse_hint = match (row_src, col_src) {
                (Some(r), Some(c)) if r < self.nodes.len() as u32 && c < self.nodes.len() as u32 => {
                    // Check both neighbors are valid (within the active node set)
                    if self.nodes.iter().any(|n| n.tile_id == r) && self.nodes.iter().any(|n| n.tile_id == c) {
                        // Prefer A-reuse (row adjacency is more beneficial for row-major GEMM)
                        if tn > 0 { ReuseHint::ReuseA { row_source: r } }
                        else { ReuseHint::ReuseB { col_source: c } }
                    } else { ReuseHint::None }
                }
                (Some(r), _) if r < self.nodes.len() as u32 && self.nodes.iter().any(|n| n.tile_id == r) => {
                    ReuseHint::ReuseA { row_source: r }
                }
                (_, Some(c)) if c < self.nodes.len() as u32 && self.nodes.iter().any(|n| n.tile_id == c) => {
                    ReuseHint::ReuseB { col_source: c }
                }
                _ => ReuseHint::None,
            };
        }
        self
    }

    /// Kahn topological sort: produce execution order respecting all edges.
    /// For normal grids (no K-split), this is the same as row-major.
    /// For K-split grids and sparse masks, this enables proper pipelining.
    pub fn topological_sort(&self) -> Result<Vec<usize>, String> {
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for edge in &self.edges {
            let from = edge.from as usize;
            let to = edge.to as usize;
            if from < n && to < n {
                adj[from].push(to);
                in_degree[to] += 1;
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push(next);
                }
            }
        }

        if order.len() != n {
            return Err(format!("Cycle detected: {}/{} nodes processed", order.len(), n));
        }
        Ok(order)
    }

    /// Execution order optimized for data reuse.
    /// Dense grids → row-major (A data stays hot for consecutive tiles in same row).
    /// K-split → respect KPipeline dependencies.
    /// Sparse masks → topological order plus adjacency hints.
    pub fn execution_order(&self) -> Vec<usize> {
        // For grids with explicit dependencies (K-split, sparse), use topological sort
        if self.k_splits > 0 || self.nodes.len() < (self.grid_m * self.grid_n) as usize {
            return self.topological_sort().unwrap_or_else(|_| (0..self.nodes.len()).collect());
        }
        // Dense grids: row-major for A data reuse
        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        order.sort_by_key(|&i| {
            let tn = i as u32 % self.grid_n;
            let tm = i as u32 / self.grid_n;
            // Within same column (same B data), row-major
            // Across columns (row by row, for A reuse)
            tm * (self.grid_n + 1) + tn
        });
        order
    }

    /// Returns tiles that can be fused (row-adjacent, same A data).
    /// Result: vector of (tile_idx, neighbor_idx) pairs.
    pub fn fusion_candidates(&self) -> Vec<(usize, usize)> {
        let mut candidates = Vec::new();
        for edge in &self.edges {
            if edge.kind == EdgeKind::RowAdjacent {
                candidates.push((edge.from as usize, edge.to as usize));
            }
        }
        candidates
    }

    /// Check if the given order respects all dependency edges.
    pub fn validate_order(&self, order: &[usize]) -> bool {
        let pos: std::collections::HashMap<usize, usize> = order.iter().enumerate()
            .map(|(i, &n)| (n, i)).collect();
        for edge in &self.edges {
            if let (Some(&pf), Some(&pt)) = (pos.get(&(edge.from as usize)), pos.get(&(edge.to as usize))) {
                if pf > pt { return false; }
            }
        }
        true
    }

    pub fn row_neighbors(&self, tile_idx: usize) -> Vec<usize> {
        let tm = tile_idx as u32 / self.grid_n;
        let start = (tm * self.grid_n) as usize;
        let end = start + self.grid_n as usize;
        (start..end).filter(|&i| i != tile_idx && i < self.nodes.len()).collect()
    }

    pub fn col_neighbors(&self, tile_idx: usize) -> Vec<usize> {
        let tn = tile_idx as u32 % self.grid_n;
        (0..self.grid_m).filter_map(|tm| {
            let idx = (tm * self.grid_n + tn) as usize;
            if idx != tile_idx && idx < self.nodes.len() { Some(idx) } else { None }
        }).collect()
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct TTGLayout {
    pub l1_map: Vec<u32>,
    pub l2_table: Vec<TileMetadata>,
    pub num_active_tiles: u32,
    pub variant: Option<crate::core::config::GemmVariant>,
    pub topology: Option<TileTopology>,
}

impl TTGLayout {
    pub fn suggest_order(&self) -> Vec<usize> {
        match &self.topology {
            Some(topo) => topo.execution_order(),
            None => (0..self.num_active_tiles as usize).collect(),
        }
    }

    /// Returns L1 map reordered by topology-based execution order.
    /// GPU threadgroups will process tiles in this order,
    /// improving cache reuse by keeping A/B data hot.
    pub fn l1_map_ordered(&self) -> Vec<u32> {
        if self.topology.is_none() { return self.l1_map.clone(); }
        let order = self.suggest_order();
        // Reindex: for each slot in the new order, pick the logical tile ID
        // that should execute at that position
        let mut ordered = vec![0u32; self.num_active_tiles as usize];
        for (new_pos, &old_idx) in order.iter().enumerate() {
            if old_idx < self.l1_map.len() {
                ordered[new_pos] = self.l1_map[old_idx];
            }
        }
        ordered
    }

    pub fn fusion_candidates(&self) -> Vec<(usize, usize)> {
        match &self.topology {
            Some(topo) => topo.fusion_candidates(),
            None => vec![],
        }
    }

    pub fn validate_order(&self, order: &[usize]) -> bool {
        match &self.topology {
            Some(topo) => topo.validate_order(order),
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_row_adjacency() {
        let topo = TileTopology::from_grid(3, 3, 32, 32, 96, 96, 128);
        assert_eq!(topo.nodes.len(), 9);
        let center_edges = topo.edges.iter().filter(|e| e.to == 4 || e.from == 4).count();
        assert_eq!(center_edges, 4);
    }

    #[test]
    fn test_topology_execution_order_row_major() {
        let topo = TileTopology::from_grid(2, 3, 32, 32, 64, 96, 128);
        assert_eq!(topo.execution_order(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_topology_k_split_execution_order() {
        let topo = TileTopology::from_grid_k_split(2, 2, 32, 32, 64, 64, 128, 2);
        // Grid: (0,0)→[node0,node1], (0,1)→[node2,node3], (1,0)→[node4,node5], (1,1)→[node6,node7]
        // Edges: KPipeline(0→1), KPipeline(2→3), KPipeline(4→5), KPipeline(6→7)
        //         RowAdj(0→2), RowAdj(4→6)
        //         ColAdj(0→4), ColAdj(2→6)
        let order = topo.execution_order();
        assert_eq!(order.len(), 8);
        // Validate topological: no dependency violated
        assert!(topo.validate_order(&order));
        // node0 must come before node1 (KPipeline)
        assert!(order.iter().position(|&x| x == 0) < order.iter().position(|&x| x == 1));
    }

    #[test]
    fn test_topology_sparse_mask() {
        // Only include diagonal tiles: (0,0), (1,1), (2,2)
        let topo = TileTopology::from_mask(3, 3, 32, 32, 96, 96, 128,
            |tm, tn| tm == tn);
        assert_eq!(topo.nodes.len(), 3);
        // No adjacency edges between diagonal tiles
        assert!(topo.edges.is_empty());
        let order = topo.execution_order();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_topology_fusion_candidates() {
        let topo = TileTopology::from_grid(2, 3, 32, 32, 64, 96, 128);
        let fusions = topo.fusion_candidates();
        // Row-adjacent pairs: (0,1), (1,2), (3,4), (4,5)
        assert_eq!(fusions.len(), 4);
        assert!(fusions.contains(&(0, 1)));
        assert!(fusions.contains(&(3, 4)));
    }

    #[test]
    fn test_topology_reuse_hints() {
        let topo = TileTopology::from_grid(2, 3, 32, 32, 64, 96, 128);
        // Tile 1 is row-adjacent to tile 0 → ReuseA
        // Tile 3 is col-adjacent to tile 0 → ReuseB (if tn=0) or ReuseA
        // Tile at (0,1)=idx1: tn=1>0 → ReuseA from tile 0
        assert_eq!(topo.nodes[1].reuse_hint, ReuseHint::ReuseA { row_source: 0 });
    }

    #[test]
    fn test_topological_sort_respects_edges() {
        let topo = TileTopology::from_grid_k_split(1, 1, 32, 32, 32, 32, 64, 3);
        let order = topo.topological_sort().unwrap();
        assert!(topo.validate_order(&order));
        // KPipeline edges: 0→1→2, so order must be [0, 1, 2]
        assert_eq!(order, vec![0, 1, 2]);
    }
}
