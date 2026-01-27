
pub type LogicalID = u32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileMetadata {
    pub region_m: u32,  // Logical coordinate M
    pub region_n: u32,  // Logical coordinate N
    pub k_start: u32,    // Future use: Dynamic K split
    pub k_end: u32,      // Future use
    pub role: u32,       // 0=Main, 1=Boundary, etc.
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct TTGLayout {
    pub l1_map: Vec<u32>,
    pub l2_table: Vec<TileMetadata>,
    pub num_active_tiles: u32,
}
