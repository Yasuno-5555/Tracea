use crate::runtime::manager::{BufferId, RuntimeManager, DeviceBackend, ArenaSlice};
use crate::core::ttg::TTGLayout;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DeviceTTG {
    pub l1_buffer: BufferId,
    pub l2_buffer: BufferId,
    pub l1_offset: usize,  // Offset within arena (0 if direct alloc)
    pub l2_offset: usize,  // Offset within arena (0 if direct alloc)
    pub num_active_tiles: u32,
}

impl DeviceTTG {
    /// Calculate total bytes needed for L1 and L2 buffers
    pub fn calculate_size(layout: &TTGLayout) -> (usize, usize) {
        let l1_size = layout.l1_map.len() * 4;  // Vec<u32> -> bytes
        let l2_size = layout.l2_table.len() * 20;  // TileMetadata = 5 * u32
        // Align both to 256 bytes
        let l1_aligned = (l1_size + 255) & !255;
        let l2_aligned = (l2_size + 255) & !255;
        (l1_aligned, l2_aligned)
    }

    /// Create TTG using arena-based allocation (zero runtime malloc)
    pub fn new_from_arena(
        runtime: &RuntimeManager,
        layout: &TTGLayout,
        arena_buffer: BufferId,
        l1_offset: usize,
        l2_offset: usize,
    ) -> Result<Self, String> {
        // L1 Map: Vec<u32> -> Bytes
        let l1_bytes: Vec<u8> = layout.l1_map.iter()
            .flat_map(|x| x.to_ne_bytes().to_vec())
            .collect();

        // L2 Table: Vec<TileMetadata> -> Bytes
        let l2_bytes: Vec<u8> = layout.l2_table.iter().flat_map(|meta| {
             let mut b = Vec::with_capacity(20);
             b.extend_from_slice(&meta.region_m.to_ne_bytes());
             b.extend_from_slice(&meta.region_n.to_ne_bytes());
             b.extend_from_slice(&meta.k_start.to_ne_bytes());
             b.extend_from_slice(&meta.k_end.to_ne_bytes());
             b.extend_from_slice(&meta.role.to_ne_bytes());
             b
        }).collect();

        // Copy to arena at specified offsets
        runtime.copy_to_device_at_offset(arena_buffer, l1_offset, &l1_bytes)?;
        runtime.copy_to_device_at_offset(arena_buffer, l2_offset, &l2_bytes)?;

        Ok(Self {
            l1_buffer: arena_buffer,
            l2_buffer: arena_buffer,
            l1_offset,
            l2_offset,
            num_active_tiles: layout.num_active_tiles,
        })
    }

    /// Legacy constructor (for backward compatibility, allocates individually)
    pub fn new(runtime: &RuntimeManager, layout: &TTGLayout, backend: DeviceBackend) -> Result<Self, String> {
        // L1 Map: Vec<u32> -> Bytes
        let l1_bytes: Vec<u8> = layout.l1_map.iter()
            .flat_map(|x| x.to_ne_bytes().to_vec())
            .collect();

        // L2 Table: Vec<TileMetadata> -> Bytes
        let l2_bytes: Vec<u8> = layout.l2_table.iter().flat_map(|meta| {
             let mut b = Vec::with_capacity(20);
             b.extend_from_slice(&meta.region_m.to_ne_bytes());
             b.extend_from_slice(&meta.region_n.to_ne_bytes());
             b.extend_from_slice(&meta.k_start.to_ne_bytes());
             b.extend_from_slice(&meta.k_end.to_ne_bytes());
             b.extend_from_slice(&meta.role.to_ne_bytes());
             b
        }).collect();

        let l1_id = runtime.alloc(l1_bytes.len(), backend)?;
        let l2_id = runtime.alloc(l2_bytes.len(), backend)?;

        runtime.copy_to_device(l1_id, &l1_bytes)?;
        runtime.copy_to_device(l2_id, &l2_bytes)?;

        Ok(Self {
            l1_buffer: l1_id,
            l2_buffer: l2_id,
            l1_offset: 0,
            l2_offset: 0,
            num_active_tiles: layout.num_active_tiles,
        })
    }
}
