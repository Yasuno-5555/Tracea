use crate::runtime::manager::{BufferId, RuntimeManager, DeviceBackend};
use crate::core::ttg::TTGLayout;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DeviceTTG {
    pub l1_buffer: BufferId,
    pub l2_buffer: BufferId,
    pub num_active_tiles: u32,
}

impl DeviceTTG {
    pub fn new(runtime: &RuntimeManager, layout: &TTGLayout, backend: DeviceBackend) -> Result<Self, String> {
        // L1 Map: Vec<u32> -> Bytes
        let l1_bytes: Vec<u8> = layout.l1_map.iter()
            .flat_map(|x| x.to_ne_bytes().to_vec())
            .collect();

        // L2 Table: Vec<TileMetadata> -> Bytes
        // TileMetadata is 5 * u32 = 20 bytes
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
            num_active_tiles: layout.num_active_tiles,
        })
    }
}
