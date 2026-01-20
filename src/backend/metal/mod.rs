use crate::backend::Backend;

pub mod primitives;

pub struct MetalBackend {
    // Stub fields
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {}
    }
    pub fn get_primitive_defs() -> String {
        primitives::MetalPrimitives::all_definitions()
    }
}

impl Backend for MetalBackend {
    fn device_id(&self) -> String {
        "metal_device_stub".into()
    }

    fn driver_version(&self) -> String {
        "metal_stub".into()
    }

    fn runtime_version(&self) -> String {
        "metal_stub".into()
    }

    fn max_shared_memory(&self) -> usize {
        32768 // Default Metal group memory limit
    }

    fn max_threads_per_block(&self) -> usize {
        1024
    }
}
