use crate::backend::Backend;

pub mod primitives;

pub struct RocmBackend {
    // Stub fields
}

impl RocmBackend {
    pub fn new() -> Self {
        Self {}
    }
    pub fn get_primitive_defs() -> String {
        primitives::RocmPrimitives::all_definitions()
    }
}

impl Backend for RocmBackend {
    fn device_id(&self) -> String {
        "rocm_device_stub".into()
    }

    fn driver_version(&self) -> String {
        "rocm_stub".into()
    }

    fn runtime_version(&self) -> String {
        "hip_stub".into()
    }

    fn max_shared_memory(&self) -> usize {
        65536 // Default ROCm LDS size
    }

    fn max_threads_per_block(&self) -> usize {
        1024
    }
}
