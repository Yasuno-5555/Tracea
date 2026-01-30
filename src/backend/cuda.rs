use crate::backend::Backend;
use cudarc::driver::{CudaDevice, DriverError};
use std::sync::Arc;

pub mod primitives;
pub mod capture;

pub struct CudaBackend {
    pub device: Arc<CudaDevice>,
    pub device_id: String,
}

impl CudaBackend {
    pub fn new(device_ordinal: usize) -> Result<Self, DriverError> {
        let device = CudaDevice::new(device_ordinal)?;
        
        // Construct a unique device ID (e.g., "sm_86_RTX_3070")
        // cudarc doesn't expose all props easily without unsafe or FFI, 
        // but we can name it simply for now or use available methods.
        // device.name() is available? 
        // Using "cuda:<ordinal>" as fallback or querying raw properties via separate helper if needed.
        let name = device.name().unwrap_or_else(|_| "UnknownGPU".into());
        let device_id = format!("{}_{}", name.replace(" ", "_"), device_ordinal);

        Ok(Self {
            device,
            device_id,
        })
    }
    pub fn get_primitive_defs() -> String {
        primitives::CudaPrimitives::all_definitions()
    }
}

impl Backend for CudaBackend {
    fn device_id(&self) -> String {
        self.device_id.clone()
    }

    fn driver_version(&self) -> String {
        // Placeholder: would query actual driver version via FFI
        "cuda_driver_unknown".to_string()
    }

    fn runtime_version(&self) -> String {
        // Placeholder
        "cuda_runtime_unknown".to_string()
    }

    fn max_shared_memory(&self) -> usize {
        self.device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).unwrap_or(49152) as usize
    }

    fn max_threads_per_block(&self) -> usize {
        self.device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap_or(1024) as usize
    }
}
