use crate::backend::Backend;

pub mod primitives;

pub struct CpuBackend {
    pub cpu_name: String,
    pub logical_cores: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
            
        Self {
            cpu_name: "HostCPU".to_string(), // In future use cpuid binding
            logical_cores,
        }
    }
    pub fn get_primitive_defs() -> String {
        primitives::CpuPrimitives::all_definitions()
    }
}

impl Backend for CpuBackend {
    fn device_id(&self) -> String {
        // In a real environment, we'd append CPU model, e.g. "HostCPU_Ryzen9"
        self.cpu_name.clone()
    }

    fn driver_version(&self) -> String {
        "native".to_string()
    }

    fn runtime_version(&self) -> String {
        "std".to_string()
    }

    fn max_shared_memory(&self) -> usize {
        0 // No shared memory for CPU backend in this context
    }

    fn max_threads_per_block(&self) -> usize {
        self.logical_cores
    }
}
