use crate::backend::Backend;

pub mod primitives;
pub mod capture;

#[cfg(target_os = "macos")]
use metal;

pub struct MetalBackend {
    #[cfg(target_os = "macos")]
    pub device: metal::Device,
    #[cfg(target_os = "macos")]
    pub queue: metal::CommandQueue,
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(target_os = "macos")]
        return f.debug_struct("MetalBackend")
            .field("device", &self.device.name())
            .finish();
        #[cfg(not(target_os = "macos"))]
        return f.debug_struct("MetalBackend").finish();
    }
}

impl MetalBackend {
    #[cfg(target_os = "macos")]
    pub fn new() -> Result<Self, String> {
        let device = metal::Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new() -> Result<Self, String> {
        Err("Metal is only supported on macOS".to_string())
    }

    pub fn get_primitive_defs() -> String {
        primitives::MetalPrimitives::all_definitions()
    }
}

impl Backend for MetalBackend {
    fn device_id(&self) -> String {
        #[cfg(target_os = "macos")]
        {
            self.device.name().into()
        }
        #[cfg(not(target_os = "macos"))]
        {
            "metal_stub".into()
        }
    }

    fn driver_version(&self) -> String {
        "metal_default".into()
    }

    fn runtime_version(&self) -> String {
        "metal_default".into()
    }

    fn max_shared_memory(&self) -> usize {
        #[cfg(target_os = "macos")]
        {
            // Usually 32KB per threadgroup for generic compute, but Apple Silicon varies.
            // .max_threadgroup_memory_length() is available on device.
            self.device.max_threadgroup_memory_length() as usize
        }
        #[cfg(not(target_os = "macos"))]
        {
            32768
        }
    }

    fn max_threads_per_block(&self) -> usize {
        #[cfg(target_os = "macos")]
        {
            self.device.max_threads_per_threadgroup().width as usize * 
            self.device.max_threads_per_threadgroup().height as usize * 
            self.device.max_threads_per_threadgroup().depth as usize
        }
        #[cfg(not(target_os = "macos"))]
        {
            1024
        }
    }
}
