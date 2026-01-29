use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendType {
    Cuda,
    Metal,
    Rocm,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub backend: BackendType,
    pub name: String, // "Apple M1", "NVIDIA GeForce RTX 3090"
    
    // Compute Capabilities
    pub max_threads_per_block: usize, // CUDA: 1024, Metal: 1024
    pub simd_width: usize,            // CUDA: 32, Metal: 32, AMD: 64
    pub local_memory_size: usize,     // Shared/Threadgroup Memory size
    
    // Feature Flags
    pub has_tensor_cores: bool,       // AMX or TensorCore
    pub has_fp16_storage: bool,
    
    // Alignment Constraints
    pub texture_alignment: usize,
}


impl DeviceProfile {
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            if let Some(profile) = Self::detect_metal() {
                return profile;
            }
        }

        if let Some(profile) = Self::detect_cuda() {
            return profile;
        }

        Self::default()
    }

    #[cfg(target_os = "macos")]
    fn detect_metal() -> Option<Self> {
        // macOS 'metal' crate usage
        let device = metal::Device::system_default()?;
        
        // Conservatively assume Apple Silicon features if name contains "Apple"
        let name = device.name().to_string();
        let is_apple = name.contains("Apple") || name.contains("M1") || name.contains("M2") || name.contains("M3");
        
        Some(Self {
            backend: BackendType::Metal,
            name,
            max_threads_per_block: {
                let s = device.max_threads_per_threadgroup();
                (s.width * s.height * s.depth) as usize
            },
            simd_width: 32,
            local_memory_size: device.max_threadgroup_memory_length() as usize,
            has_tensor_cores: is_apple, // Apple Silicon has AMX
            has_fp16_storage: true,
            texture_alignment: 256, // Safe default
        })
    }

    fn detect_cuda() -> Option<Self> {
        // We use cudarc to detect
        // Note: initializing CudaDevice(0) is a robust check
        match cudarc::driver::CudaDevice::new(0) {
            Ok(device) => {
                let name = device.name().unwrap_or("Unknown CUDA Device".into());
                let max_threads = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap_or(1024) as usize;
                let shared_mem = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).unwrap_or(49152) as usize;
                
                // Check TensorCores via Compute Capability
                let major = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0);
                let has_tc = major >= 7; // Volta+

                Some(Self {
                    backend: BackendType::Cuda,
                    name,
                    max_threads_per_block: max_threads,
                    simd_width: 32,
                    local_memory_size: shared_mem,
                    has_tensor_cores: has_tc,
                    has_fp16_storage: true,
                    texture_alignment: 512,
                })
            },
            Err(_) => None,
        }
    }

    /// Create a DeviceProfile from a runtime backend enum.
    /// Used by RuntimeManager to get profile without triggering detection.
    pub fn from_backend(backend: crate::runtime::manager::DeviceBackend) -> Self {
        use crate::runtime::manager::DeviceBackend;
        match backend {
            DeviceBackend::Metal => {
                #[cfg(target_os = "macos")]
                { Self::detect_metal().unwrap_or_default() }
                #[cfg(not(target_os = "macos"))]
                { Self::default() }
            },
            DeviceBackend::Cuda => {
                Self::detect_cuda().unwrap_or_default()
            },
            DeviceBackend::Rocm => {
                // ROCm profile (fallback)
                Self {
                    backend: BackendType::Rocm,
                    name: "AMD GPU".to_string(),
                    max_threads_per_block: 1024,
                    simd_width: 64,
                    local_memory_size: 65536,
                    has_tensor_cores: false,
                    has_fp16_storage: true,
                    texture_alignment: 256,
                }
            },
            DeviceBackend::Cpu => Self::default(),
            #[cfg(feature = "vulkan")]
            DeviceBackend::Vulkan => Self::default(),
        }
    }
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self {
            backend: BackendType::Cpu,
            name: "Unknown CPU".to_string(),
            max_threads_per_block: 1,
            simd_width: 8, // AVX
            local_memory_size: 0,
            has_tensor_cores: false,
            has_fp16_storage: false,
            texture_alignment: 64,
        }
    }
}
