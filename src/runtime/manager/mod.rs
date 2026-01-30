use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::doctor::Doctor;
use crate::runtime::compiler::GraphCompiler;
use crate::runtime::executor::GraphExecutor;
use crate::optimizer::tuner::MetaTuner;

// Re-export submodules
pub mod memory;
pub mod kernel;
pub mod launcher;
pub mod graph;
pub mod cache;

pub use memory::*;
pub use kernel::*;
pub use launcher::*;
pub use graph::*;
pub use cache::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelArgKind {
    Buffer,
    Scalar, 
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(pub u64);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MetalConvParams {
    pub batch: u32, pub h_in: u32, pub w_in: u32, pub c_in: u32, pub k_out: u32,
    pub h_out: u32, pub w_out: u32, pub r_sz: u32, pub s_sz: u32,
    pub stride: u32, pub pad: u32, pub dilation: u32,
    pub hw_m: u32, pub hw_s: u32,
    pub w_m: u32, pub w_s: u32,
    pub sic_m: u32, pub sic_s: u32,
    pub c_m: u32, pub c_s: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceBackend {
    Cuda,
    Rocm,
    Metal,
    Cpu,
    #[cfg(feature = "vulkan")]
    Vulkan,
}

#[derive(Debug, Clone)]
pub struct DeviceHandle {
    pub backend: DeviceBackend,
    pub cuda_dev: Option<Arc<cudarc::driver::CudaDevice>>,
    #[cfg(feature = "vulkan")]
    pub vulkan_dev: Option<Arc<crate::backend::vulkan::VulkanBackend>>,
    #[cfg(target_os = "macos")]
    pub metal_dev: Option<Arc<crate::backend::metal::MetalBackend>>,
    pub arch: String,
}

#[derive(Debug)]
pub struct RuntimeManager {
    pub devices: Mutex<HashMap<DeviceBackend, DeviceHandle>>,
    pub kernels: Mutex<HashMap<KernelId, RecordedKernel>>,
    pub source_cache: Mutex<HashMap<String, KernelId>>,
    pub buffers: Mutex<HashMap<BufferId, DeviceBuffer>>,
    pub next_kernel_id: Mutex<u64>,
    pub next_buffer_id: Mutex<u64>,
    pub compatibility_log: Mutex<Vec<String>>,
    pub doctor: Arc<Doctor>,
    /// Pre-allocated arena for graph execution (reduces malloc overhead)
    pub arena: Mutex<Option<MemoryArena>>,
    /// Graph Execution Plan Cache
    pub graph_cache: RwLock<GraphCache>,
    
    // Components
    pub compiler: Mutex<GraphCompiler>, 
    pub executor: GraphExecutor,
    pub tuner: Arc<MetaTuner>,
}

static INSTANCE: Mutex<Option<Arc<RuntimeManager>>> = Mutex::new(None);

impl RuntimeManager {
    pub fn init(pref_backend: Option<DeviceBackend>) -> Result<Arc<Self>, String> {
        let mut cache = INSTANCE.lock().map_err(|_| "Global Instance Lock Poisoned".to_string())?;
        if let Some(instance) = &*cache {
            return Ok(Arc::clone(instance));
        }

        let mut devices = HashMap::new();
        #[cfg(not(target_os = "macos"))]
        if let Ok(dev) = cudarc::driver::CudaDevice::new(0) {
            let major = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(8);
            let minor = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(6);
            devices.insert(DeviceBackend::Cuda, DeviceHandle {
                backend: DeviceBackend::Cuda,
                cuda_dev: Some(dev),
                #[cfg(feature = "vulkan")]
                vulkan_dev: None,
                #[cfg(target_os = "macos")]
                metal_dev: None,
                arch: format!("sm_{}{}", major, minor),
            });
            println!("[Doctor] 🟢 CUDA Device Registered.");
        }

        #[cfg(not(target_os = "macos"))]
        if let Some(_) = crate::emitter::rocm_driver::RocmDriverApi::get() {
             devices.insert(DeviceBackend::Rocm, DeviceHandle {
                 backend: DeviceBackend::Rocm,
                 cuda_dev: None,
                 #[cfg(feature = "vulkan")]
                 vulkan_dev: None,
                 #[cfg(target_os = "macos")]
                 metal_dev: None,
                 arch: "gfx90a".to_string(),
             });
             println!("[Doctor] 🟢 ROCm Backend Registered.");
        }

        #[cfg(feature = "vulkan")]
        {
            let mut vulkan_dev_option = None;
            if let Ok(vk_backend) = unsafe { crate::backend::vulkan::VulkanBackend::new() } {
                println!("[Runtime] 🌋 Vulkan Initialization Successful: {}", vk_backend.device_id);
                vulkan_dev_option = Some(Arc::new(vk_backend));
                devices.insert(DeviceBackend::Vulkan, DeviceHandle {
                    backend: DeviceBackend::Vulkan,
                    cuda_dev: None,
                    vulkan_dev: vulkan_dev_option.clone(),
                    #[cfg(target_os = "macos")]
                    metal_dev: None,
                    arch: "vulkan".to_string(),
                });
            }
        }

        #[cfg(target_os = "macos")]
        if let Ok(metal_backend) = crate::backend::metal::MetalBackend::new() {
             println!("[Doctor] 🍏 Metal Backend Registered: {}", metal_backend.device.name());
             devices.insert(DeviceBackend::Metal, DeviceHandle {
                 backend: DeviceBackend::Metal,
                 cuda_dev: None,
                 #[cfg(feature = "vulkan")]
                 vulkan_dev: None,
                 metal_dev: Some(Arc::new(metal_backend)),
                 arch: "apple_m_series".to_string(),
             });
        }

        let doctor = Doctor::global();
        doctor.diagnose_environment();

        let selected_backend = pref_backend.unwrap_or(DeviceBackend::Cuda);
        let instance = Arc::new_cyclic(|me| {
            let tuner = Arc::new(MetaTuner::new(me.clone()));
            
            Self {
            devices: Mutex::new(devices),
            kernels: Mutex::new(HashMap::new()),
            source_cache: Mutex::new(HashMap::new()),
            buffers: Mutex::new(HashMap::new()),
            next_kernel_id: Mutex::new(0),
            next_buffer_id: Mutex::new(0),
            compatibility_log: Mutex::new(Vec::new()),
            doctor: Arc::new(Doctor::new(crate::doctor::diagnosis::DoctorConfig::default())),
            arena: Mutex::new(None),
            graph_cache: RwLock::new(GraphCache::new()),
            compiler: Mutex::new(GraphCompiler::new()),
            executor: GraphExecutor::new(selected_backend),
            tuner,
            }
        });
        *cache = Some(Arc::clone(&instance));
        Ok(instance)
    }

    pub fn new() -> Arc<Self> {
        Self::init(None).expect("Failed to initialize RuntimeManager")
    }

    pub fn synchronize(&self) {
        let devices = self.devices.lock().unwrap();
        
        // Sync CUDA
        if let Some(handle) = devices.get(&DeviceBackend::Cuda) {
            if let Some(dev) = &handle.cuda_dev {
                let _ = dev.synchronize();
            }
        }
        
        // Sync ROCm
        if let Some(_) = devices.get(&DeviceBackend::Rocm) {
            if let Some(api) = crate::emitter::rocm_driver::RocmDriverApi::get() {
                unsafe { (api.hipDeviceSynchronize)(); }
            }
        }

        #[cfg(target_os = "macos")]
        if let Some(h) = devices.get(&DeviceBackend::Metal) {
            if let Some(b) = &h.metal_dev {
                let cb = b.queue.new_command_buffer();
                cb.commit();
                cb.wait_until_completed();
            }
        }
    }

    pub fn get_device(&self, backend: DeviceBackend) -> Result<DeviceHandle, String> {
        let devs = self.devices.lock().map_err(|_| "Lock".to_string())?;
        devs.get(&backend).cloned().ok_or_else(|| format!("Device {:?} not found", backend))
    }

    pub fn generate_kernel_id(&self) -> Result<KernelId, String> {
        let mut id = self.next_kernel_id.lock().map_err(|_| "Lock".to_string())?;
        *id += 1;
        Ok(KernelId(*id))
    }

    pub fn generate_buffer_id(&self) -> Result<BufferId, String> {
        let mut id = self.next_buffer_id.lock().map_err(|_| "Lock".to_string())?;
        *id += 1;
        Ok(BufferId(*id))
    }

    /// Initialize the memory arena with a single large allocation.
    /// Call this once before graph execution to avoid per-kernel allocs.
    pub fn init_arena(&self, size: usize, backend: DeviceBackend) -> Result<(), String> {
        let mut arena_guard = self.arena.lock().map_err(|_| "Arena lock")?;
        
        // Skip if already allocated with sufficient size
        if let Some(ref existing) = *arena_guard {
            if existing.total_size >= size && existing.backend == backend {
                return Ok(());
            }
        }
        
        // Align to 4KB page size for better performance
        let aligned_size = (size + 4095) & !4095;
        println!("[Arena] Allocating {} MB workspace", aligned_size / (1024 * 1024));
        
        let buffer_id = self.alloc(aligned_size, backend)?;
        
        *arena_guard = Some(MemoryArena {
            buffer_id,
            total_size: aligned_size,
            backend,
        });
        
        Ok(())
    }

    /// Get a slice of the arena buffer at the specified offset.
    /// Returns the arena's buffer ID and the offset for use in kernel launch.
    pub fn get_arena_slice(&self, offset: usize, size: usize) -> Result<ArenaSlice, String> {
        let arena_guard = self.arena.lock().map_err(|_| "Arena lock")?;
        let arena = arena_guard.as_ref().ok_or("Arena not initialized")?;
        arena.slice(offset, size)
    }

    /// Get the underlying Metal buffer from the arena for direct binding
    #[cfg(target_os = "macos")]
    pub fn get_arena_metal_buffer(&self, required_size: usize) -> Result<(BufferId, usize), String> {
        // Init arena if needed!
        // This helper assumes arena is sufficient. But for robustness, we should check/init.
        // But `init_arena` requires mutable access or lock.
        // Let's assume the Caller ensures `init_arena` was called, OR we check here.
        
        let mut arena_guard = self.arena.lock().map_err(|_| "Arena lock")?;
        if arena_guard.is_none() || arena_guard.as_ref().unwrap().total_size < required_size {
             drop(arena_guard);
             // Default 256MB if not present? Or use required_size.
             // But we need backend. Assume Metal since this is Metal specific fn.
             self.init_arena(std::cmp::max(required_size, 256*1024*1024), DeviceBackend::Metal)?;
             arena_guard = self.arena.lock().map_err(|_| "Arena lock")?;
        }
        
        let arena = arena_guard.as_ref().ok_or("Arena not initialized")?;
        Ok((arena.buffer_id, arena.total_size))
    }
}
