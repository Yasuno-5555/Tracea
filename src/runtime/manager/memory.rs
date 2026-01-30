use std::sync::Mutex;
use cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice};
use super::{BufferId, RuntimeManager, DeviceBackend};
#[cfg(feature = "vulkan")]
use ash::vk;

/// Arena-style memory allocation for reducing runtime malloc overhead.
/// Allocates one large buffer at startup and provides offset-based slices.
#[derive(Debug)]
pub struct MemoryArena {
    pub buffer_id: BufferId,
    pub total_size: usize,
    pub backend: DeviceBackend,
}

impl MemoryArena {
    /// Get the offset-based slice info (used for kernel args)
    pub fn slice(&self, offset: usize, size: usize) -> Result<ArenaSlice, String> {
        if offset + size > self.total_size {
            return Err(format!(
                "Arena overflow: offset {} + size {} > total {}", 
                offset, size, self.total_size
            ));
        }
        // Align offset to 256 bytes (Metal requirement)
        let aligned_offset = (offset + 255) & !255;
        Ok(ArenaSlice {
            arena_buffer_id: self.buffer_id,
            offset: aligned_offset,
            size,
        })
    }
}

/// A slice of the arena buffer, referenced by offset
#[derive(Debug, Clone, Copy)]
pub struct ArenaSlice {
    pub arena_buffer_id: BufferId,
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug)]
pub enum DeviceBuffer {
    Cuda(CudaSlice<u8>),
    Rocm(RocmBuffer), 
    External(u64),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanBuffer),
    #[cfg(target_os = "macos")]
    Metal(metal::Buffer),
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
pub struct VulkanBuffer {
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub buffer: vk::Buffer,
    pub device: ash::Device,
    pub allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            if let Some(alloc) = self.allocation.take() {
                if let Ok(mut lock) = self.allocator.lock() {
                    let _ = lock.free(alloc);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct RocmBuffer {
    pub ptr: u64,
}

impl Drop for RocmBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
             if let Some(api) = crate::emitter::rocm_driver::RocmDriverApi::get() {
                 unsafe { (api.hip_free)(self.ptr); }
             }
        }
    }
}

impl RuntimeManager {
    pub fn alloc(&self, size_bytes: usize, backend: DeviceBackend) -> Result<BufferId, String> {
        let id = self.generate_buffer_id()?;
        let buf = match backend {
            #[cfg(feature = "vulkan")]
            DeviceBackend::Vulkan => {
                let handle = self.get_device(DeviceBackend::Vulkan)?;
                let vk_backend = handle.vulkan_dev.as_ref().ok_or("Vulkan device not found")?;
                let mut allocator = vk_backend.allocator.lock().map_err(|_| "Allocator Lock")?;
                
                unsafe {
                    let buffer_info = vk::BufferCreateInfo::builder()
                        .size(size_bytes as u64)
                        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    
                    let buffer = vk_backend.device.create_buffer(&buffer_info, None).map_err(|e| e.to_string())?;
                    let requirements = vk_backend.device.get_buffer_memory_requirements(buffer);
                    
                    let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                        name: "TraceaBuffer",
                        requirements,
                        location: gpu_allocator::MemoryLocation::GpuOnly,
                        linear: true,
                        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                    }).map_err(|e| e.to_string())?;
                    
                    vk_backend.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).map_err(|e| e.to_string())?;
                    
                    DeviceBuffer::Vulkan(VulkanBuffer {
                        allocation: Some(allocation),
                        buffer,
                        device: vk_backend.device.clone(),
                        allocator: vk_backend.allocator.clone(),
                    })
                }
            }
            DeviceBackend::Cuda => {
                let devs = self.devices.lock().map_err(|_| "Lock")?;
                let d_handle = devs.get(&DeviceBackend::Cuda).ok_or("No CUDA")?;
                let d = d_handle.cuda_dev.as_ref().ok_or("No CUDA Dev")?;
                DeviceBuffer::Cuda(d.alloc_zeros::<u8>(size_bytes).map_err(|e| format!("{:?}", e))?)
            }
            DeviceBackend::Rocm => {
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("No ROCm API")?;
                let mut ptr: u64 = 0;
                unsafe {
                    let res = (api.hip_malloc)(&mut ptr, size_bytes);
                    if res != 0 { return Err(format!("hipMalloc failed: {}", res)); }
                }
                DeviceBuffer::Rocm(RocmBuffer { ptr })
            }
            #[cfg(target_os = "macos")]
            DeviceBackend::Metal => {
                let devs = self.devices.lock().map_err(|_| "Lock")?;
                let handle = devs.get(&DeviceBackend::Metal).ok_or("No Metal Device")?;
                let backend = handle.metal_dev.as_ref().ok_or("No Metal Backend")?;
                
                let options = metal::MTLResourceOptions::StorageModeShared;
                // println!("[Runtime] Metal Alloc: {} bytes", size_bytes);
                let buffer = backend.device.new_buffer(size_bytes as u64, options);
                DeviceBuffer::Metal(buffer)
            }
            _ => return Err("Alloc failed".to_string()),
        };
        self.buffers.lock().map_err(|_| "Lock")?.insert(id, buf);
        Ok(id)
    }

    pub fn get_device_ptr(&self, id: BufferId) -> Result<u64, String> {
        if id == BufferId(0) {
            return Ok(0);
        }
        let bufs = self.buffers.lock().map_err(|_| "Lock")?;
        match bufs.get(&id).ok_or("No buffer")? {
            DeviceBuffer::Cuda(slice) => Ok(*slice.device_ptr()),
            DeviceBuffer::Rocm(buf) => Ok(buf.ptr),
            DeviceBuffer::External(ptr) => Ok(*ptr),
            #[cfg(target_os = "macos")]
            DeviceBuffer::Metal(buf) => Ok(buf.gpu_address()),
            #[cfg(feature = "vulkan")]
            DeviceBuffer::Vulkan(_) => Err("Vulkan ptr not supported".to_string()),
        }
    }

    pub fn get_ptr(&self, id: BufferId) -> Option<u64> {
        self.get_device_ptr(id).ok()
    }

    pub fn register_external_ptr(&self, ptr: u64) -> Result<BufferId, String> {
        let id = self.generate_buffer_id()?;
        self.buffers.lock().map_err(|_| "Lock".to_string())?.insert(id, DeviceBuffer::External(ptr));
        Ok(id)
    }

    pub fn copy_to_device<T: Copy>(&self, id: BufferId, data: &[T]) -> Result<(), String> {
        let mut bufs = self.buffers.lock().map_err(|_| "Lock".to_string())?;
        match bufs.get_mut(&id).ok_or("No buffer".to_string())? {
            DeviceBuffer::Cuda(slice) => {
                let devs = self.devices.lock().map_err(|_| "Lock".to_string())?;
                let _ = devs.get(&DeviceBackend::Cuda).ok_or("No CUDA".to_string())?.cuda_dev.as_ref().ok_or("No CUDA".to_string())?;
                let u8_slice = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
                };
                unsafe {
                    let res = cudarc::driver::sys::lib().cuMemcpyHtoD_v2(*slice.device_ptr(), u8_slice.as_ptr() as *const _, u8_slice.len());
                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return Err(format!("cuMemcpyHtoD failed: {:?}", res)); }
                }
                Ok(())
            }
            #[cfg(target_os = "macos")]
            DeviceBuffer::Metal(buf) => {
                 let len_bytes = data.len() * std::mem::size_of::<T>();
                 let void_ptr = buf.contents();
                 if void_ptr.is_null() {
                     return Err("Metal buffer contents() is NULL".to_string());
                 }
                 let dst_slice = unsafe {
                     std::slice::from_raw_parts_mut(void_ptr as *mut u8, len_bytes)
                 };
                 let src_slice = unsafe {
                     std::slice::from_raw_parts(data.as_ptr() as *const u8, len_bytes)
                 };
                 dst_slice.copy_from_slice(src_slice);
                 Ok(())
            }
            _ => Err("Not implemented for this backend".to_string()),
        }
    }

    /// Copy data to device buffer at a specific offset (for arena-based allocation)
    pub fn copy_to_device_at_offset<T: Copy>(&self, id: BufferId, offset: usize, data: &[T]) -> Result<(), String> {
        let bufs = self.buffers.lock().map_err(|_| "Lock".to_string())?;
        match bufs.get(&id).ok_or("No buffer".to_string())? {
            #[cfg(target_os = "macos")]
            DeviceBuffer::Metal(buf) => {
                 let len_bytes = data.len() * std::mem::size_of::<T>();
                 let buf_len = buf.length() as usize;
                 if offset + len_bytes > buf_len {
                     return Err(format!("Offset {} + len {} > buffer size {}", offset, len_bytes, buf_len));
                 }
                 let void_ptr = buf.contents();
                 if void_ptr.is_null() {
                     return Err("Metal buffer contents() is NULL".to_string());
                 }
                 unsafe {
                     let dst_ptr = (void_ptr as *mut u8).add(offset);
                     std::ptr::copy_nonoverlapping(
                         data.as_ptr() as *const u8,
                         dst_ptr,
                         len_bytes
                     );
                 }
                 Ok(())
            }
            DeviceBuffer::Cuda(slice) => {
                let u8_slice = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
                };
                let dst_ptr = *slice.device_ptr() + offset as u64;
                unsafe {
                    let res = cudarc::driver::sys::lib().cuMemcpyHtoD_v2(dst_ptr, u8_slice.as_ptr() as *const _, u8_slice.len());
                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return Err(format!("cuMemcpyHtoD failed: {:?}", res)); }
                }
                Ok(())
            }
            _ => Err("Not implemented for this backend".to_string()),
        }
    }

    pub fn copy_from_device<T: Copy>(&self, id: BufferId, data: &mut [T]) -> Result<(), String> {
        let bufs = self.buffers.lock().map_err(|_| "Lock".to_string())?;
        match bufs.get(&id).ok_or("No buffer".to_string())? {
            DeviceBuffer::Cuda(slice) => {
                let devs = self.devices.lock().map_err(|_| "Lock".to_string())?;
                let _ = devs.get(&DeviceBackend::Cuda).ok_or("No CUDA".to_string())?;
                let u8_slice = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * std::mem::size_of::<T>())
                };
                unsafe {
                    let res = cudarc::driver::sys::lib().cuMemcpyDtoH_v2(u8_slice.as_mut_ptr() as *mut _, *slice.device_ptr(), u8_slice.len());
                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return Err(format!("cuMemcpyDtoH failed: {:?}", res)); }
                }
                Ok(())
            }
            #[cfg(target_os = "macos")]
            DeviceBuffer::Metal(buf) => {
                 let len_bytes = data.len() * std::mem::size_of::<T>();
                 let void_ptr = buf.contents();
                 unsafe {
                     std::ptr::copy_nonoverlapping(void_ptr as *const _, data.as_mut_ptr() as *mut _, len_bytes);
                 }
                 Ok(())
            }
            _ => Err("Not implemented for this backend".to_string()),
        }
    }

    pub fn read_buffer(&self, id: BufferId) -> Result<Vec<u8>, String> {
        let bufs = self.buffers.lock().map_err(|_| "Lock")?;
        let buf = bufs.get(&id).ok_or("No buffer")?;
        
        match buf {
            DeviceBuffer::Cuda(slice) => {
                let len = slice.len();
                let mut host = vec![0u8; len];
                unsafe {
                    let res = cudarc::driver::sys::lib().cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, *slice.device_ptr(), len);
                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { return Err(format!("cuMemcpyDtoH failed: {:?}", res)); }
                }
                Ok(host)
            }
            #[cfg(target_os = "macos")]
            DeviceBuffer::Metal(buf) => {
                 let len = buf.length() as usize;
                 let mut host = vec![0u8; len];
                 let void_ptr = buf.contents();
                 unsafe {
                     std::ptr::copy_nonoverlapping(void_ptr as *const _, host.as_mut_ptr() as *mut _, len);
                 }
                 Ok(host)
            }
            _ => Err("Not implemented for this backend".to_string()),
        }
    }

    pub fn alloc_f32(&self, size: usize, backend: DeviceBackend) -> Result<BufferId, String> {
        self.alloc(size * 4, backend)
    }

    pub fn alloc_i32(&self, size: usize, backend: DeviceBackend) -> Result<BufferId, String> {
        self.alloc(size * 4, backend)
    }

    pub fn alloc_u16(&self, size: usize, backend: DeviceBackend) -> Result<BufferId, String> {
        self.alloc(size * 2, backend)
    }

    pub fn alloc_f16(&self, size: usize, backend: DeviceBackend) -> Result<BufferId, String> {
        self.alloc(size * 2, backend)
    }
    pub fn memcpy_d2d(&self, src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, size: usize) -> Result<(), String> {
        let bufs = self.buffers.lock().map_err(|_| "Lock")?;
        let src_buf = bufs.get(&src).ok_or("No src buffer")?;
        let dst_buf = bufs.get(&dst).ok_or("No dst buffer")?;

        match (src_buf, dst_buf) {
            #[cfg(target_os = "macos")]
            (DeviceBuffer::Metal(s), DeviceBuffer::Metal(d)) => {
                 let devs = self.devices.lock().map_err(|_| "Lock")?;
                 let handle = devs.get(&DeviceBackend::Metal).ok_or("No Metal Device")?;
                 let backend = handle.metal_dev.as_ref().ok_or("No Metal Backend")?;
                 
                 let cb = backend.queue.new_command_buffer();
                 let blit = cb.new_blit_command_encoder();
                 blit.copy_from_buffer(s, src_offset as u64, d, dst_offset as u64, size as u64);
                 blit.end_encoding();
                 cb.commit();
                 Ok(())
            }
            #[cfg(target_os = "macos")]
            _ => Err("memcpy_d2d only implemented for Metal -> Metal".into()),
            #[cfg(not(target_os = "macos"))]
            _ => Err("memcpy_d2d not implemented for non-Metal".into()),
        }
    }
}
