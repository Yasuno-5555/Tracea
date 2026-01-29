use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use cudarc::driver::*;
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::ffi::{c_void, CString};
use std::io::Write;
use serde::{Serialize, Deserialize};
#[cfg(feature = "vulkan")]
use ash::vk;
#[cfg(target_os = "macos")]
use metal;
use crate::doctor::{BackendKind, JitResultInfo, AssemblerResultInfo, KernelLaunchInfo, ModuleLoadInfo};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelArgKind {
    Buffer,
    Scalar, 
}

#[derive(Debug)]
pub struct CudaModule(pub cudarc::driver::sys::CUmodule);

impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let res = cudarc::driver::sys::lib().cuModuleUnload(self.0);
                if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    eprintln!("[Runtime] ⚠️ Failed to unload CUDA module: {:?}", res);
                }
            }
        }
    }
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

#[derive(Debug, Clone, Copy)]
pub struct CudaFunction(pub cudarc::driver::sys::CUfunction);

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(pub u64);

#[derive(Debug, Clone)]
pub enum KernelArg {
    Buffer(BufferId),
    Int(i32),
    Float(f32),
    Usize(usize),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceBackend {
    Cuda,
    Rocm,
    Metal,
    Cpu,
    #[cfg(feature = "vulkan")]
    Vulkan,
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
    pub buffer: ash::vk::Buffer,
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

#[derive(Debug)]
pub struct RocmModule(pub *mut c_void);

impl Drop for RocmModule {
    fn drop(&mut self) {
        if !self.0.is_null() {
            if let Some(api) = crate::emitter::rocm_driver::RocmDriverApi::get() {
                unsafe { (api.hipModuleUnload)(self.0); }
            }
        }
    }
}

unsafe impl Send for RocmModule {}
unsafe impl Sync for RocmModule {}

#[derive(Debug, Clone, Copy)]
pub struct RocmFunction(pub *mut c_void);

unsafe impl Send for RocmFunction {}
unsafe impl Sync for RocmFunction {}

impl Drop for RocmBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
             if let Some(api) = crate::emitter::rocm_driver::RocmDriverApi::get() {
                 unsafe { (api.hipFree)(self.ptr); }
                 // println!("[Runtime] 🧹 ROCm Buffer freed: 0x{:x}", self.ptr);
             }
        }
    }
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
pub struct VulkanModule {
    pub module: ash::vk::ShaderModule,
    pub device: ash::Device,
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanModule {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

#[cfg(feature = "vulkan")]
unsafe impl Send for VulkanModule {}
#[cfg(feature = "vulkan")]
unsafe impl Sync for VulkanModule {}

#[cfg(feature = "vulkan")]
#[derive(Debug, Clone, Copy)]
pub struct VulkanFunction {
    pub pipeline: ash::vk::Pipeline,
}

#[cfg(feature = "vulkan")]
unsafe impl Send for VulkanFunction {}
#[cfg(feature = "vulkan")]
unsafe impl Sync for VulkanFunction {}

#[derive(Debug, Clone)]
pub enum KernelHandle {
    Cuda {
        func: CudaFunction,
        module: Arc<CudaModule>,
    },
    Rocm {
        func: RocmFunction,
        module: Arc<RocmModule>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        func: VulkanFunction,
        module: Arc<VulkanModule>,
    },
    #[cfg(target_os = "macos")]
    Metal {
        pipeline: metal::ComputePipelineState,
        layout: Vec<KernelArgKind>,
        // Cache source for identifying?
    },
}

#[derive(Debug, Clone)]
pub struct RecordedKernel {
    pub name: String,
    pub handle: KernelHandle,
    pub backend: DeviceBackend,
}

#[derive(Debug, Clone)]
pub struct DeviceHandle {
    pub backend: DeviceBackend,
    pub cuda_dev: Option<Arc<CudaDevice>>,
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
    pub doctor: Arc<crate::doctor::Doctor>,
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
        if let Ok(dev) = CudaDevice::new(0) {
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

        let doctor = crate::doctor::Doctor::global();
        doctor.diagnose_environment();

        let instance = Arc::new(Self {
            devices: Mutex::new(devices),
            kernels: Mutex::new(HashMap::new()),
            source_cache: Mutex::new(HashMap::new()),
            buffers: Mutex::new(HashMap::new()),
            next_kernel_id: Mutex::new(0),
            next_buffer_id: Mutex::new(0),
            compatibility_log: Mutex::new(Vec::new()),
            doctor,
        });
        *cache = Some(Arc::clone(&instance));
        Ok(instance)
    }

    pub fn new() -> Arc<Self> {
        Self::init(None).expect("Failed to initialize RuntimeManager")
    }

    pub fn compile(&self, source: &str, kernel_name: &str, backend: DeviceBackend) -> Result<KernelId, String> {
        println!("[Runtime] Debug: compile called for {}", kernel_name);
        // Source-based caching
        {
            let cache = self.source_cache.lock().map_err(|_| "Lock")?;
            if let Some(id) = cache.get(source) {
                return Ok(*id);
            }
        }

        let id = match backend {
            #[cfg(feature = "vulkan")]
            DeviceBackend::Vulkan => {
                 println!("[Runtime] JIT Compiling GLSL for Vulkan: {}", kernel_name);
                 let compiler = shaderc::Compiler::new().ok_or("Failed to create shaderc compiler")?;
                 let mut options = shaderc::CompileOptions::new().ok_or("Failed to create shaderc options")?;
                 options.set_optimization_level(shaderc::OptimizationLevel::Performance);
                 options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);

                 let binary = compiler.compile_into_spirv(
                     source,
                     shaderc::ShaderKind::Compute,
                     "source.glsl",
                     "main",
                     Some(&options)
                 ).map_err(|e| format!("GLSL Compilation Failed: {}", e))?;

                 self.load_vulkan(binary.as_binary_u8(), kernel_name)
            }
            #[cfg(target_os = "macos")]
            DeviceBackend::Metal => {
                let devices = self.devices.lock().map_err(|_| "Lock")?;
                let handle = devices.get(&DeviceBackend::Metal).ok_or("No Metal Device")?;
                let backend = handle.metal_dev.as_ref().ok_or("No Metal Backend instance")?;
                
                // SHA256/Hasher for caching (Using DefaultHasher for now as dependency minamilism)
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                source.hash(&mut hasher);
                let hash = hasher.finish();
                
                let compile_options = metal::CompileOptions::new();
                let library = backend.device.new_library_with_source(source, &compile_options).map_err(|e| format!("Metal Compile Error: {}", e))?;
                
                let func = library.get_function(kernel_name, None).map_err(|e| format!("Function '{}' not found", kernel_name))?;
                
                let pipeline = backend.device.new_compute_pipeline_state_with_function(&func).map_err(|e| format!("Pipeline Error: {}", e))?;

                let id = self.generate_kernel_id()?;
                // MVP: Implicit layout (sequential) or need explicit?
                // For now, empty layout implies "No check" or "Sequential"
                self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel {
                    name: kernel_name.to_string(),
                    handle: KernelHandle::Metal {
                        pipeline,
                        layout: Vec::new(), // TODO: Populate via reflection or explicit arg
                    },
                    backend: DeviceBackend::Metal,
                });
                return Ok(id);
            }
            DeviceBackend::Cuda => {
                let devices = self.devices.lock().map_err(|_| "Lock")?;
                let cuda_handle = devices.get(&DeviceBackend::Cuda).ok_or("No CUDA device")?;
                let target_arch = cuda_handle.arch.clone();
                let _target_sm = target_arch.replace("sm_", "");

                let mut arch_static: &'static str = "sm_80";
                if let Some(handle) = devices.get(&DeviceBackend::Cuda) {
                    arch_static = Box::leak(handle.arch.clone().into_boxed_str());
                }

                // Source-aware AOT Cache Check
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                source.hash(&mut hasher);
                let source_hash = hasher.finish();
                let cache_name = format!("{}_{:x}", kernel_name, source_hash);

                if let Ok(id) = self.load_binary(&cache_name, kernel_name, arch_static) {
                    self.source_cache.lock().unwrap().insert(source.to_string(), id);
                    return Ok(id);
                }

                // Debug dump
                if let Ok(mut f) = std::fs::File::create("last_kernel.cu") {
                    let _ = f.write_all(source.as_bytes());
                }

                let mut opts = CompileOptions {
                    arch: Some(arch_static), 
                    options: vec![
                        "-I".to_string(), 
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include".to_string(),
                        // Correct architecture flag instead of version hacking later
                        format!("--gpu-architecture=compute_{}", arch_static.replace("sm_", "")),
                    ],
                    ..Default::default()
                };
                
                let ptx_res = cudarc::nvrtc::compile_ptx_with_opts(source, opts);
                
                // Doctor Hook: NVRTC Result
                let (jit_code, jit_log) = match &ptx_res {
                    Ok(_) => (0, String::new()),
                    Err(e) => (1, format!("{:?}", e)),
                };
                 self.doctor.on_jit_result(crate::doctor::JitResultInfo {
                    backend: crate::doctor::BackendKind::Cuda,
                    kernel_name: kernel_name.to_string(),
                    return_code: jit_code,
                    source: source.to_string(),
                    stdout: String::new(),
                    stderr: jit_log,
                });

                if ptx_res.is_err() {
                     let _ = std::fs::write("failed_source.cu", source);
                     println!("[Runtime] 📝 Dumped failed source to failed_source.cu");
                }

                let ptx_src = match ptx_res {
                    Ok(ptx) => {
                        println!("[Runtime] JIT Compilation Successful for {}", kernel_name);
                        ptx.to_src()
                    }
                    Err(e) => {
                         println!("[Runtime] ❌ JIT Compilation Failed for {}: {}", kernel_name, e);
                         let _ = std::fs::write("jit_error.log", format!("Error: {}", e));
                         return self.load_prebuilt_fallback(kernel_name, backend); 
                    }
                };
                
                // PTX VERSION HACKING REMOVED AS WE USE CORRECT ARCH FLAGS NOW.

                // Direct Driver Load (Bypass ptxas)
                println!("[Runtime] Loading PTX directly via Driver JIT...");
                
                unsafe {
                    let lib = cudarc::driver::sys::lib();
                    let mut module: cudarc::driver::sys::CUmodule = std::ptr::null_mut();
                    let ptx_cstring = std::ffi::CString::new(ptx_src.clone()).unwrap();

                    let res = lib.cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const _);
                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        let err = format!("Driver JIT Failed: {:?}", res);
                        println!("[Runtime] ❌ {}", err);
                        return Err(err);
                    }
                    
                    let mut func: cudarc::driver::sys::CUfunction = std::ptr::null_mut();
                    let name_c = std::ffi::CString::new(kernel_name).unwrap();
                    let res_func = lib.cuModuleGetFunction(&mut func, module, name_c.as_ptr());
                    
                    if res_func != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                         return Err(format!("Failed to get kernel function '{}': {:?}", kernel_name, res_func));
                    }

                    let res_attr = lib.cuFuncSetAttribute(
                        func, 
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 
                        98304 // 96KB
                    );
                    
                    let id = self.generate_kernel_id()?;
                    self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel {
                        name: kernel_name.to_string(),
                        handle: KernelHandle::Cuda {
                            func: CudaFunction(func),
                            module: Arc::new(CudaModule(module)),
                        },
                        backend: DeviceBackend::Cuda,
                    });
                    
                    println!("[Runtime] Kernel '{}' loaded successfully. ID: {:?}", kernel_name, id);
                    Ok(id)
                }
                
            }
            DeviceBackend::Rocm => {
                let jit = crate::emitter::rocm_jit::ROCMJITCompiler::new().ok_or("ROCm JIT API not found")?;
                let binary = jit.compile(source, kernel_name, vec![])?;
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("ROCm Driver API not found")?;
                let mut module_ptr: *mut c_void = std::ptr::null_mut();
                unsafe {
                    let _ = (api.hipModuleLoadData)(&mut module_ptr, binary.as_ptr() as *const _);
                    let mut func_ptr: *mut c_void = std::ptr::null_mut();
                    let name_c = std::ffi::CString::new(kernel_name).unwrap();
                    let _ = (api.hipModuleGetFunction)(&mut func_ptr, module_ptr, name_c.as_ptr());
                    let id = self.generate_kernel_id()?;
                    self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel {
                        name: kernel_name.to_string(),
                        handle: KernelHandle::Rocm {
                            func: RocmFunction(func_ptr),
                            module: Arc::new(RocmModule(module_ptr)),
                        },
                        backend: DeviceBackend::Rocm,
                    });
                    Ok(id)
                }
            }
            _ => Err("Unsupported".to_string()),
        }?;

        // Cache the newly compiled kernel ID by source
        self.source_cache.lock().map_err(|_| "Poisoned")?.insert(source.to_string(), id);
        Ok(id)
    }

    pub fn launch(&self, id: KernelId, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: Vec<KernelArg>) -> Result<(), String> {
        let recorded = self.kernels.lock().map_err(|_| "Lock")?.get(&id).ok_or("No kernel")?.clone();
        
        println!("[Runtime] Launching {}: Grid{:?}, Block{:?}, Smem: {}", recorded.name, grid, block, smem);

        let mut arg_store = [0u64; 64]; 
        let mut kernel_params = [std::ptr::null_mut() as *mut c_void; 64];

        for (i, arg) in args.iter().enumerate() {
            if i >= 64 { break; }
            match arg {
                KernelArg::Int(x) => {
                    let ptr = &mut arg_store[i] as *mut u64 as *mut i32;
                    unsafe { *ptr = *x; }
                    kernel_params[i] = ptr as *mut c_void;
                }
                KernelArg::Float(x) => {
                    let ptr = &mut arg_store[i] as *mut u64 as *mut f32;
                    unsafe { *ptr = *x; }
                    kernel_params[i] = ptr as *mut c_void;
                }
                KernelArg::Usize(x) => {
                    let ptr = &mut arg_store[i] as *mut u64;
                    unsafe { *ptr = *x as u64; }
                    kernel_params[i] = ptr as *mut c_void;
                }
                KernelArg::Buffer(bid) => {
                    let ptr_val = self.get_device_ptr(*bid)?;
                    let ptr = &mut arg_store[i] as *mut u64;
                    unsafe { *ptr = ptr_val; }
                    kernel_params[i] = ptr as *mut c_void;
                }
                KernelArg::Bytes(bytes) => {
                    kernel_params[i] = bytes.as_ptr() as *mut c_void;
                }
            }
        }

        match &recorded.handle {
            KernelHandle::Cuda { func, .. } => {
                unsafe {
                    let lib = cudarc::driver::sys::lib();
                    
                    let res = lib.cuLaunchKernel(
                        func.0,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem, std::ptr::null_mut(),
                        kernel_params.as_ptr() as *mut *mut c_void,
                        std::ptr::null_mut()
                    );

                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { 
                        let msg = format!("{:?}", res);
                        self.doctor.on_kernel_launch(crate::doctor::KernelLaunchInfo {
                             backend: crate::doctor::BackendKind::Cuda,
                             kernel_name: recorded.name.clone(),
                             return_code: res as i32,
                             last_runtime_error: Some(msg),
                             grid: (grid.0, grid.1, grid.2), 
                             block: (block.0, block.1, block.2),
                             smem,
                        });
                        return Err(format!("CUDA Launch Failed: {:?}", res)); 
                    }
                    self.doctor.on_kernel_launch(crate::doctor::KernelLaunchInfo {
                             backend: crate::doctor::BackendKind::Cuda,
                             kernel_name: recorded.name.clone(),
                             return_code: 0,
                             last_runtime_error: None,
                             grid: (grid.0, grid.1, grid.2), 
                             block: (block.0, block.1, block.2),
                             smem,
                    });
                }
            }
            KernelHandle::Rocm { func, .. } => {
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("ROCm API not found")?;
                unsafe {
                    let res = (api.hipModuleLaunchKernel)(
                        func.0,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem, std::ptr::null_mut(),
                        kernel_params.as_ptr() as *mut *mut c_void,
                        std::ptr::null_mut()
                    );
                    if res != 0 { return Err(format!("ROCm Launch Failed: {}", res)); }
                }
            }
            #[cfg(feature = "vulkan")]
            KernelHandle::Vulkan { func, .. } => {
                let vk_backend = self.devices.lock().map_err(|_| "Lock")?.get(&DeviceBackend::Vulkan).ok_or("Vulkan Device not found")?.vulkan_dev.clone().ok_or("Vulkan Backend not found")?;
                unsafe {
                    let device = &vk_backend.device;
                    let command_pool_info = vk::CommandPoolCreateInfo::builder()
                        .queue_family_index(vk_backend.queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                    let pool = device.create_command_pool(&command_pool_info, None).map_err(|e| e.to_string())?;
                    
                    let alloc_info = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);
                    let cmd_buf = device.allocate_command_buffers(&alloc_info).map_err(|e| e.to_string())?[0];
                    
                    device.begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| e.to_string())?;
                    device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, func.pipeline);
                    
                    // Simple Dispatch - Parameters/Descriptors FIXME
                    device.cmd_dispatch(cmd_buf, grid.0, grid.1, grid.2);
                    
                    device.end_command_buffer(cmd_buf).map_err(|e| e.to_string())?;
                    device.queue_submit(vk_backend.queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd_buf]).build()], vk::Fence::null()).map_err(|e| e.to_string())?;
                    device.queue_wait_idle(vk_backend.queue).map_err(|e| e.to_string())?;
                    
                    device.destroy_command_pool(pool, None);
                }
            }
            #[cfg(target_os = "macos")]
            KernelHandle::Metal { pipeline, layout } => {
                let devices = self.devices.lock().map_err(|_| "Lock")?;
                let handle = devices.get(&DeviceBackend::Metal).ok_or("No Metal Device")?;
                let backend = handle.metal_dev.as_ref().ok_or("No Metal Backend instance")?;
                
                let command_buffer = backend.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                
                encoder.set_compute_pipeline_state(pipeline);
                
                // Bind Args
                for (i, arg) in args.iter().enumerate() {
                    match arg {
                        KernelArg::Buffer(bid) => {
                            let buf_guards = self.buffers.lock().map_err(|_| "Lock")?;
                            if let Some(DeviceBuffer::Metal(b)) = buf_guards.get(bid) {
                                encoder.set_buffer(i as u64, Some(b), 0);
                            } else {
                                return Err(format!("Arg {} is not a Metal buffer", i));
                            }
                        }
                        KernelArg::Int(val) => {
                             encoder.set_bytes(i as u64, std::mem::size_of::<i32>() as u64, val as *const i32 as *const c_void);
                        }
                        KernelArg::Float(val) => {
                             encoder.set_bytes(i as u64, std::mem::size_of::<f32>() as u64, val as *const f32 as *const c_void);
                        }
                        KernelArg::Usize(val) => {
                             encoder.set_bytes(i as u64, std::mem::size_of::<u64>() as u64, val as *const usize as *const c_void);
                        }
                        KernelArg::Bytes(data) => {
                             encoder.set_bytes(i as u64, data.len() as u64, data.as_ptr() as *const c_void);
                        }
                    }
                }
                
                let thread_group_count = metal::MTLSize { width: grid.0 as u64, height: grid.1 as u64, depth: grid.2 as u64 };
                let thread_group_size = metal::MTLSize { width: block.0 as u64, height: block.1 as u64, depth: block.2 as u64 };
                
                encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
                encoder.end_encoding();
                
                command_buffer.commit();
                command_buffer.wait_until_completed(); // MVP Synchronous
            }
            _ => return Err("Unsupported backend".to_string()),
        }
        Ok(())
    }

    pub fn launch_ttg(
        &self,
        kernel_id: KernelId,
        block: (u32, u32, u32),
        smem: u32,
        base_args: Vec<KernelArg>,
        ttg: &crate::runtime::ttg::DeviceTTG,
        epilogue_args: Vec<KernelArg>
    ) -> Result<(), String> {
        let mut final_args = base_args;
        
        // Append TTG Arguments (L1, L2)
        // Ensure this matches the generated kernel signature: 
        // func(..., l1_ptr, l2_ptr, ...epilogue)
        final_args.push(KernelArg::Buffer(ttg.l1_buffer));
        final_args.push(KernelArg::Buffer(ttg.l2_buffer));
        
        // Append Epilogue Arguments
        final_args.extend(epilogue_args);

        // Grid is determined by active tiles
        let grid = (ttg.num_active_tiles, 1, 1);
        
        self.launch(kernel_id, grid, block, smem, final_args)
    }

    fn load_prebuilt_fallback(&self, kernel_name: &str, backend: DeviceBackend) -> Result<KernelId, String> {
        let safe_name: String = kernel_name.chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
        let cubin_path = format!("E:/Projects/Tracea/prebuilt/{}.cubin", safe_name);
        println!("[Runtime] Attempting fallback load from {}", cubin_path);
        if std::path::Path::new(&cubin_path).exists() && backend == DeviceBackend::Cuda {
             self.load_from_file_and_register(kernel_name, &cubin_path)
        } else {
             println!("[Runtime] Fallback failed: {} not found", cubin_path);
             Err("Fallback failed".to_string())
        }
    }

    pub fn load_from_file_and_register(&self, kernel_name: &str, path: &str) -> Result<KernelId, String> {
        println!("[Runtime] Loading kernel {} from {}", kernel_name, path);
        let cubin_data = std::fs::read(path).map_err(|e| format!("Failed to read cubin: {}", e))?;
        self.load_cubin(&cubin_data, kernel_name)
    }

    pub fn load_cubin(&self, data: &[u8], name: &str) -> Result<KernelId, String> {
        unsafe {
            let mut module: cudarc::driver::sys::CUmodule = std::ptr::null_mut();
            let res = cudarc::driver::sys::lib().cuModuleLoadData(&mut module, data.as_ptr() as *const _);
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { 
                let msg = format!("{:?}", res);
                println!("[Runtime] cuModuleLoadData FAILED for {}: {:?}", name, res);
                self.doctor.on_module_load(crate::doctor::ModuleLoadInfo {
                     backend: crate::doctor::BackendKind::Cuda,
                     kernel_name: name.to_string(),
                     return_code: res as i32,
                     error_msg: Some(msg),
                });
                return Err(format!("cuModuleLoadData failed: {:?}", res)); 
            }
            self.doctor.on_module_load(crate::doctor::ModuleLoadInfo {
                     backend: crate::doctor::BackendKind::Cuda,
                     kernel_name: name.to_string(),
                     return_code: 0,
                     error_msg: None,
            });
            
            let mut func: cudarc::driver::sys::CUfunction = std::ptr::null_mut();
            let name_c = std::ffi::CString::new(name).unwrap();
            let res = cudarc::driver::sys::lib().cuModuleGetFunction(&mut func, module, name_c.as_ptr());
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { 
                println!("[Runtime] cuModuleGetFunction failed for {}: {:?}", name, res);
                return Err(format!("cuModuleGetFunction failed: {:?}", res)); 
            }
            
            println!("[Runtime] Kernel Registered: {} (handle: {:p})", name, func);

            // Set Max Shared Memory (Safe 96KB limit, up to 100KB on Ampere)
            let attr = cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
            let _ = cudarc::driver::sys::lib().cuFuncSetAttribute(func, attr, 101376); 

            let id = self.generate_kernel_id()?;
            self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel { 
                name: name.to_string(), 
                handle: KernelHandle::Cuda {
                    func: CudaFunction(func),
                    module: Arc::new(CudaModule(module)),
                },
                backend: DeviceBackend::Cuda 
            });
            Ok(id)
        }
    }

    pub fn load_rocm(&self, binary: &[u8], name: &str) -> Result<KernelId, String> {
        let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("ROCm API not found")?;
        let mut module_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = (api.hipModuleLoadData)(&mut module_ptr, binary.as_ptr() as *const _);
            if res != 0 { return Err(format!("hipModuleLoadData failed: {}", res)); }
            let mut func_ptr: *mut c_void = std::ptr::null_mut();
            let name_c = std::ffi::CString::new(name).unwrap();
            let _ = (api.hipModuleGetFunction)(&mut func_ptr, module_ptr, name_c.as_ptr());
            let id = self.generate_kernel_id()?;
            self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel { 
                name: name.to_string(), 
                handle: KernelHandle::Rocm {
                    func: RocmFunction(func_ptr),
                    module: Arc::new(RocmModule(module_ptr)),
                },
                backend: DeviceBackend::Rocm 
            });
            Ok(id)
        }
    }

    #[cfg(feature = "vulkan")]
    pub fn load_vulkan(&self, spirv: &[u8], name: &str) -> Result<KernelId, String> {
        let handle = self.get_device(DeviceBackend::Vulkan)?; 
        let vk_backend = handle.vulkan_dev.as_ref().ok_or("Vulkan device not found")?;
        let device = &vk_backend.device;
        
        unsafe {
            let shader_module_info = vk::ShaderModuleCreateInfo::builder()
                .code(bytemuck::cast_slice(spirv));
            let module = device.create_shader_module(&shader_module_info, None).map_err(|e| e.to_string())?;
            
            let entry_point_name = CString::new("main").unwrap();
            let stage_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry_point_name);
            
            let layout_info = vk::PipelineLayoutCreateInfo::builder();
            let layout = device.create_pipeline_layout(&layout_info, None).map_err(|e| e.to_string())?;
            
            let compute_info = vk::ComputePipelineCreateInfo::builder()
                .stage(stage_info.build())
                .layout(layout);
            
            let pipeline_res = device.create_compute_pipelines(vk::PipelineCache::null(), &[compute_info.build()], None);
            let pipeline = match pipeline_res {
                Ok(pipes) => pipes[0],
                Err((pipes, err)) => {
                    if !pipes.is_empty() {
                         for p in pipes { device.destroy_pipeline(p, None); }
                    }
                    return Err(format!("Pipeline creation failed: {:?}", err));
                }
            };
            
            let id = self.generate_kernel_id()?;
            self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel {
                name: name.to_string(),
                handle: KernelHandle::Vulkan {
                    func: VulkanFunction { pipeline },
                    module: Arc::new(VulkanModule { module, device: device.clone() }),
                },
                backend: DeviceBackend::Vulkan,
            });
            Ok(id)
        }
    }

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
                    let res = (api.hipMalloc)(&mut ptr, size_bytes);
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
                let buffer = backend.device.new_buffer(size_bytes as u64, options);
                DeviceBuffer::Metal(buffer)
            }
            _ => return Err("Alloc failed".to_string()),
        };
        self.buffers.lock().map_err(|_| "Lock")?.insert(id, buf);
        Ok(id)
    }

    pub fn get_device_ptr(&self, id: BufferId) -> Result<u64, String> {
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

    pub fn copy_to_device<T: Copy>(&self, id: BufferId, data: &[T]) -> Result<(), String> {
        let mut bufs = self.buffers.lock().map_err(|_| "Lock".to_string())?;
        match bufs.get_mut(&id).ok_or("No buffer".to_string())? {
            DeviceBuffer::Cuda(slice) => {
                let devs = self.devices.lock().map_err(|_| "Lock".to_string())?;
                // Ensure CUDA context exists. dev reference unused but check existence.
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
                 unsafe {
                     std::ptr::copy_nonoverlapping(data.as_ptr() as *const _, void_ptr as *mut _, len_bytes);
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

    pub fn get_max_shared_memory(&self, backend: DeviceBackend) -> usize {
        match backend {
            DeviceBackend::Cuda => {
                let devs = self.devices.lock().unwrap();
                if let Some(handle) = devs.get(&DeviceBackend::Cuda) {
                    if let Some(dev) = &handle.cuda_dev {
                        // On Ampere+ GPUs, the limit is often higher than 48KB but needs opt-in.
                        // We query the maximum possible opt-in size.
                        let optin = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).unwrap_or(0);
                        let total = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK).unwrap_or(49152);
                        let limit = std::cmp::max(optin as i32, total as i32);
                        return limit as usize;
                    }
                }
                49152 // Fallback
            }
            _ => 32768, // Conservative fallback for others
        }
    }

    pub fn get_max_threads_per_block(&self, backend: DeviceBackend) -> usize {
        match backend {
            DeviceBackend::Cuda => {
                let devs = self.devices.lock().unwrap();
                if let Some(handle) = devs.get(&DeviceBackend::Cuda) {
                    if let Some(dev) = &handle.cuda_dev {
                        return dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap_or(1024) as usize;
                    }
                }
                1024
            }
            _ => 1024,
        }
    }

    pub fn save_binary(&self, kernel_name: &str, arch: &str, cubin_path: &str) -> Result<(), String> {
        let env_id = self.doctor.get_environment_id();
        let cache_dir = format!("E:/Projects/Tracea/cache/{}/{}", env_id, arch);
        std::fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
        
        let dest_path = format!("{}/{}.cubin", cache_dir, kernel_name);
        std::fs::copy(cubin_path, &dest_path).map_err(|e| e.to_string())?;
        println!("[Runtime] 💾 Binary cached to {}", dest_path);
        Ok(())
    }

    pub fn load_binary(&self, cache_name: &str, kernel_name: &str, arch: &str) -> Result<KernelId, String> {
        let env_id = self.doctor.get_environment_id();
        let cache_path = format!("E:/Projects/Tracea/cache/{}/{}/{}.cubin", env_id, arch, cache_name);
        
        if std::path::Path::new(&cache_path).exists() {
            println!("[Runtime] 🚀 AOT Cache Hit: {}", cache_path);
            self.load_from_file_and_register(kernel_name, &cache_path)
        } else {
            Err("Cache miss".to_string())
        }
    }

    pub fn read_buffer(&self, id: BufferId) -> Result<Vec<u8>, String> {
        let bufs = self.buffers.lock().map_err(|_| "Lock")?;
        let buf = bufs.get(&id).ok_or("No buffer")?;
        
        match buf {
            DeviceBuffer::Cuda(slice) => {
                let len = slice.len();
                let mut host = vec![0u8; len];
                let _lock = self.devices.lock(); // Keep devices alive?
                
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

    pub fn launch_with_policy(
        &self,
        kernel_id: KernelId,
        args: Vec<KernelArg>,
        op: &crate::policy::types::OperatorTopology,
        t_policy: &crate::policy::types::TilePolicy,
        e_policy: &crate::policy::types::ExecPolicy,
        epilogue_args: Vec<KernelArg>,
        backend: DeviceBackend,
    ) -> Result<(), String> {
        let layout = crate::runtime::ttg_builder::TTGBuilder::from_policy(op, t_policy);
        let device_ttg = crate::runtime::ttg::DeviceTTG::new(self, &layout, backend)?;
        let block = e_policy.backend_hint.preferred_block_dim;
        let smem = 48 * 1024;
        self.launch_ttg(kernel_id, block, smem as u32, args, &device_ttg, epilogue_args)
    }

    /// Execute a graph using Policy-driven scheduling.
    /// This is the main entry point for "Universal Compute OS" execution.
    /// 
    /// # Arguments
    /// * `graph` - The operator graph topology
    /// * `input_buffers` - Map from op_id to pre-allocated input BufferId
    /// * `output_buffers` - Map from op_id to output BufferId (will be allocated if not present)
    /// * `backend` - Target execution backend
    pub fn execute_graph(
        &self,
        graph: &crate::policy::types::GraphTopology,
        input_buffers: &HashMap<u64, BufferId>,
        backend: DeviceBackend,
    ) -> Result<HashMap<u64, BufferId>, String> {
        use crate::policy::types::*;
        use crate::policy::engine::PolicyEngine;
        use crate::policy::standard::StandardPolicyEngine;
        use crate::policy::scheduler::StandardScheduler;
        use crate::core::device::DeviceProfile;

        // 0. Canonicalize Graph
        let mut graph = graph.clone();
        crate::policy::transform::canonicalize_graph(&mut graph);

        // 1. Get Device Profile
        let device = DeviceProfile::from_backend(backend);

        // 2. Policy Decision
        let mut engine = StandardPolicyEngine::new();
        let ctx = GraphContext {
            device: &device,
            graph: &graph,
        };

        // 3. Schedule
        let decision = StandardScheduler::schedule(&mut engine, &ctx);

        // 4. Memory Pool (alias-aware allocation)
        let mut memory_pool: HashMap<usize, BufferId> = HashMap::new();
        let mut output_buffers: HashMap<u64, BufferId> = HashMap::new();

        // 5. Execute Loop (sorted by operator_id as execution order proxy)
        let mut sorted_policies: Vec<_> = decision.exec_policies.iter().collect();
        sorted_policies.sort_by_key(|p| p.operator_id);

        for exec_policy in sorted_policies {
            let op_id = exec_policy.operator_id;
            
            // Find corresponding tile policy and operator
            let tile_policy = decision.tile_policies.iter()
                .find(|t| t.operator_id() == op_id);
            let operator = graph.operators.iter()
                .find(|o| o.op_id() == op_id);

            let (tile_policy, operator) = match (tile_policy, operator) {
                (Some(t), Some(o)) => (t, o),
                _ => continue, // Skip if not found
            };

            // Resolve output buffer (with alias support)
            let buf_id = self.resolve_buffer_with_alias(
                op_id,
                &exec_policy.memory_alias_hint,
                &mut memory_pool,
                Self::estimate_output_size(operator),
                backend,
            )?;
            output_buffers.insert(op_id, buf_id);

            // Phase F: TTG Builder Integration
            // 1. Generate TTGLayout from Policy
            let layout = crate::runtime::ttg_builder::TTGBuilder::from_policy(operator, tile_policy);
            
            // 2. Upload TTGLayout to GPU (optional - only for operators that use TTG)
            // For now, we store the layout for potential future use
            #[allow(unused_variables)]
            let device_ttg_result = match backend {
                #[cfg(target_os = "macos")]
                crate::runtime::manager::DeviceBackend::Metal => {
                    Some(crate::runtime::ttg::DeviceTTG::new(self, &layout, backend))
                },
                crate::runtime::manager::DeviceBackend::Cuda => {
                    Some(crate::runtime::ttg::DeviceTTG::new(self, &layout, backend))
                },
                _ => None, // CPU and other backends don't use device TTG
            };
            
            // Handle TTG upload errors gracefully (don't fail entire graph)
            if let Some(Err(e)) = device_ttg_result {
                eprintln!("[TTG] Warning: Failed to upload TTG for op {}: {}", op_id, e);
            }
            
            // 3. Log TTG generation (for verification)
            #[cfg(debug_assertions)]
            {
                eprintln!("[TTG] Op {} -> {} active tiles, variant: {:?}", 
                    op_id, layout.num_active_tiles, layout.variant);
            }
            
            // Note: Actual kernel launch is handled by launch_with_policy
            // which already integrates with TTGBuilder::from_policy
        }

        Ok(output_buffers)
    }

    /// Resolve buffer with alias support.
    /// If alias_hint has an offset, try to reuse existing buffer from pool.
    fn resolve_buffer_with_alias(
        &self,
        _op_id: u64,
        alias_hint: &crate::policy::types::MemoryAliasPolicy,
        pool: &mut HashMap<usize, BufferId>,
        size: usize,
        backend: DeviceBackend,
    ) -> Result<BufferId, String> {
        if let Some(offset) = alias_hint.output_offset {
            // Check if we have a buffer at this offset
            if let Some(&existing_id) = pool.get(&offset) {
                return Ok(existing_id);
            }
            // Allocate new and store in pool
            let new_id = self.alloc(size, backend)?;
            pool.insert(offset, new_id);
            Ok(new_id)
        } else {
            // No alias, fresh allocation
            self.alloc(size, backend)
        }
    }

    /// Estimate output buffer size for an operator (in bytes).
    fn estimate_output_size(op: &crate::policy::types::OperatorTopology) -> usize {
        use crate::policy::types::OperatorTopology;
        match op {
            OperatorTopology::Gemm { m, n, .. } => (*m as usize) * (*n as usize) * 4, // fp32
            OperatorTopology::Attention { b, s, h, d, .. } => {
                (*b as usize) * (*h as usize) * (*s as usize) * (*d as usize) * 2 // fp16
            },
            OperatorTopology::Conv2d { n, k, h, w, .. } => {
                (*n as usize) * (*k as usize) * (*h as usize) * (*w as usize) * 4
            },
            OperatorTopology::Relu { .. } | OperatorTopology::Elementwise { .. } => {
                1024 * 1024 // Placeholder 1MB
            },
        }
    }
}

impl RecordedKernel {
    fn backend_copy(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::types::{GraphTopology, OperatorTopology, TopologyKind};

    #[test]
    #[cfg(target_os = "macos")]
    fn test_execute_graph_simple() {
        // Create a simple 2-op graph: GEMM1 -> GEMM2
        let graph = GraphTopology {
            operators: vec![
                OperatorTopology::Gemm {
                    op_id: 1,
                    name: "gemm1".into(),
                    m: 64, n: 64, k: 64,
                    kind: TopologyKind::Dense,
                },
                OperatorTopology::Gemm {
                    op_id: 2,
                    name: "gemm2".into(),
                    m: 64, n: 64, k: 64,
                    kind: TopologyKind::Dense,
                },
            ],
            dependencies: vec![(1, 2)], // gemm1 -> gemm2
        };

        // Use Metal backend (macOS)
        let runtime = RuntimeManager::new();
        let input_buffers = HashMap::new();

        // Execute graph - should return output buffer map
        let result = runtime.execute_graph(&graph, &input_buffers, DeviceBackend::Metal);
        
        // Should succeed and return buffers for both ops
        assert!(result.is_ok(), "execute_graph failed: {:?}", result.err());
        let output_buffers = result.unwrap();
        assert!(output_buffers.contains_key(&1), "Missing buffer for op 1");
        assert!(output_buffers.contains_key(&2), "Missing buffer for op 2");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_execute_graph_with_reuse() {
        // Create a graph with two independent chains that should reuse memory
        // Chain 1: Op1 -> Op2
        // Chain 2: Op3 -> Op4
        // Op1 and Op3 should be able to share memory with Op2 and Op4 respectively
        let graph = GraphTopology {
            operators: vec![
                OperatorTopology::Gemm {
                    op_id: 1, name: "gemm1".into(),
                    m: 128, n: 128, k: 128, kind: TopologyKind::Dense,
                },
                OperatorTopology::Gemm {
                    op_id: 2, name: "gemm2".into(),
                    m: 128, n: 128, k: 128, kind: TopologyKind::Dense,
                },
                OperatorTopology::Gemm {
                    op_id: 3, name: "gemm3".into(),
                    m: 128, n: 128, k: 128, kind: TopologyKind::Dense,
                },
                OperatorTopology::Gemm {
                    op_id: 4, name: "gemm4".into(),
                    m: 128, n: 128, k: 128, kind: TopologyKind::Dense,
                },
            ],
            dependencies: vec![(1, 2), (3, 4)], // Two independent chains
        };

        let runtime = RuntimeManager::new();
        let input_buffers = HashMap::new();

        let result = runtime.execute_graph(&graph, &input_buffers, DeviceBackend::Metal);
        assert!(result.is_ok(), "execute_graph failed: {:?}", result.err());
        
        let output_buffers = result.unwrap();
        
        // All 4 ops should have buffers
        assert_eq!(output_buffers.len(), 4);
        
        // Due to memory aliasing, some BufferIds might be shared
        // The scheduler should have assigned same offset to disjoint ops
        // We verify that the system works without crashing
    }
}
