use std::sync::{Arc, Mutex};
use std::hash::{Hash, Hasher};
use std::ffi::{c_void, CString};
use std::io::Write;
use super::{KernelId, DeviceBackend, RuntimeManager};
use crate::doctor::{JitResultInfo, ModuleLoadInfo};

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

#[derive(Debug)]
pub struct RocmModule(pub *mut c_void);

impl Drop for RocmModule {
    fn drop(&mut self) {
        if !self.0.is_null() {
            if let Some(api) = crate::emitter::rocm_driver::RocmDriverApi::get() {
                unsafe { (api.hip_module_unload)(self.0); }
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
        layout: Vec<crate::runtime::manager::KernelArgKind>,
    },
}

#[derive(Debug, Clone)]
pub struct RecordedKernel {
    pub name: String,
    pub handle: KernelHandle,
    pub backend: DeviceBackend,
}

impl RecordedKernel {
    pub fn backend_copy(&self) -> Self {
        self.clone()
    }
}

impl RuntimeManager {
    pub fn compile(&self, source: &str, kernel_name: &str, backend: DeviceBackend) -> Result<KernelId, String> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(source, &mut hasher);
        let hash = std::hash::Hasher::finish(&hasher);
        let cache_key = format!("{}_{:x}", kernel_name, hash);

        {
            let cache = self.source_cache.lock().map_err(|_| "Lock")?;
            if let Some(id) = cache.get(&cache_key) {
                return Ok(*id);
            }
        }
        
        println!("[Runtime] Debug: compiling {} (Hash: {:x})", kernel_name, hash);

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
                
                let compile_options = metal::CompileOptions::new();
                let library = backend.device.new_library_with_source(source, &compile_options).map_err(|e| format!("Metal Compile Error: {}", e))?;
                
                let func = library.get_function(kernel_name, None).map_err(|e| format!("Function '{}' not found", kernel_name))?;
                
                let pipeline = backend.device.new_compute_pipeline_state_with_function(&func).map_err(|e| format!("Pipeline Error: {}", e))?;

                let id = self.generate_kernel_id()?;
                self.source_cache.lock().map_err(|_| "Lock")?.insert(cache_key, id);
                self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel {
                    name: kernel_name.to_string(),
                    handle: KernelHandle::Metal {
                        pipeline,
                        layout: Vec::new(),
                    },
                    backend: DeviceBackend::Metal,
                });
                return Ok(id);
            }
            DeviceBackend::Cuda => {
                let devices = self.devices.lock().map_err(|_| "Lock")?;
                let cuda_handle = devices.get(&DeviceBackend::Cuda).ok_or("No CUDA device")?;
                let target_arch = cuda_handle.arch.clone();

                let mut arch_static: &'static str = "sm_80";
                if let Some(handle) = devices.get(&DeviceBackend::Cuda) {
                    arch_static = Box::leak(handle.arch.clone().into_boxed_str());
                }

                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                source.hash(&mut hasher);
                let source_hash = hasher.finish();
                let cache_name = format!("{}_{:x}", kernel_name, source_hash);

                if let Ok(id) = self.load_binary(&cache_name, kernel_name, arch_static) {
                    self.source_cache.lock().unwrap().insert(source.to_string(), id);
                    return Ok(id);
                }

                if let Ok(mut f) = std::fs::File::create("last_kernel.cu") {
                    let _ = f.write_all(source.as_bytes());
                }

                let opts = cudarc::nvrtc::CompileOptions {
                    arch: Some(arch_static), 
                    options: vec![
                        "-I".to_string(), 
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include".to_string(),
                        format!("--gpu-architecture=compute_{}", arch_static.replace("sm_", "")),
                    ],
                    ..Default::default()
                };
                
                let ptx_res = cudarc::nvrtc::compile_ptx_with_opts(source, opts);
                
                let (jit_code, jit_log) = match &ptx_res {
                    Ok(_) => (0, String::new()),
                    Err(e) => (1, format!("{:?}", e)),
                };
                 self.doctor.on_jit_result(JitResultInfo {
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
                         return self.load_prebuilt_fallback(kernel_name, backend); 
                    }
                };
                
                println!("[Runtime] Loading PTX directly via Driver JIT...");
                
                unsafe {
                    let lib = cudarc::driver::sys::lib();
                    let mut module: cudarc::driver::sys::CUmodule = std::ptr::null_mut();
                    let ptx_cstring = CString::new(ptx_src.clone()).unwrap();

                    let res = lib.cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const _);
                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        let err = format!("Driver JIT Failed: {:?}", res);
                        println!("[Runtime] ❌ {}", err);
                        return Err(err);
                    }
                    
                    let mut func: cudarc::driver::sys::CUfunction = std::ptr::null_mut();
                    let name_c = CString::new(kernel_name).unwrap();
                    let res_func = lib.cuModuleGetFunction(&mut func, module, name_c.as_ptr());
                    
                    if res_func != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                         return Err(format!("Failed to get kernel function '{}': {:?}", kernel_name, res_func));
                    }

                    let _ = lib.cuFuncSetAttribute(
                        func, 
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 
                        98304
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
                    let _ = (api.hip_module_load_data)(&mut module_ptr, binary.as_ptr() as *const _);
                    let mut func_ptr: *mut c_void = std::ptr::null_mut();
                    let name_c = CString::new(kernel_name).unwrap();
                    let _ = (api.hip_module_get_function)(&mut func_ptr, module_ptr, name_c.as_ptr());
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

        self.source_cache.lock().map_err(|_| "Poisoned")?.insert(source.to_string(), id);
        Ok(id)
    }

    pub fn load_cubin(&self, data: &[u8], name: &str) -> Result<KernelId, String> {
        unsafe {
            let lib = cudarc::driver::sys::lib();
            let mut module: cudarc::driver::sys::CUmodule = std::ptr::null_mut();
            let res = lib.cuModuleLoadData(&mut module, data.as_ptr() as *const _);
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { 
                let msg = format!("{:?}", res);
                self.doctor.on_module_load(ModuleLoadInfo {
                     backend: crate::doctor::BackendKind::Cuda,
                     kernel_name: name.to_string(),
                     return_code: res as i32,
                     error_msg: Some(msg),
                });
                return Err(format!("cuModuleLoadData failed: {:?}", res)); 
            }
            self.doctor.on_module_load(ModuleLoadInfo {
                     backend: crate::doctor::BackendKind::Cuda,
                     kernel_name: name.to_string(),
                     return_code: 0,
                     error_msg: None,
            });
            
            let mut func: cudarc::driver::sys::CUfunction = std::ptr::null_mut();
            let name_c = CString::new(name).unwrap();
            let res = lib.cuModuleGetFunction(&mut func, module, name_c.as_ptr());
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { 
                return Err(format!("cuModuleGetFunction failed: {:?}", res)); 
            }
            
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
            let res = (api.hip_module_load_data)(&mut module_ptr, binary.as_ptr() as *const _);
            if res != 0 { return Err(format!("hipModuleLoadData failed: {}", res)); }
            let mut func_ptr: *mut c_void = std::ptr::null_mut();
            let name_c = CString::new(name).unwrap();
            let _ = (api.hip_module_get_function)(&mut func_ptr, module_ptr, name_c.as_ptr());
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
            let shader_module_info = ash::vk::ShaderModuleCreateInfo::builder()
                .code(bytemuck::cast_slice(spirv));
            let module = device.create_shader_module(&shader_module_info, None).map_err(|e| e.to_string())?;
            
            let entry_point_name = CString::new("main").unwrap();
            let stage_info = ash::vk::PipelineShaderStageCreateInfo::builder()
                .stage(ash::vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry_point_name);
            
            let layout_info = ash::vk::PipelineLayoutCreateInfo::builder();
            let layout = device.create_pipeline_layout(&layout_info, None).map_err(|e| e.to_string())?;
            
            let compute_info = ash::vk::ComputePipelineCreateInfo::builder()
                .stage(stage_info.build())
                .layout(layout);
            
            let pipeline_res = device.create_compute_pipelines(ash::vk::PipelineCache::null(), &[compute_info.build()], None);
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

    pub fn load_from_file_and_register(&self, kernel_name: &str, path: &str) -> Result<KernelId, String> {
        let cubin_data = std::fs::read(path).map_err(|e| format!("Failed to read cubin: {}", e))?;
        self.load_cubin(&cubin_data, kernel_name)
    }

    pub fn save_binary(&self, kernel_name: &str, arch: &str, cubin_path: &str) -> Result<(), String> {
        let env_id = self.doctor.get_environment_id();
        let cache_dir = format!("E:/Projects/Tracea/cache/{}/{}", env_id, arch);
        std::fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
        
        let dest_path = format!("{}/{}.cubin", cache_dir, kernel_name);
        std::fs::copy(cubin_path, &dest_path).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load_binary(&self, cache_name: &str, kernel_name: &str, arch: &str) -> Result<KernelId, String> {
        let env_id = self.doctor.get_environment_id();
        let cache_path = format!("E:/Projects/Tracea/cache/{}/{}/{}.cubin", env_id, arch, cache_name);
        
        if std::path::Path::new(&cache_path).exists() {
            self.load_from_file_and_register(kernel_name, &cache_path)
        } else {
            Err("Cache miss".to_string())
        }
    }

    fn load_prebuilt_fallback(&self, kernel_name: &str, backend: DeviceBackend) -> Result<KernelId, String> {
        let safe_name: String = kernel_name.chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
        let cubin_path = format!("E:/Projects/Tracea/prebuilt/{}.cubin", safe_name);
        if std::path::Path::new(&cubin_path).exists() && backend == DeviceBackend::Cuda {
             self.load_from_file_and_register(kernel_name, &cubin_path)
        } else {
             Err("Fallback failed".to_string())
        }
    }

    pub fn get_max_shared_memory(&self, backend: DeviceBackend) -> usize {
        match backend {
            DeviceBackend::Cuda => {
                let devs = self.devices.lock().unwrap();
                if let Some(handle) = devs.get(&DeviceBackend::Cuda) {
                    if let Some(dev) = &handle.cuda_dev {
                        let optin = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).unwrap_or(0);
                        let total = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK).unwrap_or(49152);
                        return std::cmp::max(optin as i32, total as i32) as usize;
                    }
                }
                49152
            }
            _ => 32768,
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
}
