use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use cudarc::driver::*;
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::ffi::c_void;
use std::io::Write;
use serde::{Serialize, Deserialize};
use crate::doctor::{BackendKind, JitResultInfo, AssemblerResultInfo, KernelLaunchInfo, ModuleLoadInfo};

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceBackend {
    Cuda,
    Rocm,
    Metal,
    Cpu,
}

#[derive(Debug)]
pub enum DeviceBuffer {
    Cuda(CudaSlice<u8>),
    Rocm(RocmBuffer), 
    External(u64),
}

#[derive(Debug)]
pub struct RocmBuffer {
    pub ptr: u64,
}

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

#[derive(Debug, Clone)]
pub struct RecordedKernel {
    pub name: String,
    pub raw: u64, // handle to CUfunction
    pub module: u64, // handle to CUmodule
    pub backend: DeviceBackend,
}

#[derive(Debug, Clone)]
pub struct DeviceHandle {
    pub backend: DeviceBackend,
    pub cuda_dev: Option<Arc<CudaDevice>>,
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
    pub fn init(_pref_backend: Option<DeviceBackend>) -> Result<Arc<Self>, String> {
        let mut cache = INSTANCE.lock().map_err(|_| "Global Instance Lock Poisoned".to_string())?;
        if let Some(instance) = &*cache {
            return Ok(Arc::clone(instance));
        }

        let mut devices = HashMap::new();
        if let Ok(dev) = CudaDevice::new(0) {
            let major = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(8);
            let minor = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(6);
            devices.insert(DeviceBackend::Cuda, DeviceHandle {
                backend: DeviceBackend::Cuda,
                cuda_dev: Some(dev),
                arch: format!("sm_{}{}", major, minor),
            });
            println!("[Doctor] 🟢 CUDA Device Registered.");
        }

        if let Some(_) = crate::emitter::rocm_driver::RocmDriverApi::get() {
             devices.insert(DeviceBackend::Rocm, DeviceHandle {
                 backend: DeviceBackend::Rocm,
                 cuda_dev: None,
                 arch: "gfx90a".to_string(),
             });
             println!("[Doctor] 🟢 ROCm Backend Registered.");
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
            DeviceBackend::Cuda => {
                let devices = self.devices.lock().map_err(|_| "Lock")?;
                let cuda_handle = devices.get(&DeviceBackend::Cuda).ok_or("No CUDA device")?;
                let target_arch = cuda_handle.arch.clone();
                let _target_sm = target_arch.replace("sm_", "");

                let arch_static: &'static str = Box::leak(target_arch.clone().into_boxed_str());

                // Source-aware AOT Cache Check
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                source.hash(&mut hasher);
                let source_hash = hasher.finish();
                let cache_name = format!("{}_{:x}", kernel_name, source_hash);

                if let Ok(id) = self.load_binary(&cache_name, kernel_name, &target_arch) {
                    self.source_cache.lock().unwrap().insert(source.to_string(), id);
                    return Ok(id);
                }

                let mut opts = CompileOptions {
                    arch: Some(arch_static), 
                    options: vec![
                        "-I".to_string(), 
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include".to_string(),
                        // "--use_fast_math".to_string(),
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

                let mut ptx_src = match ptx_res {
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
                
                ptx_src = ptx_src.replace(".version 8.5", ".version 7.0")
                                  .replace(".version 8.4", ".version 7.0")
                                  .replace(".target sm_86", ".target sm_80");

                let safe_name: String = kernel_name.chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
                let ptx_path = format!("E:/Projects/Tracea/debug_dump_{}.ptx", safe_name);
                let _ = std::fs::write(&ptx_path, &ptx_src);
                let cubin_path = format!("E:/Projects/Tracea/debug_dump_{}.cubin", safe_name);
                
                let mut cmd = std::process::Command::new("ptxas");
                cmd.arg("-v").arg("--gpu-name").arg(&target_arch).arg(&ptx_path).arg("-o").arg(&cubin_path);
                
                println!("[Runtime] Executing: ptxas -v --gpu-name {} {} -o {}", target_arch, ptx_path, cubin_path);

                let output = cmd.output().map_err(|e| format!("Failed to run ptxas: {}", e))?;
                
                // Doctor Hook: PTXAS Result
                let pt_stderr = String::from_utf8_lossy(&output.stderr).to_string();
                self.doctor.on_assembler_result(crate::doctor::AssemblerResultInfo {
                    backend: crate::doctor::BackendKind::Cuda,
                    arch: target_arch.clone(),
                    return_code: output.status.code().unwrap_or(-1),
                    stderr: pt_stderr.clone(),
                    ptx_content: ptx_src.clone(),
                    cubin_size: Some(std::fs::metadata(&cubin_path).map(|m| m.len()).unwrap_or(0)),
                });

                if !output.status.success() {
                    let err = String::from_utf8_lossy(&output.stderr);
                    let out = String::from_utf8_lossy(&output.stdout);
                    println!("[Runtime] ptxas FAILED. Code: {:?}", output.status.code());
                    println!("[Runtime] stderr: {}", err);
                    println!("[Runtime] stdout: {}", out);
                    return Err(format!("ptxas failed: {} (stdout: {})", err, out));
                }
                println!("[Runtime] ptxas SUCCESS: {}", pt_stderr);

                let id = self.load_from_file_and_register(kernel_name, &cubin_path)?;
                
                // Save to AOT Cache with source-aware name
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                source.hash(&mut hasher);
                let source_hash = hasher.finish();
                let cache_name = format!("{}_{:x}", kernel_name, source_hash);
                
                let _ = self.save_binary(&cache_name, &target_arch, &cubin_path);
                
                Ok(id)
            }
            DeviceBackend::Rocm => {
                let jit = crate::emitter::rocm_jit::ROCMJITCompiler::new().ok_or("ROCm JIT API not found")?;
                let binary = jit.compile(source, kernel_name, vec![])?;
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("ROCm Driver API not found")?;
                let mut module: *mut c_void = std::ptr::null_mut();
                unsafe {
                    let _ = (api.hipModuleLoadData)(&mut module, binary.as_ptr() as *const _);
                    let mut func: *mut c_void = std::ptr::null_mut();
                    let name_c = std::ffi::CString::new(kernel_name).unwrap();
                    let _ = (api.hipModuleGetFunction)(&mut func, module, name_c.as_ptr());
                    let id = self.generate_kernel_id()?;
                    self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel {
                        name: kernel_name.to_string(),
                        raw: func as u64,
                        module: module as u64,
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
            }
        }

        match recorded.backend {
            DeviceBackend::Cuda => {
                unsafe {
                    let lib = cudarc::driver::sys::lib();
                    
                    let res = lib.cuLaunchKernel(
                        recorded.raw as sys::CUfunction,
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
            DeviceBackend::Rocm => {
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("ROCm API not found")?;
                unsafe {
                    let res = (api.hipModuleLaunchKernel)(
                        recorded.raw as *mut _,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem, std::ptr::null_mut(),
                        kernel_params.as_ptr() as *mut *mut c_void,
                        std::ptr::null_mut()
                    );
                    if res != 0 { return Err(format!("ROCm Launch Failed: {}", res)); }
                }
            }
            _ => return Err("Unsupported backend".to_string()),
        }
        Ok(())
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
                raw: func as u64, 
                module: module as u64,
                backend: DeviceBackend::Cuda 
            });
            Ok(id)
        }
    }

    pub fn load_rocm(&self, binary: &[u8], name: &str) -> Result<KernelId, String> {
        let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("ROCm API not found")?;
        let mut module: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = (api.hipModuleLoadData)(&mut module, binary.as_ptr() as *const _);
            if res != 0 { return Err(format!("hipModuleLoadData failed: {}", res)); }
            let mut func: *mut c_void = std::ptr::null_mut();
            let name_c = std::ffi::CString::new(name).unwrap();
            let _ = (api.hipModuleGetFunction)(&mut func, module, name_c.as_ptr());
            let id = self.generate_kernel_id()?;
            self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel { 
                name: name.to_string(), 
                raw: func as u64, 
                module: module as u64,
                backend: DeviceBackend::Rocm 
            });
            Ok(id)
        }
    }

    pub fn alloc(&self, size_bytes: usize, backend: DeviceBackend) -> Result<BufferId, String> {
        let id = self.generate_buffer_id()?;
        let buf = match backend {
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
}

impl RecordedKernel {
    fn backend_copy(&self) -> Self {
        Self { name: self.name.clone(), raw: self.raw, module: self.module, backend: self.backend }
    }
}
