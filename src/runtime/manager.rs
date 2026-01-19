use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use cudarc::driver::*;
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::ffi::c_void;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[derive(Clone)]
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
}

pub enum DeviceBuffer {
    Cuda(CudaSlice<u8>),
    Rocm(u64), 
    External(u64),
}

pub struct RecordedKernel {
    pub safe: Option<CudaFunction>, // Only for CUDA
    pub raw: u64, // Raw handle (CUfunction or hipFunction_t)
    pub backend: DeviceBackend,
}

#[derive(Clone)]
pub struct DeviceHandle {
    pub backend: DeviceBackend,
    pub cuda_dev: Option<Arc<CudaDevice>>,
    pub arch: String,
}

pub struct RuntimeManager {
    // Multi-Device Support
    pub devices: Mutex<HashMap<DeviceBackend, DeviceHandle>>,
    
    // Resource registries (Backend-agnostic identifiers)
    kernels: Mutex<HashMap<KernelId, RecordedKernel>>,
    buffers: Mutex<HashMap<BufferId, DeviceBuffer>>,
    
    // Simple ID generators
    next_kernel_id: Mutex<u64>,
    next_buffer_id: Mutex<u64>,
    
    pub compatibility_log: Mutex<Vec<String>>,
}

static INSTANCE: Mutex<Option<Arc<RuntimeManager>>> = Mutex::new(None);

impl RuntimeManager {
    pub fn init(_pref_backend: Option<DeviceBackend>) -> Result<Arc<Self>, String> {
        let mut cache = INSTANCE.lock().map_err(|_| "Global Instance Lock Poisoned".to_string())?;
        if let Some(instance) = &*cache {
            return Ok(Arc::clone(instance));
        }

        let mut devices = HashMap::new();

        // 1. Try CUDA
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

        // 2. Try ROCm
        if let Some(_) = crate::emitter::rocm_driver::RocmDriverApi::get() {
             devices.insert(DeviceBackend::Rocm, DeviceHandle {
                 backend: DeviceBackend::Rocm,
                 cuda_dev: None,
                 arch: "gfx90a".to_string(), // In practice, detect ISA
             });
             println!("[Doctor] 🟢 ROCm Backend Registered.");
        }

        let instance = Arc::new(Self {
            devices: Mutex::new(devices),
            kernels: Mutex::new(HashMap::new()),
            buffers: Mutex::new(HashMap::new()),
            next_kernel_id: Mutex::new(0),
            next_buffer_id: Mutex::new(0),
            compatibility_log: Mutex::new(Vec::new()),
        });
        
        *cache = Some(Arc::clone(&instance));
        Ok(instance)
    }

    pub fn log_compatibility(&self, msg: &str) {
        let mut log = self.compatibility_log.lock().unwrap_or_else(|e| e.into_inner());
        log.push(format!("[{:.2}] {}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(), msg));
        println!("[Doctor] Compatibility: {}", msg);
    }
    
    fn generate_kernel_id(&self) -> Result<KernelId, String> {
        let mut lock = self.next_kernel_id.lock().map_err(|_| "Next Kernel ID Lock Poisoned".to_string())?;
        let id = *lock;
        *lock += 1;
        Ok(KernelId(id))
    }

    fn generate_buffer_id(&self) -> Result<BufferId, String> {
        let mut lock = self.next_buffer_id.lock().map_err(|_| "Next Buffer ID Lock Poisoned".to_string())?;
        let id = *lock;
        *lock += 1;
        Ok(BufferId(id))
    }

    pub fn compile(&self, source: &str, kernel_name: &str, backend: DeviceBackend) -> Result<KernelId, String> {
        self.log_compatibility(&format!("Compiling {} for {:?}...", kernel_name, backend));
        
        match backend {
            DeviceBackend::Cuda => {
                let devices = self.devices.lock().map_err(|_| "Lock Poisoned")?;
                let handle = devices.get(&DeviceBackend::Cuda).ok_or("No CUDA device available")?;
                let cuda_dev = handle.cuda_dev.as_ref().ok_or("CUDA device handle missing")?;

                let driver_ver = crate::emitter::jit::JITCompiler::get_driver_version().unwrap_or(0);
                
                let mut target_arch = "compute_86";
                let mut target_sm = "sm_86";
                
                if driver_ver > 0 && driver_ver < 12040 {
                     target_arch = "compute_80";
                     target_sm = "sm_80";
                }
                
                let _ = std::fs::write("E:/Projects/Tracea/debug_last_source.cu", source);
                
                let opts = CompileOptions {
                    ftz: Some(false),
                    prec_div: Some(true),
                    prec_sqrt: Some(true),
                    fmad: Some(true),
                    arch: Some(target_arch), 
                    options: vec![
                        "-I".to_string(),
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include".to_string()
                    ],
                    ..Default::default()
                };
                
                let ptx_res = cudarc::nvrtc::compile_ptx_with_opts(source, opts);
                let mut ptx_src = if let Ok(ptx) = ptx_res {
                     ptx.to_src()
                } else {
                     let err = ptx_res.err().unwrap();
                     eprintln!("[Doctor] NVRTC Compilation Failed: {:?}", err);
                     return self.load_prebuilt_fallback(kernel_name, backend);
                };
                
                // PTX patching logic
                ptx_src = ptx_src.replace(".version 8.5", ".version 7.0")
                                  .replace(".version 8.4", ".version 7.0")
                                  .replace(".target sm_86", ".target sm_80");

                let ptx_path = format!("E:/Projects/Tracea/debug_dump_{}.ptx", kernel_name);
                let _ = std::fs::write(&ptx_path, &ptx_src);
                let cubin_path = format!("E:/Projects/Tracea/debug_dump_{}.cubin", kernel_name);
                
                let mut cmd = std::process::Command::new("ptxas");
                cmd.arg("-v").arg("--gpu-name").arg(target_sm).arg(&ptx_path).arg("-o").arg(&cubin_path);
                
                if let Ok(out) = cmd.output() {
                    if !out.status.success() {
                         return self.load_prebuilt_fallback(kernel_name, backend);
                    }
                }

                self.load_from_file_and_register(cuda_dev, &cubin_path, kernel_name)
            }
            DeviceBackend::Rocm => {
                // ROCm Compilation (similar to existing logic but generic)
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
                        safe: None,
                        raw: func as u64,
                        backend: DeviceBackend::Rocm,
                    });
                    Ok(id)
                }
            }
            _ => Err(format!("Unsupported backend {:?} for compilation", backend)),
        }
    }

    pub fn alloc(&self, size_bytes: usize, backend: DeviceBackend) -> Result<BufferId, String> {
        let id = self.generate_buffer_id()?;
        let buf = match backend {
            DeviceBackend::Cuda => {
                let devices = self.devices.lock().map_err(|_| "Lock")?;
                let dev = devices.get(&DeviceBackend::Cuda).ok_or("No CUDA")?.cuda_dev.as_ref().ok_or("No CUDA Dev")?;
                let slice = dev.alloc_zeros::<u8>(size_bytes).map_err(|e| format!("{:?}", e))?;
                DeviceBuffer::Cuda(slice)
            }
            DeviceBackend::Rocm => {
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("No ROCm API")?;
                let mut ptr: u64 = 0;
                unsafe {
                    let res = (api.hipMalloc)(&mut ptr, size_bytes);
                    if res != 0 { return Err(format!("hipMalloc failed: {}", res)); }
                }
                DeviceBuffer::Rocm(ptr)
            }
            _ => return Err("Unsupported backend".to_string()),
        };
        self.buffers.lock().map_err(|_| "Lock Poisoned")?.insert(id, buf);
        Ok(id)
    }

    fn load_prebuilt_fallback(&self, kernel_name: &str, backend: DeviceBackend) -> Result<KernelId, String> {
        let prebuilt_dir = std::env::var("TRACEA_PREBUILT_DIR").unwrap_or_else(|_| "E:/Projects/Tracea/prebuilt".to_string());
        let cubin_path = format!("{}/{}.cubin", prebuilt_dir, kernel_name);
        
        if std::path::Path::new(&cubin_path).exists() && backend == DeviceBackend::Cuda {
             let devices = self.devices.lock().map_err(|_| "Lock")?;
             let dev = devices.get(&DeviceBackend::Cuda).ok_or("No CUDA")?.cuda_dev.as_ref().ok_or("No CUDA Dev")?;
             self.load_from_file_and_register(dev, &cubin_path, kernel_name)
        } else {
             Err(format!("Fallback failed for {} on {:?}", kernel_name, backend))
        }
    }

    fn load_from_file_and_register(&self, device: &Arc<CudaDevice>, path: &str, kernel_name: &str) -> Result<KernelId, String> {
        let id = self.generate_kernel_id()?;
        let module_name = format!("tracea_mod_{}", id.0);
        let module_name_static = Box::leak(module_name.into_boxed_str());
        let kernel_name_static = Box::leak(kernel_name.to_string().into_boxed_str());

        device.load_ptx(Ptx::from_file(path), module_name_static, &[kernel_name_static])
             .map_err(|e| format!("Load Module Error: {:?}", e))?;

        let kernel = device.get_func(module_name_static, kernel_name_static)
             .ok_or_else(|| format!("Function {} not found", kernel_name))?;

        // Handle internal CUfunction for smem configuration
        let mut raw_handle: Option<cudarc::driver::sys::CUfunction> = None;
        unsafe {
             let base_ptr = &kernel as *const CudaFunction as *const *mut c_void;
             for i in 0..24 { 
                 let candidate = *base_ptr.add(i);
                 if candidate.is_null() { continue; }
                 let mut val: i32 = 0;
                 let res = cudarc::driver::sys::lib().cuFuncGetAttribute(&mut val, cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, candidate as _);
                 if res == cudarc::driver::sys::CUresult::CUDA_SUCCESS && val >= 32 && val <= 2048 {
                     raw_handle = Some(candidate as _);
                     break;
                 }
             }
        }

        if let Some(h) = raw_handle {
             unsafe {
                 cudarc::driver::sys::lib().cuFuncSetAttribute(h, cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 100 * 1024);
             }
             self.kernels.lock().map_err(|_| "Poisoned")?.insert(id, RecordedKernel { 
                 safe: Some(kernel), 
                 raw: h as u64,
                 backend: DeviceBackend::Cuda,
             });
             Ok(id)
        } else {
             Err("Failed to locate raw handle".to_string())
        }
    }
    
    fn get_device_ptr(&self, id: &BufferId) -> Result<u64, String> {
        let buffers = self.buffers.lock().map_err(|_| "Buffers Lock Poisoned")?;
        match buffers.get(id).ok_or_else(|| format!("Buffer {:?} not found", id))? {
            DeviceBuffer::Cuda(s) => Ok(*s.device_ptr()),
            DeviceBuffer::Rocm(p) => Ok(*p),
            DeviceBuffer::External(p) => Ok(*p),
        }
    }



    pub fn alloc_f32(&self, len: usize, backend: DeviceBackend) -> Result<BufferId, String> { self.alloc(len * 4, backend) }
    pub fn alloc_i32(&self, len: usize, backend: DeviceBackend) -> Result<BufferId, String> { self.alloc(len * 4, backend) }
    pub fn alloc_u16(&self, len: usize, backend: DeviceBackend) -> Result<BufferId, String> { self.alloc(len * 2, backend) }
    
    pub fn copy_h2d(&self, buf_id: BufferId, data: &[u8]) -> Result<(), String> {
        let ptr = self.get_device_ptr(&buf_id)?;
        if let Some(_) = crate::emitter::rocm_driver::RocmDriverApi::get() {
             let buffers = self.buffers.lock().unwrap();
             if let Some(DeviceBuffer::Rocm(_)) = buffers.get(&buf_id) {
                 let api = crate::emitter::rocm_driver::RocmDriverApi::get().unwrap();
                 unsafe { (api.hipMemcpyHtoD)(ptr, data.as_ptr() as *const _, data.len()); }
                 return Ok(());
             }
        }
        unsafe {
             let _ = cudarc::driver::sys::lib().cuMemcpyHtoD_v2(ptr, data.as_ptr() as *const _, data.len());
        }
        Ok(())
    }

    pub fn launch(&self, kernel_id: KernelId, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: Vec<KernelArg>) -> Result<(), String> {
        let kernels = self.kernels.lock().map_err(|_| "Kernels Lock Poisoned")?;
        let recorded = kernels.get(&kernel_id).ok_or("Kernel not found")?;
        let func_handle = recorded.raw;

        let mut arg_store_i32: Vec<i32> = Vec::with_capacity(args.len());
        let mut arg_store_u64: Vec<u64> = Vec::with_capacity(args.len());
        let mut kernel_params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(args.len());

        for arg in &args {
            match arg {
                KernelArg::Int(x) => {
                    arg_store_i32.push(*x);
                    kernel_params.push(arg_store_i32.last_mut().unwrap() as *mut i32 as *mut std::ffi::c_void);
                }
                KernelArg::Float(x) => {
                    arg_store_i32.push(x.to_bits() as i32);
                    kernel_params.push(arg_store_i32.last_mut().unwrap() as *mut i32 as *mut std::ffi::c_void);
                }
                KernelArg::Usize(x) => {
                    arg_store_u64.push(*x as u64);
                    kernel_params.push(arg_store_u64.last_mut().unwrap() as *mut u64 as *mut std::ffi::c_void);
                }
                KernelArg::Buffer(id) => {
                    let ptr = self.get_device_ptr(id)?;
                    arg_store_u64.push(ptr);
                    kernel_params.push(arg_store_u64.last_mut().unwrap() as *mut u64 as *mut std::ffi::c_void);
                }
            }
        }

        match recorded.backend {
            DeviceBackend::Cuda => {
                unsafe {
                    let res = cudarc::driver::sys::lib().cuLaunchKernel(
                        func_handle as _,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem, std::ptr::null_mut(),
                        kernel_params.as_mut_ptr(), std::ptr::null_mut()
                    );
                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("CUDA Launch Failed: {:?}", res));
                    }
                }
            }
            DeviceBackend::Rocm => {
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("ROCm API not found")?;
                unsafe {
                    let res = (api.hipModuleLaunchKernel)(
                        func_handle as _,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem, std::ptr::null_mut(),
                        kernel_params.as_mut_ptr(), std::ptr::null_mut()
                    );
                    if res != 0 { return Err(format!("ROCm Launch Failed: {}", res)); }
                }
            }
            _ => return Err("Unsupported backend for launch".to_string()),
        }
        Ok(())
    }

    pub fn get_device(&self, backend: DeviceBackend) -> Option<Arc<CudaDevice>> {
        let devices = self.devices.lock().ok()?;
        devices.get(&backend)?.cuda_dev.clone()
    }
    
    pub fn register_external_ptr(&self, ptr: u64) -> Result<BufferId, String> {
        let id = self.generate_buffer_id()?;
        self.buffers.lock().map_err(|_| "Buffers Lock Poisoned")?.insert(id, DeviceBuffer::External(ptr));
        Ok(id)
    }

    pub fn get_ptr(&self, buf_id: BufferId) -> Option<u64> {
         self.get_device_ptr(&buf_id).ok()
    }

    pub fn synchronize(&self, backend: DeviceBackend) -> Result<(), String> {
        match backend {
            DeviceBackend::Cuda => {
                let dev = self.get_device(DeviceBackend::Cuda).ok_or("No CUDA")?;
                dev.synchronize().map_err(|e| format!("{:?}", e))
            }
            DeviceBackend::Rocm => {
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("No ROCm")?;
                unsafe { (api.hipDeviceSynchronize)(); }
                Ok(())
            }
            _ => Err("Unsupported".to_string()),
        }
    }
}
