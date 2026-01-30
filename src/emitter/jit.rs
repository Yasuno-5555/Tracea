use cudarc::driver::safe::{CudaDevice, CudaFunction};
// use cudarc::driver::LaunchAsync; (not found or unused)
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ffi::c_void;


pub struct JITCompiler {
    pub device: Arc<CudaDevice>,
    pub emitter: crate::emitter::cuda::CUDAEmitter,
    kernel_cache: Mutex<HashMap<String, CudaFunction>>,
    module_counter: Mutex<u32>,
}

impl JITCompiler {
    pub fn new() -> Result<Self, String> {
        let device = CudaDevice::new(0)
            .map_err(|e| format!("CUDA Init Error: {:?}", e))?;
        
        Ok(Self {
            device,
            emitter: crate::emitter::cuda::CUDAEmitter::new(),
            kernel_cache: Mutex::new(HashMap::new()),
            module_counter: Mutex::new(0),
        })
    }

    fn get_compute_capability(device: &CudaDevice) -> Result<usize, String> {
        use cudarc::driver::sys::CUdevice_attribute_enum;
        let major = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
             .map_err(|e| format!("{:?}", e))?;
        let minor = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
             .map_err(|e| format!("{:?}", e))?;
        Ok((major as usize) * 10 + (minor as usize))
    }

    pub fn get_driver_version() -> Result<i32, String> {
        unsafe {
            let driver = crate::emitter::driver::get_driver_api()
                .map_err(|e| format!("Failed to get driver API: {}", e))?;
            let func_get_ver = &driver.cu_driver_get_version;
            let mut version: i32 = 0;
            let res = func_get_ver(&mut version);
            if res == 0 {
                Ok(version)
            } else {
                Err(format!("cuDriverGetVersion failed: {}", res))
            }
        }
    }

    pub fn compile_cuda(&self, source: &str, kernel_name: &str) -> Result<CudaFunction, String> {
        eprintln!("[Tracea Debug] ENTERING compile_cuda for {}", kernel_name);

        let cwd = std::env::current_dir().unwrap_or_default();
        eprintln!("[Tracea Debug] CWD: {:?}", cwd);

        let cache_key = format!("{}:{}", kernel_name, source.len());
            let cache = self.kernel_cache.lock().map_err(|_| "Cache Lock Poisoned".to_string())?;
            if let Some(kernel) = cache.get(&cache_key) {
                return Ok(kernel.clone());
            }
        
        let compute_cap = Self::get_compute_capability(&self.device).unwrap_or(75);
        let effective_cap = if compute_cap > 80 { 80 } else { compute_cap };
        let arch = format!("compute_{}", effective_cap);
        let arch_static = Box::leak(arch.into_boxed_str());
        
        let opts = CompileOptions {
            arch: Some(arch_static),
            ftz: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            fmad: Some(true),
            options: vec![
                "--include-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include".to_string()
            ],
            ..Default::default()
        };

        let ptx = compile_ptx_with_opts(source, opts)
            .map_err(|e| {
                let err_msg = format!("NVRTC Error: {:?}", e);
                let _ = std::fs::write("jit_error.log", &err_msg);
                eprintln!("[JIT] {}", err_msg);
                err_msg
            })?;
            
        let mut ptx_src = ptx.to_src();
        
        // Smart PTX Version Patching (Phase X: Compatibility Layer)
        let driver_ver = Self::get_driver_version().unwrap_or(0);
        eprintln!("[Doctor] Detected Driver Version: {}", driver_ver);
        
        // CUDA 12.4 -> 12040. If driver < 12.4, it won't support PTX 8.4.
        if driver_ver > 0 && driver_ver < 12040 {
             eprintln!("[Doctor] Driver is older than 12.4. Applying PTX downgrade patch to 7.0/sm_80.");
             let ptx_src_raw = ptx_src.clone();
             let ptx_src_patched = ptx_src_raw
                .replace(".version 8.4", ".version 7.0")
                .replace(".version 8.3", ".version 7.0")
                .replace(".version 8.2", ".version 7.0")
                .replace(".version 8.1", ".version 7.0")
                .replace(".version 8.0", ".version 7.0")
                .replace(".target sm_86", ".target sm_80"); // Force sm_80 for broad compatibility
             ptx_src = ptx_src_patched;
        } else {
             eprintln!("[Doctor] Driver {} is sufficient for native PTX.", driver_ver);
             // Still patch .target sm_86 to sm_80 if we capped effective_cap at 80?
             // effective_cap is set to 80 above. The generated PTX likely says .target sm_80 unless NVRTC defaults higher.
             // If NVRTC generates .target sm_86 but we asked for compute_80, it might be weird.
             // Let's safe-guard sm_86 -> sm_80 anyway if effective_cap is 80.
             if effective_cap <= 80 {
                 ptx_src = ptx_src.replace(".target sm_86", ".target sm_80");
             }
        }
        
        // Dump the final PTX to disk for debugging
        let dump_path = format!("E:/Projects/Tracea/debug_dump_{}.ptx", kernel_name);
        if let Ok(mut file) = std::fs::File::create(&dump_path) {
             let _ = std::io::Write::write_all(&mut file, ptx_src.as_bytes());
        }

        // let head = if ptx_src.len() > 500 { &ptx_src[..500] } else { &ptx_src };
        
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
        
        let mut counter = self.module_counter.lock().map_err(|_| "Counter Lock Poisoned".to_string())?;
        let module_name = format!("tracea_module_{}", *counter);
        *counter += 1;
        let module_name_static = Box::leak(module_name.into_boxed_str());
        
        let kernel_name_static = Box::leak(kernel_name.to_string().into_boxed_str());
        let function_names = vec![kernel_name_static as &str];
        
        self.device.load_ptx(ptx, module_name_static, &function_names)
            .map_err(|e| format!("Load PTX Error: {:?}", e))?;

        let kernel = self.device.get_func(module_name_static, kernel_name_static)
            .ok_or_else(|| format!("Function {} not found", kernel_name))?;

        // Query and log register pressure (Phase 5: Reliability)
        unsafe {
            if let Ok(driver) = crate::emitter::driver::get_driver_api() {
                let mut num_regs: i32 = 0;
                let base_ptr = &kernel as *const CudaFunction as *const *mut c_void;
                let mut h_ptr: *mut c_void = std::ptr::null_mut();
                for i in 0..10 {
                    let candidate = *base_ptr.add(i);
                    if !candidate.is_null() {
                        let mut val: i32 = 0;
                        if (driver.cu_func_get_attribute)(&mut val, 0, candidate) == 0 && val >= 32 && val <= 1024 {
                             h_ptr = candidate; break;
                        }
                    }
                }
                if !h_ptr.is_null() {
                    if (driver.cu_func_get_attribute)(&mut num_regs, 4, h_ptr) == 0 {
                        eprintln!("[Doctor] JIT Performance Audit: Kernel '{}' uses {} registers.", kernel_name, num_regs);
                    }
                }
            }
        }

        let mut cache = self.kernel_cache.lock().map_err(|_| "Cache Lock Poisoned".to_string())?;
        cache.insert(cache_key, kernel.clone());
        
        Ok(kernel)
    }

    pub fn compile_nvrtc(&self, source: &str, kernel_name: &str) -> Result<CudaFunction, String> {
        self.compile_cuda(source, kernel_name)
    }
    
    pub fn set_max_dynamic_shared_mem(&self, kernel: &CudaFunction, bytes: u32) -> Result<(), String> {
        unsafe {
            let driver = crate::emitter::driver::get_driver_api()
                .map_err(|e| format!("Failed to get driver API: {}", e))?;
            let func_set = &driver.cu_func_set_attribute;
            let func_get = &driver.cu_func_get_attribute;
            
            let base_ptr = kernel as *const CudaFunction as *const *mut c_void;
            let mut handle: Option<*mut c_void> = None;
            
            for i in 0..10 {
                let candidate = *base_ptr.add(i);
                if candidate.is_null() { continue; }
                
                let mut val: i32 = 0;
                let res = func_get(&mut val, 0, candidate);
                if res == 0 && val >= 32 && val <= 1024 {
                    let mut shmem: i32 = 0;
                    if func_get(&mut shmem, 1, candidate) == 0 && shmem >= 0 && shmem < 1000000 {
                        handle = Some(candidate);
                        break;
                    }
                }
            }
            
            if let Some(h) = handle {
                let result = func_set(h, 8, bytes as i32);
                if result != 0 {
                    return Err(format!("cuFuncSetAttribute returned {}", result));
                }
                Ok(())
            } else {
                Err("Could not locate CUfunction handle".into())
            }
        }
    }

    pub fn compile_cuda_with_opt(&self, source: &str, kernel_name: &str, _compute_cap: Option<u32>) -> Result<CudaFunction, String> {
         self.compile_cuda(source, kernel_name)
    }

    pub fn clear_cache(&self) {
        let mut cache = self.kernel_cache.lock().unwrap(); // Keep unwrap for clear methods for now or fix silently
        cache.clear();
    }

    pub fn compile_fused_attention(&self, op: &crate::core::op::FusedAttentionOp) -> Result<Arc<CudaFunction>, String> {
        // Use UnifiedOpIR and UniversalEmitter/CUDAEmitter logic
        use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::FusedAttention {
                b: op.b.as_static().unwrap_or(1), 
                s: op.s.as_static().unwrap_or(128), 
                d: op.d.as_static().unwrap_or(64), 
                h: op.h.as_static().unwrap_or(8), 
                dh: op.dh.as_static().unwrap_or(64), 
                causal: op.causal
            },
            precison: "f16".to_string(),
            tiling: crate::PipelineConfig::new(2, 64, 64, op.dh.as_static().unwrap_or(64)),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        
        let source = self.emitter.generate_from_ir(&ir);
        let key = format!("fused_attn_dh{}_c{}_s{}", op.dh.as_static().unwrap_or(64), op.causal, op.scale_inv_sqrt_d);
        
        let mut cache = self.kernel_cache.lock().map_err(|_| "Cache Lock Poisoned".to_string())?;
        if let Some(kernel) = cache.get(&key) {
            return Ok(Arc::new(kernel.clone()));
        }
        
        let kernel = self.compile_nvrtc(&source, "flash_attention_v2_kernel")
            .map_err(|e| format!("NVRTC Error: {}", e))?;
            
        cache.insert(key, kernel.clone());
        Ok(Arc::new(kernel))
    }

    pub fn load_static_raw(&self, ptx_source: &str, kernel_name: &str) -> Result<*mut c_void, String> {

        use std::ffi::CString;
        use std::ptr;
        use libloading::Symbol;

        #[allow(non_snake_case)]
        type CuModuleLoadData = unsafe extern "system" fn(*mut *mut c_void, *const c_void) -> i32;
        #[allow(non_snake_case)]
        type CuModuleGetFunction = unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;

        let ptx_c = CString::new(ptx_source).map_err(|_| "Invalid PTX source".to_string())?;
        let name_c = CString::new(kernel_name).map_err(|_| "Invalid kernel name".to_string())?;

        unsafe {
            use crate::emitter::driver::get_driver_api;
            let driver = get_driver_api().map_err(|e| format!("Driver API access failed: {}", e))?;
            let lib = driver.lib; 
            
            let load_data: Symbol<CuModuleLoadData> = lib.get(b"cuModuleLoadData").map_err(|e| format!("Sym load failed: {}", e))?;
            let get_func: Symbol<CuModuleGetFunction> = lib.get(b"cuModuleGetFunction").map_err(|e| format!("Sym load failed: {}", e))?;

            let mut module: *mut c_void = ptr::null_mut();
            let res = load_data(&mut module, ptx_c.as_ptr() as *const _);
            if res != 0 {
                return Err(format!("cuModuleLoadData failed: {}", res));
            }
            
            let mut func: *mut c_void = ptr::null_mut();
            let res = get_func(&mut func, module, name_c.as_ptr());
            if res != 0 {
                return Err(format!("cuModuleGetFunction failed: {}", res));
            }

            Ok(func)
        }
    }
}
