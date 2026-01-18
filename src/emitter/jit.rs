use cudarc::driver::safe::{CudaDevice, CudaFunction};
use cudarc::driver::LaunchAsync;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
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

    pub fn compile_cuda(&self, source: &str, kernel_name: &str) -> Result<CudaFunction, String> {
        eprintln!("[Tracea Debug] ENTERING compile_cuda for {}", kernel_name);
        let cwd = std::env::current_dir().unwrap_or_default();
        eprintln!("[Tracea Debug] CWD: {:?}", cwd);

        let cache_key = format!("{}:{}", kernel_name, source.len());
            let cache = self.kernel_cache.lock().unwrap();
            if let Some(kernel) = cache.get(&cache_key) {
                return Ok(kernel.clone());
            }
        
        
        let compute_cap = Self::get_compute_capability(&self.device).unwrap_or(75);
        // Cap at 80 because some NVRTC versions in the path might not support 86
        // But we need at least 80 for cp.async
        let effective_cap = if compute_cap > 80 { 80 } else { compute_cap };
        let arch = format!("compute_{}", effective_cap);
        let arch_static = Box::leak(arch.into_boxed_str());
        
        let opts = CompileOptions {
            arch: Some(arch_static),
            ftz: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            fmad: Some(true),
            ..Default::default()
        };

        let ptx = compile_ptx_with_opts(source, opts)
            .map_err(|e| format!("NVRTC Error: {:?}", e))?;
            
        let mut ptx_src = ptx.to_src();
        let mut ptx_src = ptx.to_src();
        let mut ptx_src = ptx.to_src();
        
        // Robust PTX Version Patching
        let mut lines: Vec<&str> = ptx_src.lines().collect();
        let mut new_lines = Vec::new();
        
        eprintln!("[Tracea Debug] PTX Dump Head:");
        for (i, line) in lines.iter().take(5).enumerate() {
            eprintln!("  L{}: {}", i, line);
        }

        for line in lines {
            if line.trim().starts_with(".version") {
                eprintln!("[Tracea Debug] Replacing Version: {} -> .version 7.0", line);
                new_lines.push(".version 7.0");
            } else if line.trim().starts_with(".target") {
                 eprintln!("[Tracea Debug] Replacing Target: {} -> .target sm_80", line);
                 new_lines.push(".target sm_80");
            } else {
                new_lines.push(line);
            }
        }
        ptx_src = new_lines.join("\n");
        eprintln!("[Tracea Debug] PTX Patching Done.");
        
        // Dump the final PTX to disk for debugging
        let dump_path = format!("E:/Projects/Tracea/debug_dump_{}.ptx", kernel_name);
        if let Ok(mut file) = std::fs::File::create(&dump_path) {
             let _ = std::io::Write::write_all(&mut file, ptx_src.as_bytes());
        }

        let head = if ptx_src.len() > 500 { &ptx_src[..500] } else { &ptx_src };
        panic!("DEBUG EXFILTRATION: {}", head);
        
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
        
        let mut counter = self.module_counter.lock().unwrap();
        let module_name = format!("tracea_module_{}", *counter);
        *counter += 1;
        let module_name_static = Box::leak(module_name.into_boxed_str());
        
        let kernel_name_static = Box::leak(kernel_name.to_string().into_boxed_str());
        let function_names = vec![kernel_name_static as &str];
        
        self.device.load_ptx(ptx, module_name_static, &function_names)
            .map_err(|e| format!("Load PTX Error: {:?}", e))?;

        let kernel = self.device.get_func(module_name_static, kernel_name_static)
            .ok_or_else(|| format!("Function {} not found", kernel_name))?;

        let mut cache = self.kernel_cache.lock().unwrap();
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
        let mut cache = self.kernel_cache.lock().unwrap();
        cache.clear();
    }

    pub fn compile_fused_attention(&self, op: &crate::core::op::FusedAttentionOp) -> Result<Arc<CudaFunction>, String> {
        // Keeping Arc for Attention for now as it's used elsewhere as Arc
        let source = self.emitter.generate_fused_attention(op.clone());
        let key = format!("fused_attn_dh{}_c{}_s{}", op.dh, op.causal, op.scale_inv_sqrt_d);
        
        let mut cache = self.kernel_cache.lock().unwrap();
        if let Some(kernel) = cache.get(&key) {
            return Ok(Arc::new(kernel.clone()));
        }
        
        
        let kernel = self.compile_nvrtc(&source, "fused_attention_kernel")
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
