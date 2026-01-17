use cudarc::driver::safe::{CudaDevice, CudaFunction};
use cudarc::driver::LaunchAsync;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ffi::c_void;

// Function signatures for CUDA Driver API
type CuFuncSetAttribute = unsafe extern "system" fn(*mut c_void, i32, i32) -> i32;
type CuFuncGetAttribute = unsafe extern "system" fn(*mut i32, i32, *mut c_void) -> i32;

pub struct JITCompiler {
    pub device: Arc<CudaDevice>,
    kernel_cache: Mutex<HashMap<String, CudaFunction>>,
    module_counter: Mutex<u32>,
}

impl JITCompiler {
    pub fn new() -> Result<Self, String> {
        let device = CudaDevice::new(0)
            .map_err(|e| format!("CUDA Init Error: {:?}", e))?;
        
        // println!("Tracea JIT: Device Compute Capability: {}", Self::get_compute_capability(&device).unwrap_or(0));
        
        Ok(Self {
            device,
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
        let cache_key = format!("{}:{}", kernel_name, source.len());
        {
            let cache = self.kernel_cache.lock().unwrap();
            if let Some(kernel) = cache.get(&cache_key) {
                return Ok(kernel.clone());
            }
        }
        
        println!("Tracea JIT: Compiling {}...", kernel_name);
        
        let compute_cap = Self::get_compute_capability(&self.device).unwrap_or(75);
        let arch = format!("compute_{}", compute_cap);
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
        if ptx_src.contains(".version 9.") {
             ptx_src = ptx_src.replace(".version 9.1", ".version 7.8");
             ptx_src = ptx_src.replace(".version 9.0", ".version 7.8");
        }
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
    
    pub fn set_max_dynamic_shared_mem(&self, kernel: &CudaFunction, bytes: u32) -> Result<(), String> {
        unsafe {
            let lib = libloading::Library::new("nvcuda.dll")
                .map_err(|e| format!("Failed to load nvcuda.dll: {}", e))?;
                
            let func_set: libloading::Symbol<CuFuncSetAttribute> = lib.get(b"cuFuncSetAttribute")
                .map_err(|e| format!("Failed to load cuFuncSetAttribute: {}", e))?;
                
            let func_get: libloading::Symbol<CuFuncGetAttribute> = lib.get(b"cuFuncGetAttribute")
                .map_err(|e| format!("Failed to load cuFuncGetAttribute: {}", e))?;
            
            let base_ptr = kernel as *const CudaFunction as *const *mut c_void;
            let mut handle: Option<*mut c_void> = None;
            
            for i in 0..2 {
                let candidate = *base_ptr.add(i);
                if candidate.is_null() { continue; }
                
                let mut val: i32 = 0;
                let res = func_get(&mut val, 0, candidate);
                if res == 0 && val > 0 && val <= 1024 {
                    handle = Some(candidate);
                    break;
                }
            }
            
            if let Some(h) = handle {
                let result = func_set(h, 8, bytes as i32);
                if result != 0 {
                    return Err(format!("cuFuncSetAttribute returned {}", result));
                }
                
                let mut check_val: i32 = 0;
                let res = func_get(&mut check_val, 8, h);
                if res == 0 {
                    println!("Tracea JIT: Verified MaxDynamicSharedMemory is now {} bytes", check_val);
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



    pub fn load_static_raw(&self, ptx_source: &str, kernel_name: &str) -> Result<*mut c_void, String> {
        // Reuse singleton if possible, but load_static_raw uses cuModuleLoadData which we might also want to cache.
        // For now, let's just cache the launch kernel as that's the hot path. 
        // Note: We could cache module loading too, but that happens once per compilation/warmup.
        // The LAUNCH is what happens 10000 times.
        
        println!("Tracea JIT: Static Load Bypass invoked for {}", kernel_name);

        use std::ffi::CString;
        use std::ptr;
        use libloading::{Library, Symbol};

        // Define function signatures for Windows (stdcall/system)
        #[allow(non_snake_case)]
        type CuModuleLoadData = unsafe extern "system" fn(*mut *mut c_void, *const c_void) -> i32;
        #[allow(non_snake_case)]
        type CuModuleGetFunction = unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;

        let ptx_c = CString::new(ptx_source).map_err(|_| "Invalid PTX source".to_string())?;
        let name_c = CString::new(kernel_name).map_err(|_| "Invalid kernel name".to_string())?;

        unsafe {
            use crate::emitter::driver::get_driver_api;
            let driver = get_driver_api().map_err(|e| format!("Driver API access failed: {}", e))?;
            let lib = driver.lib; // Re-use the static library handle
            
            let load_data: Symbol<CuModuleLoadData> = lib.get(b"cuModuleLoadData").map_err(|e| format!("Sym load failed: {}", e))?;
            let get_func: Symbol<CuModuleGetFunction> = lib.get(b"cuModuleGetFunction").map_err(|e| format!("Sym load failed: {}", e))?;

            let mut module: *mut c_void = ptr::null_mut();
            let res = load_data(&mut module, ptx_c.as_ptr() as *const _);
            if res != 0 {
                return Err(format!("cuModuleLoadData failed: {}", res));
            }
            
            // Leak module 
            let mut func: *mut c_void = ptr::null_mut();
            let res = get_func(&mut func, module, name_c.as_ptr());
            if res != 0 {
                return Err(format!("cuModuleGetFunction failed: {}", res));
            }

            Ok(func)
        }
    }
}

// Global Driver Singleton moved to driver.rs


