use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use cudarc::driver::*;
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::ffi::c_void;

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

// Wrapper to hold both the Safe CudaFunction (keeps module alive) and the Raw Handle
pub struct RecordedKernel {
    pub safe: CudaFunction,
    pub raw: cudarc::driver::sys::CUfunction,
}

unsafe impl Send for RecordedKernel {}
unsafe impl Sync for RecordedKernel {}

pub struct RuntimeManager {
    device: Arc<CudaDevice>,
    arch: String,
    
    // Resource registries
    kernels: Mutex<HashMap<KernelId, RecordedKernel>>,
    buffers_i32: Mutex<HashMap<BufferId, CudaSlice<i32>>>,
    buffers_f32: Mutex<HashMap<BufferId, CudaSlice<f32>>>,
    buffers_u16: Mutex<HashMap<BufferId, CudaSlice<u16>>>,
    external_buffers: Mutex<HashMap<BufferId, u64>>,
    
    // Simple ID generators
    next_kernel_id: Mutex<u64>,
    next_buffer_id: Mutex<u64>,
}

static INSTANCE: Mutex<Option<Arc<RuntimeManager>>> = Mutex::new(None);

impl RuntimeManager {
    pub fn init(arch: &str) -> Result<Arc<Self>, String> {
        let mut cache = INSTANCE.lock().unwrap();
        if let Some(instance) = &*cache {
            return Ok(Arc::clone(instance));
        }

        let device = CudaDevice::new(0).map_err(|e| format!("CUDA Init Error: {:?}", e))?;
        let instance = Arc::new(Self {
            device,
            arch: arch.to_string(),
            kernels: Mutex::new(HashMap::new()),
            buffers_i32: Mutex::new(HashMap::new()),
            buffers_f32: Mutex::new(HashMap::new()),
            buffers_u16: Mutex::new(HashMap::new()),
            external_buffers: Mutex::new(HashMap::new()),
            next_kernel_id: Mutex::new(0),
            next_buffer_id: Mutex::new(0),
        });
        
        *cache = Some(Arc::clone(&instance));
        Ok(instance)
    }
    
    fn generate_kernel_id(&self) -> KernelId {
        let mut lock = self.next_kernel_id.lock().unwrap();
        let id = *lock;
        *lock += 1;
        KernelId(id)
    }

    fn generate_buffer_id(&self) -> BufferId {
        let mut lock = self.next_buffer_id.lock().unwrap();
        let id = *lock;
        *lock += 1;
        BufferId(id)
    }

    pub fn compile(&self, source: &str, kernel_name: &str) -> Result<KernelId, String> {
        println!("[RuntimeManager] Compiling {} (len={})...", kernel_name, source.len());
        let arch_static: &'static str = "compute_86"; 
        let opts = CompileOptions {
            ftz: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            fmad: Some(true),
            arch: Some(arch_static), 
            include_paths: vec!["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\include".into()],
            ..Default::default()
        };
        
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, opts)
            .map_err(|e| format!("NVRTC Error: {:?}", e))?;
            
        let ptx_src = ptx.to_src();
        
        let ptx_path = format!("E:/Projects/Tracea/debug_dump_{}.ptx", kernel_name);
        std::fs::write(&ptx_path, &ptx_src).map_err(|e| format!("Write PTX error: {:?}", e))?;

        // Offline compilation using ptxas to avoid JIT issues
        let cubin_path = format!("E:/Projects/Tracea/debug_dump_{}.cubin", kernel_name);
        let mut cmd = std::process::Command::new("ptxas");
        cmd.arg("-v")
           .arg("--gpu-name").arg("sm_86") // Match RTX 3070
           .arg(&ptx_path)
           .arg("-o").arg(&cubin_path);
        
        println!("[RuntimeManager] Running ptxas for {} (target sm_86)...", kernel_name);
        let output = cmd.output().map_err(|e| format!("Failed to run ptxas: {:?}", e))?;
        if !output.status.success() {
             let err_msg = String::from_utf8_lossy(&output.stderr);
             println!("[RuntimeManager] ptxas FAILED: {}", err_msg);
             return Err(format!("ptxas failed: {}", err_msg));
        } else {
             println!("[RuntimeManager] ptxas success: {}", String::from_utf8_lossy(&output.stderr));
        }

        let id = self.generate_kernel_id();
        let module_name = format!("tracea_mod_{}", id.0);
        let module_name_static = Box::leak(module_name.into_boxed_str());
        let kernel_name_static = Box::leak(kernel_name.to_string().into_boxed_str());
 
        println!("[RuntimeManager] Loading CUBIN from {}...", cubin_path);
        // Direct CUBIN loading to bypass JIT entirely
        self.device.load_ptx(Ptx::from_file(cubin_path), module_name_static, &[kernel_name_static])
             .map_err(|e| {
                 let msg = format!("Load CUBIN Error: {:?}", e);
                 println!("[RuntimeManager] {}", msg);
                 msg
             })?;

        println!("[RuntimeManager] Module loaded. Finding function {}...", kernel_name);
        let kernel = self.device.get_func(module_name_static, kernel_name_static)
             .ok_or_else(|| format!("Function {} not found", kernel_name))?;

        // Resolve Raw Handle using Hack
        let mut raw_handle: Option<cudarc::driver::sys::CUfunction> = None;
        unsafe {
             let base_ptr = &kernel as *const CudaFunction as *const *mut c_void;
             for i in 0..16 { // Scan wide
                 let candidate = *base_ptr.add(i);
                 if candidate.is_null() { continue; }

                 let mut val: i32 = 0;
                 let res = cudarc::driver::sys::lib().cuFuncGetAttribute(
                     &mut val, 
                     cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 
                     candidate as cudarc::driver::sys::CUfunction
                 );
                 
                 // If getting threads per block works and is reasonable, it's likely the handle
                 if res == cudarc::driver::sys::CUresult::CUDA_SUCCESS && val >= 32 && val <= 2048 {
                     raw_handle = Some(candidate as cudarc::driver::sys::CUfunction);
                     break;
                 }
             }
        }
        
        // Configure Large Shared Memory if handle found
        if let Some(h) = raw_handle {
             unsafe {
                 let max_smem = 100 * 1024; // 100KB
                 let res = cudarc::driver::sys::lib().cuFuncSetAttribute(
                     h, 
                     cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 
                     max_smem
                 );
                  if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS && max_smem > 48 * 1024 {
                       println!("[RuntimeManager] Warning: Failed to set max smem: {:?}", res);
                  }
             }
        } else {
             println!("[RuntimeManager] Warning: Could not locate Raw CUfunction handle! Launch might fail if Smem > 48KB.");
        }
        
        // If we couldn't find handle, we can store dummy or rely on safe wrapper (implied risk)
        // But for our generic launch we NEED raw handle if we use cuLaunchKernel.
        // If not found, we can't use our launch.
        let h = raw_handle.ok_or("Failed to locate raw kernel handle (internal error)")?;

        self.kernels.lock().unwrap().insert(id, RecordedKernel { safe: kernel, raw: h });
        Ok(id)
    }
    
    pub fn alloc_f32(&self, len: usize) -> Result<BufferId, String> {
        let slice = self.device.alloc_zeros::<f32>(len).map_err(|e| format!("{:?}", e))?;
        let id = self.generate_buffer_id();
        self.buffers_f32.lock().unwrap().insert(id, slice);
        Ok(id)
    }

    pub fn alloc_i32(&self, len: usize) -> Result<BufferId, String> {
        let slice = self.device.alloc_zeros::<i32>(len).map_err(|e| format!("{:?}", e))?;
        let id = self.generate_buffer_id();
        self.buffers_i32.lock().unwrap().insert(id, slice);
        Ok(id)
    }

    pub fn alloc_u16(&self, len: usize) -> Result<BufferId, String> {
        let slice = self.device.alloc_zeros::<u16>(len).map_err(|e| format!("{:?}", e))?;
        let id = self.generate_buffer_id();
        self.buffers_u16.lock().unwrap().insert(id, slice);
        Ok(id)
    }
    
    pub fn copy_h2d_i32(&self, buf_id: BufferId, data: &[i32]) -> Result<(), String> {
        let mut map = self.buffers_i32.lock().unwrap();
        let slice = map.get_mut(&buf_id).ok_or("Buffer not found")?;
        unsafe {
             let res = cudarc::driver::sys::lib().cuMemcpyHtoD_v2(*slice.device_ptr(), data.as_ptr() as *const _, data.len() * 4);
             if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                 return Err(format!("Copy H2D Failed: {:?}", res));
             }
        }
        Ok(())
    }

    pub fn copy_raw_to_buffer_i32(&self, buf_id: BufferId, src_ptr: u64, len: usize) -> Result<(), String> {
        let mut map = self.buffers_i32.lock().unwrap();
        let slice = map.get_mut(&buf_id).ok_or("Buffer not found")?;
        
        unsafe {
             let res = cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*slice.device_ptr(), src_ptr, len * 4);
             if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                 return Err(format!("Copy Failed (Raw): {:?}", res));
             }
        }
        Ok(())
    }

    pub fn copy_raw_to_buffer_f32(&self, buf_id: BufferId, src_ptr: u64, len: usize) -> Result<(), String> {
        let mut map = self.buffers_f32.lock().unwrap();
        let slice = map.get_mut(&buf_id).ok_or("Buffer not found")?;
        unsafe {
             let res = cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*slice.device_ptr(), src_ptr, len * 4);
             if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                 return Err(format!("Copy Failed (Raw): {:?}", res));
             }
        }
        Ok(())
    }

    pub fn copy_raw_to_buffer_u16(&self, buf_id: BufferId, src_ptr: u64, len: usize) -> Result<(), String> {
        let mut map = self.buffers_u16.lock().unwrap();
        let slice = map.get_mut(&buf_id).ok_or("Buffer not found")?;
        unsafe {
             let res = cudarc::driver::sys::lib().cuMemcpyDtoD_v2(*slice.device_ptr(), src_ptr, len * 2);
             if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                 return Err(format!("Copy Failed (Raw): {:?}", res));
             }
        }
        Ok(())
    }
    
    // Manual Launch without cudarc wrapper logic
    pub fn launch(&self, kernel_id: KernelId, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: Vec<KernelArg>) -> Result<(), String> {
        println!("[RuntimeManager] Launching Kernel {:?} Grid={:?} Block={:?} Smem={}", kernel_id, grid, block, smem);
        
        let kernels = self.kernels.lock().unwrap();
        let recorded = kernels.get(&kernel_id).ok_or("Kernel not found")?;
        let func_handle = recorded.raw;

        let mut raw_args: Vec<u64> = Vec::new(); 
        
        let f32_map = self.buffers_f32.lock().unwrap();
        let i32_map = self.buffers_i32.lock().unwrap();
        let u16_map = self.buffers_u16.lock().unwrap();
        let ext_map = self.external_buffers.lock().unwrap();

        for arg in &args {
            let val = match arg {
                KernelArg::Int(x) => *x as u64, 
                KernelArg::Usize(x) => *x as u64,
                KernelArg::Float(x) => x.to_bits() as u64, 
                KernelArg::Buffer(id) => {
                    if let Some(s) = f32_map.get(id) { *s.device_ptr() }
                    else if let Some(s) = i32_map.get(id) { *s.device_ptr() }
                    else if let Some(s) = u16_map.get(id) { *s.device_ptr() }
                    else if let Some(ptr) = ext_map.get(id) { *ptr }
                    else { return Err(format!("Buffer {:?} not found in any registry", id)); }
                }
            };
            raw_args.push(val);
        }
        
        // Prepare generic kernel arguments for cuLaunchKernel
        // We only support 32-bit integers, floats, and 64-bit pointers (passed as u64)
        // GEMM Signature: (u64, u64, u64, i32, i32, i32)
        // We must construct *mut c_void array pointing to these values.
        
        let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(args.len());
        
        // We need to keep values alive while taking pointers
        // Since `raw_args` holds u64 representation, we need to be careful about 32-bit types.
        // If the kernel expects `int` (32-bit), we must pass pointer to `i32`.
        // `raw_args` stores everything as `u64`.
        // We need to re-parse from `KernelArg` to get correct size pointers.
        
        // Second pass to stable storage vectors
        let mut arg_store_i32: Vec<i32> = Vec::with_capacity(args.len());
        let mut arg_store_u64: Vec<u64> = Vec::with_capacity(args.len());
        
        for arg in &args {
            match arg {
                KernelArg::Int(x) => arg_store_i32.push(*x),
                KernelArg::Float(x) => arg_store_i32.push(x.to_bits() as i32),
                KernelArg::Usize(x) => arg_store_u64.push(*x as u64),
                KernelArg::Buffer(id) => {
                     let ptr = if let Some(s) = f32_map.get(id) { *s.device_ptr() }
                              else if let Some(s) = i32_map.get(id) { *s.device_ptr() }
                              else if let Some(s) = u16_map.get(id) { *s.device_ptr() }
                              else if let Some(ptr) = ext_map.get(id) { *ptr }
                              else { return Err(format!("Buffer {:?} not found", id)); };
                     arg_store_u64.push(ptr);
                }
            }
        }
        
        let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(args.len());
        let mut i32_ptr = 0;
        let mut u64_ptr = 0;
        
        for arg in &args {
            match arg {
                KernelArg::Int(_) | KernelArg::Float(_) => {
                    kernel_params.push(&mut arg_store_i32[i32_ptr] as *mut i32 as *mut c_void);
                    i32_ptr += 1;
                },
                KernelArg::Usize(_) | KernelArg::Buffer(_) => {
                    kernel_params.push(&mut arg_store_u64[u64_ptr] as *mut u64 as *mut c_void);
                    u64_ptr += 1;
                }
            }
        }
        
        println!("[RuntimeManager] Invoking cuLaunchKernel with {} args...", kernel_params.len());
        
        unsafe {
             let res = cudarc::driver::sys::lib().cuLaunchKernel(
                 func_handle,
                 grid.0, grid.1, grid.2,
                 block.0, block.1, block.2,
                 smem,
                 std::ptr::null_mut(), // stream
                 kernel_params.as_mut_ptr(),
                 std::ptr::null_mut() // formatting extensions
             );
             
             if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                 return Err(format!("cuLaunchKernel Failed: {:?}", res));
             }
        }
        
        println!("[RuntimeManager] Launch Success.");
        Ok(())
    }
    
    pub fn get_device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }
    
    pub fn register_external_ptr(&self, ptr: u64) -> BufferId {
        let id = self.generate_buffer_id();
        self.external_buffers.lock().unwrap().insert(id, ptr);
        id
    }

    pub fn get_ptr(&self, buf_id: BufferId) -> Option<u64> {
         let f32_map = self.buffers_f32.lock().unwrap();
         if let Some(s) = f32_map.get(&buf_id) { return Some(*s.device_ptr()); }
         
         let i32_map = self.buffers_i32.lock().unwrap();
         if let Some(s) = i32_map.get(&buf_id) { return Some(*s.device_ptr()); }
         
         let u16_map = self.buffers_u16.lock().unwrap();
         if let Some(s) = u16_map.get(&buf_id) { return Some(*s.device_ptr()); }

         let ext_map = self.external_buffers.lock().unwrap();
         if let Some(ptr) = ext_map.get(&buf_id) { return Some(*ptr); }
         
         None
    }
}
