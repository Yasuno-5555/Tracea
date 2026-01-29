use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::sync::Arc;
use crate::emitter::jit::JITCompiler;
use crate::optimizer::{AutoTuner, HardwareProfile};

pub struct CppContext {
    pub jit: Arc<JITCompiler>,
    pub tuner: AutoTuner,
    pub device: Arc<cudarc::driver::CudaDevice>,
}

// Opaque struct meant to be used as a pointer
pub struct TraceaContextOpaque;
pub type TraceaContextHandle = *mut TraceaContextOpaque;

#[no_mangle]
pub extern "C" fn tracea_create_context(device_name: *const c_char) -> TraceaContextHandle {
    let c_str = unsafe {
        if device_name.is_null() {
            return std::ptr::null_mut();
        }
        CStr::from_ptr(device_name)
    };

    let device_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let bit_arch = if device_str.contains("A100") {
        HardwareProfile::a100()
    } else {
        HardwareProfile::rtx3070()
    };

    let jit = match JITCompiler::new() {
        Ok(j) => j,
        Err(_) => return std::ptr::null_mut(),
    };
    let jit_arc = Arc::new(jit);
    let device = jit_arc.device.clone();

    let ctx = CppContext {
        jit: jit_arc,
        tuner: AutoTuner::new(bit_arch),
        device,
    };

    Box::into_raw(Box::new(ctx)) as TraceaContextHandle
}

#[no_mangle]
pub extern "C" fn tracea_destroy_context(ctx: TraceaContextHandle) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx as *mut CppContext);
        }
    }
}

use crate::emitter::driver::{get_driver_api, CuLaunchKernel, SyncPtr};

static mut KERNEL_CACHE: SyncPtr<std::ffi::c_void> = SyncPtr(std::ptr::null_mut());
static mut LAUNCH_FN: SyncPtr<std::ffi::c_void> = SyncPtr(std::ptr::null_mut());

#[no_mangle]
pub extern "C" fn tracea_compile_empty(ctx_handle: TraceaContextHandle) -> c_int {
    let result = std::panic::catch_unwind(|| {
        if ctx_handle.is_null() { return -1; }
        let ctx = unsafe { &*(ctx_handle as *mut CppContext) };
        
        // Ensure driver loaded AND cache the function pointer
        let driver = match get_driver_api() {
             Ok(d) => d,
             Err(e) => {
                 println!("Driver Init Error: {}", e);
                 return -4;
             }
        };
        
        // Cache API pointer for hot path
        unsafe {
            // Transmute Symbol to raw pointer
            let fn_ptr: CuLaunchKernel = *driver.launch_kernel; 
            LAUNCH_FN = SyncPtr(fn_ptr as *mut std::ffi::c_void);
        }

        // Trivial empty kernel (with dummy arg to satisfy LaunchAsync trait bounds)
        let empty_ptx = r#"
.version 7.0
.target sm_75
.address_size 64
.visible .entry empty_kernel(.param .u32 dummy) { ret; }
"#;

        match ctx.jit.load_static_raw(empty_ptx, "empty_kernel") {
            Ok(func) => {
                unsafe { KERNEL_CACHE = SyncPtr(func); }
                0
            },
            Err(e) => {
                println!("Compile Error: {}", e);
                -2
            },
        }
    });

    match result {
        Ok(code) => code,
        Err(e) => {
            println!("Rust Panic: {:?}", e);
            -99
        }
    }
}

#[no_mangle]
pub extern "C" fn tracea_launch_empty(ctx_handle: TraceaContextHandle) -> c_int {
    if ctx_handle.is_null() { return -1; }
    
    // Hot Path: No allocations, no dynamic loads, just pointer reads.
    unsafe {
        if KERNEL_CACHE.0.is_null() { return -2; }
        if LAUNCH_FN.0.is_null() { return -5; }
        
        let launch_kernel: CuLaunchKernel = std::mem::transmute(LAUNCH_FN.0);

        let mut dummy_arg: i32 = 0;
        let mut args: [*mut std::ffi::c_void; 1] = [&mut dummy_arg as *mut _ as *mut std::ffi::c_void];

        let res = launch_kernel(
            KERNEL_CACHE.0,
            1, 1, 1, // grid dim
            32, 1, 1, // block dim
            0, // shared mem
            std::ptr::null_mut(), // stream 0
            args.as_mut_ptr(),
            std::ptr::null_mut() // extra
        );
        
        if res != 0 {
            return -3;
        }
        0
    }
}



