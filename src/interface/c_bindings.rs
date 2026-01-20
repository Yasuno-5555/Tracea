use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;
use crate::runtime::manager::{RuntimeManager, DeviceBackend};
use crate::core::tuning::{tune_kernel, SearchMode, TunableKernel};
use crate::kernels::gemm::cpu_adapter::{GemmAdapter, GemmProblem};
use crate::kernels::attention::cuda_adapter::{Fa2Adapter, Fa2Problem};
use crate::optimizer::benchmark::{Conv2dProblem, NVRTCConvBenchmark};
use crate::optimizer::{AutoTuner, GPUInfo, OptimizationGoal};
use crate::backend::cpu::CpuBackend;

#[repr(C)]
pub struct TraceaResult {
    pub success: bool,
    pub error_msg: *mut c_char,
    pub score: f32, // For tuning: performance score
    
    // Config outputs (generic union would be better, but for C-API simplicity let's use void ptr or specific structs)
    pub config_ptr: *mut c_void, 
}

// Global Runtime Instance Helper
static mut GLOBAL_RUNTIME: Option<Arc<RuntimeManager>> = None;

#[no_mangle]
pub extern "C" fn tracea_init() -> c_int {
    // Initialize Runtime
    match RuntimeManager::init(Some(DeviceBackend::Cuda)) {
        Ok(rt) => {
            unsafe { GLOBAL_RUNTIME = Some(rt); }
            0
        }
        Err(_) => 1
    }
}

#[no_mangle]
pub extern "C" fn tracea_shutdown() {
    unsafe { GLOBAL_RUNTIME = None; }
}

#[no_mangle]
pub extern "C" fn tracea_tune_gemm(m: usize, n: usize, k: usize) -> TraceaResult {
    let backend = CpuBackend::new();
    let problem = GemmProblem { m, n, k };
    let adapter = GemmAdapter::new(backend, problem);

    // Run Tuning
    let best_config = tune_kernel(&adapter, SearchMode::GridSearch);
    
    // Evaluate Score (Re-run benchmark on best)
    let score = adapter.benchmark(&best_config).unwrap_or(0.0);

    // Serialize Config to JSON string for passing back? Or opaque struct?
    // For simplicity, let's return a serialized JSON string pointer.
    let json = serde_json::to_string(&best_config).unwrap_or_default();
    let c_str = CString::new(json).unwrap();
    
    TraceaResult {
        success: true,
        error_msg: std::ptr::null_mut(),
        score,
        config_ptr: c_str.into_raw() as *mut c_void, // Caller must free this char*!
    }
}

#[no_mangle]
pub extern "C" fn tracea_tune_fa2(b: usize, h: usize, s: usize, d: usize, causal: bool) -> TraceaResult {
    let rt = unsafe { 
        match &GLOBAL_RUNTIME {
            Some(r) => r.clone(),
            None => return TraceaResult {
                success: false,
                error_msg: CString::new("Runtime not initialized").unwrap().into_raw(),
                score: 0.0,
                config_ptr: std::ptr::null_mut(),
            }
        }
    };

    let problem = Fa2Problem { b, h, s, d, is_causal: causal };
    let adapter = Fa2Adapter::new(rt, problem);

    let best_config = tune_kernel(&adapter, SearchMode::GridSearch);
    let score = adapter.benchmark(&best_config).unwrap_or(0.0);

    let json = serde_json::to_string(&best_config).unwrap_or_default();
    let c_str = CString::new(json).unwrap();

    TraceaResult {
        success: true,
        error_msg: std::ptr::null_mut(),
        score,
        config_ptr: c_str.into_raw() as *mut c_void,
    }
}


#[no_mangle]
pub extern "C" fn tracea_tune_conv2d(
    n: usize, c: usize, h: usize, w: usize, k: usize, 
    r: usize, s: usize, stride: usize, pad: usize, dilation: usize
) -> TraceaResult {
    let rt = unsafe { 
        match &GLOBAL_RUNTIME {
            Some(r) => r.clone(),
            None => return TraceaResult {
                success: false,
                error_msg: CString::new("Runtime not initialized").unwrap().into_raw(),
                score: 0.0,
                config_ptr: std::ptr::null_mut(),
            }
        }
    };

    let problem = Conv2dProblem::new("CustomConv", n, h, w, c, k, r, s, stride, pad, dilation);
    
    let benchmark = NVRTCConvBenchmark::new(rt.clone(), problem); // Clone rt

    // Create temporary Tuner
    let gpu = GPUInfo {
        name: "Generic GPU".to_string(), 
        backend: DeviceBackend::Cuda,
        shared_memory_per_block: 102400,
        max_registers_per_thread: 255,
        max_warps_per_sm: 32,
        wavefront_size: 32,
        max_blocks_per_sm: 16,
        shared_memory_per_sm: 102400,
        has_specialized_units: true,
    };
    let mut tuner = AutoTuner::new(gpu);
    tuner.runtime = Some(rt); // Set runtime

    let goal = OptimizationGoal::MaximizeTFLOPS;
    let config = tuner.optimize_conv(&benchmark, 20, goal);

    let json = serde_json::to_string(&config).unwrap_or_default();
    let c_str = CString::new(json).unwrap();
    let score = 0.0; 

    TraceaResult {
        success: true,
        error_msg: std::ptr::null_mut(),
        score,
        config_ptr: c_str.into_raw() as *mut c_void,
    }
}

#[no_mangle]
pub extern "C" fn tracea_free_string(s: *mut c_char) {
    if s.is_null() { return; }
    unsafe {
        let _ = CString::from_raw(s);
    }
}
