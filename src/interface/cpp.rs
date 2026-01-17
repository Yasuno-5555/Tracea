use std::ffi::{c_char, CStr};
use crate::semantic::fusion::EpilogueOp;

/// Opaque pointer for the Tracea Context/Tuner
pub struct TraceaContext {
    pub tuner: crate::optimizer::AutoTuner,
}

#[unsafe(no_mangle)]
pub extern "C" fn tracea_context_create(device_name: *const c_char) -> *mut TraceaContext {
    let c_str = unsafe { CStr::from_ptr(device_name) };
    let device = c_str.to_str().unwrap_or("A100").to_string();
    let gpu = crate::optimizer::GPUInfo::a100(); // Simplified for FFI
    let tuner = crate::optimizer::AutoTuner::new(gpu);
    Box::into_raw(Box::new(TraceaContext { tuner }))
}

#[unsafe(no_mangle)]
pub extern "C" fn tracea_context_destroy(ctx: *mut TraceaContext) {
    if !ctx.is_null() {
        unsafe { drop(Box::from_raw(ctx)) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tracea_execute_fused(
    _a_ptr: *const f32,
    _b_ptr: *const f32,
    _c_ptr: *mut f32,
    m: u32, n: u32, k: u32,
    epilogue_ops: *const u32, // Array of op types (0: None, 1: ReLU, 2: Gelu, 3: BiasAdd)
    num_ops: u32
) {
    let mut internal_ops = Vec::new();
    let ops_slice = unsafe { std::slice::from_raw_parts(epilogue_ops, num_ops as usize) };
    
    for &op_code in ops_slice {
        match op_code {
            1 => internal_ops.push(EpilogueOp::ReLU),
            2 => internal_ops.push(EpilogueOp::Gelu),
            3 => internal_ops.push(EpilogueOp::BiasAdd { bias_ptr: 0 }), // Simplified for demo
            _ => {}
        }
    }
    
    println!("Tracea C++ Bridge: Executing {}x{}x{} GEMM with {} fused ops", m, n, k, internal_ops.len());
}
