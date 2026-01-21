//! Tracea FFI - C ABI for C++ Integration
//!
//! Design principles:
//! - All extern "C" functions use catch_unwind
//! - No panic across FFI boundary
//! - Memory ownership stays with caller

use std::cell::RefCell;
use std::ffi::c_void;
use std::panic;
use std::ptr;

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

// ============================================================================
// C ABI Types
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceaStatus {
    Success = 0,
    InvalidParams = 1,
    UnsupportedConfig = 2,
    CudaError = 3,
    CpuError = 4,
    UnknownError = 99,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceaDType {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int8 = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TraceaTensorView {
    /// Raw pointer (CPU or GPU)
    pub ptr: *mut c_void,
    /// Number of dimensions
    pub rank: u32,
    /// Shape array (caller-owned, length = rank)
    pub shape: *const u64,
    /// Stride array (caller-owned, length = rank)
    pub stride: *const i64,
    /// Data type
    pub dtype: TraceaDType,
    /// Device ID (-1 = CPU, 0+ = CUDA device)
    pub device_id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TraceaConv2dParams {
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding_h: u32,
    pub padding_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub groups: u32,
    /// cudaStream_t (pass-through, can be null for default stream)
    pub stream: *mut c_void,
}

// ============================================================================
// Error Handling
// ============================================================================

fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg;
    });
}

/// Get last error message
/// 
/// # Safety
/// `buf` must be a valid pointer to a buffer of at least `len` bytes
#[no_mangle]
pub unsafe extern "C" fn tracea_get_last_error(buf: *mut u8, len: usize) -> i32 {
    if buf.is_null() || len == 0 {
        return -1;
    }
    
    LAST_ERROR.with(|e| {
        let msg = e.borrow();
        let bytes = msg.as_bytes();
        let copy_len = bytes.len().min(len - 1);
        ptr::copy_nonoverlapping(bytes.as_ptr(), buf, copy_len);
        *buf.add(copy_len) = 0; // null terminate
        copy_len as i32
    })
}

/// Clear last error
#[no_mangle]
pub extern "C" fn tracea_clear_error() {
    LAST_ERROR.with(|e| {
        e.borrow_mut().clear();
    });
}

// ============================================================================
// Conv2d FFI
// ============================================================================

/// 2D Convolution
/// 
/// # Safety
/// - All pointers must be valid
/// - `b` can be null (no bias)
/// - `out` must point to pre-allocated output tensor
#[no_mangle]
pub unsafe extern "C" fn tracea_conv2d(
    x: TraceaTensorView,
    w: TraceaTensorView,
    b: *const TraceaTensorView,
    out: *mut TraceaTensorView,
    params: TraceaConv2dParams,
) -> TraceaStatus {
    // Catch all panics at FFI boundary
    let result = panic::catch_unwind(|| {
        conv2d_impl(x, w, b, out, params)
    });
    
    match result {
        Ok(Ok(())) => TraceaStatus::Success,
        Ok(Err(status)) => status,
        Err(_) => {
            set_last_error("Panic in tracea_conv2d".to_string());
            TraceaStatus::UnknownError
        }
    }
}

unsafe fn conv2d_impl(
    x: TraceaTensorView,
    w: TraceaTensorView,
    b: *const TraceaTensorView,
    out: *mut TraceaTensorView,
    params: TraceaConv2dParams,
) -> Result<(), TraceaStatus> {
    // Validate inputs
    if x.ptr.is_null() || w.ptr.is_null() || out.is_null() {
        set_last_error("Null pointer in input tensors".to_string());
        return Err(TraceaStatus::InvalidParams);
    }
    
    // Phase 7.1: Only support common configurations
    if params.groups != 1 {
        set_last_error(format!("Unsupported groups: {}", params.groups));
        return Err(TraceaStatus::UnsupportedConfig);
    }
    
    if params.dilation_h != 1 || params.dilation_w != 1 {
        set_last_error("Dilation != 1 not yet supported".to_string());
        return Err(TraceaStatus::UnsupportedConfig);
    }
    
    // FP32 only for Phase 7.1
    if x.dtype != TraceaDType::Float32 || w.dtype != TraceaDType::Float32 {
        set_last_error("Only float32 supported in Phase 7.1".to_string());
        return Err(TraceaStatus::UnsupportedConfig);
    }
    
    // Extract shapes
    let x_shape = std::slice::from_raw_parts(x.shape, x.rank as usize);
    let w_shape = std::slice::from_raw_parts(w.shape, w.rank as usize);
    
    if x.rank != 4 || w.rank != 4 {
        set_last_error("Expected 4D tensors (NCHW)".to_string());
        return Err(TraceaStatus::InvalidParams);
    }
    
    let _n = x_shape[0];
    let _c_in = x_shape[1];
    let h_in = x_shape[2];
    let w_in = x_shape[3];
    
    let c_out = w_shape[0];
    let _c_in_w = w_shape[1];
    let kh = w_shape[2];
    let kw = w_shape[3];
    
    // Calculate output shape
    let h_out = (h_in + 2 * params.padding_h as u64 - kh) / params.stride_h as u64 + 1;
    let w_out = (w_in + 2 * params.padding_w as u64 - kw) / params.stride_w as u64 + 1;
    
    // Dispatch based on device
    if x.device_id >= 0 {
        // CUDA path
        conv2d_cuda(&x, &w, b, out, &params, c_out, h_out, w_out)?;
    } else {
        // CPU path
        conv2d_cpu(&x, &w, b, out, &params, c_out, h_out, w_out)?;
    }
    
    Ok(())
}

unsafe fn conv2d_cuda(
    _x: &TraceaTensorView,
    _w: &TraceaTensorView,
    _b: *const TraceaTensorView,
    _out: *mut TraceaTensorView,
    _params: &TraceaConv2dParams,
    _c_out: u64,
    _h_out: u64,
    _w_out: u64,
) -> Result<(), TraceaStatus> {
    // TODO: Call Tracea CUDA kernel
    // let stream = params.stream as cudaStream_t;
    set_last_error("CUDA conv2d not yet implemented".to_string());
    Err(TraceaStatus::CudaError)
}

unsafe fn conv2d_cpu(
    _x: &TraceaTensorView,
    _w: &TraceaTensorView,
    _b: *const TraceaTensorView,
    _out: *mut TraceaTensorView,
    _params: &TraceaConv2dParams,
    _c_out: u64,
    _h_out: u64,
    _w_out: u64,
) -> Result<(), TraceaStatus> {
    // TODO: Call Tracea CPU kernel
    set_last_error("CPU conv2d not yet implemented".to_string());
    Err(TraceaStatus::CpuError)
}

// ============================================================================
// Version Info
// ============================================================================

#[no_mangle]
pub extern "C" fn tracea_version_major() -> u32 { 0 }

#[no_mangle]
pub extern "C" fn tracea_version_minor() -> u32 { 1 }

#[no_mangle]
pub extern "C" fn tracea_version_patch() -> u32 { 0 }
