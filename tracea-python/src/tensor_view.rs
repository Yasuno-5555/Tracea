//! TensorView - Zero-copy view into PyTorch tensors
//! 
//! SAFETY RULES:
//! - Never take memory ownership (no Vec::from_raw_parts, Box::from_raw)
//! - Never deallocate tensor memory
//! - Always keep _keepalive to prevent GC

use pyo3::prelude::*;
use crate::device::{DeviceKind, DType, detect_device, detect_dtype, is_contiguous};

/// Zero-copy view into a PyTorch tensor
pub struct TensorView {
    pub ptr: *mut u8,
    pub shape: Vec<usize>,
    pub stride: Vec<isize>,
    pub dtype: DType,
    pub device: DeviceKind,
    /// Keep Python tensor alive to prevent GC
    pub _keepalive: PyObject,
}

impl TensorView {
    /// Create a TensorView from a PyTorch tensor
    /// 
    /// Phase 7.0 guards:
    /// - Returns None if non-contiguous
    /// - Returns None if dtype != float32
    pub fn from_torch(py: Python<'_>, t: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        // Guard 1: Contiguity check
        if !is_contiguous(t)? {
            return Ok(None); // Signal fallback needed
        }
        
        // Guard 2: DType check (Phase 7.0 = FP32 only)
        let dtype = detect_dtype(t)?;
        if dtype != DType::Float32 {
            return Ok(None); // Signal fallback needed
        }
        
        let keepalive = t.clone().unbind();
        let data_ptr: u64 = t.call_method0("data_ptr")?.extract()?;
        let shape: Vec<usize> = t.getattr("shape")?.extract()?;
        let stride: Vec<isize> = t.getattr("stride")?.extract()?;
        let device = detect_device(t)?;
        
        Ok(Some(TensorView {
            ptr: data_ptr as *mut u8,
            shape,
            stride,
            dtype,
            device,
            _keepalive: keepalive,
        }))
    }
}

/// Allocate output tensor via PyTorch's caching allocator
/// 
/// This ensures memory is managed by PyTorch, not Rust
pub fn allocate_output(py: Python<'_>, x: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<PyObject> {
    let py_shape = pyo3::types::PyTuple::new_bound(py, shape);
    let out = x.call_method1("new_empty", (py_shape,))?;
    Ok(out.into())
}
