//! Device detection and type definitions for PyTorch integration

use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceKind {
    Cpu,
    Cuda(i32), // device index
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Other,
}

/// Detect device from PyTorch tensor
pub fn detect_device(t: &Bound<'_, PyAny>) -> PyResult<DeviceKind> {
    let device_str: String = t.getattr("device")?.call_method0("__str__")?.extract()?;
    parse_device(&device_str)
}

/// Parse device string like "cuda:0" or "cpu"
pub fn parse_device(s: &str) -> PyResult<DeviceKind> {
    if s == "cpu" {
        Ok(DeviceKind::Cpu)
    } else if s.starts_with("cuda") {
        let idx = s.strip_prefix("cuda:")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        Ok(DeviceKind::Cuda(idx))
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported device: {}", s)
        ))
    }
}

/// Detect dtype from PyTorch tensor
pub fn detect_dtype(t: &Bound<'_, PyAny>) -> PyResult<DType> {
    let dtype_str: String = t.getattr("dtype")?.call_method0("__str__")?.extract()?;
    Ok(match dtype_str.as_str() {
        "torch.float32" => DType::Float32,
        "torch.float16" => DType::Float16,
        "torch.bfloat16" => DType::BFloat16,
        _ => DType::Other,
    })
}

/// Check if tensor is contiguous
pub fn is_contiguous(t: &Bound<'_, PyAny>) -> PyResult<bool> {
    t.call_method0("is_contiguous")?.extract()
}
