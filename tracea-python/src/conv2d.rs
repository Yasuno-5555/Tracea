//! Conv2d implementation with fallback logic
//! 
//! Phase 7.0 Constraints:
//! - Supported: groups=1, dilation=(1,1), simple padding
//! - Fallback: groups!=1, dilation!=(1,1), exotic padding, non-FP32, non-contiguous

use pyo3::prelude::*;
use crate::device::DeviceKind;
use crate::tensor_view::{TensorView, allocate_output};

/// Main conv2d entry point
/// 
/// Automatically falls back to PyTorch for unsupported configurations
#[pyfunction]
#[pyo3(signature = (x, w, b=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1))]
pub fn conv2d(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    b: Option<&Bound<'_, PyAny>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> PyResult<PyObject> {
    // Phase 7.0: Unsupported conditions → immediate fallback
    if groups != 1 || dilation != (1, 1) {
        return fallback_conv2d(py, x, w, b, stride, padding, dilation, groups);
    }
    
    // Try to create TensorViews (returns None if guards fail)
    let x_view = match TensorView::from_torch(py, x)? {
        Some(v) => v,
        None => return fallback_conv2d(py, x, w, b, stride, padding, dilation, groups),
    };
    
    let w_view = match TensorView::from_torch(py, w)? {
        Some(v) => v,
        None => return fallback_conv2d(py, x, w, b, stride, padding, dilation, groups),
    };
    
    // Dispatch based on device
    match x_view.device {
        DeviceKind::Cuda(_) => conv2d_cuda(py, x, w, &x_view, &w_view, b, stride, padding),
        DeviceKind::Cpu => conv2d_cpu(py, x, w, &x_view, &w_view, b, stride, padding),
    }
}

/// CUDA conv2d implementation
fn conv2d_cuda(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    x_view: &TensorView,
    w_view: &TensorView,
    b: Option<&Bound<'_, PyAny>>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> PyResult<PyObject> {
    // Calculate output shape: [N, C_out, H_out, W_out]
    let n = x_view.shape[0];
    let c_out = w_view.shape[0];
    let h_in = x_view.shape[2];
    let w_in = x_view.shape[3];
    let kh = w_view.shape[2];
    let kw = w_view.shape[3];
    
    let h_out = (h_in + 2 * padding.0 - kh) / stride.0 + 1;
    let w_out = (w_in + 2 * padding.1 - kw) / stride.1 + 1;
    
    let out_shape = [n, c_out, h_out, w_out];
    let _out = allocate_output(py, x, &out_shape)?;
    
    // TODO: Call Tracea CUDA kernel
    // For now, fallback to PyTorch until kernel is ready
    fallback_conv2d(py, x, w, b, stride, padding, (1, 1), 1)
}

/// CPU conv2d implementation
fn conv2d_cpu(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    x_view: &TensorView,
    w_view: &TensorView,
    b: Option<&Bound<'_, PyAny>>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> PyResult<PyObject> {
    // Calculate output shape
    let n = x_view.shape[0];
    let c_out = w_view.shape[0];
    let h_in = x_view.shape[2];
    let w_in = x_view.shape[3];
    let kh = w_view.shape[2];
    let kw = w_view.shape[3];
    
    let h_out = (h_in + 2 * padding.0 - kh) / stride.0 + 1;
    let w_out = (w_in + 2 * padding.1 - kw) / stride.1 + 1;
    
    let _out_shape = [n, c_out, h_out, w_out];
    
    // TODO: Call Tracea CPU kernel
    // For now, fallback to PyTorch until kernel is ready
    fallback_conv2d(py, x, w, b, stride, padding, (1, 1), 1)
}

/// Fallback to torch.nn.functional.conv2d
fn fallback_conv2d(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    b: Option<&Bound<'_, PyAny>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> PyResult<PyObject> {
    let torch = py.import_bound("torch")?;
    let f = torch.getattr("nn")?.getattr("functional")?;
    
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("stride", stride)?;
    kwargs.set_item("padding", padding)?;
    kwargs.set_item("dilation", dilation)?;
    kwargs.set_item("groups", groups)?;
    
    let none = py.None();
    let bias: &Bound<'_, PyAny> = match b {
        Some(bias) => bias,
        None => none.bind(py),
    };
    
    let result = f.call_method("conv2d", (x, w, bias), Some(&kwargs))?;
    Ok(result.into())
}
