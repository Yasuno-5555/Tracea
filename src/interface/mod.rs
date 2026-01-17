pub mod python;
pub mod cpp;
pub use self::python::{PyPipelineConfig, PyContext, PyProfilingScope, PyEpilogueOp, PyEpilogueType, matmul, execute_fused};

#[derive(Debug, Clone)]
pub struct TensorView {
    pub shape: Vec<u32>,
    pub data_ptr: *mut f32,
}
