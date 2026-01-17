pub mod python;
// pub mod cpp;
pub use self::python::{
    PyPipelineConfig, PyContext, PyProfilingScope, PyEpilogueOp, PyEpilogueType, PyOptimizationGoal, PyGraph,
    PyDeviceBufferF32, PyDeviceBufferU16,
    python_relu, python_gelu, python_bias_add
};


#[derive(Debug, Clone)]
pub struct TensorView {
    pub shape: Vec<u32>,
    pub data_ptr: *mut f32,
}
