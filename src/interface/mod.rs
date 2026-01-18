#[cfg(feature = "python")]
pub mod python;
#[cfg(feature = "cpp")]
pub mod cpp;
#[cfg(feature = "python")]
pub mod nn;
#[cfg(feature = "python")]
pub use self::python::{
    PyPipelineConfig, PyContext, PyProfilingScope, PyEpilogueOp, PyEpilogueType, PyOptimizationGoal, PyGraph,
    PyDeviceBufferF32, PyDeviceBufferU16, PyDeviceBufferI32, PyDecision,
    python_relu, python_gelu, python_bias_add
};


#[derive(Debug, Clone)]
pub struct TensorView {
    pub shape: Vec<u32>,
    pub data_ptr: *mut f32,
}
