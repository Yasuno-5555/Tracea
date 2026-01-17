#![allow(unused_unsafe)]
#![allow(unsafe_op_in_unsafe_fn)]
pub mod interface;
pub mod core;
pub mod semantic;
pub mod optimized;
pub mod emitter;
pub mod optimizer;

pub use crate::core::op::GemmOp;
pub use crate::core::config::PipelineConfig;
pub use crate::semantic::transition::{Phase, PhaseTransition, SyncRequirement};
pub use crate::semantic::mapping::{LaneMapping, MatrixLayout};
pub use crate::semantic::fragment::{Fragment, FragmentType, FragmentOp, FragmentRole};
pub use crate::optimizer::{AutoTuner, GPUInfo};
pub use crate::emitter::CUDAEmitter;

use pyo3::prelude::*;
use crate::interface::*;

#[pymodule]
fn tracea(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPipelineConfig>()?;
    m.add_class::<PyContext>()?;
    m.add_class::<PyDeviceBufferF32>()?;
    m.add_class::<PyDeviceBufferU16>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyProfilingScope>()?;
    m.add_class::<PyEpilogueOp>()?;
    m.add_class::<PyEpilogueType>()?;
    m.add_class::<PyOptimizationGoal>()?;
    m.add_function(wrap_pyfunction!(python_relu, m)?)?;
    m.add_function(wrap_pyfunction!(python_gelu, m)?)?;
    m.add_function(wrap_pyfunction!(python_bias_add, m)?)?;
    Ok(())
}
