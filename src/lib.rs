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
pub use crate::optimizer::{AutoTuner, GPUInfo};
pub use crate::emitter::{CUDAEmitter, HIPEmitter, SYCLEmitter};

use pyo3::prelude::*;
use crate::interface::*;

#[pymodule]
fn tracea(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPipelineConfig>()?;
    m.add_class::<PyContext>()?;
    m.add_class::<PyProfilingScope>()?;
    m.add_class::<PyEpilogueOp>()?;
    m.add_class::<PyEpilogueType>()?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(execute_fused, m)?)?;
    Ok(())
}
