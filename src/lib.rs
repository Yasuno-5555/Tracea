#![allow(unused_unsafe)]
#![allow(unsafe_op_in_unsafe_fn)]

//! # Tracea: Universal GPU Kernel Optimization Framework 🏛️
//!
//! Tracea is a research-grade compiler and runtime for generating mathematically verified, 
//! high-performance CUDA kernels. It bridges the gap between high-level graph definitions
//! and low-level hardware instructions (PTX/SASS).
//!
//! ## Core Modules
//!
//! - **[`core`]**: Defines the High-Level IR (Graph, Operations) and Semantic IR (Tiling, Swizzle).
//! - **[`optimizer`]**: Bayesian Auto-Tuner that searches for optimal implementation configurations.
//! - **[`emitter`]**: Generates optimized CUDA C++ / PTX code.
//! - **[`interface`]**: Python and C++ bindings (FFI).
//!
//! ## Features
//!
//! - `python`: Enables `pyo3` bindings for the Python Interface.
//! - `cpp`: Enables C-ABI export functions for C++ integration.

pub mod interface;
// pub mod bindings; // REMOVED
pub mod kernels;
pub mod core;
pub(crate) mod semantic;
pub(crate) mod optimized;
pub mod emitter;
pub mod runtime;
pub mod backend;
pub mod optimizer;
pub mod doctor;

pub use crate::core::op::GemmOp;
pub use crate::core::config::{PipelineConfig, SwizzleMode, SoftmaxGranularity, BarrierMode};
pub use crate::semantic::transition::{Phase, PhaseTransition, SyncRequirement};
pub use crate::semantic::mapping::{LaneMapping, MatrixLayout};
pub use crate::semantic::fragment::{Fragment, FragmentType, FragmentOp, FragmentRole};
pub use crate::optimizer::{AutoTuner, HardwareProfile};
pub use crate::optimizer::benchmark::{MicroBenchmark, SimulatedBenchmark};
pub use crate::emitter::cuda::CUDAEmitter;
pub use half;


#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn tracea(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<interface::python::PyContext>()?;
    m.add_class::<interface::python::PyPipelineConfig>()?;
    m.add_class::<interface::python::PyProfilingScope>()?;
    m.add_class::<interface::python::PyGraph>()?;
    m.add_class::<interface::python::PyDeviceBufferF32>()?;
    m.add_class::<interface::python::PyDeviceBufferU16>()?;
    m.add_class::<interface::python::PyDeviceBufferI32>()?; // Int4 packed
    
    // Enum exports
    m.add_class::<interface::python::PyEpilogueType>()?;
    m.add_class::<interface::python::PyOptimizationGoal>()?;
    m.add_class::<interface::python::PyDecision>()?;
    m.add_class::<interface::python::PyDoctor>()?;
    m.add_class::<interface::python::PyEnvironmentReport>()?;
    m.add_class::<interface::python::PyDoctorErrorReport>()?;
    m.add_class::<interface::python::PyDoctorArtifacts>()?;
    m.add_class::<interface::python::PyTuner>()?;

    // Register NN module
    // Register NN module
    interface::nn::register_nn_module(_py, m)?;



    Ok(())
}
