use pyo3::prelude::*;
use crate::core::config::PipelineConfig;
use crate::optimizer::{AutoTuner, GPUInfo};
use pyo3::{PyRef, PyRefMut};

#[pyclass]
#[derive(Clone)]
pub struct PyPipelineConfig {
    pub inner: PipelineConfig,
}

#[pymethods]
impl PyPipelineConfig {
    #[new]
    pub fn new(num_stages: u32, m_tile: u32, n_tile: u32, k_tile: u32) -> Self {
        Self {
            inner: PipelineConfig::new(num_stages, m_tile, n_tile, k_tile),
        }
    }

    #[getter]
    pub fn num_stages(&self) -> u32 { self.inner.num_stages }

    pub fn add_relu(&mut self) {
        self.inner.epilogue.push(crate::semantic::fusion::EpilogueOp::ReLU);
    }

    pub fn add_bias(&mut self, ptr: usize) {
        self.inner.epilogue.push(crate::semantic::fusion::EpilogueOp::BiasAdd { bias_ptr: ptr });
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyContext {
    pub tuner: AutoTuner,
}

#[pymethods]
impl PyContext {
    #[new]
    pub fn new(device_name: String) -> Self {
        let bit_arch = if device_name.contains("A100") {
            GPUInfo::a100()
        } else {
            GPUInfo::a100()
        };
        Self {
            tuner: AutoTuner::new(bit_arch),
        }
    }

    pub fn matmul(
        &self,
        _a_ptr: usize, _b_ptr: usize, _c_ptr: usize,
        m: u32, n: u32, k: u32,
        epilogue: PyEpilogueOp // Builder based
    ) -> PyResult<()> {
        println!("Tracea Context: Executing {}x{}x{} on optimized device", m, n, k);
        Ok(())
    }

    pub fn profiling(&self) -> PyResult<PyProfilingScope> {
        Ok(PyProfilingScope {})
    }
}

#[pyclass]
pub struct PyProfilingScope {}

#[pymethods]
impl PyProfilingScope {
    fn __enter__(&self) {}
    fn __exit__(&self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {}
}

#[pyclass]
#[derive(Clone)]
pub enum PyEpilogueType {
    ReLU,
    Gelu,
    BiasAdd,
}

#[pyclass]
#[derive(Clone)]
pub struct PyEpilogueOp {
    pub ops: Vec<(PyEpilogueType, Option<usize>)>,
}

#[pymethods]
impl PyEpilogueOp {
    #[new]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn relu(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.ops.push((PyEpilogueType::ReLU, None));
        Ok(slf)
    }

    pub fn gelu(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.ops.push((PyEpilogueType::Gelu, None));
        Ok(slf)
    }

    pub fn bias_add(mut slf: PyRefMut<'_, Self>, ptr: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.ops.push((PyEpilogueType::BiasAdd, Some(ptr)));
        Ok(slf)
    }

    /// Operator overloading for >>
    fn __rshift__(slf: PyRef<'_, Self>, other: PyRef<'_, Self>) -> PyResult<Self> {
        let mut new_ops = slf.ops.clone();
        new_ops.extend(other.ops.clone());
        Ok(Self { ops: new_ops })
    }

    #[staticmethod]
    pub fn empty() -> Self { Self::new() }
}

#[pyfunction]
pub fn execute_fused(
    _a_ptr: usize, _b_ptr: usize, _c_ptr: usize,
    m: u32, n: u32, k: u32,
    epilogue: PyEpilogueOp, // Now takes the builder object
    device: String
) -> PyResult<()> {
    // Convert PyEpilogueOp to internal EpilogueOp
    let mut internal_ops = Vec::new();
    for (op_type, bias_ptr) in epilogue.ops {
        match op_type {
            PyEpilogueType::ReLU => internal_ops.push(crate::semantic::fusion::EpilogueOp::ReLU),
            PyEpilogueType::Gelu => internal_ops.push(crate::semantic::fusion::EpilogueOp::Gelu),
            PyEpilogueType::BiasAdd => {
                if let Some(ptr) = bias_ptr {
                    internal_ops.push(crate::semantic::fusion::EpilogueOp::BiasAdd { bias_ptr: ptr });
                }
            }
        }
    }
    
    println!("Tracea JIT: Executing Fused GEMM ({} ops) on {}", internal_ops.len(), device);
    Ok(())
}

#[pyfunction]
pub fn matmul(
    _a_ptr: usize, 
    _b_ptr: usize, 
    _c_ptr: usize, 
    m: u32, n: u32, k: u32,
    device: String
) -> PyResult<()> {
    // 1. Check Cache
    // 2. If miss, Tune
    // 3. Generate Code
    // 4. JIT Compile
    // 5. Launch Kernel
    
    println!("Tracea JIT: Executing {}x{}x{} GEMM on {}", m, n, k, device);
    Ok(())
}
