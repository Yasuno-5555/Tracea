use pyo3::prelude::*;
use crate::core::config::PipelineConfig;
use crate::optimizer::{AutoTuner, GPUInfo, OptimizationGoal};
use crate::emitter::CUDAEmitter;
use crate::emitter::jit::JITCompiler;
use pyo3::{PyRef, PyRefMut};

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use cudarc::driver::safe::DevicePtr;
use std::sync::Arc;
use pyo3::types::PyLong;

#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferF32 {
    pub inner: Arc<CudaSlice<f32>>,
}

#[pymethods]
impl PyDeviceBufferF32 {
    #[staticmethod]
    pub fn unsafe_from_ptr(ptr: usize, len: usize, device: &PyContext) -> Self {
        unsafe {
            let slice = device.device.upgrade_device_ptr::<f32>(ptr as u64, len);
             Self { inner: Arc::new(slice) }
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferU16 {
    pub inner: Arc<CudaSlice<u16>>,
}

#[pymethods]
impl PyDeviceBufferU16 {
   #[staticmethod]
    pub fn unsafe_from_ptr(ptr: usize, len: usize, device: &PyContext) -> Self {
        unsafe {
            let slice = device.device.upgrade_device_ptr::<u16>(ptr as u64, len);
             Self { inner: Arc::new(slice) }
        }
    }
}

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

#[pyclass(name = "Graph")]
pub struct PyGraph {
    pub inner: crate::core::graph::Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    pub fn new() -> Self {
        Self { inner: crate::core::graph::Graph::new() }
    }

    pub fn add_gemm(&mut self, m: u32, n: u32, k: u32) -> usize {
        self.inner.add_gemm(m, n, k, Vec::new())
    }
}

#[pyclass(name = "Context")]
#[derive(Clone)]
pub struct PyContext {
    pub tuner: AutoTuner,
    pub device: Arc<CudaDevice>,
    pub jit: Arc<JITCompiler>,
    pub scratch_a: PyDeviceBufferF32,
    pub scratch_b: PyDeviceBufferF32,
    pub scratch_c: PyDeviceBufferF32,
    pub scratch_a_h: PyDeviceBufferU16,
    pub scratch_b_h: PyDeviceBufferU16,
    pub best_config: Arc<std::sync::Mutex<PipelineConfig>>,
}

#[pyclass(name = "OptimizationGoal")]
#[derive(Clone, Copy)]
pub enum PyOptimizationGoal {
    MaximizeTFLOPS,
    MinimizeLatency,
}

#[pymethods]
impl PyContext {
    #[new]
    pub fn new(device_name: String) -> PyResult<Self> {
        let bit_arch = if device_name.contains("A100") {
            GPUInfo::a100()
        } else {
            GPUInfo::rtx3070()
        };

        // Initialize JIT (owns Device)
        let jit = JITCompiler::new().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("JIT Init Error: {}", e)))?;
        let jit_arc = Arc::new(jit);
        let device = jit_arc.device.clone();
        
        // Allocate Scratchpad Memory (256MB each for A, B, C - up to 8k x 8k float32)
        let size = 8192 * 8192; 
        let scratch_a = device.alloc_zeros::<f32>(size).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Alloc Error: {:?}", e)))?;
        let scratch_b = device.alloc_zeros::<f32>(size).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Alloc Error: {:?}", e)))?;
        let scratch_c = device.alloc_zeros::<f32>(size).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Alloc Error: {:?}", e)))?;

        let scratch_a_h = device.alloc_zeros::<u16>(size).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Alloc Error: {:?}", e)))?;
        let scratch_b_h = device.alloc_zeros::<u16>(size).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Alloc Error: {:?}", e)))?;
        
        // Initial Default Config
        let mut initial_config = PipelineConfig::new(2, 128, 128, 32);
        initial_config.use_tensor_cores = true;

        Ok(Self {
            tuner: AutoTuner::new(bit_arch),
            device,
            jit: jit_arc,
            scratch_a: PyDeviceBufferF32 { inner: Arc::new(scratch_a) },
            scratch_b: PyDeviceBufferF32 { inner: Arc::new(scratch_b) },
            scratch_c: PyDeviceBufferF32 { inner: Arc::new(scratch_c) },
            scratch_a_h: PyDeviceBufferU16 { inner: Arc::new(scratch_a_h) },
            scratch_b_h: PyDeviceBufferU16 { inner: Arc::new(scratch_b_h) },
            best_config: Arc::new(std::sync::Mutex::new(initial_config)),
        })
    }


    pub fn matmul(
        slf: &Bound<'_, Self>,
        a: &Bound<'_, PyAny>, // Accepts either F32 or U16 buffer
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyDeviceBufferF32>,
        m: u32, n: u32, k: u32,
        epilogue: PyEpilogueOp
    ) -> PyResult<()> {
        let mut config = {
            let ctx = slf.borrow();
            let cfg = ctx.best_config.lock().unwrap().clone();
            cfg
        };
        
        // Convert Epilogue
        let mut rust_epilogue = Vec::new();
        for (op_type, ptr_opt) in epilogue.ops {
            let op = match op_type {
                PyEpilogueType::ReLU => crate::semantic::fusion::EpilogueOp::ReLU,
                PyEpilogueType::Gelu => crate::semantic::fusion::EpilogueOp::Gelu,
                PyEpilogueType::BiasAdd => {
                    if let Some(ptr) = ptr_opt {
                        crate::semantic::fusion::EpilogueOp::BiasAdd { bias_ptr: ptr }
                    } else {
                         return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("BiasAdd requires pointer"));
                    }
                }
            };
            rust_epilogue.push(op);
        }
        config.epilogue = rust_epilogue;

        // 2. Emit Source
        let emitter = CUDAEmitter::new();
        let source = emitter.generate_pipelined_gemm(config.clone());
        let kernel_name = "gemm_pipelined_kernel";

        // 3. JIT Compile
        let ctx = slf.borrow();
        let kernel = ctx.jit.compile_cuda(&source, kernel_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("JIT Compilation Failed: {}", e)))?;
            
        // 3.5 Set Dynamic SMEM Limit
        ctx.jit.set_max_dynamic_shared_mem(&kernel, 99000)
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set Shared Mem Limit: {}", e)))?;

        // 4. Launch Kernel (Adaptive Tiled Pipelined)
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        let n_stages = config.num_stages;
        
        let grid_dim = ((m + mt - 1) / mt, (n + nt - 1) / nt, 1);
        let block_dim = (128, 1, 1); // 4 warps
        
        let smem_size = if config.use_tensor_cores {
            (n_stages * mt * kt + n_stages * kt * nt) * 2 // __half is 2 bytes
        } else {
            let s_a = kt + 4;
            let s_b = nt + 4;
            (n_stages * mt * s_a + n_stages * kt * s_b) * 4 // float is 4 bytes
        };

        let launch_config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: smem_size,
        };

        unsafe {
             let c_ptr = *c.borrow().inner.device_ptr();
             
             let (final_a, final_b) = if config.use_tensor_cores {
                // Must be U16 buffers
                let a_buf = a.extract::<PyDeviceBufferU16>().map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("TensorCore requires U16/Half buffers for A"))?;
                let b_buf = b.extract::<PyDeviceBufferU16>().map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("TensorCore requires U16/Half buffers for B"))?;
                (*a_buf.inner.device_ptr() as u64, *b_buf.inner.device_ptr() as u64)
            } else {
                // Must be F32 buffers
                let a_buf = a.extract::<PyDeviceBufferF32>().map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("CUDA Core requires F32 buffers for A"))?;
                let b_buf = b.extract::<PyDeviceBufferF32>().map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("CUDA Core requires F32 buffers for B"))?;
                (*a_buf.inner.device_ptr() as u64, *b_buf.inner.device_ptr() as u64)
            };

            kernel.launch(launch_config, (final_a, final_b, c_ptr as u64, m as i32, n as i32, k as i32))
                 .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Kernel Launch Failed: {:?}", e)))?;
        }
        
        Ok(())
    }

    #[getter]
    pub fn scratch_a(&self) -> PyDeviceBufferF32 { self.scratch_a.clone() }
    
    #[getter]
    pub fn scratch_b(&self) -> PyDeviceBufferF32 { self.scratch_b.clone() }
    
    #[getter]
    pub fn scratch_c(&self) -> PyDeviceBufferF32 { self.scratch_c.clone() }

    #[getter]
    pub fn scratch_a_h(&self) -> PyDeviceBufferU16 { self.scratch_a_h.clone() }
    
    #[getter]
    pub fn scratch_b_h(&self) -> PyDeviceBufferU16 { self.scratch_b_h.clone() }

    #[allow(unused_unsafe)]
    pub fn profiling(_slf: &Bound<'_, Self>) -> PyResult<PyProfilingScope> {
        Ok(PyProfilingScope {})
    }

    pub fn synchronize(&self) -> PyResult<()> {
        self.device.synchronize().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Sync Error: {:?}", e)))
    }

    pub fn optimize_graph(&mut self, graph: &PyGraph, iterations: usize, goal: Option<PyOptimizationGoal>) -> PyResult<()> {
        println!("[Tracea] 🕸️ Optimizing Graph with {} nodes...", graph.inner.nodes.len());
        
        let strategy = crate::core::graph::PrioritySchedule;
        let schedule = crate::core::graph::ScheduleStrategy::schedule(&strategy, &graph.inner);

        for &node_id in &schedule {
            let node = &graph.inner.nodes[node_id];
            match &node.op {
                crate::core::graph::Operation::Gemm(gemm) => {
                    self.auto_tune(gemm.m.0, gemm.n.0, gemm.k.0, iterations, goal)?;
                }
            }
        }
        println!("[Tracea] ✅ Graph Optimization Complete.");
        Ok(())
    }

    pub fn auto_tune(&mut self, m: u32, n: u32, k: u32, iterations: usize, goal: Option<PyOptimizationGoal>) -> PyResult<()> {
        let benchmark = crate::optimizer::benchmark::NVRTCBenchmark::new(
            self.jit.clone(),
            m, n, k,
        );
        
        // Convert Python goal to Internal goal
        let internal_goal = match goal.unwrap_or(PyOptimizationGoal::MaximizeTFLOPS) {
            PyOptimizationGoal::MaximizeTFLOPS => crate::optimizer::OptimizationGoal::MaximizeTFLOPS,
            PyOptimizationGoal::MinimizeLatency => crate::optimizer::OptimizationGoal::MinimizeLatency,
        };
        
        println!("[Tracea] Auto-tuning for {}x{}x{} ({} iterations)...", m, n, k, iterations);
        let winner = self.tuner.optimize(&benchmark, iterations, internal_goal);
        
        let mut best_config_guard = self.best_config.lock().unwrap();
        *best_config_guard = winner;
        
        Ok(())
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

#[pyclass(name = "Epilogue")]
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

    pub fn relu(slf: &Bound<'_, Self>) -> PyResult<Self> {
        let mut inner = slf.borrow_mut();
        inner.ops.push((PyEpilogueType::ReLU, None));
        Ok(inner.clone())
    }

    pub fn gelu(slf: &Bound<'_, Self>) -> PyResult<Self> {
        let mut inner = slf.borrow_mut();
        inner.ops.push((PyEpilogueType::Gelu, None));
        Ok(inner.clone())
    }

    pub fn bias_add(slf: &Bound<'_, Self>, ptr: usize) -> PyResult<Self> {
        let mut inner = slf.borrow_mut();
        inner.ops.push((PyEpilogueType::BiasAdd, Some(ptr)));
        Ok(inner.clone())
    }

    /// Operator overloading for >>
    fn __rshift__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        let inner_slf = slf.borrow();
        let inner_other = other.borrow();
        let mut new_ops = inner_slf.ops.clone();
        new_ops.extend(inner_other.ops.clone());
        Ok(Self { ops: new_ops })
    }

    #[staticmethod]
    pub fn empty() -> Self { Self::new() }
}

#[pyfunction(name = "ReLU")]
pub fn python_relu() -> PyEpilogueOp {
    let mut op = PyEpilogueOp::new();
    op.ops.push((PyEpilogueType::ReLU, None));
    op
}

#[pyfunction(name = "Gelu")]
pub fn python_gelu() -> PyEpilogueOp {
    let mut op = PyEpilogueOp::new();
    op.ops.push((PyEpilogueType::Gelu, None));
    op
}

#[pyfunction(name = "BiasAdd")]
pub fn python_bias_add(ptr: usize) -> PyEpilogueOp {
    let mut op = PyEpilogueOp::new();
    op.ops.push((PyEpilogueType::BiasAdd, Some(ptr)));
    op
}
