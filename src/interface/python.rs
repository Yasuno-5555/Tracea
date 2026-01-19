#![allow(unused)]
use pyo3::prelude::*;
use crate::core::config::PipelineConfig;
use crate::optimizer::{AutoTuner, GPUInfo, OptimizationGoal};
use crate::optimizer::cache::{TuningCache, CacheKey};
use crate::optimizer::benchmark::NVRTCBenchmark;
use crate::runtime::{RuntimeManager, BufferId, KernelArg, DeviceBackend};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use std::path::PathBuf;
use std::sync::Arc;
use std::io::{self, Write};
use serde::{Serialize, Deserialize};

#[pyclass(name = "Context")]
#[derive(Clone)]
pub struct PyContext {
    pub tuner: AutoTuner,
    pub runtime: Arc<RuntimeManager>,
    #[pyo3(get)]
    pub scratch_a: PyDeviceBufferF32,
    #[pyo3(get)]
    pub scratch_b: PyDeviceBufferF32,
    #[pyo3(get)]
    pub scratch_c: PyDeviceBufferF32,
    #[pyo3(get)]
    pub scratch_a_h: PyDeviceBufferU16,
    #[pyo3(get)]
    pub scratch_b_h: PyDeviceBufferU16,
    pub best_config: Arc<std::sync::Mutex<PipelineConfig>>,
}

#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferF32 {
    pub id: BufferId,
    pub runtime: Arc<RuntimeManager>,
}

#[pymethods]
impl PyDeviceBufferF32 {
    #[staticmethod]
    pub fn unsafe_from_ptr(ptr: usize, _len: usize, device: &PyContext) -> PyResult<Self> {
        let id = device.runtime.register_external_ptr(ptr as u64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { id, runtime: device.runtime.clone() })
    }
    
    pub fn device_ptr(&self) -> PyResult<usize> {
        self.runtime.get_ptr(self.id)
            .map(|p| p as usize)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Buffer dropped or invalid"))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferU16 {
    pub id: BufferId,
    pub runtime: Arc<RuntimeManager>,
}

#[pymethods]
impl PyDeviceBufferU16 {
   #[staticmethod]
    pub fn unsafe_from_ptr(ptr: usize, _len: usize, device: &PyContext) -> PyResult<Self> {
        let id = device.runtime.register_external_ptr(ptr as u64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { id, runtime: device.runtime.clone() })
    }
    
    pub fn device_ptr(&self) -> PyResult<usize> {
        self.runtime.get_ptr(self.id)
            .map(|p| p as usize)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Buffer dropped or invalid"))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferI32 {
    pub id: BufferId,
    pub runtime: Arc<RuntimeManager>,
}

#[pymethods]
impl PyDeviceBufferI32 {
    #[staticmethod]
    pub fn unsafe_from_ptr(ptr: usize, _len: usize, device: &PyContext) -> PyResult<Self> {
        let id = device.runtime.register_external_ptr(ptr as u64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { id, runtime: device.runtime.clone() })
    }
    
    pub fn device_ptr(&self) -> PyResult<usize> {
        self.runtime.get_ptr(self.id)
            .map(|p| p as usize)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Buffer dropped or invalid"))
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
    pub fn swizzle_mode(&self) -> u32 { 
        match self.inner.swizzle_mode {
            crate::core::config::SwizzleMode::None => 0,
            crate::core::config::SwizzleMode::Xor2 => 1,
            crate::core::config::SwizzleMode::Xor4 => 2,
            crate::core::config::SwizzleMode::Xor8 => 3,
        }
    }

    #[getter]
    pub fn num_stages(&self) -> u32 { self.inner.num_stages }

    pub fn add_relu(&mut self) {
        self.inner.epilogue.push(crate::core::op::EpilogueOp::ReLU);
    }

    pub fn add_bias(&mut self, ptr: usize) {
        self.inner.epilogue.push(crate::core::op::EpilogueOp::BiasAdd { bias_ptr: ptr });
    }

    #[getter]
    pub fn force_num_warps(&self) -> Option<u32> { self.inner.force_num_warps }

    #[setter]
    pub fn set_force_num_warps(&mut self, val: Option<u32>) { self.inner.force_num_warps = val; }
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

    #[pyo3(signature = (m, n, k, deps=Vec::new()))]
    pub fn add_gemm(&mut self, m: u32, n: u32, k: u32, deps: Vec<usize>) -> usize {
        self.inner.add_gemm(m, n, k, deps)
    }

    pub fn lower(&self) -> Self {
        Self { inner: self.inner.lower() }
    }

    pub fn node_count(&self) -> usize {
        self.inner.nodes.len()
    }

    pub fn optimize_fusion(&self) -> Self {
        Self { inner: self.inner.optimize_fusion() }
    }

    pub fn __len__(&self) -> usize {
        self.inner.nodes.len()
    }
}



#[pyclass(name = "OptimizationGoal")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyOptimizationGoal {
    MaximizeTFLOPS,
    MinimizeLatency,
}

#[pyclass(name = "Decision")]
#[derive(Clone)]
pub struct PyDecision {
    #[pyo3(get)]
    pub variant_id: Option<String>,
    #[pyo3(get)]
    pub backend: String,
    #[pyo3(get)]
    pub strategy: String,
    #[pyo3(get)]
    pub reason: String,
}

#[pymethods]
impl PyContext {
    #[new]
    #[pyo3(signature = (arch=None))]
    pub fn new(arch: Option<String>) -> PyResult<Self> {
        let caps = crate::doctor::get_capabilities();
        
        let target_arch = arch.unwrap_or_else(|| {
            if let Some(cuda) = caps.get_backend(crate::doctor::BackendKind::Cuda) {
                format!("sm_{}", cuda.arch_code)
            } else {
                "gfx90a".to_string() 
            }
        });

        println!("[Tracea] ⚕️ Initializing Heterogeneous Context (Target: {})", target_arch);
        
        let runtime = RuntimeManager::init(None).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
        let primary_backend = if target_arch.contains("sm") { DeviceBackend::Cuda } else { DeviceBackend::Rocm };

        let bit_arch = if target_arch.contains("80") {
             GPUInfo::a100()
        } else if target_arch.contains("gfx") {
             GPUInfo::mi250()
        } else {
             GPUInfo::rtx3070()
        };

        let size = 8192 * 8192; 
        let id_a = runtime.alloc_f32(size, primary_backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let id_b = runtime.alloc_f32(size, primary_backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let id_c = runtime.alloc_f32(size, primary_backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let id_a_h = runtime.alloc_u16(size, primary_backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let id_b_h = runtime.alloc_u16(size, primary_backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let mut initial_config = PipelineConfig::new(1, 64, 64, 32);
        initial_config.instruction = if primary_backend == DeviceBackend::Cuda {
             crate::core::config::SpecializedInstruction::CudaMMA
        } else {
             crate::core::config::SpecializedInstruction::RocmMFMA
        };

        Ok(Self {
            tuner: AutoTuner::new(bit_arch).with_runtime(runtime.clone()),
            runtime: runtime.clone(),
            scratch_a: PyDeviceBufferF32 { id: id_a, runtime: runtime.clone() },
            scratch_b: PyDeviceBufferF32 { id: id_b, runtime: runtime.clone() },
            scratch_c: PyDeviceBufferF32 { id: id_c, runtime: runtime.clone() },
            scratch_a_h: PyDeviceBufferU16 { id: id_a_h, runtime: runtime.clone() },
            scratch_b_h: PyDeviceBufferU16 { id: id_b_h, runtime: runtime.clone() },
            best_config: Arc::new(std::sync::Mutex::new(initial_config)),
        })
    }

    pub fn plan_kernel(&self, kernel_id: String, precision: String) -> PyResult<PyDecision> {
        let policy = match precision.to_uppercase().as_str() {
            "FP32" => crate::doctor::PrecisionPolicy::FP32,
            "FP16" => crate::doctor::PrecisionPolicy::FP16,
            "BF16" => crate::doctor::PrecisionPolicy::BF16,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid precision")),
        };

        let request = crate::doctor::KernelRequestContext {
            precision_policy: policy,
            latency_vs_throughput: 0.5,
            allow_fallback: true,
        };

        let decision = crate::doctor::plan_kernel(&kernel_id, request);
        
        let variant = decision.selected_variant.map(|s| s.to_string());
        let backend_str = if let Some(vid) = &variant {
             let variants = crate::doctor::registry::get_variants_for_id(vid);
             if let Some(v) = variants.first() {
                 format!("{:?}", v.backend)
             } else { "Unknown".to_string() }
        } else { "None".to_string() };

        Ok(PyDecision {
            variant_id: variant,
            backend: backend_str,
            strategy: format!("{:?}", decision.compile_strategy),
            reason: format!("{:?}", decision.reason),
        })
    }


    #[pyo3(signature = (a, b, c, m, n, k, epilogue=None))]
    pub fn matmul(
        slf: &Bound<'_, Self>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
        _m: u32, _n: u32, _k: u32,
        epilogue: Option<Bound<'_, PyEpilogueOp>>
    ) -> PyResult<()> {
        Ok(())
    }

    #[pyo3(signature = (q, k, v, o, b_in, h_in, s_in, d_in, dh_in, causal=false, scale_sqrt=true, m_tile=None, n_tile=None, stages=None, warps=None))]
    pub fn attention(
        slf: &Bound<'_, Self>,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        o: &Bound<'_, PyAny>,
        b_in: u32, h_in: u32, s_in: u32, d_in: u32, dh_in: u32,
        causal: bool,
        scale_sqrt: bool,
        m_tile: Option<u32>,
        n_tile: Option<u32>,
        stages: Option<u32>,
        warps: Option<u32>,
    ) -> PyResult<u64> {
        let ctx = slf.borrow();
        let op = crate::core::op::FusedAttentionOp {
            b: b_in, s: s_in, d: d_in, h: h_in, dh: dh_in, causal, scale_inv_sqrt_d: scale_sqrt,
        };

        let config = if let (Some(m), Some(n), Some(s)) = (m_tile, n_tile, stages) {
            let mut c = crate::core::config::PipelineConfig::new(s, m, n, dh_in);
            c.force_num_warps = warps;
            Some(c)
        } else {
            None
        };

        let backend = ctx.tuner.gpu.backend;
        let emitter = UniversalEmitter::new(backend);
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::FusedAttention {
                b: op.b, s: op.s, d: op.d, h: op.h, dh: op.dh, causal: op.causal
            },
            precison: "f16".to_string(), // Attention usually f16
            tiling: config.clone().unwrap_or_else(|| PipelineConfig::new(2, 64, 64, dh_in)),
        };
        let source = emitter.generate(ir);
        
        let kernel_id = ctx.runtime.compile(&source, "flash_attention_v2_kernel", backend)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Attention Compilation Failed: {}", e)))?;

        fn get_arg(obj: &Bound<'_, PyAny>) -> PyResult<KernelArg> {
            if let Ok(buf) = obj.extract::<PyDeviceBufferU16>() {
                return Ok(KernelArg::Buffer(buf.id));
            }
            if let Ok(buf) = obj.extract::<PyDeviceBufferF32>() {
                return Ok(KernelArg::Buffer(buf.id));
            }
            if let Ok(ptr_obj) = obj.call_method0("data_ptr") {
                let ptr = ptr_obj.extract::<usize>()?;
                return Ok(KernelArg::Usize(ptr));
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected Buffer or data_ptr()"))
        }

        let arg_q = get_arg(q)?;
        let arg_k = get_arg(k)?;
        let arg_v = get_arg(v)?;
        let arg_o = get_arg(o)?;

        let final_config = config.unwrap_or_else(|| PipelineConfig::new(2, 64, 64, dh_in));
        let mt = final_config.m_tile;
        let nt = final_config.n_tile;

        let scale_val = if scale_sqrt { 1.0 / (dh_in as f32).sqrt() } else { 1.0 };

        let num_warps = final_config.force_num_warps.unwrap_or(mt / 16) + 1;
        let block = (num_warps * 32, 1, 1);
        let grid = ( (s_in + mt - 1) / mt, h_in, b_in );
        
        // Kernel needs: 
        // sS: num_warps * 256 floats (1024 bytes/warp)
        // sO: num_warps * 16 * dh_in floats (64 * dh_in bytes/warp)
        // sP: num_warps * 256 halves (512 bytes/warp)
        // sK: 2 * nt * (dh_in + 8) halves 
        // sV: 2 * nt * (dh_in + 8) halves
        let smem_bytes = (num_warps as usize) * (1536 + 64 * (dh_in as usize)) + (8 * (nt as usize) * (dh_in as usize + 8));

        ctx.runtime.launch(
            kernel_id, grid, block, smem_bytes as u32,
            vec![
                arg_q, arg_k, arg_v, arg_o,
                KernelArg::Usize(b_in as usize),
                KernelArg::Usize(h_in as usize),
                KernelArg::Usize(s_in as usize),
                KernelArg::Usize(dh_in as usize),
                KernelArg::Float(scale_val)
            ]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Attention Launch Error: {}", e)))?;

        Ok(kernel_id.0)
    }

    pub fn launch_kernel(&self, py: Python<'_>, id: u64, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: Vec<PyObject>) -> PyResult<()> {
         let mut k_args = Vec::new();
         for obj in args {
             let bound = obj.into_bound(py);
             if let Ok(val) = bound.extract::<i32>() { k_args.push(KernelArg::Int(val)); continue; }
             if let Ok(val) = bound.extract::<f32>() { k_args.push(KernelArg::Float(val)); continue; }
             if let Ok(val) = bound.extract::<usize>() { k_args.push(KernelArg::Usize(val)); continue; }
             if let Ok(buf) = bound.extract::<PyDeviceBufferU16>() { k_args.push(KernelArg::Buffer(buf.id)); continue; }
             if let Ok(buf) = bound.extract::<PyDeviceBufferF32>() { k_args.push(KernelArg::Buffer(buf.id)); continue; }
             if let Ok(buf) = bound.extract::<PyDeviceBufferI32>() { k_args.push(KernelArg::Buffer(buf.id)); continue; }
             
             if let Ok(ptr_obj) = bound.call_method0("data_ptr") {
                 let ptr = ptr_obj.extract::<usize>()?;
                 k_args.push(KernelArg::Usize(ptr));
                 continue;
             }
             return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported kernel argument type"));
         }
         self.runtime.launch(crate::runtime::KernelId(id), grid, block, smem, k_args).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn compile_custom(&self, source: String, name: String) -> PyResult<u64> {
         let backend = self.tuner.gpu.backend;
         let id = self.runtime.compile(&source, &name, backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
         Ok(id.0)
    }

    #[pyo3(signature = (q, k, v, o, b_in, h_in, s_in, _d_in, dh_in, scale_sqrt=true, m_tile=64, n_tile=64, _stages=2, warps=4))]
    pub fn get_attention_params(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        o: &Bound<'_, PyAny>,
        b_in: u32, h_in: u32, s_in: u32, _d_in: u32, dh_in: u32,
        scale_sqrt: bool,
        m_tile: u32, n_tile: u32, _stages: u32, warps: u32,
    ) -> PyResult<( (u32, u32, u32), (u32, u32, u32), u32, Vec<PyObject> )> {
        // Just return the grid, block, smem, and args for low-level launch
        let num_warps = warps + 1;
        let block = (num_warps * 32, 1, 1);
        let grid = ( (s_in + m_tile - 1) / m_tile, h_in, b_in );
        let smem_bytes = (num_warps as usize) * (1536 + 64 * (dh_in as usize)) + (8 * (n_tile as usize) * (dh_in as usize + 8));
        let scale_val = if scale_sqrt { 1.0 / (dh_in as f32).sqrt() } else { 1.0 };
        
        Ok((
            grid, block, smem_bytes as u32,
            vec![
                q.as_any().clone().unbind(),
                k.as_any().clone().unbind(),
                v.as_any().clone().unbind(),
                o.as_any().clone().unbind(),
                (b_in as i64).to_object(q.py()),
                (h_in as i64).to_object(q.py()),
                (s_in as i64).to_object(q.py()),
                (dh_in as i64).to_object(q.py()),
                (scale_val as f64).to_object(q.py()),
            ]
        ))
    }

    #[pyo3(signature = (a, b, c, iterations=10))]
    pub fn benchmark_gemm(
        &self,
        _a: &Bound<'_, PyAny>,
        _b: &Bound<'_, PyAny>,
        _c: &Bound<'_, PyAny>,
        _iterations: usize
    ) -> PyResult<f64> {
         Ok(0.0) 
    }

    #[allow(unused_unsafe)]
    pub fn profiling(_slf: Bound<'_, Self>) -> PyResult<PyProfilingScope> {
        Ok(PyProfilingScope {})
    }

    pub fn synchronize(&self) -> PyResult<()> {
        self.runtime.synchronize();
        Ok(())
    }

    pub fn optimize_graph(&mut self, _graph: &PyGraph, _iterations: usize, _goal: Option<PyOptimizationGoal>) -> PyResult<()> {
        Ok(())
    }

    #[pyo3(signature = (m, n, k, iterations, goal=None, epilogue=None))]
    pub fn auto_tune(&mut self, _m: u32, _n: u32, _k: u32, _iterations: usize, _goal: Option<PyOptimizationGoal>, _epilogue: Option<PyEpilogueOp>) -> PyResult<()> {
        Ok(())
    }

    #[getter]
    pub fn doctor(&self) -> PyDoctor {
        PyDoctor { inner: self.runtime.doctor.clone() }
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
#[derive(Clone, Debug)]
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

#[pyclass(name = "Doctor")]
#[derive(Clone)]
pub struct PyDoctor {
    pub inner: std::sync::Arc<crate::doctor::Doctor>,
}

#[pymethods]
impl PyDoctor {
    pub fn diagnose(&self) -> PyResult<PyEnvironmentReport> {
        let report = self.inner.diagnose_environment();
        Ok(PyEnvironmentReport {
            timestamp: report.timestamp,
            cuda_version: report.cuda_version,
            ptxas_version: report.ptxas_version,
            gpu_info: report.gpu_info,
            rocm_info: report.rocm_info,
            status: format!("{:?}", report.status),
            summary: report.summary,
            issues: report.issues,
        })
    }

    pub fn last_error(&self) -> Option<PyDoctorErrorReport> {
        self.inner.last_error().map(|err| {
            PyDoctorErrorReport {
                kind: format!("{:?}", err.kind),
                backend: format!("{:?}", err.backend),
                message: err.message,
                suggestion: err.suggestion,
                artifacts: PyDoctorArtifacts {
                    source_path: err.artifacts.source_path.map(|p| p.to_string_lossy().to_string()),
                    ptx_path: err.artifacts.ptx_path.map(|p| p.to_string_lossy().to_string()),
                    asm_log_path: err.artifacts.asm_log_path.map(|p| p.to_string_lossy().to_string()),
                    cubin_path: err.artifacts.cubin_path.map(|p| p.to_string_lossy().to_string()),
                    launch_snapshot_path: err.artifacts.launch_snapshot_path.map(|p| p.to_string_lossy().to_string()),
                }
            }
        })
    }
}

#[pyclass(name = "EnvironmentReport")]
pub struct PyEnvironmentReport {
    #[pyo3(get)]
    pub timestamp: u64,
    #[pyo3(get)]
    pub cuda_version: Option<String>,
    #[pyo3(get)]
    pub ptxas_version: Option<String>,
    #[pyo3(get)]
    pub gpu_info: Option<String>,
    #[pyo3(get)]
    pub rocm_info: Option<String>,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub summary: String,
    #[pyo3(get)]
    pub issues: Vec<String>,
}

#[pyclass(name = "DoctorArtifacts")]
#[derive(Clone)]
pub struct PyDoctorArtifacts {
    #[pyo3(get)] pub source_path: Option<String>,
    #[pyo3(get)] pub ptx_path: Option<String>,
    #[pyo3(get)] pub asm_log_path: Option<String>,
    #[pyo3(get)] pub cubin_path: Option<String>,
    #[pyo3(get)] pub launch_snapshot_path: Option<String>,
}

#[pyclass(name = "DoctorErrorReport")]
#[derive(Clone)]
pub struct PyDoctorErrorReport {
    #[pyo3(get)] pub kind: String,
    #[pyo3(get)] pub backend: String,
    #[pyo3(get)] pub message: String,
    #[pyo3(get)] pub suggestion: String,
    #[pyo3(get)] pub artifacts: PyDoctorArtifacts,
}
