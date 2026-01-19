use pyo3::prelude::*;
use crate::core::config::PipelineConfig;
use crate::optimizer::{AutoTuner, GPUInfo, OptimizationGoal};
use crate::optimizer::cache::{TuningCache, CacheKey};
use crate::optimizer::benchmark::NVRTCBenchmark;
use crate::runtime::{RuntimeManager, BufferId, KernelArg, DeviceBackend};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use std::io::Write; 
use std::sync::Arc;

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

#[pyclass(name = "Context")]
#[derive(Clone)]
pub struct PyContext {
    pub tuner: AutoTuner,
    pub runtime: Arc<RuntimeManager>,
    pub scratch_a: PyDeviceBufferF32,
    pub scratch_b: PyDeviceBufferF32,
    pub scratch_c: PyDeviceBufferF32,
    pub scratch_a_h: PyDeviceBufferU16,
    pub scratch_b_h: PyDeviceBufferU16,
    pub best_config: Arc<std::sync::Mutex<PipelineConfig>>,
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
            tuner: AutoTuner::new(bit_arch),
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
        m: u32, n: u32, k: u32,
        epilogue: Option<Bound<'_, PyEpilogueOp>>
    ) -> PyResult<()> {
        let (m, n, k) = (m as u32, n as u32, k as u32);
        let (config, kernel_id) = {
            let mut ctx_val = slf.borrow_mut();
            let ctx = &mut *ctx_val;

            let mut rust_epilogue = Vec::new();
            if let Some(epi_bound) = epilogue {
                let epi = epi_bound.borrow();
                for (op_type, ptr_opt) in &epi.ops {
                    let op = match op_type {
                        PyEpilogueType::ReLU => crate::core::op::EpilogueOp::ReLU,
                        PyEpilogueType::Gelu => crate::core::op::EpilogueOp::Gelu,
                        PyEpilogueType::BiasAdd => {
                            let ptr = ptr_opt.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("BiasAdd requires a pointer"))?;
                            crate::core::op::EpilogueOp::BiasAdd { bias_ptr: ptr as usize }
                        }
                    };
                    rust_epilogue.push(op);
                }
            }

            let env = AutoTuner::get_env_info(&ctx.runtime.get_device(ctx.tuner.gpu.backend));
            
            let is_b_int32 = b.extract::<PyDeviceBufferI32>().is_ok();
            let is_a_f32 = a.extract::<PyDeviceBufferF32>().is_ok();

            let quant_mode = if is_b_int32 { 
                crate::core::config::QuantizationMode::Int4 
            } else { 
                crate::core::config::QuantizationMode::None 
            };

            let dtype_str = if is_b_int32 {
                "int4".to_string()
            } else if is_a_f32 {
                "f32".to_string()
            } else {
                "f16".to_string()
            };

            let key = CacheKey {
                backend: ctx.tuner.gpu.backend,
                gpu: ctx.tuner.gpu.name.clone(),
                m, n, k,
                dtype: dtype_str.clone(),
                epilogue: rust_epilogue.clone(),
                env_version: env.api_version.clone(),
                arch: env.arch.clone(),
            };

            let mut cache = TuningCache::new();
            let opt_cfg = cache.get(&key);
            
            let mut final_config = if let Some(cfg) = opt_cfg {
                cfg
            } else {
                let benchmark = NVRTCBenchmark::new(ctx.runtime.clone(), m, n, k);
                println!("[Tracea] 🚀 Launching Shape-Aware Auto-Tuner for {}x{}x{} (Quant: {:?})...", m, n, k, quant_mode); std::io::stdout().flush().unwrap();
                let mut winner = ctx.tuner.optimize(&benchmark, 5, OptimizationGoal::MaximizeTFLOPS, rust_epilogue.clone());
                println!("[Tracea Debug] Optimization Done."); std::io::stdout().flush().unwrap();
                winner.quantization = quant_mode;
                winner
            };
            
            final_config.epilogue = rust_epilogue;
            final_config.quantization = quant_mode;

            let backend = ctx.tuner.gpu.backend;
            let emitter = UniversalEmitter::new(backend);
            let ir = UnifiedOpIR {
                op_type: UnifiedOpType::Gemm { m, n, k },
                precison: dtype_str.clone(),
                tiling: final_config.clone(),
            };
            let source = emitter.generate(ir);
            
            let kernel_id = ctx.runtime.compile(&source, "gemm_kernel", backend)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Compilation Failed: {}", e)))?;
                
            (final_config, kernel_id)
        };

        let (mt, nt, kt, n_stages) = (config.m_tile, config.n_tile, config.k_tile, config.num_stages);
        let grid_dim = ((n + nt - 1) / nt, (m + mt - 1) / mt, 1);
        let block_dim = (128, 1, 1);
        
        let is_int4 = config.quantization == crate::core::config::QuantizationMode::Int4;
        let smem_size = if config.use_tensor_cores() {
            if is_int4 {
                 (n_stages * mt * kt * 2) + (n_stages * kt * nt / 2)
            } else {
                 (n_stages * mt * kt + n_stages * kt * nt) * 2
            }
        } else {
            let s_a = kt + 4;
            let s_b = nt + 4;
            (n_stages * mt * s_a + n_stages * kt * s_b) * 4
        };
        
        fn get_arg(obj: &Bound<'_, PyAny>) -> PyResult<KernelArg> {
            if let Ok(buf) = obj.extract::<PyDeviceBufferF32>() {
                return Ok(KernelArg::Buffer(buf.id));
            }
            if let Ok(buf) = obj.extract::<PyDeviceBufferU16>() {
                return Ok(KernelArg::Buffer(buf.id));
            }
            if let Ok(buf) = obj.extract::<PyDeviceBufferI32>() {
                 return Ok(KernelArg::Buffer(buf.id));
            }
            if let Ok(ptr_obj) = obj.call_method0("data_ptr") {
                let ptr = ptr_obj.extract::<usize>()?;
                return Ok(KernelArg::Usize(ptr));
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected PyDeviceBuffer or Object with data_ptr()"))
        }

        let arg_a = get_arg(a)?;
        let arg_b = get_arg(b)?;
        let arg_c = get_arg(c)?;
        
        // Ensure we pass arguments in the EXACT order expected by the kernel signature
        // The Emitter generates: (A, B, C, M, N, K)
        
        let ctx = slf.borrow();
        ctx.runtime.launch(
            kernel_id, 
            grid_dim, 
            block_dim, 
            smem_size as u32, 
            vec![
                arg_a,
                arg_b,
                arg_c,
                KernelArg::Int(m as i32),
                KernelArg::Int(n as i32),
                KernelArg::Int(k as i32)
            ]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Launch Error: {}", e)))?;
        
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
    ) -> PyResult<()> {
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

        let num_warps = final_config.force_num_warps.unwrap_or(mt / 16);
        let block = (num_warps * 32, 1, 1);
        let grid = ( (s_in + mt - 1) / mt, h_in, b_in );
        
        // Single-buffer Smem calculation: sQ(mt*D) + sK(nt*D) + sV(nt*D) + sP(mt*nt)
        let smem_bytes = (mt * dh_in + 2 * nt * dh_in + mt * nt) * 2;

        ctx.runtime.launch(
            kernel_id, grid, block, smem_bytes as u32,
            vec![
                arg_q, arg_k, arg_v, arg_o,
                KernelArg::Int(b_in as i32),
                KernelArg::Int(h_in as i32),
                KernelArg::Int(s_in as i32),
                KernelArg::Int(dh_in as i32),
                KernelArg::Float(scale_val)
            ]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Attention Launch Error: {}", e)))?;

        Ok(())
    }

    #[pyo3(signature = (a, b, c, iterations=10))]
    pub fn benchmark_gemm(
        slf: &Bound<'_, Self>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
        iterations: usize
    ) -> PyResult<f64> {
         Ok(0.0) 
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
        let backend = self.tuner.gpu.backend;
        self.runtime.synchronize(backend)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Sync Error: {:?}", e)))
    }

    pub fn optimize_graph(&mut self, graph: &PyGraph, iterations: usize, goal: Option<PyOptimizationGoal>) -> PyResult<()> {
        let lowered_graph = graph.inner.lower().optimize_fusion();
        println!("[Tracea] 🕸️ Optimizing Graph (Lowered & Fused to {} nodes)...", lowered_graph.nodes.len());
        
        let strategy = crate::core::graph::PrioritySchedule;
        let schedule = crate::core::graph::ScheduleStrategy::schedule(&strategy, &lowered_graph);

        for &node_id in &schedule {
            let node = &lowered_graph.nodes[node_id];
            match &node.op {
                crate::core::graph::Operation::Gemm(gemm) => {
                    self.auto_tune(gemm.m.0, gemm.n.0, gemm.k.0, iterations, goal, None)?;
                },
                crate::core::graph::Operation::FusedGemm(fused) => {
                     let mut py_epi = PyEpilogueOp::new();
                     for op in &fused.epilogue {
                         match op {
                             crate::core::op::EpilogueOp::ReLU => py_epi.ops.push((PyEpilogueType::ReLU, None)),
                             crate::core::op::EpilogueOp::Gelu => py_epi.ops.push((PyEpilogueType::Gelu, None)),
                             crate::core::op::EpilogueOp::BiasAdd { bias_ptr } => py_epi.ops.push((PyEpilogueType::BiasAdd, Some(*bias_ptr))),
                             _ => {}
                         }
                     }
                     self.auto_tune(fused.base.m.0, fused.base.n.0, fused.base.k.0, iterations, goal, Some(py_epi))?;
                },
                _ => {}
            }
        }
        println!("[Tracea] ✅ Graph Optimization Complete.");
        Ok(())
    }

    #[pyo3(signature = (m, n, k, iterations, goal=None, epilogue=None))]
    pub fn auto_tune(&mut self, m: u32, n: u32, k: u32, iterations: usize, goal: Option<PyOptimizationGoal>, epilogue: Option<PyEpilogueOp>) -> PyResult<()> {
        let benchmark = crate::optimizer::benchmark::NVRTCBenchmark::new(
            self.runtime.clone(),
            m, n, k,
        );
        
        let internal_goal = match goal.unwrap_or(PyOptimizationGoal::MaximizeTFLOPS) {
            PyOptimizationGoal::MaximizeTFLOPS => crate::optimizer::OptimizationGoal::MaximizeTFLOPS,
            PyOptimizationGoal::MinimizeLatency => crate::optimizer::OptimizationGoal::MinimizeLatency,
        };

        let mut rust_epilogue = Vec::new();
        if let Some(py_epi) = epilogue {
            for (op_type, ptr_opt) in py_epi.ops {
                let op = match op_type {
                    PyEpilogueType::ReLU => crate::core::op::EpilogueOp::ReLU,
                    PyEpilogueType::Gelu => crate::core::op::EpilogueOp::Gelu,
                    PyEpilogueType::BiasAdd => {
                        if let Some(ptr) = ptr_opt {
                            crate::core::op::EpilogueOp::BiasAdd { bias_ptr: ptr }
                        } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("BiasAdd requires pointer"));
                        }
                    }
                };
                rust_epilogue.push(op);
            }
        }
        
        let winner = self.tuner.optimize(&benchmark, iterations, internal_goal, rust_epilogue);
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
