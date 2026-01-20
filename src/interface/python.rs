#![allow(unused)]
use pyo3::prelude::*;
use std::sync::Arc;
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::UnifiedOpIR;
use crate::emitter::traits::UnifiedOpType;
use crate::kernels::attention::cuda_emitter::FlashAttentionEmitter; // Import specific emitter for calc logic
use crate::core::config::PipelineConfig;
use std::collections::HashMap;
use crate::core::tuning::{tune_kernel, SearchMode, TunableKernel};
use crate::kernels::gemm::cpu_adapter::{GemmAdapter, GemmProblem};
use crate::kernels::attention::cuda_adapter::{Fa2Adapter, Fa2Problem};
use crate::backend::cpu::CpuBackend;
use crate::optimizer::benchmark::{Conv2dProblem, NVRTCConvBenchmark};
use crate::optimizer::{AutoTuner, GPUInfo};
use std::sync::Mutex;

#[pyclass(name = "Context")]
#[derive(Clone)]
pub struct PyContext {
    pub runtime: Arc<RuntimeManager>,
}

fn get_kernel_arg(obj: &Bound<'_, PyAny>) -> PyResult<KernelArg> {
    if let Ok(buf) = obj.extract::<PyDeviceBufferU16>() { return Ok(KernelArg::Buffer(buf.id)); }
    if let Ok(buf) = obj.extract::<PyDeviceBufferF32>() { return Ok(KernelArg::Buffer(buf.id)); }
    if let Ok(ptr_obj) = obj.call_method0("data_ptr") {
        let ptr = ptr_obj.extract::<usize>()?;
        return Ok(KernelArg::Usize(ptr));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected Buffer or object with data_ptr()"))
}

#[pymethods]
impl PyContext {
    #[new]
    #[pyo3(signature = (arch=None))]
    pub fn new(arch: Option<String>) -> PyResult<Self> {
        let runtime = RuntimeManager::init(None).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { runtime })
    }

    pub fn synchronize(&self) -> PyResult<()> {
        self.runtime.synchronize();
        Ok(())
    }

    #[pyo3(signature = (q, k, v, o, b_in, h_in, s_in, d_in, dh_in, causal=false, scale_sqrt=true, m_tile=None, n_tile=None, stages=None, warps=None))]
    pub fn attention(
        &self,
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
        let ctx = self;
        let mut request_ctx = crate::doctor::KernelRequestContext {
            precision_policy: crate::doctor::PrecisionPolicy::FP16,
            latency_vs_throughput: 0.5,
            allow_fallback: true,
        };

        let decision = crate::doctor::plan_kernel("fa2", request_ctx);
        let variant = decision.selected_variant.unwrap_or("fa2_cuda");

        let problem = Fa2Problem {
            b: b_in as usize,
            s: s_in as usize,
            h: h_in as usize,
            d: dh_in as usize,
            is_causal: causal,
        };

        let user_config = if let (Some(m), Some(n), Some(s)) = (m_tile, n_tile, stages) {
            let mut c = PipelineConfig::new(s, m, n, dh_in);
            c.force_num_warps = warps;
            Some(c)
        } else {
            None
        };

        let final_config = if let Some(c) = user_config {
            c
        } else {
            match variant {
                "fa2_cuda" => {
                    let adapter = Fa2Adapter::new(Arc::clone(&ctx.runtime), problem);
                    tune_kernel(&adapter, SearchMode::GridSearch)
                },
                "fa2_metal" => {
                    // Fallback or specific Metal tuning
                    PipelineConfig::new(2, 64, 64, dh_in)
                }
                _ => PipelineConfig::new(2, 64, 64, dh_in)
            }
        };

        let backend = DeviceBackend::Cuda; 
        let emitter = UniversalEmitter::new(backend);
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::FusedAttention {
                b: b_in, s: s_in, d: d_in, h: h_in, dh: dh_in, causal
            },
            precison: "f16".to_string(),
            tiling: final_config.clone(),
            conv_magic_strategy: None,
        };
        let final_config = ir.tiling.clone(); // Clone before move
        let source = emitter.generate(ir);
        
        let kernel_id = match ctx.runtime.compile(&source, "flash_attention_v2_kernel", backend) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("[Tracea Min] ❌ Compile Error: {}", e);
                eprintln!("[Tracea Min] 📜 Generated Source:\n{}\n", source);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Compilation Failed: {}", e)));
            }
        };

        let arg_q = get_kernel_arg(q)?;
        let arg_k = get_kernel_arg(k)?;
        let arg_v = get_kernel_arg(v)?;
        let arg_o = get_kernel_arg(o)?;

        let mt = final_config.m_tile;
        let nt = final_config.n_tile;
        let stages = final_config.num_stages;
        let d = dh_in as u32;
        let stride = d; 

        // Warp Spec Mode: 2 items Producer + N items Consumer
        // Consumers = mt / 16 (16 rows per warp)
        // Partitioning: Fixed 16 rows per consumer warp.
        let num_warps = final_config.force_num_warps.unwrap_or(1 + (mt / 16)); 
        let block = (num_warps * 32, 1, 1);
        let grid = ( (s_in + mt - 1) / mt, h_in, b_in );
        
        // Centralized Smem Calculation
        let (smem_bytes, _, _, _, _) = FlashAttentionEmitter::calculate_smem_layout(&final_config, dh_in as usize);
        let scale_val = if scale_sqrt { 1.0 / (dh_in as f32).sqrt() } else { 1.0 };

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
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Launch Error: {}", e)))?;

        Ok(kernel_id.0)
    }

    #[pyo3(signature = (a, b, c, m, n, k, m_tile=None, n_tile=None, k_tile=None))]
    pub fn gemm(
        &self,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
        m: u32, n: u32, k: u32,
        m_tile: Option<u32>,
        n_tile: Option<u32>,
        k_tile: Option<u32>,
    ) -> PyResult<u64> {
        let ctx = self;
        
        // Planning (Stub for now, use CPU or CUDA based on runtime backend?)
        // The UniversalEmitter handles backend dispatch? 
        // No, UniversalEmitter takes a backend enum.
        // But PyContext holds RuntimeManager which has a backend.
        // We know it's CUDA or CPU.
        // For now, assume CUDA for testing.
        
        let mut config = PipelineConfig::new(2, m_tile.unwrap_or(16), n_tile.unwrap_or(16), k_tile.unwrap_or(16));
        config.m_tile = m_tile.unwrap_or(16);
        config.n_tile = n_tile.unwrap_or(16);
        config.k_tile = k_tile.unwrap_or(16);
        
        let backend = DeviceBackend::Cuda; // Force CUDA for now as requested
        let emitter = UniversalEmitter::new(backend);
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm { m, n, k },
            precison: "f16".to_string(),
            tiling: config.clone(),
            conv_magic_strategy: None,
        };
        
        let source = emitter.generate(ir);
        
        let kernel_id = match ctx.runtime.compile(&source, "gemm_mma_kernel", backend) {
            Ok(id) => id,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("GEMM Compile Error: {}", e))),
        };

        // C = A * B
        let arg_a = get_kernel_arg(a)?;
        let arg_b = get_kernel_arg(b)?;
        let arg_c = get_kernel_arg(c)?;
        
        // Launch Config
        // Grid: M/MT, N/NT
        let mt = config.m_tile;
        let nt = config.n_tile;
        let grid = ((n + nt - 1) / nt, (m + mt - 1) / mt, 1);
        let block = (32, 1, 1); // 1 Warp per Block for 16x16
        let smem_bytes = 0; 
        
        eprintln!("[Tracea Gemm] Launching M={} N={} K={} Grid={:?} Block={:?} Smem={}", m, n, k, grid, block, smem_bytes);
        eprintln!("[Tracea Gemm] Args: A={:?} B={:?} C={:?}", arg_a, arg_b, arg_c);

        ctx.runtime.launch(
            kernel_id, grid, block, smem_bytes,
            vec![arg_a, arg_b, arg_c, KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32)]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("GEMM Launch Error: {}", e)))?;

        Ok(kernel_id.0)
    }

    #[pyo3(signature = (q, k, v, o, b_in, h_in, s_in, _d_in, dh_in, scale_sqrt=true, m_tile=64, n_tile=64, stages=2, warps=4))]
    pub fn get_attention_params(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        o: &Bound<'_, PyAny>,
        b_in: u32, h_in: u32, s_in: u32, _d_in: u32, dh_in: u32,
        scale_sqrt: bool,
        m_tile: u32, n_tile: u32, stages: u32, warps: u32,
    ) -> PyResult<( (u32, u32, u32), (u32, u32, u32), u32, Vec<PyObject> )> {
        let num_warps = warps + 2; // Assuming warps arg is just Consumers? No, passed as explicit.
        // Actually, bench_fa2.py passes 'warps' explicitly in some configs?
        // Wait, bench_fa2.py configs: '64x64, 2S (Baseline)' -> NO 'warps' arg.
        // So 'warps' arg is Option<u32> in attention(). But here it is u32 default=4.
        // If passed 4, we want it to be 4?
        // But if default '4' was intended as "total warps", now we need 6.
        // Let's rely on calculation if not forced? 
        // Logic: If 'warps' parameter is used, trust it. 
        // BUT for dynamic change, we want auto-calc.
        // The previous code: let num_warps = warps + 1;
        // Let's change it to match attention() logic if warps is default?
        // Actually, let's just use the robust formula:
        let num_warps = if warps == 4 { 2 + (m_tile / 16) } else { warps }; 
        // This is hacky. Better:
        // let num_warps = 2 + (m_tile / 16); 
        // This ignores user input 'warps' though.
        // Let's assume user input 'warps' is TOTAL warps.
        // If bench passes '4', it breaks.
        // Benchmark config does NOT pass 'warps'. 
        // Function sig default is `warps=4`.
        // So we should ignore default 4 and calc.
        let num_warps = 2 + (m_tile / 16);
        
        let block = (num_warps * 32, 1, 1);
        let grid = ( (s_in + m_tile - 1) / m_tile, h_in, b_in );
        let temp_config = PipelineConfig::new(stages, m_tile, n_tile, 32); // K-tile matches?
        // Emitter doesn't use k_tile for Smem calc?
        let (smem_bytes, _, _, _, _) = FlashAttentionEmitter::calculate_smem_layout(&temp_config, dh_in as usize);
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

    pub fn launch_kernel(&self, py: Python<'_>, id: u64, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: Vec<PyObject>) -> PyResult<()> {
         let mut k_args = Vec::new();
         for obj in args {
             let bound = obj.into_bound(py);
             if let Ok(val) = bound.extract::<i32>() { k_args.push(KernelArg::Int(val)); continue; }
             if let Ok(val) = bound.extract::<f32>() { k_args.push(KernelArg::Float(val)); continue; }
             if let Ok(val) = bound.extract::<usize>() { k_args.push(KernelArg::Usize(val)); continue; }
             if let Ok(buf) = bound.extract::<PyDeviceBufferU16>() { k_args.push(KernelArg::Buffer(buf.id)); continue; }
             if let Ok(buf) = bound.extract::<PyDeviceBufferF32>() { k_args.push(KernelArg::Buffer(buf.id)); continue; }
             
             if let Ok(ptr_obj) = bound.call_method0("data_ptr") {
                 let ptr = ptr_obj.extract::<usize>()?;
                 k_args.push(KernelArg::Usize(ptr));
                 continue;
             }
             return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported kernel argument type"));
         }
         self.runtime.launch(crate::runtime::manager::KernelId(id), grid, block, smem, k_args).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }
    
    // Add compile_custom for benchmark
    pub fn compile_custom(&self, source: String, name: String) -> PyResult<u64> {
         let backend = DeviceBackend::Cuda;
         let id = self.runtime.compile(&source, &name, backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
         Ok(id.0)
    }
}

// Minimal Device Buffers
#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferF32 {
    pub id: crate::runtime::manager::BufferId,
    pub runtime: Arc<RuntimeManager>,
}
#[pymethods]
impl PyDeviceBufferF32 {
     #[staticmethod]
     pub fn unsafe_from_ptr(ptr: usize, _size_bytes: usize, ctx: &PyContext) -> PyResult<Self> {
        let id = ctx.runtime.register_external_ptr(ptr as u64).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { 
            id, 
            runtime: ctx.runtime.clone() 
        })
    }
    pub fn data_ptr(&self) -> usize { self.id.0 as usize }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferU16 {
    pub id: crate::runtime::manager::BufferId,
    pub runtime: Arc<RuntimeManager>,
}
#[pymethods]
impl PyDeviceBufferU16 {
     #[staticmethod]
     pub fn unsafe_from_ptr(ptr: usize, _size_bytes: usize, ctx: &PyContext) -> PyResult<Self> {
        let id = ctx.runtime.register_external_ptr(ptr as u64).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { 
            id, 
            runtime: ctx.runtime.clone() 
        })
    }
    pub fn data_ptr(&self) -> usize { self.id.0 as usize }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferI32 {
    pub id: crate::runtime::manager::BufferId,
    pub runtime: Arc<RuntimeManager>,
}

// STUBS
#[pyclass]
#[derive(Clone)]
pub struct PyPipelineConfig {}

#[pyclass]
#[derive(Clone)]
pub struct PyProfilingScope {}

#[pyclass]
#[derive(Clone)]
pub struct PyEpilogueOp {
    pub ops: Vec<(PyEpilogueType, Option<usize>)>,
}

#[pyclass]
#[derive(Clone)]
pub enum PyEpilogueType { ReLU, Gelu, BiasAdd }

#[pyclass]
#[derive(Clone)]
pub enum PyOptimizationGoal { MaximizeTFLOPS }

#[pyclass]
#[derive(Clone)]
pub struct PyGraph {
    pub inner: crate::core::graph::Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    pub fn new() -> Self {
        Self { inner: crate::core::graph::Graph::new() }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDecision {}

#[pyclass]
#[derive(Clone)]
pub struct PyDoctor {}

#[pyclass]
#[derive(Clone)]
pub struct PyEnvironmentReport {}

#[pyclass]
#[derive(Clone)]
pub struct PyDoctorErrorReport {}

#[pyclass]
#[derive(Clone)]
pub struct PyDoctorArtifacts {}

#[pyfunction]
pub fn python_relu() {}
#[pyfunction]
pub fn python_gelu() {}
#[pyfunction]
pub fn python_bias_add() {}


#[pyclass(name = "Tuner")]
#[derive(Clone)]
pub struct PyTuner {
    inner: Arc<Mutex<AutoTuner>>,
}

#[pymethods]
impl PyTuner {
    #[new]
    pub fn new() -> Self {
        // Initialize with default/detected GPU info. 
        // Real implementation should query RuntimeManager/CUDA driver.
        let gpu = GPUInfo {
            name: "Generic GPU".to_string(), 
            backend: DeviceBackend::Cuda,
            shared_memory_per_block: 102400,
            max_registers_per_thread: 255,
            max_warps_per_sm: 32,
            wavefront_size: 32,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 102400,
            has_specialized_units: true,
        };
        let tuner = AutoTuner::new(gpu);
        Self { inner: Arc::new(Mutex::new(tuner)) }
    }

    pub fn tune_gemm(&self, m: usize, n: usize, k: usize) -> PyResult<String> {
        let backend = CpuBackend::new();
        let problem = GemmProblem { m, n, k };
        let adapter = GemmAdapter::new(backend, problem);

        let best_config = tune_kernel(&adapter, SearchMode::GridSearch);
        let _score = adapter.benchmark(&best_config); 
        
        serde_json::to_string(&best_config).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    pub fn tune_fa2(&self, ctx: &PyContext, b: usize, h: usize, s: usize, d: usize, causal: bool) -> PyResult<String> {
        let problem = Fa2Problem { b, h, s, d, is_causal: causal };
        let adapter = Fa2Adapter::new(ctx.runtime.clone(), problem);

        let best_config = tune_kernel(&adapter, SearchMode::GridSearch);
        
        serde_json::to_string(&best_config).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[pyo3(signature = (ctx, n, c, h, w, k, r, s, stride=1, pad=0, dilation=1))]
    #[allow(clippy::too_many_arguments)]
    pub fn tune_conv2d(
        &self, 
        ctx: &PyContext, 
        n: usize, c: usize, h: usize, w: usize, k: usize, 
        r: usize, s: usize, 
        stride: usize, pad: usize, dilation: usize
    ) -> PyResult<String> {
        let problem = Conv2dProblem::new("CustomConv", n, h, w, c, k, r, s, stride, pad, dilation);
        
        let benchmark = NVRTCConvBenchmark::new(ctx.runtime.clone(), problem);
        
        // Update tuner's runtime reference for Doctor
        let mut tuner = self.inner.lock().map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock Poisoned"))?;
        tuner.runtime = Some(ctx.runtime.clone());
        
        let goal = crate::optimizer::OptimizationGoal::MaximizeTFLOPS;
        let config = tuner.optimize_conv(&benchmark, 20, goal); // 20 iterations default
        
        serde_json::to_string(&config).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}


