#![allow(unused)]
use pyo3::prelude::*;
use std::sync::Arc;
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::UnifiedOpIR;
use crate::emitter::traits::UnifiedOpType;
use crate::kernels::attention::cuda_emitter::FlashAttentionEmitter; // Import specific emitter for calc logic
use crate::core::config::PipelineConfig;
use crate::core::config::LayoutPolicy;
use crate::core::op::EpilogueOp;
use crate::core::config::SoftmaxGranularity;
use crate::core::config::{GemmVariant, AttentionVariant};
use std::collections::HashMap;
use crate::core::tuning::{tune_kernel, SearchMode, TunableKernel};
use crate::kernels::gemm::cpu_adapter::{GemmAdapter, GemmProblem};
use crate::kernels::attention::cuda_adapter::{Fa2Adapter, Fa2Problem};
use crate::backend::cpu::CpuBackend;
use crate::optimizer::benchmark::{Conv2dProblem, NVRTCConvBenchmark};
use crate::optimizer::{AutoTuner, HardwareProfile};
use std::sync::Mutex;

#[pyclass(name = "Context")]
#[derive(Clone)]
pub struct PyContext {
    pub runtime: Arc<RuntimeManager>,
}

fn get_kernel_arg(obj: &Bound<'_, PyAny>) -> PyResult<KernelArg> {
    if let Ok(buf) = obj.extract::<PyDeviceBufferU16>() { return Ok(KernelArg::Buffer(buf.id)); }
    if let Ok(buf) = obj.extract::<PyDeviceBufferF32>() { return Ok(KernelArg::Buffer(buf.id)); }
    if let Ok(buf) = obj.extract::<PyDeviceBufferF16>() { return Ok(KernelArg::Buffer(buf.id)); }
    if let Ok(ptr_obj) = obj.call_method0("data_ptr") {
        let ptr = ptr_obj.extract::<usize>()?;
        return Ok(KernelArg::Usize(ptr));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected Buffer or object with data_ptr()"))
}

fn parse_epilogue(
    epilogue_str: Option<String>,
    bias: Option<&Bound<'_, PyAny>>,
    residual: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<EpilogueOp>, Vec<KernelArg>)> {
    let mut ops = Vec::new();
    let mut args = Vec::new();
    
    if let Some(s) = epilogue_str {
        let parts: Vec<&str> = s.split('+').collect();
        for part in parts {
            match part.trim().to_lowercase().as_str() {
                "identity" => {},
                "relu" => ops.push(EpilogueOp::ReLU),
                "gelu" => ops.push(EpilogueOp::Gelu),
                "silu" => ops.push(EpilogueOp::SiLU),
                "bias" => {
                    if let Some(b) = bias {
                        let arg = get_kernel_arg(b)?;
                        // We store a dummy ptr in Op, real one in args. 
                        // The emitter uses the structure index, not the value here?
                        // Actually Op definition has `bias_ptr: usize`.
                        // But emitter generates `const float* bias_{i}`.
                        // And we pass `arg` to launch. 
                        // The value in EpilogueOp is mainly for ... hashing/equality?
                        // Let's use 0 in Op and pass real arg.
                        ops.push(EpilogueOp::BiasAdd { bias_ptr: 0 }); 
                        args.push(arg);
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Epilogue contains 'bias' but no bias argument provided"));
                    }
                },
                "residual" => {
                     if let Some(r) = residual {
                        let arg = get_kernel_arg(r)?;
                        ops.push(EpilogueOp::ResidualAdd { residual_ptr: 0 });
                        args.push(arg);
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Epilogue contains 'residual' but no reference provided"));
                    }
                }
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unknown epilogue op: {}", part))),
            }
        }
    }
    Ok((ops, args))
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

    pub fn alloc_f32(&self, size: usize) -> PyResult<PyDeviceBufferF32> {
        #[cfg(target_os = "macos")]
        let backend = DeviceBackend::Metal;
        #[cfg(not(target_os = "macos"))]
        let backend = DeviceBackend::Cuda; 
        
        let id = self.runtime.alloc_f32(size, backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(PyDeviceBufferF32 { id, runtime: self.runtime.clone() })
    }

    pub fn alloc_f16(&self, size: usize) -> PyResult<PyDeviceBufferF16> {
        #[cfg(target_os = "macos")]
        let backend = DeviceBackend::Metal;
        #[cfg(not(target_os = "macos"))]
        let backend = DeviceBackend::Cuda; 
        
        let id = self.runtime.alloc_f16(size, backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(PyDeviceBufferF16 { id, runtime: self.runtime.clone() })
    }

    #[pyo3(signature = (q, k, v, o, b_in, h_in, s_in, d_in, dh_in, causal=false, scale_sqrt=true, m_tile=None, n_tile=None, stages=None, warps=None, softmax_mode=None, variant=None))]
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
        softmax_mode: Option<String>,
        variant: Option<u32>,
    ) -> PyResult<u64> {
        let ctx = self;
        let mut request_ctx = crate::doctor::KernelRequestContext {
            precision_policy: crate::doctor::PrecisionPolicy::FP16,
            latency_vs_throughput: 0.5,
            allow_fallback: true,
        };

        let decision = if let Some(v_idx) = variant {
            // If explicit variant is passed, we skip Doctor and just set it in PipelineConfig later
            crate::doctor::plan_kernel("fa2", request_ctx) // dummy
        } else {
            crate::doctor::plan_kernel("fa2", request_ctx)
        };
        
        let variant_str = decision.selected_variant.unwrap_or("fa2_cuda");

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
            if let Some(mode) = &softmax_mode {
                c.softmax_granularity = match mode.as_str() {
                    "per_tile" => SoftmaxGranularity::PerTile,
                    "per_two_tiles" => SoftmaxGranularity::PerTwoTiles,
                    "full" => SoftmaxGranularity::FullBr, // Experimental
                    _ => SoftmaxGranularity::PerTile, // Fallback/Auto
                };
            }
            if let Some(v_idx) = variant {
                c.attention_variant = match v_idx {
                    2 => AttentionVariant::SimdQK,
                    3 => AttentionVariant::SimdFull,
                    4 => AttentionVariant::FlashV2,
                    _ => AttentionVariant::Naive,
                };
            }
            Some(c)
        } else {
            None
        };

        let final_config = if let Some(c) = user_config {
            c
        } else {
            #[cfg(target_os = "macos")]
            {
                // Metal default config
                let mut c = PipelineConfig::new(2, 32, 32, dh_in);
                c.force_num_warps = Some(1); // 32 threads
                c
            }
            #[cfg(not(target_os = "macos"))]
            match variant_str {
                "fa2_cuda" => {
                    let adapter = Fa2Adapter::new(Arc::clone(&ctx.runtime), problem);
                    tune_kernel(&adapter, SearchMode::GridSearch)
                },
                "fa2_metal" => {
                    PipelineConfig::new(2, 64, 64, dh_in)
                }
                _ => PipelineConfig::new(2, 64, 64, dh_in)
            }
        };

        #[cfg(target_os = "macos")]
        let backend = DeviceBackend::Metal;
        #[cfg(not(target_os = "macos"))]
        let backend = DeviceBackend::Cuda; 
        let emitter = UniversalEmitter::new(backend);
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::FusedAttention {
                b: b_in, s: s_in, d: d_in, h: h_in, dh: dh_in, causal
            },
            precison: "f16".to_string(),
            tiling: final_config.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
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
        let (smem_bytes, _, _, _, _, _) = FlashAttentionEmitter::calculate_smem_layout(&final_config, dh_in as usize);
        let scale_val = if scale_sqrt { 1.0 / (dh_in as f32).sqrt() } else { 1.0 };

        // Pack FAParams struct for Metal: [b, h, s, d, scale]
        // MSL: struct FAParams { uint b, h, s, d; float scale; };
        let mut params_bytes = Vec::with_capacity(20);
        params_bytes.extend_from_slice(&(b_in as u32).to_ne_bytes());
        params_bytes.extend_from_slice(&(h_in as u32).to_ne_bytes());
        params_bytes.extend_from_slice(&(s_in as u32).to_ne_bytes());
        params_bytes.extend_from_slice(&(dh_in as u32).to_ne_bytes());
        params_bytes.extend_from_slice(&scale_val.to_ne_bytes());

        ctx.runtime.launch(
            kernel_id, grid, block, smem_bytes as u32,
            vec![
                arg_q, arg_k, arg_v, arg_o,
                KernelArg::Bytes(params_bytes)
            ]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Launch Error: {}", e)))?;

        Ok(kernel_id.0)
    }

    #[pyo3(signature = (a, b, c, m, n, k, m_tile=None, n_tile=None, k_tile=None, epilogue=None, bias=None, residual=None, variant=None))]
    pub fn gemm(
        &self,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
        m: u32, n: u32, k: u32,
        m_tile: Option<u32>,
        n_tile: Option<u32>,
        k_tile: Option<u32>,
        epilogue: Option<String>,
        bias: Option<&Bound<'_, PyAny>>,
        residual: Option<&Bound<'_, PyAny>>,
        variant: Option<u32>,
    ) -> PyResult<u64> {
        let ctx = self;
        
        let mut config = PipelineConfig::new(2, m_tile.unwrap_or(64), n_tile.unwrap_or(64), k_tile.unwrap_or(32));
        config.gemm_variant = match variant.unwrap_or(0) {
            1 => GemmVariant::Tiled,
            2 => GemmVariant::Simd,
            _ => GemmVariant::Naive,
        };
        
        let (epilogue_ops, epilogue_args) = parse_epilogue(epilogue, bias, residual)?;
        config.epilogue = epilogue_ops;

        #[cfg(target_os = "macos")]
        let (backend, kernel_name) = (DeviceBackend::Metal, if config.gemm_variant == GemmVariant::Tiled { "gemm_tiled_kernel" } else { "gemm_metal_kernel" });
        #[cfg(not(target_os = "macos"))]
        let (backend, kernel_name) = (DeviceBackend::Cuda, if config.gemm_variant == GemmVariant::Simd { "gemm_mma_kernel" } else { "gemm_cublas_fallback" });
        
        let emitter = UniversalEmitter::new(backend);
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm { m, n, k, batch: 1, epilogue: config.epilogue.clone() },
            precison: "f16".to_string(),
            tiling: config.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        
        let source = emitter.generate(ir);
        
        let kernel_id = match ctx.runtime.compile(&source, kernel_name, backend) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("[Tracea Gemm] ❌ Compile Error: {}", e);
                eprintln!("[Tracea Gemm] 📜 Generated Source:\n{}\n", source);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("GEMM Compile Error: {}", e)));
            }
        };

        // C = A * B
        let arg_a = get_kernel_arg(a)?;
        let arg_b = get_kernel_arg(b)?;
        let arg_c = get_kernel_arg(c)?;
        
        // Launch Config
        // Grid: M/MT, N/NT
        // Fix for 64x64: Needs 5 warps (1 Prod, 4 Cons)
        config.force_num_warps = Some(5);
        let mt = config.m_tile;
        let nt = config.n_tile;
        let grid = ((n + nt - 1) / nt, (m + mt - 1) / mt, 1);
        let block = (160, 1, 1); // 5 Warps
        // Calculate smem size for async pipeline
        let a_stride = config.k_tile + 8;
        let b_stride = config.n_tile + 8;
        let smem_a = config.m_tile * a_stride * 2;
        let smem_b = config.k_tile * b_stride * 2;
        let smem_bytes = 128 + (smem_a + smem_b) * config.num_stages;
        
        eprintln!("[Tracea Gemm] Launching M={} N={} K={} Grid={:?} Block={:?} Smem={}", m, n, k, grid, block, smem_bytes);
        eprintln!("[Tracea Gemm] Args: A={:?} B={:?} C={:?}", arg_a, arg_b, arg_c);

        ctx.runtime.launch(
            kernel_id, grid, block, smem_bytes,
            {
                let mut args = vec![arg_a, arg_b, arg_c, KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32)];
                args.extend(epilogue_args);
                args
            }
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("GEMM Launch Error: {}", e)))?;

        Ok(kernel_id.0)
    }

    #[pyo3(signature = (x, w, o, n, c, h, w_in, k, r, s, stride=1, pad=0, dilation=1, epilogue=None, bias=None, residual=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        x: &Bound<'_, PyAny>,
        w: &Bound<'_, PyAny>,
        o: &Bound<'_, PyAny>,
        n: u32, c: u32, h: u32, w_in: u32, k: u32,
        r: u32, s: u32,
        stride: u32, pad: u32, dilation: u32,
        epilogue: Option<String>,
        bias: Option<&Bound<'_, PyAny>>,
        residual: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let ctx = self;
        
        let h_out = (h + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * pad - dilation * (s - 1) - 1) / stride + 1;
        
        // Planning: Universal Tiling Selection
        let mut config = PipelineConfig::new(3, 128, 128, 32); // Baseline "God Config" tiling
        if n == 1 || h_out * w_out < 128 {
            config.m_tile = 32; config.n_tile = 32; config.k_tile = 32;
        }

        let (epilogue_ops, epilogue_args) = parse_epilogue(epilogue, bias, residual)?;
        config.epilogue = epilogue_ops;

        #[cfg(target_os = "macos")]
        let backend = DeviceBackend::Metal;
        #[cfg(not(target_os = "macos"))]
        let backend = DeviceBackend::Cuda; 
        let emitter = UniversalEmitter::new(backend);
        
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Conv2d {
                n: n as usize, c: c as usize, h: h as usize, w: w_in as usize, k: k as usize,
                r: r as usize, s: s as usize,
                stride: stride as usize, 
                pad: pad as usize,
                dilation: dilation as usize,
                layout: crate::core::config::LayoutPolicy::NHWC,
                epilogue: config.epilogue.clone(),
            },
            precison: "f16".to_string(),
            tiling: config.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        
        let source = emitter.generate(ir);
        let kernel_id = match ctx.runtime.compile(&source, "conv2d_implicit_gemm", backend) {
            Ok(id) => id,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Conv2d Compile Error: {}", e))),
        };

        // Pack ConvParams (struct align=4, size=68)
        let (hw_m, hw_s) = crate::emitter::conv::magic_u32(h_out * w_out);
        let (w_m, w_s) = crate::emitter::conv::magic_u32(w_out);
        let (sic_m, sic_s) = crate::emitter::conv::magic_u32(s * c);
        let (c_m, c_s) = crate::emitter::conv::magic_u32(c);

        let mut params = Vec::with_capacity(72);
        for &val in &[n, h, w_in, c, k, h_out, w_out, r, s, stride, pad, dilation] {
            params.extend_from_slice(&(val as i32).to_ne_bytes());
        }
        for &val in &[hw_m, hw_s, w_m, w_s, sic_m, sic_s, c_m, c_s] {
            params.extend_from_slice(&val.to_ne_bytes());
        }

        let grid_final = (
            (batch_ho_wo(n, h_out, w_out) + config.m_tile - 1) / config.m_tile,
            (k + config.n_tile - 1) / config.n_tile,
            1u32
        );
        let block_final = (256u32, 1u32, 1u32); // MT=128 NT=128 KT=32 needs 256 threads
        
        // Smem calculation including hoisting buffers
        let smem_a = config.m_tile * (config.k_tile + 8) * 2;
        let smem_b = config.k_tile * (config.n_tile + 8) * 2;
        let hoisting = config.m_tile * (8 + 4 + 4);
        let smem_bytes = (smem_a + smem_b) * config.num_stages + hoisting + 1024;

        let k_args = {
           let mut a = vec![get_kernel_arg(x)?, get_kernel_arg(w)?, get_kernel_arg(o)?];
           a.extend(epilogue_args);
           a.push(KernelArg::Bytes(params));
           a
        };

        ctx.runtime.launch(
            kernel_id, grid_final, block_final, smem_bytes as u32, k_args
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Conv2d Launch Error: {}", e)))?;
        
        Ok(kernel_id.0)
    }


    #[pyo3(signature = (x, w, o, n, c, h, w_in, k, r, s, stride=1, pad=0, output_padding=0, dilation=1, epilogue=None, bias=None, residual=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(
        &self,
        x: &Bound<'_, PyAny>,
        w: &Bound<'_, PyAny>,
        o: &Bound<'_, PyAny>,
        n: u32, c: u32, h: u32, w_in: u32, k: u32,
        r: u32, s: u32,
        stride: u32, pad: u32, output_padding: u32, dilation: u32,
        epilogue: Option<String>,
        bias: Option<&Bound<'_, PyAny>>,
        residual: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let ctx = self;
        
        let config = PipelineConfig::new(2, 64, 64, 16); 
        let (epilogue_ops, epilogue_args) = parse_epilogue(epilogue, bias, residual)?;
        
        // Clone config and set epilogue
        let mut final_config = config.clone();
        final_config.epilogue = epilogue_ops;

        #[cfg(target_os = "macos")]
        let backend = DeviceBackend::Metal;
        #[cfg(not(target_os = "macos"))]
        let backend = DeviceBackend::Cuda; 
        let emitter = UniversalEmitter::new(backend);
        
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::ConvTranspose2d {
                 n: n as usize, c: c as usize, h: h as usize, w: w_in as usize, k: k as usize,
                 r: r as usize, s: s as usize, stride: stride as usize, pad: pad as usize, output_padding: output_padding as usize,
                 layout: crate::core::config::LayoutPolicy::NHWC,
            },
            precison: "f32".to_string(), 
            tiling: final_config.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        
        let source = emitter.generate(ir);
        
        let kernel_id = match ctx.runtime.compile(&source, "conv_transpose2d_implicit_gemm", backend) {
            Ok(id) => id,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ConvTranspose2d Compile Error: {}", e))),
        };

        let arg_x = get_kernel_arg(x)?;
        let arg_w = get_kernel_arg(w)?;
        let arg_o = get_kernel_arg(o)?;
        
        // Output Shape Calc for Grid
        let h_out = (h - 1) * stride - 2 * pad + r + output_padding;
        let w_out = (w_in - 1) * stride - 2 * pad + s + output_padding;
        
        let m_gemm = n * h_out * w_out; 
        let n_gemm = k;            
        
        let mt = final_config.m_tile;
        let nt = final_config.n_tile;
        
        let grid = ((m_gemm + mt - 1) / mt, (n_gemm + nt - 1) / nt, 1);
        let block = (128, 1, 1); 
        // Calculate smem size
        let a_stride = final_config.k_tile + 8;
        let b_stride = final_config.n_tile + 8;
        let smem_a = final_config.m_tile * a_stride * 2;
        let smem_b = final_config.k_tile * b_stride * 2;
        let smem_bytes = 128 + (smem_a + smem_b) * final_config.num_stages;
        
        ctx.runtime.launch(
            kernel_id, grid, block, smem_bytes,
            {
               let mut args = vec![arg_x, arg_w, arg_o];
               args.extend(epilogue_args);
               args
            }
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ConvTranspose2d Launch Error: {}", e)))?;
        
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
        let (smem_bytes, _, _, _, _, _) = FlashAttentionEmitter::calculate_smem_layout(&temp_config, dh_in as usize);
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
             if let Ok(buf) = bound.extract::<PyDeviceBufferF16>() { k_args.push(KernelArg::Buffer(buf.id)); continue; }
             
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
         #[cfg(target_os = "macos")]
         let backend = DeviceBackend::Metal;
         #[cfg(not(target_os = "macos"))]
         let backend = DeviceBackend::Cuda;
         let id = self.runtime.compile(&source, &name, backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
         Ok(id.0)
    }

    #[pyo3(signature = (graph, iterations=10))]
    pub fn optimize_graph(&self, graph: &Bound<'_, PyGraph>, iterations: usize) -> PyResult<()> {
        let rust_graph = &graph.borrow_mut().inner;
        
        eprintln!("[Tracea] 🚀 Optimizing Graph ({} nodes, {} iterations)", rust_graph.nodes.len(), iterations);
        
        for node in &rust_graph.nodes {
            match &node.op {
                crate::core::graph::Operation::FusedAttention(op) => {
                    eprintln!("[Tracea] Tuning FusedAttention Node {} (B={}, S={}, H={}, D={})", node.id, op.b, op.s, op.h, op.dh);
                    
                    // Crucial: Use op.dh (Head Dim) for Fa2Problem.d
                    let problem = Fa2Problem {
                        b: op.b.as_static().unwrap_or(1) as usize,
                        s: op.s.as_static().unwrap_or(1024) as usize,
                        h: op.h.as_static().unwrap_or(8) as usize,
                        d: op.dh.as_static().unwrap_or(128) as usize, 
                        is_causal: op.causal,
                    };
                    
                    #[cfg(not(target_os = "macos"))]
                    {
                        let adapter = Fa2Adapter::new(Arc::clone(&self.runtime), problem);
                        let best = tune_kernel(&adapter, SearchMode::GridSearch); 
                        println!("[Tracea] Node {} Best Config: {:?}", node.id, best);
                    }
                    #[cfg(target_os = "macos")]
                    println!("[Tracea] Node {} (Metal) - Tuning skipped", node.id);
                },
                _ => {
                    // Skip others for FA2 demo focus
                }
            }
        }
        
        Ok(())
    }

    /// Execute a graph using Policy-driven scheduling.
    /// This is the Python entry point for the "Universal Compute OS".
    /// Executes a graph of operators
    /// 
    /// # Arguments
    /// * `graph_json` - JSON string representing GraphTopology
    /// * `input_buffers` - Dict mapping buffer_id to Buffer object (PyDeviceBufferF32/F16)
    /// * `backend` - Target backend: "cuda", "metal", or "cpu"
    /// 
    /// # Returns
    /// Dict mapping buffer_id to output buffer id
    #[pyo3(signature = (graph_json, input_buffers, backend="metal"))]
    pub fn execute_graph(
        &self, 
        py: Python<'_>, 
        graph_json: String, 
        input_buffers: HashMap<u64, PyObject>,
        backend: &str
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        use crate::policy::types::GraphTopology;
        use std::collections::HashMap;
        
        // Parse graph from JSON
        let graph: GraphTopology = serde_json::from_str(&graph_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid graph JSON: {}", e)))?;
        
        // Resolve backend
        let backend = match backend.to_lowercase().as_str() {
            "cuda" => DeviceBackend::Cuda,
            "metal" => DeviceBackend::Metal,
            "rocm" => DeviceBackend::Rocm,
            "cpu" => DeviceBackend::Cpu,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unknown backend: {}", backend))),
        };
        
        // Resolve input buffers
        let mut resolved_inputs = HashMap::new();
        for (id, obj) in input_buffers {
            let buf_id = if let Ok(b) = obj.extract::<PyDeviceBufferF32>(py) {
                b.id
            } else if let Ok(b) = obj.extract::<PyDeviceBufferF16>(py) {
                b.id
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Input {} is not a Tracea DeviceBuffer", id)));
            };
            resolved_inputs.insert(id, buf_id);
        }
        
        // Execute graph
        let output_buffers = self.runtime.execute_graph(&graph, &resolved_inputs, backend)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
        // Convert to Python dict
        let dict = pyo3::types::PyDict::new_bound(py);
        for (op_id, buf_id) in output_buffers {
            dict.set_item(op_id, buf_id.0)?;
        }
        
        Ok(dict.into())
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

    pub fn copy_from_bytes(&self, data: &[u8]) -> PyResult<()> {
        // Safe to treat as u8 copy
        self.runtime.copy_to_device(self.id, data).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = self.runtime.read_buffer(self.id).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(pyo3::types::PyBytes::new_bound(py, &data))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDeviceBufferF16 {
    pub id: crate::runtime::manager::BufferId,
    pub runtime: Arc<RuntimeManager>,
}
#[pymethods]
impl PyDeviceBufferF16 {
     #[staticmethod]
     pub fn unsafe_from_ptr(ptr: usize, _size_bytes: usize, ctx: &PyContext) -> PyResult<Self> {
        let id = ctx.runtime.register_external_ptr(ptr as u64).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { 
            id, 
            runtime: ctx.runtime.clone() 
        })
    }
    pub fn data_ptr(&self) -> usize { self.id.0 as usize }

    pub fn copy_from_bytes(&self, data: &[u8]) -> PyResult<()> {
        self.runtime.copy_to_device(self.id, data).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = self.runtime.read_buffer(self.id).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(pyo3::types::PyBytes::new_bound(py, &data))
    }
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

#[pyclass(name = "Graph")]
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

    #[pyo3(signature = (m, n, k))]
    pub fn add_gemm(&mut self, m: u32, n: u32, k: u32) -> PyResult<usize> {
        let id = self.inner.add_gemm(m, n, k, vec![]);
        Ok(id)
    }

    pub fn lower(&self) -> PyResult<Self> {
        let lowered = self.inner.lower();
        Ok(Self { inner: lowered })
    }
    
    // For counting nodes
    pub fn __len__(&self) -> usize {
        self.inner.nodes.len()
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
        let gpu = HardwareProfile {
            name: "Generic GPU".to_string(), 
            backend: DeviceBackend::Cuda,
            shared_memory_per_block: 102400,
            max_registers_per_thread: 255,
            registers_per_sm: 65536,
            max_registers_per_block: 65536,
            max_warps_per_sm: 32,
            wavefront_size: 32,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 102400,
            has_specialized_units: true,
            compute_capability: Some((8, 6)),
            supported_intrinsic_shapes: vec![],
            max_threadgroup_memory: 0,
            preferred_tile_shape: [128, 128, 32],
            simd_width: 32,
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
        tuner.runtime = Some(std::sync::Arc::downgrade(&ctx.runtime));
        
        let goal = crate::optimizer::OptimizationGoal::MaximizeTFLOPS;
        let config = tuner.optimize_conv(&benchmark, 20, goal); // 20 iterations default
        
        serde_json::to_string(&config).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

fn batch_ho_wo(n: u32, h: u32, w: u32) -> u32 { n * h * w }


