use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use crate::core::config::{PipelineConfig, LayoutPolicy, SoftmaxGranularity};
use crate::core::op::EpilogueOp;
use crate::kernels::attention::cuda_emitter::FlashAttentionEmitter;

// Minimal Repr C structs
#[repr(C)]
pub struct TraceaResult {
    pub success: bool,
    pub error_msg: *mut c_char,
    pub score: f32,
    pub config_ptr: *mut c_void, 
}

#[repr(C)]
pub enum TraceaEpilogueKind {
    Identity = 0,
    BiasAdd = 1,
    ReLU = 2,
    Gelu = 3,
    SiLU = 4,
    BiasReLU = 5,
    BiasSiLU = 6,
    Residual = 7,
    ResidualReLU = 8,
}

#[repr(C)]
pub enum TraceaSoftmaxGranularity {
    Auto = 0,
    PerTile = 1,
    PerTwoTiles = 2,
}

#[repr(C)]
pub struct TraceaConv2dParams {
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding_h: u32,
    pub padding_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub groups: u32,
    pub epilogue: TraceaEpilogueKind,
    pub stream: *mut c_void,
}

#[repr(C)]
pub struct TraceaConvTranspose2dParams {
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding_h: u32,
    pub padding_w: u32,
    pub output_padding_h: u32,
    pub output_padding_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub groups: u32,
    pub epilogue: TraceaEpilogueKind,
    pub stream: *mut c_void,
}

#[repr(C)]
pub struct TraceaGemmParams {
    pub epilogue: TraceaEpilogueKind,
    pub stream: *mut c_void,
}

#[repr(C)]
pub struct TraceaAttentionParams {
    pub causal: u8,
    pub softmax_mode: TraceaSoftmaxGranularity,
    pub scale: f32, 
    pub stream: *mut c_void,
}

#[repr(C)]
pub struct TraceaTensorView {
    pub ptr: *mut c_void,
    pub rank: u32,
    pub shape: *const u64,
    pub stride: *const i64,
    pub dtype: i32,
    pub device_id: i32,
}

#[repr(C)]
pub enum TraceaStatus {
    Success = 0,
    InvalidParams = 1,
    UnsupportedConfig = 2,
    CudaError = 3,
    CpuError = 4,
    UnknownError = 99,
}

// Helpers
static mut GLOBAL_RUNTIME: Option<Arc<RuntimeManager>> = None;

#[no_mangle]
pub extern "C" fn tracea_init() -> c_int {
    match RuntimeManager::init(Some(DeviceBackend::Cuda)) {
        Ok(rt) => {
            unsafe { GLOBAL_RUNTIME = Some(rt); }
            0
        }
        Err(_) => 1
    }
}

#[no_mangle]
pub extern "C" fn tracea_shutdown() {
    unsafe { GLOBAL_RUNTIME = None; }
}

#[no_mangle]
pub extern "C" fn tracea_free_string(s: *mut c_char) {
    if s.is_null() { return; }
    unsafe { let _ = CString::from_raw(s); }
}

// --- Internal Helpers ---

impl TraceaTensorView {
    unsafe fn dims(&self) -> Vec<usize> {
        if self.shape.is_null() { return vec![]; }
        let mut d = Vec::with_capacity(self.rank as usize);
        for i in 0..self.rank {
            d.push(*self.shape.add(i as usize) as usize);
        }
        d
    }

    fn to_arg(&self) -> KernelArg {
        KernelArg::Usize(self.ptr as usize)
    }
}

unsafe fn parse_epilogue_args(
    kind: TraceaEpilogueKind, 
    bias: *const TraceaTensorView, 
    residual: *const TraceaTensorView
) -> (Vec<EpilogueOp>, Vec<KernelArg>) {
    let mut ops = Vec::new();
    let mut args = Vec::new();

    match kind {
        TraceaEpilogueKind::Identity => {},
        TraceaEpilogueKind::BiasAdd => {
            if !bias.is_null() {
                ops.push(EpilogueOp::BiasAdd { bias_ptr: 0 });
                args.push((*bias).to_arg());
            }
        },
        TraceaEpilogueKind::ReLU => {
            ops.push(EpilogueOp::ReLU);
        },
        TraceaEpilogueKind::Gelu => {
            ops.push(EpilogueOp::Gelu);
        },
        TraceaEpilogueKind::SiLU => {
            ops.push(EpilogueOp::SiLU);
        },
        TraceaEpilogueKind::BiasReLU => {
            if !bias.is_null() {
                ops.push(EpilogueOp::BiasAdd { bias_ptr: 0 });
                args.push((*bias).to_arg());
            }
            ops.push(EpilogueOp::ReLU);
        },
        TraceaEpilogueKind::BiasSiLU => {
             if !bias.is_null() {
                ops.push(EpilogueOp::BiasAddSiLU { bias_ptr: 0 });
                args.push((*bias).to_arg());
            }
        },
        TraceaEpilogueKind::Residual => {
             if !residual.is_null() {
                ops.push(EpilogueOp::ResidualAdd { residual_ptr: 0 });
                args.push((*residual).to_arg());
            }
        },
        TraceaEpilogueKind::ResidualReLU => {
             if !residual.is_null() {
                ops.push(EpilogueOp::ResidualAdd { residual_ptr: 0 });
                args.push((*residual).to_arg());
            }
            ops.push(EpilogueOp::ReLU);
        },
    }
    (ops, args)
}

fn get_runtime() -> Option<Arc<RuntimeManager>> {
    unsafe { GLOBAL_RUNTIME.clone() }
}

// --- Implementations ---

#[no_mangle]
pub extern "C" fn tracea_conv2d(
    x: TraceaTensorView, w: TraceaTensorView, 
    bias: *const TraceaTensorView, residual: *const TraceaTensorView,
    out: *mut TraceaTensorView,
    params: TraceaConv2dParams
) -> TraceaStatus {
    let runtime = if let Some(rt) = get_runtime() { rt } else { return TraceaStatus::UnknownError; };
    
    // N, H, W, C
    let x_dims = unsafe { x.dims() };
    let w_dims = unsafe { w.dims() };
    if x_dims.len() != 4 || w_dims.len() != 4 { return TraceaStatus::InvalidParams; }

    let n = x_dims[0];
    let h_in = x_dims[1];
    let w_in = x_dims[2];
    let c = x_dims[3];
    let k = w_dims[0]; // K, R, S, C (Tracea usually implies layout)
    // Wait, Python binding assumed KRSC? Or implicit?
    // Let's assume standard KRSC or similar.
    let r = w_dims[1];
    let s = w_dims[2];

    let mut config = PipelineConfig::new(2, 64, 64, 16);
    unsafe {
        let (e_ops, e_args) = parse_epilogue_args(params.epilogue, bias, residual);
        config.epilogue = e_ops;
        
        // Launch Config
        let backend = DeviceBackend::Cuda;
        let emitter = UniversalEmitter::new(backend);
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Conv2d {
                n, c, h: h_in, w: w_in, k, r, s,
                stride: params.stride_h as usize, // Assume h=w for now or update UnifiedOpType?
                // UnifiedOpType only checks stride (scalar)?
                // Checking UnifiedOpType definition (Step 2086): stride, pad, dilation are usize (scalar).
                // So we assume symmetric.
                pad: params.padding_h as usize,
                dilation: params.dilation_h as usize,
                layout: LayoutPolicy::NHWC,
            },
            precison: "f16".to_string(), // Fixed f16 for now
            tiling: config.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };

        let source = emitter.generate(ir);
        // Kernel name match!
        let kernel_name = "conv2d_implicit_gemm";
        let kernel_id = match runtime.compile(&source, kernel_name, backend) {
            Ok(id) => id,
            Err(_) => return TraceaStatus::CudaError,
        };

        // Grid Calc
        let h_out = (h_in + 2*params.padding_h as usize - r) / params.stride_h as usize + 1;
        let w_out = (w_in + 2*params.padding_w as usize - s) / params.stride_h as usize + 1;
        let m_gemm = n * h_out * w_out;
        let n_gemm = k;
        let mt = config.m_tile as usize;
        let nt = config.n_tile as usize;
        let grid = ((m_gemm + mt - 1) / mt, (n_gemm + nt - 1) / nt, 1);
        let block = (128, 1, 1); // 4 Warps

        // Smem Calc
        let kt = config.k_tile as usize;
        let stages = config.num_stages as usize;
        let a_stride = kt + 8;
        let b_stride = nt + 8;
        let smem_bytes = 128 + (mt * a_stride * 2 + kt * b_stride * 2) * stages;

        let mut args = vec![x.to_arg(), w.to_arg(), (*out).to_arg()];
        args.extend(e_args);

        match runtime.launch(kernel_id, 
            (grid.0 as u32, grid.1 as u32, grid.2 as u32),
            block, smem_bytes as u32, args) 
        {
            Ok(_) => TraceaStatus::Success,
            Err(_) => TraceaStatus::CudaError,
        }
    }
}

#[no_mangle]
pub extern "C" fn tracea_conv_transpose2d(
    x: TraceaTensorView, w: TraceaTensorView, 
    bias: *const TraceaTensorView, residual: *const TraceaTensorView,
    out: *mut TraceaTensorView,
    params: TraceaConvTranspose2dParams
) -> TraceaStatus {
    let runtime = if let Some(rt) = get_runtime() { rt } else { return TraceaStatus::UnknownError; };
    
    let x_dims = unsafe { x.dims() };
    let w_dims = unsafe { w.dims() }; // K, R, S, C (Transposed Weight Layout?)
    // In Python binding: w was (K, R, S, C). 
    
    let n = x_dims[0];
    let h_in = x_dims[1];
    let w_in = x_dims[2];
    let c = x_dims[3];
    let k = w_dims[0];
    let r = w_dims[1];
    let s = w_dims[2];

    let mut config = PipelineConfig::new(2, 64, 64, 16);
    unsafe {
        let (e_ops, e_args) = parse_epilogue_args(params.epilogue, bias, residual);
        config.epilogue = e_ops;
        
        let backend = DeviceBackend::Cuda;
        let emitter = UniversalEmitter::new(backend);
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::ConvTranspose2d {
                n, c, h: h_in, w: w_in, k, r, s,
                stride: params.stride_h as usize,
                pad: params.padding_h as usize,
                output_padding: params.output_padding_h as usize,
                layout: LayoutPolicy::NHWC,
            },
            precison: "f32".to_string(), // ConvTranspose used f32 in Python
            tiling: config.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };

        let source = emitter.generate(ir);
        let kernel_name = "conv_transpose2d_implicit_gemm";
        let kernel_id = match runtime.compile(&source, kernel_name, backend) {
            Ok(id) => id,
            Err(_) => return TraceaStatus::CudaError,
        };

        // Grid Calc
        let str = params.stride_h as usize;
        let pad = params.padding_h as usize;
        let opad = params.output_padding_h as usize;
        let h_out = (h_in - 1) * str - 2 * pad + r + opad;
        let w_out = (w_in - 1) * str - 2 * pad + s + opad;
        let m_gemm = n * h_out * w_out;
        let n_gemm = k;
        let mt = config.m_tile as usize;
        let nt = config.n_tile as usize;
        let grid = ((m_gemm + mt - 1) / mt, (n_gemm + nt - 1) / nt, 1);
        let block = (128, 1, 1);

        // Smem Calc
        let kt = config.k_tile as usize;
        let stages = config.num_stages as usize;
        let a_stride = kt + 8;
        let b_stride = nt + 8;
        let smem_bytes = 128 + (mt * a_stride * 2 + kt * b_stride * 2) * stages;

        let mut args = vec![x.to_arg(), w.to_arg(), (*out).to_arg()];
        args.extend(e_args);

        match runtime.launch(kernel_id, 
            (grid.0 as u32, grid.1 as u32, grid.2 as u32),
            block, smem_bytes as u32, args) 
        {
            Ok(_) => TraceaStatus::Success,
            Err(_) => TraceaStatus::CudaError,
        }
    }
}

#[no_mangle]
pub extern "C" fn tracea_gemm(
    a: TraceaTensorView, b: TraceaTensorView, 
    bias: *const TraceaTensorView, residual: *const TraceaTensorView,
    c: *mut TraceaTensorView,
    params: TraceaGemmParams
) -> TraceaStatus {
    let runtime = if let Some(rt) = get_runtime() { rt } else { return TraceaStatus::UnknownError; };
    
    let a_dims = unsafe { a.dims() }; // M, K
    let b_dims = unsafe { b.dims() }; // K, N
    if a_dims.len() != 2 || b_dims.len() != 2 { return TraceaStatus::InvalidParams; }
    let m = a_dims[0];
    let k = a_dims[1];
    let n = b_dims[1];

    let mut config = PipelineConfig::new(2, 64, 64, 16);
    unsafe {
        let (e_ops, e_args) = parse_epilogue_args(params.epilogue, bias, residual);
        config.epilogue = e_ops;
        config.force_num_warps = Some(5); // Stable config 64x64

        let backend = DeviceBackend::Cuda;
        let emitter = UniversalEmitter::new(backend);
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm { m: m as u32, n: n as u32, k: k as u32 },
            precison: "f16".to_string(),
            tiling: config.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };

        let source = emitter.generate(ir);
        let kernel_name = "gemm_mma_kernel";
        let kernel_id = match runtime.compile(&source, kernel_name, backend) {
            Ok(id) => id,
            Err(_) => return TraceaStatus::CudaError,
        };

        let mt = config.m_tile as usize;
        let nt = config.n_tile as usize;
        let grid = ((n + nt - 1) / nt, (m + mt - 1) / mt, 1);
        let block = (160, 1, 1); 

        // Smem Calc
        let kt = config.k_tile as usize;
        let stages = config.num_stages as usize;
        let a_stride = kt + 8;
        let b_stride = nt + 8;
        let smem_bytes = 128 + (mt * a_stride * 2 + kt * b_stride * 2) * stages;

        let mut args = vec![a.to_arg(), b.to_arg(), (*c).to_arg(), 
            KernelArg::Int(m as i32), KernelArg::Int(n as i32), KernelArg::Int(k as i32)];
        args.extend(e_args);

        match runtime.launch(kernel_id, 
            (grid.0 as u32, grid.1 as u32, grid.2 as u32),
            block, smem_bytes as u32, args) 
        {
            Ok(_) => TraceaStatus::Success,
            Err(_) => TraceaStatus::CudaError,
        }
    }
}

#[no_mangle]
pub extern "C" fn tracea_attention(
    q: TraceaTensorView, k: TraceaTensorView, v: TraceaTensorView,
    o: *mut TraceaTensorView,
    params: TraceaAttentionParams
) -> TraceaStatus {
    let runtime = if let Some(rt) = get_runtime() { rt } else { return TraceaStatus::UnknownError; };
    
    let q_dims = unsafe { q.dims() }; // B, H, S, D
    if q_dims.len() != 4 { return TraceaStatus::InvalidParams; }
    let b = q_dims[0];
    let h = q_dims[1];
    let s = q_dims[2];
    let d = q_dims[3];
    let causality = params.causal != 0;

    let config = PipelineConfig::new(2, 64, 64, d as u32);

    let backend = DeviceBackend::Cuda;
    let emitter = UniversalEmitter::new(backend);
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::FusedAttention {
            b: b as u32, s: s as u32, d: d as u32, h: h as u32, dh: d as u32,
            causal: causality,
        },
        precison: "f16".to_string(),
        tiling: config.clone(),
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let source = emitter.generate(ir);
    let kernel_name = "flash_attention_v2_kernel";
    let kernel_id = match runtime.compile(&source, kernel_name, backend) {
        Ok(id) => id,
        Err(_) => return TraceaStatus::CudaError,
    };

    let mt = config.m_tile as usize;
    let grid = ((s + mt - 1) / mt, h, b);
    let num_warps = 2 + (mt / 16);
    let block = ((num_warps * 32) as u32, 1, 1);
    
    let (smem_bytes, _, _, _, _, _) = FlashAttentionEmitter::calculate_smem_layout(&config, d);

    let args = vec![q.to_arg(), k.to_arg(), v.to_arg(), unsafe { (*o).to_arg() },
        KernelArg::Usize(b), KernelArg::Usize(h), KernelArg::Usize(s), KernelArg::Usize(d),
        KernelArg::Float(params.scale)];

    match runtime.launch(kernel_id, 
        (grid.0 as u32, grid.1 as u32, grid.2 as u32),
        block, smem_bytes as u32, args) 
    {
        Ok(_) => TraceaStatus::Success,
        Err(_) => TraceaStatus::CudaError,
    }
}


// Stubs for tuning
#[no_mangle]
pub extern "C" fn tracea_tune_gemm(_m: usize, _n: usize, _k: usize) -> TraceaResult {
    TraceaResult { success: false, error_msg: std::ptr::null_mut(), score: 0.0, config_ptr: std::ptr::null_mut() }
}
#[no_mangle]
pub extern "C" fn tracea_tune_fa2(_b: usize, _h: usize, _s: usize, _d: usize, _causal: bool) -> TraceaResult {
    TraceaResult { success: false, error_msg: std::ptr::null_mut(), score: 0.0, config_ptr: std::ptr::null_mut() }
}
#[no_mangle]
pub extern "C" fn tracea_tune_conv2d(
    _n: usize, _c: usize, _h: usize, _w: usize, _k: usize, 
    _r: usize, _s: usize, _stride: usize, _pad: usize, _dilation: usize
) -> TraceaResult {
    TraceaResult { success: false, error_msg: std::ptr::null_mut(), score: 0.0, config_ptr: std::ptr::null_mut() }
}
