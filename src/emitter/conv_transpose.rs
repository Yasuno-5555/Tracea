use crate::emitter::traits::UnifiedOpIR;
use crate::backend::cuda::CudaBackend;
use crate::emitter::traits::UnifiedOpType;
use crate::core::op::EpilogueOp;

// Magic Number Helper (shared with conv.rs)
fn magic_u32(n: u32) -> (u32, u32) {
    if n == 0 { return (1, 0); }
    let nc = n as u64;
    for p in 32..64 {
        let m = ((1u64 << p) + nc - 1) / nc;
        if m < (1u64 << 32) { 
             return (m as u32, p as u32);
        }
    }
    (1, 0)
}


/// Generate ConvTranspose2d (Deconvolution) kernel using implicit GEMM approach.
/// 
/// This implementation reuses ~90% of Conv2d logic with key differences:
/// 1. Output shape calculation: H_out = (H_in - 1) * stride - 2 * pad + kernel_size + output_padding
/// 2. Input index calculation: inverted mapping from output to input coordinates
/// 
/// Constraints (v3.1):
/// - groups = 1
/// - dilation = 1
/// - padding_mode = "zeros"
/// - dtype = FP32
pub fn generate_conv_transpose(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::ConvTranspose2d { 
        n: batch, h: h_in, w: w_in, c: c_in, k: k_out, 
        r, s, stride, pad, output_padding, layout: _ 
    } = ir.op_type {
        // ConvTranspose2d output shape:
        // H_out = (H_in - 1) * stride - 2 * pad + kernel_size + output_padding
        let h_out = (h_in - 1) * stride - 2 * pad + r + output_padding;
        let w_out = (w_in - 1) * stride - 2 * pad + s + output_padding;
        
        // Implicit GEMM dimensions for transposed convolution:
        // M = batch * h_out * w_out (output positions)
        // N = k_out (output channels)
        // K = c_in * r * s (input channels * kernel area)
        let m_gemm = batch * h_out * w_out;
        let n_gemm = k_out;
        let k_gemm = c_in * r * s;

        let mt = ir.tiling.m_tile;
        let nt = ir.tiling.n_tile;
        let kt = ir.tiling.k_tile;
        let num_warps = ir.tiling.force_num_warps.unwrap_or(4);
        let stages = ir.tiling.num_stages;
        
        let strategy = ir.conv_magic_strategy.unwrap_or(crate::core::config::MagicNumberStrategy::Standard);
        let (hw_magic, hw_shift) = magic_u32((h_out * w_out) as u32);
        let (w_magic, w_shift) = magic_u32(w_out as u32);
        let (rs_magic, rs_shift) = magic_u32((r * s) as u32);
        let (s_magic, s_shift) = magic_u32(s as u32);
        
        let div_helper = match strategy {
            crate::core::config::MagicNumberStrategy::PowerOfTwo => {
                format!(r#"
__device__ __forceinline__ void fast_divmod(int val, int magic, int shift, int divisor, int& div, int& mod) {{
    div = val >> shift;
    mod = val & (divisor - 1);
}}
"#)
            },
            crate::core::config::MagicNumberStrategy::FastSmall => {
                format!(r#"
__device__ __forceinline__ void fast_divmod(int val, int magic, int shift, int divisor, int& div, int& mod) {{
    div = (int)(((unsigned int)val * (unsigned int)magic) >> shift);
    mod = val - div * divisor;
}}
"#)
            },
            crate::core::config::MagicNumberStrategy::Standard => {
                format!(r#"
__device__ __forceinline__ void fast_divmod(int val, unsigned int magic, unsigned int shift, int divisor, int& div, int& mod) {{
    unsigned long long res = (unsigned long long)(unsigned int)val * (unsigned long long)magic;
    div = (int)(res >> shift);
    mod = val - div * divisor;
}}
"#)
            }

        };

        let a_stride = kt + 8;
        let b_stride = nt + 8;

        let mut epilogue_args = String::new();
        let mut epilogue_apply = String::new();
        
        for (i, op) in ir.tiling.epilogue.iter().enumerate() {
            match op {
                EpilogueOp::BiasAdd { .. } => {
                    epilogue_args.push_str(&format!(", const float* __restrict__ bias_{}", i));
                    epilogue_apply.push_str(&format!("tracea::epilogue::BiasAdd op_{}; op_{}.bias = bias_{}; val = op_{}(val, n_glob);\n", i, i, i, i));
                }
                EpilogueOp::ReLU => {
                    epilogue_apply.push_str(&format!("tracea::epilogue::ReLU op_{}; val = op_{}(val);\n", i, i));
                }
                EpilogueOp::Gelu => {
                    epilogue_apply.push_str(&format!("tracea::epilogue::Gelu op_{}; val = op_{}(val);\n", i, i));
                }
                EpilogueOp::SiLU => {
                    epilogue_apply.push_str(&format!("tracea::epilogue::SiLU op_{}; val = op_{}(val);\n", i, i));
                }
                EpilogueOp::ResidualAdd { .. } => {
                    epilogue_args.push_str(&format!(", const float* __restrict__ residual_{}", i));
                    epilogue_apply.push_str(&format!("tracea::epilogue::ResidualAdd op_{}; op_{}.residual = residual_{}; val = op_{}(val, (long long)m_glob * K_OUT + n_glob);\n", i, i, i, i));
                }
                EpilogueOp::BiasAddSiLU { .. } => {
                     epilogue_args.push_str(&format!(", const float* __restrict__ bias_{}", i));
                     epilogue_apply.push_str(&format!("tracea::epilogue::BiasAddSiLU op_{}; op_{}.bias = bias_{}; val = op_{}(val, n_glob);\n", i, i, i, i));
                }
                _ => {}
            }
        }

        format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

{epilogue_defs}

using namespace nvcuda;

#define MT {mt}
#define NT {nt}
#define KT {kt}
#define N_WARPS {num_warps}
#define STAGES {stages}

// ConvTranspose2d Constants
#define BATCH {batch}
#define H_IN {h_in}
#define W_IN {w_in}
#define C_IN {c_in}
#define K_OUT {k_out}
#define H_OUT {h_out}
#define W_OUT {w_out}
#define R_SZ {r}
#define S_SZ {s}
#define STRIDE {stride}
#define PAD {pad}
#define OUTPUT_PAD {output_padding}

#define HW_MAGIC {hw_magic}
#define HW_SHIFT {hw_shift}
#define W_MAGIC {w_magic}
#define W_SHIFT {w_shift}
#define RS_MAGIC {rs_magic}
#define RS_SHIFT {rs_shift}
#define S_MAGIC {s_magic}
#define S_SHIFT {s_shift}

{div_helper}

#define A_STRIDE {a_stride}
#define B_STRIDE {b_stride}

{primitives}

// ConvTranspose2d kernel using implicit GEMM
// Input: [BATCH, H_IN, W_IN, C_IN] (NHWC)
// Weight: [C_IN, K_OUT, R_SZ, S_SZ] (for transposed conv, weight layout differs)
// Output: [BATCH, H_OUT, W_OUT, K_OUT] (NHWC)
extern "C" __global__ void __launch_bounds__(N_WARPS * 32, 1) conv_transpose2d_implicit_gemm(
    const float* __restrict__ Input,
    const float* __restrict__ Weight,
    float* __restrict__ Output{epilogue_args}
) {{
    int tid = threadIdx.x;
    
    // Grid Swizzling
    int swizzled_bid = (int)(((long long)blockIdx.x * 101) % gridDim.x);
    int m_block_start = swizzled_bid * MT;
    int n_block_start = blockIdx.y * NT;
    
    extern __shared__ char smem[];
    float* sA = (float*)smem;
    float* sB = (float*)(smem + MT * KT * 4);
    float* sC = (float*)(smem + MT * KT * 4 + KT * NT * 4);

    // Initialize shared accumulator to zero
    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {{
        sC[i] = 0.0f;
    }}
    __syncthreads();

    int total_k_tiles = ({k_gemm} + KT - 1) / KT;
    
    for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {{
        int k_step = k_tile * KT;
        
        // Load A tile
        for (int i = tid; i < MT * KT; i += N_WARPS * 32) {{
            int row = i / KT;
            int col = i % KT;
            int m_glob = m_block_start + row;
            int k_glob = k_step + col;
            float val = 0.0f;
            
            if (m_glob < {m_gemm} && k_glob < {k_gemm}) {{
                int b, ho, wo;
                int rem_m;
                fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                
                int c, rs, rr, ss;
                fast_divmod(k_glob, RS_MAGIC, RS_SHIFT, R_SZ * S_SZ, c, rs);
                fast_divmod(rs, S_MAGIC, S_SHIFT, S_SZ, rr, ss);
                
                int ho_adj = ho + PAD - rr;
                int wo_adj = wo + PAD - ss;
                
                if (ho_adj >= 0 && wo_adj >= 0 && (ho_adj % STRIDE == 0) && (wo_adj % STRIDE == 0)) {{
                    int hi = ho_adj / STRIDE;
                    int wi = wo_adj / STRIDE;
                    if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {{
                        val = Input[((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c];
                    }}
                }}
            }}
            sA[row * KT + col] = val;
        }}
        
        // Load B tile
        for (int i = tid; i < KT * NT; i += N_WARPS * 32) {{
            int row = i / NT;
            int col = i % NT;
            int k_glob = k_step + row;
            int n_glob = n_block_start + col;
            float val = 0.0f;
            if (k_glob < {k_gemm} && n_glob < {n_gemm}) {{
                val = Weight[(long long)k_glob * K_OUT + n_glob];
            }}
            sB[col * KT + row] = val;
        }}
        __syncthreads();
        
        // Compute: SIMT FMA
        for (int k_idx = 0; k_idx < KT; ++k_idx) {{
            int k_glob = k_step + k_idx;
            if (k_glob >= {k_gemm}) break;
            
            for (int i = tid; i < MT * NT; i += N_WARPS * 32) {{
                int m_local = i / NT;
                int n_local = i % NT;
                sC[i] += sA[m_local * KT + k_idx] * sB[n_local * KT + k_idx];
            }}
        }}
        __syncthreads();
    }}
    
    // Store output
    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {{
        int r = i / NT;
        int c = i % NT;
        int m_glob = m_block_start + r;
        int n_glob = n_block_start + c;
        if (m_glob < {m_gemm} && n_glob < {n_gemm}) {{
            float val = sC[i];
            {epilogue_apply}
            Output[(long long)m_glob * K_OUT + n_glob] = val;
        }}
    }}
}}





"#,
        primitives=CudaBackend::get_primitive_defs(), mt=mt, nt=nt, kt=kt, stages=stages, num_warps=num_warps, 
        a_stride=a_stride, b_stride=b_stride,
        m_gemm=m_gemm, n_gemm=n_gemm, k_gemm=k_gemm,
        hw_magic=hw_magic, hw_shift=hw_shift,

        w_magic=w_magic, w_shift=w_shift,
        rs_magic=rs_magic, rs_shift=rs_shift,
        s_magic=s_magic, s_shift=s_shift,
        epilogue_defs=include_str!("../kernels/gpu/epilogue.cuh"),
        epilogue_args=epilogue_args,
        epilogue_apply=epilogue_apply)
    } else {
        panic!("Using conv_transpose emitter for non-ConvTranspose2d op");
    }
}
