use crate::emitter::traits::UnifiedOpIR;
use crate::backend::cuda::CudaBackend;
use crate::emitter::traits::UnifiedOpType;
use crate::core::op::EpilogueOp;

// Magic Number Helper
fn magic_u32(n: u32) -> (u32, u32) {
    if n == 0 { return (1, 0); }
    if (n & (n - 1)) == 0 { return (0, n.trailing_zeros()); }
    let nc = n as u64;
    for p in 32..64 {
        let m = ((1u64 << p) + nc - 1) / nc;
        if m < (1u64 << 32) { 
             return (m as u32, p as u32);
        }
    }
    (1, 0)
}


pub fn generate_conv(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::Conv2d { n: batch, h: h_in, w: w_in, c: c_in, k: k_out, r, s, stride, pad, dilation, layout: _ } = ir.op_type {
        let h_out = (h_in + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * pad - dilation * (s - 1) - 1) / stride + 1;
        
        let m_gemm = batch * h_out * w_out;
        let n_gemm = k_out;
        let k_gemm = c_in * r * s;

        let mt = ir.tiling.m_tile;
        let nt = ir.tiling.n_tile;
        let kt = ir.tiling.k_tile;
        let num_warps = ir.tiling.force_num_warps.unwrap_or(4);
        let stages = ir.tiling.num_stages;
        
        let (hw_magic, hw_shift) = magic_u32((h_out * w_out) as u32);
        let (w_magic, w_shift) = magic_u32(w_out as u32);
        let (s_magic, s_shift) = magic_u32(s as u32);
        let (c_magic, c_shift) = magic_u32(c_in as u32);
        
        // Use robust unsigned fast_divmod
        let div_helper = format!(r#"
__device__ __forceinline__ void fast_divmod(int val, unsigned int magic, unsigned int shift, int divisor, int& div, int& mod) {{
    if (magic == 0) {{
        div = val >> shift;
        mod = val & (divisor - 1);
    }} else {{
        unsigned long long res = (unsigned long long)(unsigned int)val * (unsigned long long)magic;
        div = (int)(res >> shift);
        mod = val - div * divisor;
    }}
}}
"#);

        let a_stride = kt; 
        let b_stride = nt;
        let smem_a_bytes = mt * a_stride * 2;
        let smem_b_bytes = kt * b_stride * 2;
        let smem_c_bytes = mt * nt * 4; // float accumulator

        let mut epilogue_args = String::new();
        let mut epilogue_apply = String::new();
        
        let use_cp_async = (c_in % 8 == 0) && (k_out % 8 == 0);
        let producer_warps = if use_cp_async { 1 } else { 0 };

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
#define PRODUCER_WARPS {producer_warps}
#define USE_CP_ASYNC {use_cp_async_val}

// Conv Constants
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
#define DILATION {dilation}

#define HW_MAGIC {hw_magic}
#define HW_SHIFT {hw_shift}
#define W_MAGIC {w_magic}
#define W_SHIFT {w_shift}
#define S_MAGIC {s_magic}
#define S_SHIFT {s_shift}
#define C_MAGIC {c_magic}
#define C_SHIFT {c_shift}

{div_helper}

#define A_STRIDE {a_stride}
#define B_STRIDE {b_stride}

{primitives}

extern "C" __global__ void __launch_bounds__(N_WARPS * 32, 1) conv2d_implicit_gemm(
    const half* __restrict__ Input,
    const half* __restrict__ Weight,
    half* __restrict__ Output{epilogue_args}
) {{
    int tid = threadIdx.x;
    int warp_id = tid / 32;

#if USE_CP_ASYNC
    bool is_producer = (warp_id < PRODUCER_WARPS);
    
    // Grid Swizzling
    int swizzled_bid = (int)(((long long)blockIdx.x * 101) % gridDim.x);
    int m_block_start = swizzled_bid * MT;
    int n_block_start = blockIdx.y * NT;

    extern __shared__ char smem[];
    int a_smem_offset = 128; 
    int b_smem_offset = a_smem_offset + {smem_a_bytes} * STAGES;
    int c_smem_offset = b_smem_offset + {smem_b_bytes} * STAGES;
    float* sC = (float*)(smem + c_smem_offset);

    // Initialize sC to zero
    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {{
        sC[i] = 0.0f;
    }}
    __syncthreads();

    int total_k_tiles = ({k_gemm} + KT - 1) / KT;

    // PROLOGUE: Load initial stages
    if (is_producer) {{
        for (int s_idx = 0; s_idx < STAGES - 1; ++s_idx) {{
            if (s_idx < total_k_tiles) {{
                half* sA = (half*)(smem + a_smem_offset + s_idx * {smem_a_bytes});
                half* sB = (half*)(smem + b_smem_offset + s_idx * {smem_b_bytes});
                int k_step = s_idx * KT;

                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {{
                    int r_tile = (i * 8) / KT;
                    int k_tile = (i * 8) % KT;
                    int m_glob = m_block_start + r_tile;
                    int k_glob = k_step + k_tile;
                    
                    if (m_glob < {m_gemm} && k_glob < {k_gemm}) {{
                        int b, ho, wo, r_idx, s_idx_rem, c_idx;
                        int rem_m, rem_k;
                        fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                        fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                        fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c_idx);
                        fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r_idx, s_idx_rem);
                        int hi = ho * STRIDE - PAD + r_idx * DILATION;
                        int wi = wo * STRIDE - PAD + s_idx_rem * DILATION;
                        if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {{
                            cp_async_ampere(sA + r_tile * A_STRIDE + k_tile, Input + ((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c_idx, 16);
                        }}
                    }}
                }}
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {{
                    int k_tile = (i * 8) / NT;
                    int n_tile = (i * 8) % NT;
                    int k_glob = k_step + k_tile;
                    int n_glob = n_block_start + n_tile;
                    if (k_glob < {k_gemm} && n_glob < {n_gemm}) {{
                        cp_async_ampere(sB + k_tile * B_STRIDE + n_tile, Weight + (long long)k_glob * K_OUT + n_glob, 16);
                    }}
                }}
                cp_async_commit_group();
            }}
        }}
        cp_async_wait_group<STAGES - 2>();
    }}
    __syncthreads();

    // MAIN LOOP
    for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {{
        if (!is_producer) {{
            int stage = k_tile % STAGES;
            half* sA = (half*)(smem + a_smem_offset + stage * {smem_a_bytes});
            half* sB = (half*)(smem + b_smem_offset + stage * {smem_b_bytes});
            
            // SIMT FMA Compute
            for (int k_idx = 0; k_idx < KT; ++k_idx) {{
                int k_glob = k_tile * KT + k_idx;
                if (k_glob >= {k_gemm}) break;
                
                for (int i = warp_id - PRODUCER_WARPS; i < MT * NT / 32; i += (N_WARPS - PRODUCER_WARPS)) {{
                    int m_local = (i * 32 + (tid % 32)) / NT;
                    int n_local = (i * 32 + (tid % 32)) % NT;
                    if (m_local < MT && n_local < NT) {{
                         sC[m_local * NT + n_local] += (float)sA[m_local * KT + k_idx] * (float)sB[k_idx * B_STRIDE + n_local];
                    }}
                }}
            }}
        }}

        if (is_producer) {{
            int next_k = k_tile + STAGES - 1;
            if (next_k < total_k_tiles) {{
                int stage = next_k % STAGES;
                half* sA = (half*)(smem + a_smem_offset + stage * {smem_a_bytes});
                half* sB = (half*)(smem + b_smem_offset + stage * {smem_b_bytes});
                int k_step = next_k * KT;

                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {{
                    int r_tile = (i * 8) / KT;
                    int k_tile = (i * 8) % KT;
                    int m_glob = m_block_start + r_tile;
                    int k_glob = k_step + k_tile;
                    if (m_glob < {m_gemm} && k_glob < {k_gemm}) {{
                        int b, ho, wo, r_idx, s_idx_rem, c_idx;
                        int rem_m, rem_k;
                        fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                        fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                        fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c_idx);
                        fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r_idx, s_idx_rem);
                        int hi = ho * STRIDE - PAD + r_idx * DILATION;
                        int wi = wo * STRIDE - PAD + s_idx_rem * DILATION;
                        if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {{
                            cp_async_ampere(sA + r_tile * A_STRIDE + k_tile, Input + ((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c_idx, 16);
                        }} else {{
                            // Zero padding for robustness
                            *(uint4*)(sA + r_tile * A_STRIDE + k_tile) = make_uint4(0, 0, 0, 0);
                        }}
                    }}
                }}
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {{
                    int k_tile = (i * 8) / NT;
                    int n_tile = (i * 8) % NT;
                    int k_glob = k_step + k_tile;
                    int n_glob = n_block_start + n_tile;
                    if (k_glob < {k_gemm} && n_glob < {n_gemm}) {{
                        cp_async_ampere(sB + k_tile * B_STRIDE + n_tile, Weight + (long long)k_glob * K_OUT + n_glob, 16);
                    }}
                }}
                cp_async_commit_group();
                cp_async_wait_group<STAGES - 2>();
            }} else {{
                cp_async_wait_group<0>();
            }}
        }}
        __syncthreads();
    }}

    // Epilogue
    if (!is_producer) {{
        for (int i = tid - 32 * PRODUCER_WARPS; i < MT * NT; i += 32 * (N_WARPS - PRODUCER_WARPS)) {{
            int r = i / NT;
            int c = i % NT;
            int m_glob = m_block_start + r;
            int n_glob = n_block_start + c;
            if (m_glob < {m_gemm} && n_glob < {n_gemm}) {{
                float val = sC[i];
                {epilogue_apply}
                Output[(long long)m_glob * K_OUT + n_glob] = (half)val;
            }}
        }}
    }}
#else
    // Fallback for unaligned/non-Ampere (SIMT FMA)
    int m_block_start = blockIdx.x * MT;
    int n_block_start = blockIdx.y * NT;
    extern __shared__ char smem[];
    half* sA = (half*)smem;
    half* sB = (half*)(smem + MT * KT * 2);
    float* sC = (float*)(smem + MT * KT * 2 + KT * NT * 2);

    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {{
        sC[i] = 0.0f;
    }}
    __syncthreads();

    int total_k_tiles = ({k_gemm} + KT - 1) / KT;
    for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {{
        int k_step = k_tile * KT;
        for (int i = tid; i < MT * KT; i += N_WARPS * 32) {{
             int row = i / KT; int col = i % KT;
             int m_glob = m_block_start + row; int k_glob = k_step + col;
             half val = 0.0;
             if (m_glob < {m_gemm} && k_glob < {k_gemm}) {{
                 int b, ho, wo, r_idx, s_idx_rem, c_idx;
                 int rem_m, rem_k;
                 fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                 fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                 fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c_idx);
                 fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r_idx, s_idx_rem);
                 
                 int hi = ho * STRIDE - PAD + r_idx * DILATION; 
                 int wi = wo * STRIDE - PAD + s_idx_rem * DILATION;
                 if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {{
                     val = Input[((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c_idx];
                 }}
             }}
             sA[row * KT + col] = val;
        }}
        for (int i = tid; i < NT * KT; i += N_WARPS * 32) {{
             int row = i / NT; int col = i % NT;
             int k_glob = k_step + row; int n_glob = n_block_start + col;
             half val = 0.0;
             if (k_glob < {k_gemm} && n_glob < {n_gemm}) val = Weight[(long long)k_glob * K_OUT + n_glob];
             sB[col * KT + row] = val; // Note: row as col for transposed load
        }}
        __syncthreads();
        
        for (int k_idx = 0; k_idx < KT; ++k_idx) {{
            int k_glob = k_step + k_idx;
            if (k_glob >= {k_gemm}) break;
            for (int i = tid; i < MT * NT; i += N_WARPS * 32) {{
                int m_local = i / NT;
                int n_local = i % NT;
                sC[i] += (float)sA[m_local * KT + k_idx] * (float)sB[n_local * KT + k_idx];
            }}
        }}
        __syncthreads();
    }}
    
    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {{
        int r = i / NT; int c = i % NT;
        int m_glob = m_block_start + r; int n_glob = n_block_start + c;
        if (m_glob < {m_gemm} && n_glob < {n_gemm}) {{ 
            float val = sC[i];
            {epilogue_apply}
            Output[(long long)m_glob * K_OUT + n_glob] = (half)val;
        }}
    }}
#endif
}}
"#,
        primitives=CudaBackend::get_primitive_defs(), mt=mt, nt=nt, kt=kt, stages=stages, num_warps=num_warps, 
        smem_a_bytes=smem_a_bytes, smem_b_bytes=smem_b_bytes, 
        use_cp_async_val=if use_cp_async { "1" } else { "0" },
        m_gemm=m_gemm, n_gemm=n_gemm, k_gemm=k_gemm,
        producer_warps=producer_warps,
        hw_magic=hw_magic, hw_shift=hw_shift,
        w_magic=w_magic, w_shift=w_shift,
        s_magic=s_magic, s_shift=s_shift,
        c_magic=c_magic, c_shift=c_shift,
        epilogue_defs=include_str!("../kernels/gpu/epilogue.cuh"),
        epilogue_args=epilogue_args,
        epilogue_apply=epilogue_apply,
        dilation=dilation)
    } else {
        panic!("Using conv emitter for non-conv op");
    }
}
