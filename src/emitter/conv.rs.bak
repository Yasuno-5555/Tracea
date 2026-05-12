use crate::emitter::traits::UnifiedOpIR;
use crate::backend::cuda::CudaBackend;
use crate::emitter::traits::UnifiedOpType;
use crate::core::op::EpilogueOp;

// Magic Number Helper
pub fn magic_u32(n: u32) -> (u32, u32) {
    if n <= 1 { return (0, 0); }
    if (n & (n - 1)) == 0 { return (0, n.trailing_zeros()); }
    for p in 32..64 {
        let m = ((1u64 << p) + (n as u64) - 1) / (n as u64);
        if m < (1u64 << 32) { 
             return (m as u32, p as u32);
        }
    }
    (0, 0)
}

pub fn calculate_smem_usage(ir: &UnifiedOpIR) -> usize {
    if let UnifiedOpType::Conv2d { .. } = ir.op_type {
        let mt = ir.tiling.m_tile;
        let nt = ir.tiling.n_tile;
        let kt = ir.tiling.k_tile;
        let stages = ir.tiling.num_stages.max(2);
        
        let sa_stride = kt + 8;
        let sb_stride = nt + 8;
        let smem_a_bytes = mt * sa_stride * 2; // half is 2 bytes
        let smem_b_bytes = kt * sb_stride * 2;
        let total_smem_tiles = (smem_a_bytes + smem_b_bytes) * stages;
        
        let hoisting_bytes = mt * (8 + 4 + 4);
        (total_smem_tiles + hoisting_bytes + 1024) as usize
    } else {
        0
    }
}

pub fn generate_conv(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::Conv2d { n: batch, h: h_in, w: w_in, c: c_in, k: k_out, r, s, stride, pad, dilation, layout: _, epilogue: _ } = ir.op_type {
        let h_out = (h_in + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * pad - dilation * (s - 1) - 1) / stride + 1;
        
        let m_gemm = batch * h_out * w_out;
        let n_gemm = k_out;
        let k_gemm = c_in * r * s;

        let mt = ir.tiling.m_tile;
        let nt = ir.tiling.n_tile;
        let kt = ir.tiling.k_tile;
        let num_warps = ir.tiling.force_num_warps.unwrap_or(8);
        let stages = ir.tiling.num_stages.max(2);
        let m_frags = (mt / 16).max(1);
        let n_frags = (nt / 16).max(1);

        let (_c_in_padded, k_gemm_padded) = if let Some(ref strategy) = ir.polyhedral_strategy {
            let mut cp = c_in;
            for (dim_idx, pad) in &strategy.padding_needed {
                if dim_idx == &4 { // dim_c is index 4 in our Conv2d mapper
                    cp += *pad as usize;
                }
            }
            (cp, cp * r * s) 
        } else {
            (c_in, c_in * r * s)
        };
        let m_gemm = batch * h_out * w_out;
        let k_gemm_final = k_gemm_padded;
        
        // Padded strides to avoid bank conflicts
        let sa_stride = kt + 8;
        let sb_stride = nt + 8;
        
        let smem_a_bytes = mt * sa_stride * 2;
        let smem_b_bytes = kt * sb_stride * 2;
        let hoisting_bytes = mt * (8 + 4 + 4);
        let total_smem_tiles = (smem_a_bytes + smem_b_bytes) * stages;
        
        let can_hoist = (total_smem_tiles + hoisting_bytes + 1024) < 96000;

        let (hw_magic, hw_shift) = magic_u32((h_out * w_out) as u32);
        let (w_magic, w_shift) = magic_u32(w_out as u32);
        let (sic_magic, sic_shift) = magic_u32((s * c_in) as u32);
        let (c_magic, c_shift) = magic_u32(c_in as u32);
        
        let mut epilogue_args = String::new();
        let mut epilogue_apply = String::new();
        
        // Use Ampere path if possible
        let use_ampere = (c_in % 8 == 0) && (k_out % 8 == 0);

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
{primitives}

using namespace nvcuda;

#define MT {mt}
#define NT {nt}
#define KT {kt}
#define STAGES {stages}
#define N_WARPS {num_warps}

struct ConvParams {{
    int batch, h_in, w_in, c_in, k_out;
    int h_out, w_out, r_sz, s_sz;
    int stride, pad, dilation;
    unsigned int hw_m, hw_s;
    unsigned int w_m, w_s;
    unsigned int sic_m, sic_s;
    unsigned int c_m, c_s;
}};

__device__ __forceinline__ void fast_divmod(int n, unsigned int m, int s, int d, int& q, int& r) {{
    if (d == 1) {{ q = n; r = 0; }}
    else if (m == 0) {{ q = n >> s; r = n & (d - 1); }}
    else {{
        q = (int)(((unsigned long long)n * m) >> (32 + s));
        r = n - q * d;
    }}
}}

extern "C" __global__ void kernel_main(
    const half* __restrict__ Input,
    const half* __restrict__ Weight,
    half* __restrict__ Output{epilogue_args},
    ConvParams p
) {{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ char smem[];
    half* sA = (half*)smem;
    half* sB = (half*)(smem + MT * {sa_stride} * STAGES * 2);
    float* sOut = (float*)smem;

    int m_block = blockIdx.x * MT;
    int n_block = blockIdx.y * NT;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[{n_frags}];
    #pragma unroll
    for(int i=0; i<{n_frags}; ++i) nvcuda::wmma::fill_fragment(acc[i], 0.0f);

    // Hybrid Hoisting Logic
    __shared__ long long block_base_off[MT];
    __shared__ int block_ho[MT], block_wo[MT];

    if ({can_hoist}) {{
        for (int i = tid; i < MT; i += N_WARPS * 32) {{
            int m_glob = m_block + i;
            if (m_glob < {m_gemm}) {{
                int bn, ho, wo, rem_m;
                fast_divmod(m_glob, p.hw_m, p.hw_s, p.h_out * p.w_out, bn, rem_m);
                fast_divmod(rem_m, p.w_m, p.w_s, p.w_out, ho, wo);
                block_base_off[i] = (long long)bn * p.h_in * p.w_in * p.c_in;
                block_ho[i] = ho; block_wo[i] = wo;
            }}
        }}
        __syncthreads();
    }}

    int total_k_tiles = ({k_gemm} + KT - 1) / KT;

    // Pipeline
    for (int k_tile = 0; k_tile < total_k_tiles + STAGES - 1; ++k_tile) {{
        if (k_tile + STAGES - 1 < total_k_tiles) {{
            int tk = k_tile + STAGES - 1;
            int s_in = tk % STAGES;
            int k_tile_start = tk * KT;
            
            // Load sA
            for (int i = tid; i < (MT * KT) / 8; i += N_WARPS * 32) {{
                int m_local = (i * 8) / KT; int k_local = (i * 8) % KT;
                half* dst = sA + s_in * MT * {sa_stride} + m_local * {sa_stride} + k_local;
                int m_glob = m_block + m_local;
                if (m_glob < {m_gemm}) {{
                    int r_filt, s_filt, ci, rem_sic, k_glob = k_tile_start + k_local;
                    fast_divmod(k_glob, p.sic_m, p.sic_s, p.s_sz * p.c_in, r_filt, rem_sic);
                    fast_divmod(rem_sic, p.c_m, p.c_s, p.c_in, s_filt, ci);
                    
                    int hi, wi; long long base_off;
                    if ({can_hoist}) {{
                        hi = block_ho[m_local] * p.stride - p.pad + r_filt * p.dilation;
                        wi = block_wo[m_local] * p.stride - p.pad + s_filt * p.dilation;
                        base_off = block_base_off[m_local];
                    }} else {{
                        int bn, ho, wo, rem_m;
                        fast_divmod(m_glob, p.hw_m, p.hw_s, p.h_out * p.w_out, bn, rem_m);
                        fast_divmod(rem_m, p.w_m, p.w_s, p.w_out, ho, wo);
                        hi = ho * p.stride - p.pad + r_filt * p.dilation;
                        wi = wo * p.stride - p.pad + s_filt * p.dilation;
                        base_off = (long long)bn * p.h_in * p.w_in * p.c_in;
                    }}
                    bool pred = (hi >= 0 && hi < p.h_in && wi >= 0 && wi < p.w_in);
                    const half* src_ptr = Input + base_off + (long long)(hi * p.w_in + wi) * p.c_in + ci;
                    
                    if (p.c_in % 8 == 0) {{
                        cp_async_ampere(dst, src_ptr, pred);
                    }} else {{
                        // Virtual Padding Strategy: Use predicated vector load
                        if (pred) {{
                            int remaining = p.c_in - ci;
                            float4 v = load_float4_predicated((const float*)src_ptr, remaining);
                            float4 v2 = load_float4_predicated(((const float*)src_ptr) + 4, remaining - 4);
                            *((float4*)dst) = v;
                            *((float4*)(dst + 4)) = v2;
                        }} else {{
                            *((float4*)dst) = make_float4(0,0,0,0);
                            *((float4*)(dst+4)) = make_float4(0,0,0,0);
                        }}
                    }}
                }} else {{
                    *((float4*)dst) = make_float4(0,0,0,0);
                    *((float4*)(dst+4)) = make_float4(0,0,0,0);
                }}
            }}
            // Load sB
            for (int i = tid; i < (KT * NT) / 8; i += N_WARPS * 32) {{
                int k_local = (i * 8) / NT; int n_local = (i * 8) % NT;
                half* dst = sB + s_in * KT * {sb_stride} + k_local * {sb_stride} + n_local;
                int k_glob = k_tile_start + k_local; int n_glob = n_block + n_local;
                const half* src_ptr = Weight + (long long)k_glob * p.k_out + n_glob;
                
                if (p.k_out % 8 == 0) {{
                    cp_async_ampere(dst, src_ptr, (k_glob < {k_gemm} && n_glob < p.k_out));
                }} else {{
                    if (k_glob < {k_gemm} && n_glob < p.k_out) {{
                        #pragma unroll
                        for(int v=0; v<8; ++v) dst[v] = src_ptr[v];
                    }} else {{
                        #pragma unroll
                        for(int v=0; v<8; ++v) dst[v] = 0.0f;
                    }}
                }}
            }}
            cp_async_commit_group();
        }}
        
        if (k_tile < total_k_tiles) {{
            cp_async_wait_group<STAGES - 2>();
            __syncthreads();
            int stage = k_tile % STAGES;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_A[2];
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_B[2];
            nvcuda::wmma::load_matrix_sync(frag_A[0], sA + stage * MT * {sa_stride} + warp_id * 16 * {sa_stride}, {sa_stride});
            nvcuda::wmma::load_matrix_sync(frag_A[1], sA + stage * MT * {sa_stride} + warp_id * 16 * {sa_stride} + 16, {sa_stride});
            #pragma unroll
            for(int j=0; j<{n_frags}; ++j) {{
                nvcuda::wmma::load_matrix_sync(frag_B[0], sB + stage * KT * {sb_stride} + j * 16, {sb_stride});
                nvcuda::wmma::load_matrix_sync(frag_B[1], sB + stage * KT * {sb_stride} + 16 * {sb_stride} + j * 16, {sb_stride});
                nvcuda::wmma::mma_sync(acc[j], frag_A[0], frag_B[0], acc[j]);
                nvcuda::wmma::mma_sync(acc[j], frag_A[1], frag_B[1], acc[j]);
            }}
        }}
    }}

    __syncthreads();
    
    // Alignment-Safe Epilogue
    #pragma unroll
    for(int j=0; j<{n_frags}; ++j) {{
        int m_glob_warp = m_block + warp_id * 16;
        int n_glob_tile = n_block + j * 16;
        if (m_glob_warp < {m_gemm} && n_glob_tile < p.k_out) {{
            nvcuda::wmma::store_matrix_sync(sOut + warp_id * 256, acc[j], 16, nvcuda::wmma::mem_row_major);
            __syncwarp();
            half* p_half = (half*)smem + warp_id * 256;
            for(int k=lane_id; k<256; k+=32) p_half[k] = (half)sOut[warp_id * 256 + k];
            __syncwarp();
            
            int r_idx = lane_id / 2; int c_start = (lane_id % 2) * 8;
            int m_glob = m_glob_warp + r_idx;
            int n_glob = n_glob_tile + c_start;
            
            if (m_glob < {m_gemm}) {{
                // Robust Store: check OC alignment
                if (p.k_out % 8 == 0 && n_glob + 8 <= p.k_out) {{
                    #pragma unroll
                    for(int cc=0; cc<8; ++cc) {{
                        float val = p_half[r_idx * 16 + c_start + cc];
                        int n_glob_c = n_glob + cc;
                        {epilogue_apply}
                        p_half[r_idx * 16 + c_start + cc] = (half)val;
                    }}
                    *((uint4*)&Output[(long long)m_glob * p.k_out + n_glob]) = *((uint4*)&p_half[r_idx * 16 + c_start]);
                }} else {{
                    #pragma unroll
                    for(int cc=0; cc<8; ++cc) {{
                        if (n_glob + cc < p.k_out) {{
                            float val = p_half[r_idx * 16 + c_start + cc];
                            int n_glob_c = n_glob + cc;
                            {epilogue_apply}
                            Output[(long long)m_glob * p.k_out + n_glob + cc] = (half)val;
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
    "#, 
        epilogue_defs=include_str!("../kernels/gpu/epilogue.cuh"),
        primitives=CudaBackend::get_primitive_defs(),
        mt=mt, nt=nt, kt=kt, stages=stages, num_warps=num_warps,
        sa_stride=sa_stride, sb_stride=sb_stride,
        can_hoist=if can_hoist { "true" } else { "false" },
        m_gemm=m_gemm, k_gemm=k_gemm_final,
        epilogue_args=epilogue_args, epilogue_apply=epilogue_apply
        )
    } else {
        panic!("Using conv emitter for non-conv op");
    }
}
