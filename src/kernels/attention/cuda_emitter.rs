use crate::core::config::PipelineConfig;
use crate::backend::cuda::CudaBackend;

pub struct FlashAttentionEmitter {
    pub config: PipelineConfig,
}

impl FlashAttentionEmitter {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub fn calculate_smem_layout(config: &PipelineConfig, d: usize) -> (usize, usize, usize, usize, usize, usize) {
        let mt = config.m_tile;
        let nt = config.n_tile;
        let stages = config.num_stages; 
        let stride = d + 8; // BANK PADDING
        
        // Offset mapping:
        // 1. Q (half) [MT x D]
        let q_offset = 128; // alignment
        let sQ_bytes = (mt as usize) * d * 2;
        
        // 2. K (half) [STAGES x NT x STRIDE]
        let k_offset = (q_offset + sQ_bytes + 127) / 128 * 128;
        let sK_bytes = (stages as usize) * (nt as usize) * stride * 2;
        
        // 3. V (half) [STAGES x NT x STRIDE]
        let v_offset = (k_offset + sK_bytes + 127) / 128 * 128;
        let sV_bytes = (v_offset + (stages as usize) * (nt as usize) * stride * 2 + 127) / 128 * 128;
        
        // Note: We reuse smem_K_base for the epilogue (float accumulation staging)
        // MT x D * 4 bytes
        let sO_float_bytes = (mt as usize) * d * 4;
        let total = std::cmp::max(sV_bytes, k_offset + sO_float_bytes);
        
        (total, 0, 0, q_offset, k_offset, v_offset) 
    }

    pub fn generate_kernel(&self, _h: usize, d: usize, is_causal: bool) -> String {
        let mt = self.config.m_tile;
        let nt = self.config.n_tile; 
        let stages = self.config.num_stages;
        let num_warps = self.config.force_num_warps.unwrap_or(mt / 16); 
        let d_over_16 = d / 16;

        let (total_bytes, _, _, q_offset, k_offset, v_offset) = Self::calculate_smem_layout(&self.config, d);

        format!(r#"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// HARDWARE CONSTRAINTS (RTX 3070 Ampere)
#define MAX_SMEM 98304
#define MT {mt}
#define KT {nt}
#define D_VAL {d}
#define STAGES {stages}
#define STRIDE_K ({d} + 8)
#define D_OVER_16 ({d}/16)

{primitives}

__device__ __forceinline__ void check_smem_limits() {{
    static_assert({total_bytes} <= MAX_SMEM, "FlashAttention-2 kernel Smem footprint exceeds 96KB hardware limit");
}}

extern "C" __global__ void __launch_bounds__({num_threads}, 1) flash_attention_v2_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    long long B, long long H, long long S, long long D,
    float scale
) {{
    int tile_idx = blockIdx.x; int h = blockIdx.y; int b = blockIdx.z;
    int tid = threadIdx.x; int warp_id = tid / 32; int lane_id = tid % 32;
    
    long long offset_base = b * H * S * D + h * S * D;
    Q += offset_base; K += offset_base; V += offset_base; O += offset_base;

    extern __shared__ char smem[];
    half* smem_Q = (half*)(smem + {q_offset});
    half* smem_K_base = (half*)(smem + {k_offset});
    half* smem_V_base = (half*)(smem + {v_offset});

    int q_row_glob = tile_idx * MT + warp_id * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q[D_OVER_16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[D_OVER_16];
    for(int k=0; k<D_OVER_16; ++k) wmma::fill_fragment(acc_O[k], 0.0f);
    
    float m_prev[16]; float l_prev[16];
    for(int i=0; i<16; ++i) {{ m_prev[i] = -50000.0f; l_prev[i] = 0.0f; }}

    // Collective Q Load
    for (int idx = tid * 8; idx < MT * D_VAL; idx += blockDim.x * 8) {{
        int r = idx / D_VAL; int c = idx % D_VAL;
        if (tile_idx * MT + r < S)
            *((uint4*)&smem_Q[r * D_VAL + c]) = *((uint4*)&Q[(tile_idx * MT + r) * D + c]);
    }}
    __syncthreads();

    if (q_row_glob < S) {{
        for(int k=0; k<D_OVER_16; ++k) 
            wmma::load_matrix_sync(frag_Q[k], smem_Q + warp_id * 16 * D_VAL + k * 16, D_VAL);
    }}

    int total_tiles = (S + KT - 1) / KT;

    // PROLOGUE (Collective Async)
    for (int s = 0; s < STAGES - 1; ++s) {{
        if (s < total_tiles) {{
            int k_start = s * KT;
            for (int idx = tid * 8; idx < KT * D_VAL; idx += blockDim.x * 8) {{
                int r = idx / D_VAL; int c = idx % D_VAL;
                half* k_dst = smem_K_base + s * KT * STRIDE_K + r * STRIDE_K + c;
                half* v_dst = smem_V_base + s * KT * STRIDE_K + r * STRIDE_K + c;
                cp_async_ampere(k_dst, &K[(k_start + r) * D + c], (k_start + r < S) ? 16 : 0);
                cp_async_ampere(v_dst, &V[(k_start + r) * D + c], (k_start + r < S) ? 16 : 0);
            }}
            cp_async_commit_group();
        }}
    }}

    for (int k_tile = 0; k_tile < total_tiles; ++k_tile) {{
        int next_s = k_tile + STAGES - 1;
        if (next_s < total_tiles) {{
            int stage = next_s % STAGES;
            int k_start = next_s * KT;
            for (int idx = tid * 8; idx < KT * D_VAL; idx += blockDim.x * 8) {{
                int r = idx / D_VAL; int c = idx % D_VAL;
                half* k_dst = smem_K_base + stage * KT * STRIDE_K + r * STRIDE_K + c;
                half* v_dst = smem_V_base + stage * KT * STRIDE_K + r * STRIDE_K + c;
                cp_async_ampere(k_dst, &K[(k_start + r) * D + c], (k_start + r < S) ? 16 : 0);
                cp_async_ampere(v_dst, &V[(k_start + r) * D + c], (k_start + r < S) ? 16 : 0);
            }}
            cp_async_commit_group();
        }}
        
        cp_async_wait_group<STAGES - 1>();
        __syncthreads();

        if (q_row_glob < S) {{
            int stage = k_tile % STAGES;
            half* my_sK = smem_K_base + stage * KT * STRIDE_K;
            half* my_sV = smem_V_base + stage * KT * STRIDE_K;

            for (int step = 0; step < KT / 16; ++step) {{
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S;
                wmma::fill_fragment(acc_S, 0.0f);
                for(int k=0; k<D_OVER_16; ++k) {{
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;
                    wmma::load_matrix_sync(frag_K, my_sK + step * 16 * STRIDE_K + k * 16, STRIDE_K);
                    wmma::mma_sync(acc_S, frag_Q[k], frag_K, acc_S);
                }}

                float m0 = fmaxf(acc_S.x[0], fmaxf(acc_S.x[1], fmaxf(acc_S.x[4], acc_S.x[5])));
                float m1 = fmaxf(acc_S.x[2], fmaxf(acc_S.x[3], fmaxf(acc_S.x[6], acc_S.x[7])));
                unsigned int mask = 0xffffffff;
                for(int i=1; i<4; i*=2) {{
                    m0 = fmaxf(m0, __shfl_xor_sync(mask, m0, i));
                    m1 = fmaxf(m1, __shfl_xor_sync(mask, m1, i));
                }}
                m0 *= scale; m1 *= scale;

                // Causal implemented if needed
                if ({is_causal}) {{
                    // int col_glob_start = k_tile * KT + step * 16;
                    // ...
                }}

                float m_next[2]; 
                m_next[0] = fmaxf(m_prev[lane_id/4], m0);
                m_next[1] = fmaxf(m_prev[(lane_id/4)+8], m1);

                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;
                for(int i=0; i<8; ++i) {{
                    float val = acc_S.x[i] * scale;
                    float sub_m = (i==0||i==1||i==4||i==5) ? m_next[0] : m_next[1];
                    frag_P.x[i] = __float2half(expf(val - sub_m));
                }}

                float r0 = expf(m_prev[lane_id/4] - m_next[0]);
                float r1 = expf(m_prev[(lane_id/4)+8] - m_next[1]);
                for(int k=0; k<D_OVER_16; ++k) {{
                    acc_O[k].x[0] *= r0; acc_O[k].x[1] *= r0; acc_O[k].x[4] *= r0; acc_O[k].x[5] *= r0;
                    acc_O[k].x[2] *= r1; acc_O[k].x[3] *= r1; acc_O[k].x[6] *= r1; acc_O[k].x[7] *= r1;
                }}

                for(int k=0; k<D_OVER_16; ++k) {{
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V;
                    wmma::load_matrix_sync(frag_V, my_sV + step * 16 * STRIDE_K + k * 16, STRIDE_K);
                    wmma::mma_sync(acc_O[k], frag_P, frag_V, acc_O[k]);
                }}

                float s0 = 0, s1 = 0;
                for(int i=0; i<8; ++i) {{
                    float val = __half2float(frag_P.x[i]);
                    if(i==0||i==1||i==4||i==5) s0 += val; else s1 += val;
                }}
                for(int i=1; i<4; i*=2) {{
                    s0 += __shfl_xor_sync(mask, s0, i);
                    s1 += __shfl_xor_sync(mask, s1, i);
                }}

                float p_cap = (lane_id % 4 == 0) ? s0 : 0.0f;
                float p1_cap = (lane_id % 4 == 0) ? s1 : 0.0f;
                float m_cap = (lane_id % 4 == 0) ? m_next[0] : -50000.0f;
                float m1_cap = (lane_id % 4 == 0) ? m_next[1] : -50000.0f;

                for(int i=0; i<16; ++i) {{
                    int src = (i % 8) * 4;
                    float ps = __shfl_sync(mask, (i < 8) ? p_cap : p1_cap, src);
                    float mn = __shfl_sync(mask, (i < 8) ? m_cap : m1_cap, src);
                    l_prev[i] = l_prev[i] * expf(m_prev[i] - mn) + ps;
                    m_prev[i] = mn;
                }}
            }}
        }}
    }}

    __syncthreads();
    if (q_row_glob < S) {{
        float* my_o_smem = (float*)smem_K_base + warp_id * 16 * D_VAL;
        for(int k=0; k<D_OVER_16; ++k) 
            wmma::store_matrix_sync(my_o_smem + k * 16, acc_O[k], D_VAL, wmma::mem_row_major);
        __syncwarp();
        if (lane_id < 16) {{
            float lp = l_prev[lane_id];
            for (int k=0; k<D_OVER_16; ++k) {{
                float* s_row = my_o_smem + lane_id * D_VAL + k * 16;
                half* g_row = O + (q_row_glob + lane_id) * D + k * 16;
                for(int c=0; c<16; ++c) g_row[c] = __float2half(s_row[c] / (lp + 1e-9f));
            }}
        }}
    }}
}}
"# , 
        mt=mt, nt=nt, num_threads=num_warps*32, stages=stages, d=d, 
        q_offset=q_offset, k_offset=k_offset, v_offset=v_offset, 
        is_causal=if is_causal { "true" } else { "false" }, 
        primitives=crate::backend::cuda::CudaBackend::get_primitive_defs())
    }
}
