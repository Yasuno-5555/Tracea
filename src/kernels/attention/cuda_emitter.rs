use crate::core::config::PipelineConfig;
use crate::backend::cuda::CudaBackend;

pub struct FlashAttentionEmitter {
    pub config: PipelineConfig,
}

impl FlashAttentionEmitter {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub fn calculate_smem_layout(config: &PipelineConfig, d: usize) -> (usize, usize, usize, usize, usize) {
        let mt = config.m_tile;
        let nt = config.n_tile;
        let stages = config.num_stages; 
        let stride = d + 8; // BANK PADDING
        
        // Barriers
        let barrier_bytes = (stages * 16) as usize;
        let s_offset = (barrier_bytes + 127) / 128 * 128;

        // 2. S / O Buffer (Aliased)
        let pad_s = 8;
        let sS_bytes = (mt * (nt + pad_s) * 4) as usize; // float P scores (padded)
        let sO_bytes = (mt as usize) * d * 4;  // float O results
        let shared_so_bytes = std::cmp::max(sS_bytes, sO_bytes);
        
        let p_offset = s_offset + (shared_so_bytes + 127) / 128 * 128;
        let sP_bytes = (mt * (std::cmp::max(nt, d as u32)) * 2) as usize; // half P scores or Q load buffer

        let k_offset = p_offset + (sP_bytes + 127) / 128 * 128;
        let sK_bytes = (stages as usize) * (nt as usize) * stride * 2;
        
        let v_offset = k_offset + (sK_bytes + 127) / 128 * 128;
        let sV_bytes = (stages as usize) * (nt as usize) * stride * 2;
        
        let total = v_offset + (sV_bytes + 127) / 128 * 128;
        (total, s_offset, p_offset, k_offset, v_offset)
    }

    pub fn generate_kernel(&self, _h: usize, d: usize, is_causal: bool) -> String {
        let mt = self.config.m_tile;
        let nt = self.config.n_tile; 
        let num_warps = self.config.force_num_warps.unwrap_or(1 + mt / 16); 
        let stages = self.config.num_stages;
        let stride = d + 8; 
        let d_over_16 = d / 16;

        let (_total_bytes, s_offset, p_offset, k_offset, v_offset) = Self::calculate_smem_layout(&self.config, d);

        format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define MT {mt}
#define KT {nt}
#define NUM_WARPS {num_warps}
#define STAGES {stages}
#define D_VAL {d}
#define STRIDE {stride}
#define STRIDE_S ({nt} + 8)
#define D_OVER_16 {d_over_16}
#define SOFTMAX_MODE {softmax_mode}

{primitives}

extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32, 1) flash_attention_v2_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    long long B, long long H, long long S, long long D,
    float scale
) {{
    int tile_idx = blockIdx.x; int h = blockIdx.y; int b = blockIdx.z;
    int tid = threadIdx.x; int warp_id = tid / 32; int lane_id = tid % 32;
    
    const int PRODUCER_WARPS = 1; 
    bool is_producer = (warp_id < PRODUCER_WARPS);

    long long offset_base = b * H * S * D + h * S * D;
    Q += offset_base; K += offset_base; V += offset_base; O += offset_base;

    extern __shared__ char smem[];
    
    half* smem_K_base = (half*)(smem + {k_offset});
    half* smem_V_base = (half*)(smem + {v_offset});
    float* smem_S_ptr = (float*)(smem + {s_offset});
    float* smem_O_ptr = smem_S_ptr; 
    half* smem_P_ptr = (half*)(smem + {p_offset});

    int cons_warp = warp_id - PRODUCER_WARPS;
    
    // Q Logic: Load Once
    int q_row_start = cons_warp * 16;
    int q_row_glob = tile_idx * MT + q_row_start; 
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q[D_OVER_16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[D_VAL/16]; 
    #pragma unroll
    for(int k=0; k<D_VAL/16; ++k) wmma::fill_fragment(acc_O[k], 0.0f);
    
    float m_prev[16]; float l_prev[16];
    #pragma unroll
    for(int i=0; i<16; ++i) {{ m_prev[i] = -50000.0f; l_prev[i] = 0.0f; }}

    if (!is_producer && cons_warp * 16 < MT) {{
        half* my_sq_buf = smem_P_ptr + cons_warp * 16 * D_VAL; 
        #pragma unroll
        for(int k=0; k<D_OVER_16; ++k) {{
             int r_ld = lane_id / 2; int c_bs = (lane_id % 2) * 8;
             long long sqr = (q_row_glob + r_ld < S) ? (q_row_glob + r_ld) : (S - 1);
             if (sqr < 0) sqr = 0;
             if (q_row_glob + r_ld < S && k*16 + c_bs < D) {{
                 *((uint4*)&my_sq_buf[r_ld*D_VAL + k*16 + c_bs]) = *((uint4*)&Q[sqr*D + k*16 + c_bs]);
             }} else {{
                 *((uint4*)&my_sq_buf[r_ld*D_VAL + k*16 + c_bs]) = make_uint4(0,0,0,0);
             }}
             __syncwarp();
             wmma::load_matrix_sync(frag_Q[k], my_sq_buf + k * 16, D_VAL);
             __syncwarp();
        }}
    }}
    __syncthreads(); 

    int total_tiles = (S + KT - 1) / KT;
    
    // PROLOGUE
    if (is_producer) {{
        for (int s = 0; s < STAGES - 1; ++s) {{
            if (s < total_tiles) {{
                half* sK = smem_K_base + s * KT * STRIDE;
                half* sV = smem_V_base + s * KT * STRIDE;
                int k_start = s * KT;
                for (int idx = tid * 8; idx < KT * D_VAL; idx += 32 * 8) {{
                     int r = idx / D_VAL; int c = idx % D_VAL;
                     long long safe_r = (k_start + r < S) ? (k_start + r) : (S - 1);
                     if (safe_r < 0) safe_r = 0;
                     half* k_dst = &sK[r*STRIDE + c]; half* v_dst = &sV[r*STRIDE + c];
                     cp_async_ampere(k_dst, &K[safe_r * D + c], (k_start + r < S) ? 16 : 0);
                     cp_async_ampere(v_dst, &V[safe_r * D + c], (k_start + r < S) ? 16 : 0);
                     if (k_start + r >= S) {{ *((uint4*)k_dst) = make_uint4(0,0,0,0); *((uint4*)v_dst) = make_uint4(0,0,0,0); }}
                }}
                cp_async_commit_group();
            }}
        }}
    }}
    cp_async_wait_group<0>();
    __syncthreads();

    // MAIN LOOP
    for (int k_tile = 0; k_tile < total_tiles; ++k_tile) {{
        // Producer: Pre-load next
        if (is_producer) {{
            int next_k = k_tile + STAGES - 1;
            if (next_k < total_tiles) {{
                int stage = next_k % STAGES;
                half* sK = smem_K_base + stage * KT * STRIDE;
                half* sV = smem_V_base + stage * KT * STRIDE;
                int k_start = next_k * KT;
                for (int idx = tid * 8; idx < KT * D_VAL; idx += 32 * 8) {{
                     int r = idx / D_VAL; int c = idx % D_VAL;
                     long long safe_r = (k_start + r < S) ? (k_start + r) : (S - 1);
                     if (safe_r < 0) safe_r = 0;
                     half* k_dst = &sK[r*STRIDE + c]; half* v_dst = &sV[r*STRIDE + c];
                     cp_async_ampere(k_dst, &K[safe_r * D + c], (k_start + r < S) ? 16 : 0);
                     cp_async_ampere(v_dst, &V[safe_r * D + c], (k_start + r < S) ? 16 : 0);
                     if (k_start + r >= S) {{ *((uint4*)k_dst) = make_uint4(0,0,0,0); *((uint4*)v_dst) = make_uint4(0,0,0,0); }}
                }}
                cp_async_commit_group();
            }}
        }}
        cp_async_wait_group<STAGES - 1>();
        __syncthreads();

        // Consumer
        if (!is_producer && cons_warp * 16 < MT) {{
            int stage = k_tile % STAGES;
            half* sK_base = smem_K_base + stage * KT * STRIDE;
            half* sV_base = smem_V_base + stage * KT * STRIDE;
            float* my_sS = smem_S_ptr + cons_warp * 16 * STRIDE_S; 
            half* my_sP = smem_P_ptr + cons_warp * 16 * KT;

            for (int step = 0; step < KT / 16; ++step) {{
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S;
                wmma::fill_fragment(acc_S, 0.0f);
                for(int k=0; k<D_OVER_16; ++k) {{
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;
                    wmma::load_matrix_sync(frag_K, sK_base + step * 16 * STRIDE + k * 16, STRIDE);
                    wmma::mma_sync(acc_S, frag_Q[k], frag_K, acc_S);
                }}
                wmma::store_matrix_sync(my_sS + step * 16, acc_S, STRIDE_S, wmma::mem_row_major);
                __syncwarp(); 

                float m_new_vals[16]; float row_p_sum = 0.0f;
                float row_m_curr = -50000.0f;
                if (lane_id < 16) {{ 
                    for(int c=0; c<16; ++c) {{ 
                        int col_glob = k_tile * KT + step * 16 + c;
                        float sv = my_sS[lane_id * STRIDE_S + step * 16 + c] * scale; 
                        if (col_glob >= S || q_row_glob + lane_id >= S) sv = -50000.0f;
                        if ({is_causal} && col_glob > q_row_glob + lane_id) sv = -50000.0f;
                        if(sv > row_m_curr) row_m_curr = sv; 
                    }} 
                }}
                for(int i=0; i<16; ++i) {{
                     float cur_m = __shfl_sync(0xffffffff, (lane_id < 16) ? row_m_curr : -50000.0f, i);
                     m_new_vals[i] = fmaxf(m_prev[i], cur_m);
                }}
                if (lane_id < 16) {{ 
                    for(int c=0; c<16; ++c) {{ 
                        int col_glob = k_tile * KT + step * 16 + c;
                        float sv = my_sS[lane_id * STRIDE_S + step * 16 + c] * scale;
                        if (col_glob >= S || q_row_glob + lane_id >= S) sv = -50000.0f; 
                        if ({is_causal} && col_glob > q_row_glob + lane_id) sv = -50000.0f;
                        float s = (sv > -45000.0f) ? expf(sv - m_new_vals[lane_id]) : 0.0f; 
                        my_sP[lane_id * KT + step * 16 + c] = __float2half(s); 
                        row_p_sum += s; 
                    }} 
                }}
                __syncwarp();

                for(int k=0; k<D_VAL/16; ++k) {{
                    float m_p = m_prev[lane_id / 4]; float m_n = m_new_vals[lane_id / 4];
                    float m_p2 = m_prev[lane_id / 4 + 8]; float m_n2 = m_new_vals[lane_id / 4 + 8];
                    float exp_a = expf(m_p - m_n); float exp_b = expf(m_p2 - m_n2);
                    for(int i=0; i<acc_O[k].num_elements; ++i) acc_O[k].x[i] *= (i < 4 ? exp_a : exp_b);
                }}

                for(int i=0; i<16; ++i) {{
                     float cur_ps = __shfl_sync(0xffffffff, (lane_id < 16) ? row_p_sum : 0.0f, i);
                     l_prev[i] = l_prev[i] * expf(m_prev[i] - m_new_vals[i]) + cur_ps;
                     m_prev[i] = m_new_vals[i];
                }}
                
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;
                wmma::load_matrix_sync(frag_P, my_sP + step * 16, KT);
                for(int k_v=0; k_v<D_VAL/16; ++k_v) {{ 
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V;
                    wmma::load_matrix_sync(frag_V, sV_base + step * 16 * STRIDE + k_v * 16, STRIDE);
                    wmma::mma_sync(acc_O[k_v], frag_P, frag_V, acc_O[k_v]);
                }}
            }}
        }}
        __syncthreads();
    }}

    // EPILOGUE
    if (!is_producer && cons_warp * 16 < MT) {{
         if (q_row_glob < S) {{
             float* my_sO = smem_O_ptr + cons_warp * 16 * D_VAL;
             for(int k=0; k<D_VAL/16; ++k) wmma::store_matrix_sync(my_sO + k * 16, acc_O[k], D_VAL, wmma::mem_row_major);
             __syncwarp();
             if (lane_id < 16 && q_row_glob + lane_id < S) {{
                 float lp = l_prev[lane_id];
                 for (int k=0; k<D_VAL/16; ++k) {{
                     float* s_row = my_sO + lane_id * D_VAL + k * 16;
                     half* g_row = O + (q_row_glob + lane_id) * D + k * 16;
                     for(int c=0; c<16; ++c) g_row[c] = __float2half(s_row[c] / (lp + 1e-6f));
                 }}
             }}
         }}
    }}
}}
"# , 
        mt=mt, nt=nt, num_warps=num_warps, stages=stages, d=d, stride=stride, d_over_16=d_over_16, 
        s_offset=s_offset, p_offset=p_offset, k_offset=k_offset, v_offset=v_offset, 
        is_causal=is_causal, 
        softmax_mode=self.config.softmax_granularity.to_f32() as u32,
        primitives=crate::backend::cuda::CudaBackend::get_primitive_defs())
    }
}
