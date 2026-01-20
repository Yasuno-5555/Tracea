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

        // 1. Barriers
        let barrier_bytes = (stages * 16) as usize;
        let s_offset = (barrier_bytes + 127) / 128 * 128;

        // 2. S / O Buffer (Aliased)
        let sS_bytes = (mt * nt * 4) as usize; // float P scores
        let sO_bytes = (mt as usize) * d * 4;  // float O results
        let shared_so_bytes = std::cmp::max(sS_bytes, sO_bytes);
        
        let p_offset = s_offset + (shared_so_bytes + 127) / 128 * 128;
        let sP_bytes = (mt * nt * 2) as usize; // half P scores

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
        let stride = d + 8; // BANK PADDING
        let d_over_16 = d / 16;

        let (_total_bytes, s_offset, p_offset, k_offset, v_offset) = Self::calculate_smem_layout(&self.config, d);

        format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define TR {mt}
#define TC {nt}
#define NUM_WARPS {num_warps}
#define STAGES {stages}
#define D_VAL {d}
#define STRIDE {stride}
#define D_OVER_16 {d_over_16}

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
    uint64_t* bar_full = (uint64_t*)smem; 
    uint64_t* bar_empty = (uint64_t*)&bar_full[STAGES]; 
    
    float* smem_S_ptr = (float*)(smem + {s_offset});
    float* smem_O_ptr = smem_S_ptr;
    half* smem_P_ptr = (half*)(smem + {p_offset});
    half* smem_K_ptr = (half*)(smem + {k_offset}); 
    half* smem_V_ptr = (half*)(smem + {v_offset}); 

    if (tid == 0) {{
         #pragma unroll
         for(int i=0; i<STAGES; ++i) mbarrier_init(&bar_full[i], 32);
         int total_consumers = NUM_WARPS - PRODUCER_WARPS;
         int needed_cons = (TR + 15) / 16;
         int active_cons = (needed_cons < total_consumers) ? needed_cons : total_consumers;
         #pragma unroll
         for(int i=0; i<STAGES; ++i) mbarrier_init(&bar_empty[i], active_cons * 32);
    }}
    __syncthreads();

    int total_ks = (S + TC - 1) / TC;
    int cons_warp = warp_id - PRODUCER_WARPS; 
    int q_row_start = cons_warp * 16;
    int q_row_glob = tile_idx * TR + q_row_start;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q[D_OVER_16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[D_VAL/16];
    #pragma unroll
    for(int k=0; k<D_VAL/16; ++k) wmma::fill_fragment(acc_O[k], 0.0f);
    
    float m_prev[16]; float l_prev[16];
    #pragma unroll
    for(int i=0; i<16; ++i) {{ m_prev[i] = -50000.0f; l_prev[i] = 0.0f; }}

    // Load Q once for this tile
    if (!is_producer && cons_warp < (TR / 16)) {{
        half* my_sq_buf = smem_P_ptr + cons_warp * 16 * TC;
        #pragma unroll
        for(int k=0; k<D_OVER_16; ++k) {{
            int r_ld = lane_id / 2; int c_bs = (lane_id % 2) * 8;
            long long sqr = (q_row_glob + r_ld < S) ? (q_row_glob + r_ld) : (S - 1);
            if (sqr < 0) sqr = 0;
            if (q_row_glob + r_ld < S && k*16 + c_bs < D) {{
                *((uint4*)&my_sq_buf[r_ld*16 + c_bs]) = *((uint4*)&Q[sqr*D + k*16 + c_bs]);
            }} else {{
                *((uint4*)&my_sq_buf[r_ld*16 + c_bs]) = make_uint4(0,0,0,0);
            }}
            __syncwarp();
            wmma::load_matrix_sync(frag_Q[k], my_sq_buf, 16);
            __syncwarp();
        }}
    }}
    __syncthreads();

    if (is_producer) {{
        for (int j_tile = 0; j_tile < total_ks; ++j_tile) {{
            int stage = j_tile % STAGES;
            int j_start = j_tile * TC;
            // DEADLOCK INVESTIGATION: Restore ^ 1 logic
            if (j_tile >= STAGES) mbarrier_wait(&bar_empty[stage], (j_tile / STAGES % 2) ^ 1);

            half* sK = smem_K_ptr + stage * TC * STRIDE;
            half* sV = smem_V_ptr + stage * TC * STRIDE;
            for (int idx = tid * 8; idx < TC * D_VAL; idx += 32 * 8) {{
                  int r = idx / D_VAL; int c = idx % D_VAL;
                  half* k_ptr_dst = &sK[r*STRIDE + c];
                  half* v_ptr_dst = &sV[r*STRIDE + c];
                  
                  long long safe_r = (j_start + r < S) ? (j_start + r) : (S - 1);
                  if (safe_r < 0) safe_r = 0; 
                  
                  cp_async_ampere(k_ptr_dst, &K[safe_r * D + c], (j_start + r < S) ? 16 : 0);
                  cp_async_ampere(v_ptr_dst, &V[safe_r * D + c], (j_start + r < S) ? 16 : 0);
                  
                  if (j_start + r >= S) {{
                      *((uint4*)k_ptr_dst) = make_uint4(0,0,0,0);
                      *((uint4*)v_ptr_dst) = make_uint4(0,0,0,0);
                  }}
             }}
            cp_async_commit_group();
            cp_async_wait_group_0();
            mbarrier_arrive(&bar_full[stage]);
        }}
    }} else if (cons_warp < (TR / 16)) {{
        for (int j_tile = 0; j_tile < total_ks; ++j_tile) {{
            int stage = j_tile % STAGES;
            mbarrier_wait(&bar_full[stage], (j_tile / STAGES % 2));

            float* my_sS = smem_S_ptr + cons_warp * 16 * TC; 
            half* my_sP = smem_P_ptr + cons_warp * 16 * TC;
            half* sK_base = smem_K_ptr + stage * TC * STRIDE;
            half* sV_base = smem_V_ptr + stage * TC * STRIDE;

            for (int step = 0; step < TC / 16; ++step) {{
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S;
                wmma::fill_fragment(acc_S, 0.0f);
                #pragma unroll
                for(int k=0; k<D_OVER_16; ++k) {{
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;
                    wmma::load_matrix_sync(frag_K, sK_base + step * 16 * STRIDE + k * 16, STRIDE);
                    wmma::mma_sync(acc_S, frag_Q[k], frag_K, acc_S);
                }}
                wmma::store_matrix_sync(my_sS + step * 16, acc_S, TC, wmma::mem_row_major);
                __syncwarp(); 

                float row_m_curr = -50000.0f;
                if (lane_id < 16) {{ 
                    #pragma unroll 
                    for(int c=0; c<16; ++c) {{ 
                        int col_glob = j_tile * TC + step * 16 + c;
                        float sv = my_sS[lane_id * TC + step * 16 + c] * scale; 
                        if (col_glob >= S || q_row_glob + lane_id >= S) sv = -50000.0f;
                        if ({is_causal} && col_glob > q_row_glob + lane_id) sv = -50000.0f;
                        if(sv > row_m_curr) row_m_curr = sv; 
                    }} 
                }}
                
                float m_new_vals[16]; float row_p_sum = 0.0f;
                #pragma unroll
                for(int i=0; i<16; ++i) {{
                     float cur_m = __shfl_sync(0xffffffff, (lane_id < 16) ? row_m_curr : -50000.0f, i);
                     m_new_vals[i] = fmaxf(m_prev[i], cur_m);
                }}

                if (lane_id < 16) {{ 
                    #pragma unroll 
                    for(int c=0; c<16; ++c) {{ 
                        int col_glob = j_tile * TC + step * 16 + c;
                        float sv = my_sS[lane_id * TC + step * 16 + c] * scale;
                        if (col_glob >= S || q_row_glob + lane_id >= S) sv = -50000.0f; 
                        if ({is_causal} && col_glob > q_row_glob + lane_id) sv = -50000.0f;
                        float s = expf(sv - m_new_vals[lane_id]); 
                        my_sP[lane_id * TC + step * 16 + c] = __float2half(s); 
                        row_p_sum += s; 
                    }} 
                }}
                __syncwarp();

                #pragma unroll
                for(int k=0; k<D_VAL/16; ++k) {{
                    float m_p = __shfl_sync(0xffffffff, (lane_id < 16) ? m_prev[lane_id] : -50000.0f, lane_id/4);
                    float m_n = __shfl_sync(0xffffffff, (lane_id < 16) ? m_new_vals[lane_id] : -50000.0f, lane_id/4);
                    float m_p2 = __shfl_sync(0xffffffff, (lane_id < 16) ? m_prev[lane_id] : -50000.0f, lane_id/4+8);
                    float m_n2 = __shfl_sync(0xffffffff, (lane_id < 16) ? m_new_vals[lane_id] : -50000.0f, lane_id/4+8);
                    float exp_a = expf(m_p - m_n); float exp_b = expf(m_p2 - m_n2);
                    for(int i=0; i<acc_O[k].num_elements; ++i) {{
                        float r_exp = (i < 4) ? exp_a : exp_b;
                        acc_O[k].x[i] *= r_exp;
                    }}
                }}

                #pragma unroll
                for(int i=0; i<16; ++i) {{
                     float cur_ps = __shfl_sync(0xffffffff, (lane_id < 16) ? row_p_sum : 0.0f, i);
                     float ep = expf(m_prev[i] - m_new_vals[i]);
                     l_prev[i] = l_prev[i] * ep + cur_ps;
                     m_prev[i] = m_new_vals[i];
                }}
                
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;
                wmma::load_matrix_sync(frag_P, my_sP + step * 16, TC);
                #pragma unroll
                for(int k_v=0; k_v<D_VAL/16; ++k_v) {{ 
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V;
                    wmma::load_matrix_sync(frag_V, sV_base + step * 16 * STRIDE + k_v * 16, STRIDE);
                    wmma::mma_sync(acc_O[k_v], frag_P, frag_V, acc_O[k_v]);
                }}
            }}
            mbarrier_arrive(&bar_empty[stage]); 
        }}

        if (q_row_glob < S) {{
             float* my_sO = smem_O_ptr + cons_warp * 16 * D_VAL;
             #pragma unroll
             for(int k=0; k<D_VAL/16; ++k) wmma::store_matrix_sync(my_sO + k * 16, acc_O[k], D_VAL, wmma::mem_row_major);
             __syncwarp();
             if (lane_id < 16) {{
                 if (q_row_glob + lane_id < S) {{
                     float lp = l_prev[lane_id];
                     #pragma unroll
                     for (int k=0; k<D_VAL/16; ++k) {{
                         float* s_row = my_sO + lane_id * D_VAL + k * 16;
                         half* g_row = O + (q_row_glob + lane_id) * D + k * 16;
                         #pragma unroll 
                         for(int c=0; c<16; ++c) g_row[c] = __float2half(s_row[c] / (lp + 1e-6f));
                     }}
                 }}
             }}
        }}
        }}
    }}
    if (tid==0 && blockIdx.x==0) printf("Kernel End\n");
}}
"#, mt=mt, nt=nt, num_warps=num_warps, stages=stages, d=d, stride=stride, d_over_16=d_over_16, s_offset=s_offset, p_offset=p_offset, k_offset=k_offset, v_offset=v_offset, is_causal=is_causal, primitives=CudaBackend::get_primitive_defs())
    }
}
