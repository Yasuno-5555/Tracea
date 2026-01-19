use crate::core::config::PipelineConfig;

pub struct FlashAttentionEmitter {
    pub config: PipelineConfig,
}

impl FlashAttentionEmitter {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub fn generate_kernel(&self, _h: usize, d: usize, _is_causal: bool) -> String {
        let mt = self.config.m_tile;
        let nt = self.config.n_tile;
        let num_warps = self.config.force_num_warps.unwrap_or(mt / 16) + 1;
        let d_over_16 = d / 16;
        
        let stride = d + 8;
        
        let sS_floats = (num_warps as usize) * 256;      
        let sO_floats = (num_warps as usize) * 16 * d;   
        let sP_halves = (num_warps as usize) * 256;      
        let sK_halves = 2 * (nt as usize) * stride;           
        let _sV_halves = 2 * (nt as usize) * stride;           

        format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define TR {mt}
#define TC {nt}
#define NUM_WARPS {num_warps}
#define D_VAL {d}
#define STRIDE {stride}
#define D_OVER_16 {d_over_16}

extern "C" __global__ void flash_attention_v2_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    long long B, long long H, long long S, long long D, float scale
) {{
    int tile_idx = blockIdx.x; 
    int h = blockIdx.y;
    int b = blockIdx.z;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (warp_id >= NUM_WARPS) return; 

    long long offset_base = b * H * S * D + h * S * D;
    Q += offset_base; K += offset_base; V += offset_base; O += offset_base;

    extern __shared__ char smem[];
    float* smem_S_ptr = (float*)smem;
    float* smem_O_ptr = (float*)&smem_S_ptr[{sS_floats}]; 
    half* smem_P_half_ptr = (half*)&smem_O_ptr[{sO_floats}]; 
    half* smem_K_ptr = (half*)&smem_P_half_ptr[{sP_halves}]; 
    half* smem_V_ptr = (half*)&smem_K_ptr[{sK_halves}]; 

    float* my_sS = smem_S_ptr + warp_id * 256;         
    half* my_P_half = smem_P_half_ptr + warp_id * 256; 

    bool is_producer = (warp_id == 0);
    const int num_k_sec = TC * STRIDE;
    const int elements_total = TC * D_VAL;

    // Consumer state
    float m_prev = -50000.0f;
    float l_prev = 0.0f;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q[D_OVER_16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[D_OVER_16];
    
    if (!is_producer) {{
        int q_row_start = (warp_id - 1) * 16;
        if (q_row_start < TR) {{
            int q_row_glob = tile_idx * TR + q_row_start;
            if (q_row_glob < S) {{
                #pragma unroll
                for(int k=0; k<D_OVER_16; ++k) wmma::load_matrix_sync(frag_Q[k], Q + q_row_glob * D + k * 16, D);
                #pragma unroll
                for(int k=0; k<D_OVER_16; ++k) wmma::fill_fragment(acc_O[k], 0.0f);
            }}
        }}
    }}

    auto load_tile_async = [&](int j_start, int stage) {{
        half* sK = smem_K_ptr + stage * num_k_sec;
        half* sV = smem_V_ptr + stage * num_k_sec;
        #pragma unroll
        for (int idx = lane_id * 8; idx < elements_total; idx += 32 * 8) {{
            int r = idx / D_VAL;
            int c = idx % D_VAL;
            if (j_start + r < S) {{
                __pipeline_memcpy_async((void*)&sK[r * STRIDE + c], &K[(j_start + r) * D_VAL + c], 16);
                __pipeline_memcpy_async((void*)&sV[r * STRIDE + c], &V[(j_start + r) * D_VAL + c], 16);
            }} else {{
                for(int k=0; k<8; ++k) {{ sK[r*STRIDE + c + k] = 0; sV[r*STRIDE + c + k] = 0; }}
            }}
        }}
        __pipeline_commit();
    }};

    if (is_producer) load_tile_async(0, 0);

    for (int j = 0; j < S; j += TC) {{
        int cur_stage = (j / TC) % 2;
        int next_j = j + TC;
        
        if (is_producer && next_j < S) load_tile_async(next_j, (next_j / TC) % 2);
        
        __syncthreads();

        if (!is_producer) {{
            __pipeline_wait_prior(1); 
            int q_row_start = (warp_id - 1) * 16;
            if (q_row_start < TR) {{
                int q_row_glob = tile_idx * TR + q_row_start;
                if (q_row_glob < S) {{
                    half* sK_curr = smem_K_ptr + cur_stage * num_k_sec;
                    half* sV_curr = smem_V_ptr + cur_stage * num_k_sec;

                    for (int step = 0; step < TC / 16; ++step) {{
                        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S;
                        wmma::fill_fragment(frag_S, 0.0f);
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K_chunk;
                        #pragma unroll
                        for(int k=0; k<D_OVER_16; ++k) {{
                             wmma::load_matrix_sync(frag_K_chunk, sK_curr + step * 16 * STRIDE + k * 16, STRIDE);
                             wmma::mma_sync(frag_S, frag_Q[k], frag_K_chunk, frag_S);
                        }}

                        wmma::store_matrix_sync(my_sS, frag_S, 16, wmma::mem_row_major);
                        float m_curr = -50000.0f;
                        if (lane_id < 16) {{ #pragma unroll for(int c=0; c<16; ++c) {{ float s = my_sS[lane_id * 16 + c] * scale; if(s > m_curr) m_curr = s; }} }}

                        float m_new = fmaxf(m_prev, m_curr);
                        float p_sum = 0.0f;
                        if (lane_id < 16) {{ #pragma unroll for(int c=0; c<16; ++c) {{ float s = expf(my_sS[lane_id * 16 + c] * scale - m_new); my_P_half[lane_id * 16 + c] = __float2half(s); p_sum += s; }} }}
                        
                        float exp_prev = expf(m_prev - m_new);
                        float l_new = l_prev * exp_prev + p_sum;
                        float rescale_prev = (l_prev * exp_prev) / l_new;
                        if (l_new == 0.0f) rescale_prev = 0.0f;
                        #pragma unroll
                        for(int k=0; k<D_OVER_16; ++k) {{ for(int i=0; i<acc_O[k].num_elements; ++i) acc_O[k].x[i] *= rescale_prev; }}

                        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;
                        wmma::load_matrix_sync(frag_P, my_P_half, 16); 
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V_chunk;
                        #pragma unroll
                        for(int k=0; k<D_OVER_16; ++k) {{
                            wmma::load_matrix_sync(frag_V_chunk, sV_curr + step * 16 * STRIDE + k * 16, STRIDE);
                            wmma::mma_sync(acc_O[k], frag_P, frag_V_chunk, acc_O[k]);
                        }}
                        m_prev = m_new; l_prev = l_new;
                    }}
                }}
            }}
        }}
        __syncthreads();
    }}

    if (!is_producer) {{
        int q_row_start = (warp_id - 1) * 16;
        if (q_row_start < TR) {{
            int q_row_glob = tile_idx * TR + q_row_start;
            if (q_row_glob < S) {{
                #pragma unroll
                for(int k=0; k<D_OVER_16; ++k) wmma::store_matrix_sync(O + q_row_glob * D + k * 16, acc_O[k], D, wmma::mem_row_major);
            }}
        }}
    }}
}}
"#, mt=mt, nt=nt, num_warps=num_warps, d=d, stride=stride, d_over_16=d_over_16, sS_floats=sS_floats, sO_floats=sO_floats, sP_halves=sP_halves, sK_halves=sK_halves)
    }
}
