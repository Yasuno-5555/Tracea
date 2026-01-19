
use crate::core::config::PipelineConfig;

pub struct FlashAttentionEmitter {
    pub config: PipelineConfig,
}

impl FlashAttentionEmitter {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub fn generate_kernel(&self, _h: usize, _d: usize, is_causal: bool) -> String {
        let br = self.config.m_tile; 
        let bc = self.config.n_tile; 

        let causal_logic = if is_causal {
            "if (gc > gr) val = -INFINITY;"
        } else { "" };

        format!(r#"
#include <cuda_fp16.h>

#define INFINITY 1e9f

extern "C" __global__ void flash_attention_v2_kernel(
    const half* __restrict__ Q, 
    const half* __restrict__ K, 
    const half* __restrict__ V, 
    half* __restrict__ O, 
    int B, int H, int S, int D,
    float output_scale
) {{
    extern __shared__ __align__(16) uint4 smem_u4[];
    half* smem = (half*)smem_u4;

    half* sQ = &smem[0];
    half* sK = &smem[{br} * D];
    half* sV = &smem[{br} * D + {bc} * D];

    int tx = threadIdx.x;
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z; 
    int num_threads = blockDim.x;

    long long batch_stride = (long long)H * S * D;
    long long head_stride = (long long)S * D;

    const half* Q_base = Q + bz * batch_stride + by * head_stride + (long long)bx * {br} * D;
    const half* K_base = K + bz * batch_stride + by * head_stride;
    const half* V_base = V + bz * batch_stride + by * head_stride;
    half* O_base = O + bz * batch_stride + by * head_stride + (long long)bx * {br} * D;
    
    // Initial Load Q
    {{
        int items = ({br} * D) / 8; 
        for (int i = tx; i < items; i += num_threads) {{
            int row = i / (D/8);
            int col = (i % (D/8)) * 8;
            if (bx * {br} + row < S) {{
                *(float4*)&sQ[row * D + col] = *(float4*)&Q_base[row * D + col];
            }} else {{
                *(float4*)&sQ[row * D + col] = make_float4(0,0,0,0);
            }}
        }}
    }}
    __syncthreads();

    float m_curr = -INFINITY;
    float l_curr = 0.0f;
    float O_row[128]; 
    for(int d=0; d<D; ++d) O_row[d] = 0.0f;

    int num_k_tiles = (S + {bc} - 1) / {bc};

    for(int kb = 0; kb < num_k_tiles; ++kb) {{
        // Load K, V
        {{
            int items = ({bc} * D) / 8;
            for (int i = tx; i < items; i += num_threads) {{
                int row = i / (D/8);
                int col = (i % (D/8)) * 8;
                if (row < {bc}) {{
                    if (kb * {bc} + row < S) {{
                        *(float4*)&sK[row * D + col] = *(float4*)&K_base[kb * {bc} * D + row * D + col];
                        *(float4*)&sV[row * D + col] = *(float4*)&V_base[kb * {bc} * D + row * D + col];
                    }} else {{
                        *(float4*)&sK[row * D + col] = make_float4(0,0,0,0);
                        *(float4*)&sV[row * D + col] = make_float4(0,0,0,0);
                    }}
                }}
            }}
        }}
        __syncthreads();

        if (tx < {br} && bx * {br} + tx < S) {{
            int gr = bx * {br} + tx;
            float m_tile = -INFINITY;
            
            float S_row[{bc}];
            for(int j=0; j<{bc}; ++j) {{
                float sum = 0.0f;
                #pragma unroll
                for(int d=0; d<64; ++d) {{ // Hardcode D=64 for sanity
                    sum += (float)sQ[tx * D + d] * (float)sK[j * D + d];
                }}
                float val = sum * output_scale;
                int gc = kb * {bc} + j;
                if (gc >= S) val = -INFINITY;
                {causal_logic}
                S_row[j] = val;
                if (val > m_tile) m_tile = val;
            }}

            float m_new = fmaxf(m_curr, m_tile);
            float alpha = expf(m_curr - m_new);
            float alpha_tile = expf(m_tile - m_new);

            float row_sum = 0.0f;
            for(int j=0; j<{bc}; ++j) {{
                S_row[j] = expf(S_row[j] - m_new);
                row_sum += S_row[j];
            }}

            l_curr = l_curr * alpha + row_sum;
            for(int d=0; d<D; ++d) {{
                float p_v = 0.0f;
                for(int j=0; j<{bc}; ++j) {{
                    p_v += S_row[j] * (float)sV[j * D + d];
                }}
                O_row[d] = O_row[d] * alpha + p_v;
            }}
            m_curr = m_new;
        }}
        __syncthreads();
    }}

    if (tx < {br} && bx * {br} + tx < S) {{
        for(int d=0; d<D; d += 8) {{
            float4 res;
            half* h_res = (half*)&res;
            for(int i=0; i<8; ++i) {{
                h_res[i] = __float2half(O_row[d + i] / l_curr);
            }}
            *(float4*)&O_base[tx * D + d] = res;
        }}
    }}
}}
"#, br=br, bc=bc, causal_logic=causal_logic)
    }
}
