import torch
import tracea
import numpy as np
import math

def verify_config(B, H, S, D, MT, KT, STAGES, iters=10):
    ctx = tracea.Context()
    causal = False
    scale = 1.0 / math.sqrt(D)
    
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    
    cpp_source = r"""
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define MT {MT}
#define KT {KT}
#define D_VAL {D}
#define STAGES {STAGES}
#define STRIDE_K ({D} + 8)
#define D_OVER_16 ({D}/16)

__device__ __forceinline__ void cp_async_ampere(void* dst, const void* src) {{
    unsigned int dst_u = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst_u), "l"(src));
}}

__device__ __forceinline__ void cp_async_commit_group() {{ asm volatile("cp.async.commit_group;\n" ::); }}
template<int N> __device__ __forceinline__ void cp_async_wait_group() {{ asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }}

extern "C" __global__ void jit_fa2_kernel(
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
    int q_offset = 128;
    int k_offset = q_offset + ((MT * D_VAL * 2 + 127) / 128 * 128);
    int v_offset = k_offset + ((STAGES * KT * STRIDE_K * 2 + 127) / 128 * 128); 
    
    half* smem_Q = (half*)(smem + q_offset);
    half* smem_K_base = (half*)(smem + k_offset);
    half* smem_V_base = (half*)(smem + v_offset);

    // Q Load
    int q_row_start = tile_idx * MT;
    for (int idx = tid * 8; idx < MT * D; idx += blockDim.x * 8) {{
        int r = idx / D; int c = idx % D;
        if (q_row_start + r < S)
            *((uint4*)&smem_Q[r * D_VAL + c]) = *((uint4*)&Q[(q_row_start + r) * D + c]);
    }}
    __syncthreads();

    const int num_warps_q = MT / 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q[D_OVER_16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[D_OVER_16];
    if (warp_id < num_warps_q) {{
        for(int k=0; k<D_OVER_16; ++k) {{
            wmma::fill_fragment(acc_O[k], 0.0f);
            wmma::load_matrix_sync(frag_Q[k], smem_Q + warp_id * 16 * D_VAL + k * 16, D_VAL);
        }}
    }}
    
    float m_prev[16]; float l_prev[16];
    for(int i=0; i<16; ++i) {{ m_prev[i] = -50000.0f; l_prev[i] = 0.0f; }}

    int total_tiles = (S + KT - 1) / KT;

    for (int s = 0; s < STAGES - 1; ++s) {{
        if (s < total_tiles) {{
            int k_tile_start = s * KT;
            for (int idx = tid * 8; idx < KT * D_VAL; idx += blockDim.x * 8) {{
                int r = idx / D_VAL; int c = idx % D_VAL;
                if (k_tile_start + r < S) {{
                    cp_async_ampere(smem_K_base + s * KT * STRIDE_K + r * STRIDE_K + c, &K[(k_tile_start + r) * D + c]);
                    cp_async_ampere(smem_V_base + s * KT * STRIDE_K + r * STRIDE_K + c, &V[(k_tile_start + r) * D + c]);
                }}
            }}
            cp_async_commit_group();
        }}
    }}

    for (int k_tile = 0; k_tile < total_tiles; ++k_tile) {{
        int next_s_abs = k_tile + STAGES - 1;
        if (next_s_abs < total_tiles) {{
            int s = next_s_abs % STAGES;
            int k_tile_start = next_s_abs * KT;
            for (int idx = tid * 8; idx < KT * D_VAL; idx += blockDim.x * 8) {{
                int r = idx / D_VAL; int c = idx % D_VAL;
                if (k_tile_start + r < S) {{
                    cp_async_ampere(smem_K_base + s * KT * STRIDE_K + r * STRIDE_K + c, &K[(k_tile_start + r) * D + c]);
                    cp_async_ampere(smem_V_base + s * KT * STRIDE_K + r * STRIDE_K + c, &V[(k_tile_start + r) * D + c]);
                }}
            }}
            cp_async_commit_group();
        }}

        cp_async_wait_group<STAGES - 1>();
        __syncthreads();

        if (warp_id < num_warps_q && q_row_start + warp_id * 16 < S) {{
            int stage = k_tile % STAGES;
            half* my_smem_K = smem_K_base + stage * KT * STRIDE_K;
            half* my_smem_V = smem_V_base + stage * KT * STRIDE_K;

            for (int step = 0; step < KT / 16; ++step) {{
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S;
                wmma::fill_fragment(acc_S, 0.0f);
                for(int k=0; k<D_OVER_16; ++k) {{
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;
                    wmma::load_matrix_sync(frag_K, my_smem_K + step * 16 * STRIDE_K + k * 16, STRIDE_K);
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
                float m_next0 = fmaxf(m_prev[lane_id/4], m0);
                float m_next1 = fmaxf(m_prev[(lane_id/4)+8], m1);

                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;
                for(int i=0; i<8; ++i) {{
                    float val = acc_S.x[i] * scale;
                    float sub_m = (i==0||i==1||i==4||i==5) ? m_next0 : m_next1;
                    frag_P.x[i] = __float2half(expf(val - sub_m));
                }}

                float r0 = expf(m_prev[lane_id/4] - m_next0);
                float r1 = expf(m_prev[(lane_id/4)+8] - m_next1);
                for(int k=0; k<D_OVER_16; ++k) {{
                    acc_O[k].x[0] *= r0; acc_O[k].x[1] *= r0; acc_O[k].x[4] *= r0; acc_O[k].x[5] *= r0;
                    acc_O[k].x[2] *= r1; acc_O[k].x[3] *= r1; acc_O[k].x[6] *= r1; acc_O[k].x[7] *= r1;
                }}

                for(int k=0; k<D_OVER_16; ++k) {{
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V;
                    wmma::load_matrix_sync(frag_V, my_smem_V + step * 16 * STRIDE_K + k * 16, STRIDE_K);
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
                float m_cap = (lane_id % 4 == 0) ? m_next0 : -50000.0f;
                float m1_cap = (lane_id % 4 == 0) ? m_next1 : -50000.0f;

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
    if (warp_id < num_warps_q && q_row_start + warp_id * 16 < S) {{
        float* my_o_smem = (float*)smem_K_base + warp_id * 16 * D_VAL;
        for(int k=0; k<D_OVER_16; ++k) wmma::store_matrix_sync(my_o_smem + k * 16, acc_O[k], D_VAL, wmma::mem_row_major);
        __syncwarp();
        if (lane_id < 16) {{
            float lp = l_prev[lane_id];
            for (int k=0; k<D_OVER_16; ++k) {{
                float* s_row = my_o_smem + lane_id * D_VAL + k * 16;
                half* g_row = O + (q_row_start + warp_id * 16 + lane_id) * D + k * 16;
                for(int c=0; c<16; ++c) g_row[c] = __float2half(s_row[c] / (lp + 1e-9f));
            }}
        }}
    }}
}}
    """
    
    source = cpp_source.format(MT=MT, KT=KT, STAGES=STAGES, D=D)
    kernel_id = ctx.compile_custom(source, "jit_fa2_kernel")
    
    tracea_out = torch.empty_like(q)
    block = (32 * (MT // 16), 1, 1)
    grid = ((S + MT - 1) // MT, H, B)
    smem_size = 98000
    
    args = [q, k, v, tracea_out, B, H, S, D, scale]
    ctx.launch_kernel(kernel_id, grid, block, smem_size, args)
    torch.cuda.synchronize()
    
    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters): ctx.launch_kernel(kernel_id, grid, block, smem_size, args)
    end_event.record()
    torch.cuda.synchronize()
    
    msec = start_event.elapsed_time(end_event) / iters
    flops = 4.0 * B * H * S * S * D
    tflops = (flops / (msec / 1e3)) / 1e12
    
    diff = (ref_out - tracea_out).abs()
    max_diff = diff.max().item()
    mse = (diff**2).mean().item()
    print(f"MSE: {mse:.6f} | Latency: {msec:.3f} ms | Throughput: {tflops:.2f} TFLOPS")
    return mse, tflops

def matrix_test():
    print("[Doctor] Starting Regression Matrix Test")
    sizes = [512, 1024, 2048, 4096]
    heads = [1, 8, 32]
    # Fixed tile config
    MT, KT, STAGES = 128, 64, 3
    
    for S in sizes:
        for H in heads:
            print(f">> TESTING S={S}, H={H}", end=" -> ")
            mse, tflops = verify_config(1, H, S, 64, MT, KT, STAGES)
            if mse > 1e-4:
                print("FAILED (Inaccurate)")
            elif tflops < 0.1:
                print("FAILED (Performance)")
            else:
                print("PASSED")

if __name__ == "__main__":
    matrix_test()
