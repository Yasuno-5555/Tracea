import torch
import tracea
import numpy as np
import math

def get_magic(d):
    assert d >= 1
    if d == 1: return 0, 0, 0
    if (d & (d - 1)) == 0: return 0, d.bit_length() - 1, 1
    for s in range(32):
        m = ((1 << (32 + s)) + d - 1) // d
        if m < (1 << 32): return int(m), int(s), 0
    return 0, 0, -1

def verify_conv(B, IC, H, W, OC, R, S, stride=1, pad=0):
    ctx = tracea.Context()
    HO = (H + 2 * pad - (R - 1) - 1) // stride + 1
    WO = (W + 2 * pad - (S - 1) - 1) // stride + 1
    
    input_tensor = torch.randn(B, H, W, IC, device='cuda', dtype=torch.float16) * 0.05
    weight_tensor = torch.randn(R, S, IC, OC, device='cuda', dtype=torch.float16) * 0.05
    
    ref_weight = weight_tensor.permute(3, 2, 0, 1).contiguous()
    ref_input = input_tensor.permute(0, 3, 1, 2).contiguous()
    ref_out = torch.nn.functional.conv2d(ref_input, ref_weight, stride=stride, padding=pad).to(torch.float16)
    ref_out_nhwc = ref_out.permute(0, 2, 3, 1).contiguous()
    
    M_GEMM = B * HO * WO
    N_GEMM = OC
    K_GEMM = IC * R * S
    
    M_HW, S_HW, F_HW = get_magic(HO * WO)
    M_W, S_W, F_W = get_magic(WO)
    M_SIC, S_SIC, F_SIC = get_magic(S * IC)
    M_IC, S_IC, F_IC = get_magic(IC)
    
    cpp_source = r"""
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define MT 128
#define NT 128
#define KT 32
#define STAGES 3

#define B_VAL {B}
#define IC_VAL {IC}
#define H_VAL {H}
#define W_VAL {W}
#define OC_VAL {OC}
#define R_VAL {R}
#define S_VAL {S}
#define STRIDE_VAL {stride}
#define PAD_VAL {pad}
#define HO_VAL {HO}
#define WO_VAL {WO}
#define M_GEMM {M_GEMM}
#define N_GEMM {N_GEMM}
#define K_GEMM {K_GEMM}

#define SA_STRIDE 40
#define SB_STRIDE 136

__device__ __forceinline__ void magic_divmod(int n, unsigned int m, int s, int flag, int d, int& q, int& r) {{
    if (d == 1) {{
        q = n; r = 0;
    }} else if (flag == 1) {{
        q = n >> s;
        r = n & (d - 1);
    }} else {{
        q = (int)(((unsigned long long)n * m) >> (32 + s));
        r = n - q * d;
    }}
}}

__device__ __forceinline__ void cp_async_ampere(void* dst, const void* src, bool p) {{
    unsigned int dst_u = __cvta_generic_to_shared(dst);
    int pred = p ? 1 : 0;
    if (pred) {{
        asm volatile("{{ .reg .pred p; setp.ne.b32 p, %2, 0; @p cp.async.ca.shared.global [%0], [%1], 16; }}\n" : : "r"(dst_u), "l"(src), "r"(pred));
    }} else {{
        *((uint4*)dst) = make_uint4(0,0,0,0);
    }}
}}

__device__ __forceinline__ void cp_async_commit_group() {{ asm volatile("cp.async.commit_group;\n" ::); }}
template<int N> __device__ __forceinline__ void cp_async_wait_group() {{ asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }}

extern "C" __global__ void jit_conv_kernel(
    const half* __restrict__ Input,
    const half* __restrict__ Weight,
    half* __restrict__ Output
) {{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ char smem[];
    half* sA = (half*)smem;
    half* sB = (half*)(smem + MT * SA_STRIDE * STAGES * 2);
    float* sOut = (float*)smem;

    int m_block = blockIdx.x * MT;
    int n_block = blockIdx.y * NT;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for(int i=0; i<8; ++i) wmma::fill_fragment(acc[i], 0.0f);

    __shared__ long long block_base_off[MT];
    __shared__ int block_ho[MT], block_wo[MT];
    for (int i = tid; i < MT; i += 256) {{
        int m_glob = m_block + i;
        if (m_glob < M_GEMM) {{
            int bn, ho, wo, rem_m;
            magic_divmod(m_glob, {M_HW}U, {S_HW}, {F_HW}, HO_VAL * WO_VAL, bn, rem_m);
            magic_divmod(rem_m, {M_W}U, {S_W}, {F_W}, WO_VAL, ho, wo);
            block_base_off[i] = (long long)bn * H_VAL * W_VAL * IC_VAL;
            block_ho[i] = ho; block_wo[i] = wo;
        }}
    }}
    __syncthreads();

    int total_k_tiles = (K_GEMM + KT - 1) / KT;

    for (int k_tile = 0; k_tile < total_k_tiles + STAGES - 1; ++k_tile) {{
        if (k_tile + STAGES - 1 < total_k_tiles) {{
            int tk = k_tile + STAGES - 1;
            int s_in = tk % STAGES;
            int k_tile_start = tk * KT;
            
            for (int it = 0; it < 2; ++it) {{
                int i = tid + it * 256;
                int m_local = (i * 8) / KT;
                int k_local = (i * 8) % KT;
                half* dst = sA + s_in * MT * SA_STRIDE + m_local * SA_STRIDE + k_local;
                
                if (m_block + m_local < M_GEMM) {{
                    int r, s_filt, ci, rem_sic, k_glob = k_tile_start + k_local;
                    magic_divmod(k_glob, {M_SIC}U, {S_SIC}, {F_SIC}, S_VAL * IC_VAL, r, rem_sic);
                    magic_divmod(rem_sic, {M_IC}U, {S_IC}, {F_IC}, IC_VAL, s_filt, ci);
                    
                    int hi = block_ho[m_local] * STRIDE_VAL - PAD_VAL + r;
                    int wi = block_wo[m_local] * STRIDE_VAL - PAD_VAL + s_filt;
                    bool p = (hi >= 0 && hi < H_VAL && wi >= 0 && wi < W_VAL);
                    cp_async_ampere(dst, Input + block_base_off[m_local] + (long long)(hi * W_VAL + wi) * IC_VAL + ci, p);
                }} else *((uint4*)dst) = make_uint4(0,0,0,0);
            }}
            for (int i = tid; i < (KT * NT) / 8; i += 256) {{
                int k_local = (i * 8) / NT; int n_local = (i * 8) % NT;
                half* dst = sB + s_in * KT * SB_STRIDE + k_local * SB_STRIDE + n_local;
                int k_glob = k_tile_start + k_local; int n_glob = n_block + n_local;
                bool p = (k_glob < K_GEMM && n_glob < N_GEMM);
                cp_async_ampere(dst, Weight + (long long)k_glob * OC_VAL + n_glob, p);
            }}
            cp_async_commit_group();
        }}
        
        if (k_tile < total_k_tiles) {{
            cp_async_wait_group<STAGES - 2>();
            __syncthreads();
            int stage = k_tile % STAGES;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
            wmma::load_matrix_sync(frag_A[0], sA + stage * MT * SA_STRIDE + warp_id * 16 * SA_STRIDE, SA_STRIDE);
            wmma::load_matrix_sync(frag_A[1], sA + stage * MT * SA_STRIDE + warp_id * 16 * SA_STRIDE + 16, SA_STRIDE);
            #pragma unroll
            for(int j=0; j<8; ++j) {{
                wmma::load_matrix_sync(frag_B[0], sB + stage * KT * SB_STRIDE + j * 16, SB_STRIDE);
                wmma::load_matrix_sync(frag_B[1], sB + stage * KT * SB_STRIDE + 16 * SB_STRIDE + j * 16, SB_STRIDE);
                wmma::mma_sync(acc[j], frag_A[0], frag_B[0], acc[j]);
                wmma::mma_sync(acc[j], frag_A[1], frag_B[1], acc[j]);
            }}
        }}
    }}

    __syncthreads();
    #pragma unroll
    for(int j=0; j<8; ++j) {{
        int m_glob_warp = m_block + warp_id * 16;
        int n_glob_tile = n_block + j * 16;
        if (m_glob_warp < M_GEMM && n_glob_tile < N_GEMM) {{
            wmma::store_matrix_sync(sOut + warp_id * 256, acc[j], 16, wmma::mem_row_major);
            __syncwarp();
            half* p_half = (half*)smem + warp_id * 256;
            for(int k=lane_id; k<256; k+=32) p_half[k] = (half)sOut[warp_id * 256 + k];
            __syncwarp();
            int r_idx = lane_id / 2; int c_start = (lane_id % 2) * 8;
            if (m_glob_warp + r_idx < M_GEMM && (n_glob_tile + c_start) < N_GEMM) {{
                *((uint4*)&Output[(long long)(m_glob_warp + r_idx) * OC_VAL + (n_glob_tile + c_start)]) = *((uint4*)&p_half[r_idx * 16 + c_start]);
            }}
        }}
    }}
}}
    """
    source = cpp_source.format(B=B, IC=IC, H=H, W=W, OC=OC, R=R, S=S, stride=stride, pad=pad, HO=HO, WO=WO, M_GEMM=M_GEMM, N_GEMM=N_GEMM, K_GEMM=K_GEMM,
        M_HW=M_HW, S_HW=S_HW, F_HW=F_HW, M_W=M_W, S_W=S_W, F_W=F_W, M_SIC=M_SIC, S_SIC=S_SIC, F_SIC=F_SIC, M_IC=M_IC, S_IC=S_IC, F_IC=F_IC)
    
    kernel_id = ctx.compile_custom(source, "jit_conv_kernel")
    tracea_out = torch.empty_like(ref_out_nhwc)
    MT, NT = 128, 128
    grid = ((M_GEMM + MT - 1) // MT, (N_GEMM + NT - 1) // NT, 1)
    smem_size = 98304
    args = [input_tensor, weight_tensor, tracea_out]
    ctx.launch_kernel(kernel_id, grid, (256,1,1), smem_size, args)
    torch.cuda.synchronize()
    
    diff = (ref_out_nhwc - tracea_out).abs()
    mse = (diff**2).mean().item()
    print(f"B={B}, IC={IC}, OC={OC} | MSE: {mse:.6e} | Max Diff: {diff.max().item():.6e}")
    # Samples
    print(f"Ref Sample[0,0,0,0:4]: {ref_out_nhwc[0, 0, 0, :4].tolist()}")
    print(f"Trace Sample[0,0,0,0:4]: {tracea_out[0, 0, 0, :4].tolist()}")
    
    # Bench
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10): ctx.launch_kernel(kernel_id, grid, (256,1,1), smem_size, args)
    end.record()
    torch.cuda.synchronize()
    msec = start.elapsed_time(end) / 10
    tflops = (2.0 * M_GEMM * N_GEMM * K_GEMM / (msec / 1e3)) / 1e12
    print(f"Latency: {msec:.3f} ms | TFLOPS: {tflops:.2f}")

if __name__ == "__main__":
    verify_conv(B=128, IC=512, H=7, W=7, OC=512, R=3, S=3, stride=1, pad=1)
