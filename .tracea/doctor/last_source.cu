
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define MT 128
#define NT 128
#define KT 32
#define STAGES 3

#define B_VAL 128
#define IC_VAL 512
#define H_VAL 7
#define W_VAL 7
#define OC_VAL 512
#define R_VAL 3
#define S_VAL 3
#define STRIDE_VAL 1
#define PAD_VAL 1
#define HO_VAL 7
#define WO_VAL 7
#define M_GEMM 6272
#define N_GEMM 512
#define K_GEMM 4608

#define SA_STRIDE 40
#define SB_STRIDE 136

__device__ __forceinline__ void magic_divmod(int n, unsigned int m, int s, int flag, int d, int& q, int& r) {
    if (d == 1) {
        q = n; r = 0;
    } else if (flag == 1) {
        q = n >> s;
        r = n & (d - 1);
    } else {
        q = (int)(((unsigned long long)n * m) >> (32 + s));
        r = n - q * d;
    }
}

__device__ __forceinline__ void cp_async_ampere(void* dst, const void* src, bool p) {
    unsigned int dst_u = __cvta_generic_to_shared(dst);
    int pred = p ? 1 : 0;
    if (pred) {
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; @p cp.async.ca.shared.global [%0], [%1], 16; }\n" : : "r"(dst_u), "l"(src), "r"(pred));
    } else {
        *((uint4*)dst) = make_uint4(0,0,0,0);
    }
}

__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N> __device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

extern "C" __global__ void jit_conv_kernel(
    const half* __restrict__ Input,
    const half* __restrict__ Weight,
    half* __restrict__ Output
) {
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
    for (int i = tid; i < MT; i += 256) {
        int m_glob = m_block + i;
        if (m_glob < M_GEMM) {
            int bn, ho, wo, rem_m;
            magic_divmod(m_glob, 87652394U, 0, 0, HO_VAL * WO_VAL, bn, rem_m);
            magic_divmod(rem_m, 613566757U, 0, 0, WO_VAL, ho, wo);
            block_base_off[i] = (long long)bn * H_VAL * W_VAL * IC_VAL;
            block_ho[i] = ho; block_wo[i] = wo;
        }
    }
    __syncthreads();

    int total_k_tiles = (K_GEMM + KT - 1) / KT;

    for (int k_tile = 0; k_tile < total_k_tiles + STAGES - 1; ++k_tile) {
        if (k_tile + STAGES - 1 < total_k_tiles) {
            int tk = k_tile + STAGES - 1;
            int s_in = tk % STAGES;
            int k_tile_start = tk * KT;
            
            for (int it = 0; it < 2; ++it) {
                int i = tid + it * 256;
                int m_local = (i * 8) / KT;
                int k_local = (i * 8) % KT;
                half* dst = sA + s_in * MT * SA_STRIDE + m_local * SA_STRIDE + k_local;
                
                if (m_block + m_local < M_GEMM) {
                    int r, s_filt, ci, rem_sic, k_glob = k_tile_start + k_local;
                    magic_divmod(k_glob, 2796203U, 0, 0, S_VAL * IC_VAL, r, rem_sic);
                    magic_divmod(rem_sic, 0U, 9, 1, IC_VAL, s_filt, ci);
                    
                    int hi = block_ho[m_local] * STRIDE_VAL - PAD_VAL + r;
                    int wi = block_wo[m_local] * STRIDE_VAL - PAD_VAL + s_filt;
                    bool p = (hi >= 0 && hi < H_VAL && wi >= 0 && wi < W_VAL);
                    cp_async_ampere(dst, Input + block_base_off[m_local] + (long long)(hi * W_VAL + wi) * IC_VAL + ci, p);
                } else *((uint4*)dst) = make_uint4(0,0,0,0);
            }
            for (int i = tid; i < (KT * NT) / 8; i += 256) {
                int k_local = (i * 8) / NT; int n_local = (i * 8) % NT;
                half* dst = sB + s_in * KT * SB_STRIDE + k_local * SB_STRIDE + n_local;
                int k_glob = k_tile_start + k_local; int n_glob = n_block + n_local;
                bool p = (k_glob < K_GEMM && n_glob < N_GEMM);
                cp_async_ampere(dst, Weight + (long long)k_glob * OC_VAL + n_glob, p);
            }
            cp_async_commit_group();
        }
        
        if (k_tile < total_k_tiles) {
            cp_async_wait_group<STAGES - 2>();
            __syncthreads();
            int stage = k_tile % STAGES;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2];
            wmma::load_matrix_sync(frag_A[0], sA + stage * MT * SA_STRIDE + warp_id * 16 * SA_STRIDE, SA_STRIDE);
            wmma::load_matrix_sync(frag_A[1], sA + stage * MT * SA_STRIDE + warp_id * 16 * SA_STRIDE + 16, SA_STRIDE);
            #pragma unroll
            for(int j=0; j<8; ++j) {
                wmma::load_matrix_sync(frag_B[0], sB + stage * KT * SB_STRIDE + j * 16, SB_STRIDE);
                wmma::load_matrix_sync(frag_B[1], sB + stage * KT * SB_STRIDE + 16 * SB_STRIDE + j * 16, SB_STRIDE);
                wmma::mma_sync(acc[j], frag_A[0], frag_B[0], acc[j]);
                wmma::mma_sync(acc[j], frag_A[1], frag_B[1], acc[j]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for(int j=0; j<8; ++j) {
        int m_glob_warp = m_block + warp_id * 16;
        int n_glob_tile = n_block + j * 16;
        if (m_glob_warp < M_GEMM && n_glob_tile < N_GEMM) {
            wmma::store_matrix_sync(sOut + warp_id * 256, acc[j], 16, wmma::mem_row_major);
            __syncwarp();
            half* p_half = (half*)smem + warp_id * 256;
            for(int k=lane_id; k<256; k+=32) p_half[k] = (half)sOut[warp_id * 256 + k];
            __syncwarp();
            int r_idx = lane_id / 2; int c_start = (lane_id % 2) * 8;
            if (m_glob_warp + r_idx < M_GEMM && (n_glob_tile + c_start) < N_GEMM) {
                *((uint4*)&Output[(long long)(m_glob_warp + r_idx) * OC_VAL + (n_glob_tile + c_start)]) = *((uint4*)&p_half[r_idx * 16 + c_start]);
            }
        }
    }
}
    