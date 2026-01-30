
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

#pragma once
#include <cuda_fp16.h>

/**
 * Tracea v3.1 Epilogue Functors
 * Used for fusing activation and residual operations into GEMM/Conv kernels.
 */

namespace tracea {
namespace epilogue {

struct Identity {
  __device__ __forceinline__ float operator()(float x) const { return x; }
};

struct ReLU {
  __device__ __forceinline__ float operator()(float x) const {
    return fmaxf(0.0f, x);
  }
};

struct Gelu {
  __device__ __forceinline__ float operator()(float x) const {
    // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 *
    // x^3)))
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
  }
};

struct SiLU {
  __device__ __forceinline__ float operator()(float x) const {
    return x / (1.0f + expf(-x));
  }
};

struct BiasAdd {
  const float *bias;
  __device__ __forceinline__ float operator()(float x, int channel) const {
    return x + bias[channel];
  }
};

struct BiasAddReLU {
  const float *bias;
  __device__ __forceinline__ float operator()(float x, int channel) const {
    return fmaxf(0.0f, x + bias[channel]);
  }
};

struct BiasAddSiLU {
  const float *bias;
  __device__ __forceinline__ float operator()(float x, int channel) const {
    float val = x + bias[channel];
    return val / (1.0f + expf(-val));
  }
};

struct ResidualAdd {
  const float *residual;
  __device__ __forceinline__ float operator()(float x, int index) const {
    return x + residual[index];
  }
};

} // namespace epilogue
} // namespace tracea


__device__ __forceinline__ void ldmatrix_m8n8_x4(uint32_t* regs, void* smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "l"(smem_ptr)
    );
}

__device__ __forceinline__ void mma_m16n8k16_f16(float* acc, uint32_t* a, uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbarrier_ptr, uint32_t expected_count) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(smem_addr), "r"(expected_count));
}

__device__ __forceinline__ void mbarrier_invalidate(uint64_t* mbarrier_ptr) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.inval.shared.b64 [%0];" : : "r"(smem_addr));
}

__device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t* mbarrier_ptr) {
    uint64_t state;
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(smem_addr));
    return state;
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* mbarrier_ptr, uint32_t tx_bytes) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;" : : "r"(smem_addr), "r"(tx_bytes));
}

// Hopper-only primitive (sm_90+). Commented for sm_80 compatibility.
/*
__device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(uint64_t* mbarrier_ptr, uint32_t tx_bytes) {
    uint64_t state;
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;" : "=l"(state) : "r"(smem_addr), "r"(tx_bytes));
    return state;
}
*/

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbarrier_ptr, uint64_t phase) {
    uint32_t mbarrier_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    uint64_t state = (phase << 63);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  wait_loop:\n\t"
        "  mbarrier.test_wait.shared.b64 p, [%0], %1;\n\t"
        "  @!p bra wait_loop;\n\t"
        "}\n\t"
        : 
        : "r"(mbarrier_addr), "l"(state)
    );
}
// Note: PTX 'mbarrier.wait' is Hopper+. For Ampere we must use test_wait loop or similar.
// I will use a tighter assembly loop.

__device__ __forceinline__ void cp_async_ampere(void* dst, const void* src, bool p) {
    #if __CUDA_ARCH__ >= 800
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(dst);
        if (p) {
            asm volatile(
                "{ .reg .pred p; setp.ne.b32 p, %2, 0; @p cp.async.ca.shared.global [%0], [%1], 16; }\n"
                : : "r"(smem_addr), "l"(src), "r"((int)p)
            );
        } else {
            *((uint4*)dst) = make_uint4(0, 0, 0, 0);
        }
    #else
        if (p) {
            *((uint4*)dst) = *((const uint4*)src);
        } else {
            *((uint4*)dst) = make_uint4(0, 0, 0, 0);
        }
    #endif
}

// Pipeline Management
__device__ __forceinline__ void cp_async_commit_group() {
    #if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;");
    #endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    #if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group %0;" :: "n"(N));
    #endif
}

// Helper for XOR Swizzling (128B aligned safe)
__device__ __forceinline__ uint32_t smem_swizzle(uint32_t addr) {
    // bits 4,5,6 XORed with bits 7,8,9
    uint32_t sw = (addr >> 4) & 0x7;
    return addr ^ (sw << 7);
}

__device__ __forceinline__ void* smem_swizzle_ptr(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t sw_addr = smem_swizzle(addr);
    return __cvta_shared_to_generic((size_t)sw_addr);
}

/**
 * Logical Padding (Zero-Padding) Load
 * Safely loads up to 4 elements from ptr. 
 * Elements beyond 'remaining' are set to 0.0f.
 */
__device__ __forceinline__ float4 load_float4_predicated(const float* ptr, int remaining) {
    if (remaining >= 4) return *((const float4*)ptr);
    float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (remaining > 0) val.x = ptr[0];
    if (remaining > 1) val.y = ptr[1];
    if (remaining > 2) val.z = ptr[2];
    return val;
}


using namespace nvcuda;

#define MT 64
#define NT 64
#define KT 32
#define STAGES 2
#define N_WARPS 8

struct ConvParams {
    int batch, h_in, w_in, c_in, k_out;
    int h_out, w_out, r_sz, s_sz;
    int stride, pad, dilation;
    unsigned int hw_m, hw_s;
    unsigned int w_m, w_s;
    unsigned int sic_m, sic_s;
    unsigned int c_m, c_s;
};

__device__ __forceinline__ void fast_divmod(int n, unsigned int m, int s, int d, int& q, int& r) {
    if (d == 1) { q = n; r = 0; }
    else if (m == 0) { q = n >> s; r = n & (d - 1); }
    else {
        q = (int)(((unsigned long long)n * m) >> (32 + s));
        r = n - q * d;
    }
}

extern "C" __global__ void kernel_main(
    const half* __restrict__ Input,
    const half* __restrict__ Weight,
    half* __restrict__ Output,
    ConvParams p
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ char smem[];
    half* sA = (half*)smem;
    half* sB = (half*)(smem + MT * 40 * STAGES * 2);
    float* sOut = (float*)smem;

    int m_block = blockIdx.x * MT;
    int n_block = blockIdx.y * NT;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for(int i=0; i<4; ++i) nvcuda::wmma::fill_fragment(acc[i], 0.0f);

    // Hybrid Hoisting Logic
    __shared__ long long block_base_off[MT];
    __shared__ int block_ho[MT], block_wo[MT];

    if (true) {
        for (int i = tid; i < MT; i += N_WARPS * 32) {
            int m_glob = m_block + i;
            if (m_glob < 4096) {
                int bn, ho, wo, rem_m;
                fast_divmod(m_glob, p.hw_m, p.hw_s, p.h_out * p.w_out, bn, rem_m);
                fast_divmod(rem_m, p.w_m, p.w_s, p.w_out, ho, wo);
                block_base_off[i] = (long long)bn * p.h_in * p.w_in * p.c_in;
                block_ho[i] = ho; block_wo[i] = wo;
            }
        }
        __syncthreads();
    }

    int total_k_tiles = (288 + KT - 1) / KT;

    // Pipeline
    for (int k_tile = 0; k_tile < total_k_tiles + STAGES - 1; ++k_tile) {
        if (k_tile + STAGES - 1 < total_k_tiles) {
            int tk = k_tile + STAGES - 1;
            int s_in = tk % STAGES;
            int k_tile_start = tk * KT;
            
            // Load sA
            for (int i = tid; i < (MT * KT) / 8; i += N_WARPS * 32) {
                int m_local = (i * 8) / KT; int k_local = (i * 8) % KT;
                half* dst = sA + s_in * MT * 40 + m_local * 40 + k_local;
                int m_glob = m_block + m_local;
                if (m_glob < 4096) {
                    int r_filt, s_filt, ci, rem_sic, k_glob = k_tile_start + k_local;
                    fast_divmod(k_glob, p.sic_m, p.sic_s, p.s_sz * p.c_in, r_filt, rem_sic);
                    fast_divmod(rem_sic, p.c_m, p.c_s, p.c_in, s_filt, ci);
                    
                    int hi, wi; long long base_off;
                    if (true) {
                        hi = block_ho[m_local] * p.stride - p.pad + r_filt * p.dilation;
                        wi = block_wo[m_local] * p.stride - p.pad + s_filt * p.dilation;
                        base_off = block_base_off[m_local];
                    } else {
                        int bn, ho, wo, rem_m;
                        fast_divmod(m_glob, p.hw_m, p.hw_s, p.h_out * p.w_out, bn, rem_m);
                        fast_divmod(rem_m, p.w_m, p.w_s, p.w_out, ho, wo);
                        hi = ho * p.stride - p.pad + r_filt * p.dilation;
                        wi = wo * p.stride - p.pad + s_filt * p.dilation;
                        base_off = (long long)bn * p.h_in * p.w_in * p.c_in;
                    }
                    bool pred = (hi >= 0 && hi < p.h_in && wi >= 0 && wi < p.w_in);
                    const half* src_ptr = Input + base_off + (long long)(hi * p.w_in + wi) * p.c_in + ci;
                    
                    if (p.c_in % 8 == 0) {
                        cp_async_ampere(dst, src_ptr, pred);
                    } else {
                        // Virtual Padding Strategy: Use predicated vector load
                        if (pred) {
                            int remaining = p.c_in - ci;
                            float4 v = load_float4_predicated((const float*)src_ptr, remaining);
                            float4 v2 = load_float4_predicated(((const float*)src_ptr) + 4, remaining - 4);
                            *((float4*)dst) = v;
                            *((float4*)(dst + 4)) = v2;
                        } else {
                            *((float4*)dst) = make_float4(0,0,0,0);
                            *((float4*)(dst+4)) = make_float4(0,0,0,0);
                        }
                    }
                } else {
                    *((float4*)dst) = make_float4(0,0,0,0);
                    *((float4*)(dst+4)) = make_float4(0,0,0,0);
                }
            }
            // Load sB
            for (int i = tid; i < (KT * NT) / 8; i += N_WARPS * 32) {
                int k_local = (i * 8) / NT; int n_local = (i * 8) % NT;
                half* dst = sB + s_in * KT * 72 + k_local * 72 + n_local;
                int k_glob = k_tile_start + k_local; int n_glob = n_block + n_local;
                const half* src_ptr = Weight + (long long)k_glob * p.k_out + n_glob;
                
                if (p.k_out % 8 == 0) {
                    cp_async_ampere(dst, src_ptr, (k_glob < 288 && n_glob < p.k_out));
                } else {
                    if (k_glob < 288 && n_glob < p.k_out) {
                        #pragma unroll
                        for(int v=0; v<8; ++v) dst[v] = src_ptr[v];
                    } else {
                        #pragma unroll
                        for(int v=0; v<8; ++v) dst[v] = 0.0f;
                    }
                }
            }
            cp_async_commit_group();
        }
        
        if (k_tile < total_k_tiles) {
            cp_async_wait_group<STAGES - 2>();
            __syncthreads();
            int stage = k_tile % STAGES;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_A[2];
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_B[2];
            nvcuda::wmma::load_matrix_sync(frag_A[0], sA + stage * MT * 40 + warp_id * 16 * 40, 40);
            nvcuda::wmma::load_matrix_sync(frag_A[1], sA + stage * MT * 40 + warp_id * 16 * 40 + 16, 40);
            #pragma unroll
            for(int j=0; j<4; ++j) {
                nvcuda::wmma::load_matrix_sync(frag_B[0], sB + stage * KT * 72 + j * 16, 72);
                nvcuda::wmma::load_matrix_sync(frag_B[1], sB + stage * KT * 72 + 16 * 72 + j * 16, 72);
                nvcuda::wmma::mma_sync(acc[j], frag_A[0], frag_B[0], acc[j]);
                nvcuda::wmma::mma_sync(acc[j], frag_A[1], frag_B[1], acc[j]);
            }
        }
    }

    __syncthreads();
    
    // Alignment-Safe Epilogue
    #pragma unroll
    for(int j=0; j<4; ++j) {
        int m_glob_warp = m_block + warp_id * 16;
        int n_glob_tile = n_block + j * 16;
        if (m_glob_warp < 4096 && n_glob_tile < p.k_out) {
            nvcuda::wmma::store_matrix_sync(sOut + warp_id * 256, acc[j], 16, nvcuda::wmma::mem_row_major);
            __syncwarp();
            half* p_half = (half*)smem + warp_id * 256;
            for(int k=lane_id; k<256; k+=32) p_half[k] = (half)sOut[warp_id * 256 + k];
            __syncwarp();
            
            int r_idx = lane_id / 2; int c_start = (lane_id % 2) * 8;
            int m_glob = m_glob_warp + r_idx;
            int n_glob = n_glob_tile + c_start;
            
            if (m_glob < 4096) {
                // Robust Store: check OC alignment
                if (p.k_out % 8 == 0 && n_glob + 8 <= p.k_out) {
                    #pragma unroll
                    for(int cc=0; cc<8; ++cc) {
                        float val = p_half[r_idx * 16 + c_start + cc];
                        int n_glob_c = n_glob + cc;
                        
                        p_half[r_idx * 16 + c_start + cc] = (half)val;
                    }
                    *((uint4*)&Output[(long long)m_glob * p.k_out + n_glob]) = *((uint4*)&p_half[r_idx * 16 + c_start]);
                } else {
                    #pragma unroll
                    for(int cc=0; cc<8; ++cc) {
                        if (n_glob + cc < p.k_out) {
                            float val = p_half[r_idx * 16 + c_start + cc];
                            int n_glob_c = n_glob + cc;
                            
                            Output[(long long)m_glob * p.k_out + n_glob + cc] = (half)val;
                        }
                    }
                }
            }
        }
    }
}
    