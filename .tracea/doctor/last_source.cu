
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


using namespace nvcuda;

#define MT 64
#define NT 64
#define KT 16
#define N_WARPS 4
#define STAGES 2
#define PRODUCER_WARPS 1
#define USE_CP_ASYNC 1

// Conv Constants
#define BATCH 1
#define H_IN 8
#define W_IN 8
#define C_IN 32
#define K_OUT 32
#define H_OUT 8
#define W_OUT 8
#define R_SZ 3
#define S_SZ 3
#define STRIDE 1
#define PAD 1
#define DILATION 1

#define HW_MAGIC 0
#define HW_SHIFT 6
#define W_MAGIC 0
#define W_SHIFT 3
#define S_MAGIC 1431655766
#define S_SHIFT 32
#define C_MAGIC 0
#define C_SHIFT 5


__device__ __forceinline__ void fast_divmod(int val, unsigned int magic, unsigned int shift, int divisor, int& div, int& mod) {
    if (magic == 0) {
        div = val >> shift;
        mod = val & (divisor - 1);
    } else {
        unsigned long long res = (unsigned long long)(unsigned int)val * (unsigned long long)magic;
        div = (int)(res >> shift);
        mod = val - div * divisor;
    }
}


#define A_STRIDE 16
#define B_STRIDE 64


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

__device__ __forceinline__ void cp_async_ampere(void* smem_ptr, const void* global_ptr, uint32_t size) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    if (size == 16) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;"
            : 
            : "r"(smem_addr), "l"(global_ptr)
        );
    }
}

// Pipeline Management
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// Helper for XOR Swizzling (128B aligned safe)
__device__ __forceinline__ uint32_t smem_swizzle(uint32_t addr) {
    uint32_t sw = (addr >> 4) & 0x7;
    return addr ^ (sw << 7);
}


extern "C" __global__ void __launch_bounds__(N_WARPS * 32, 1) conv2d_implicit_gemm(
    const half* __restrict__ Input,
    const half* __restrict__ Weight,
    half* __restrict__ Output
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;

#if USE_CP_ASYNC
    bool is_producer = (warp_id < PRODUCER_WARPS);
    
    // Grid Swizzling
    int swizzled_bid = (int)(((long long)blockIdx.x * 101) % gridDim.x);
    int m_block_start = swizzled_bid * MT;
    int n_block_start = blockIdx.y * NT;

    extern __shared__ char smem[];
    int a_smem_offset = 128; 
    int b_smem_offset = a_smem_offset + 2048 * STAGES;
    int c_smem_offset = b_smem_offset + 2048 * STAGES;
    float* sC = (float*)(smem + c_smem_offset);

    // Initialize sC to zero
    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {
        sC[i] = 0.0f;
    }
    __syncthreads();

    int total_k_tiles = (288 + KT - 1) / KT;

    // PROLOGUE: Load initial stages
    if (is_producer) {
        for (int s_idx = 0; s_idx < STAGES - 1; ++s_idx) {
            if (s_idx < total_k_tiles) {
                half* sA = (half*)(smem + a_smem_offset + s_idx * 2048);
                half* sB = (half*)(smem + b_smem_offset + s_idx * 2048);
                int k_step = s_idx * KT;

                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {
                    int r_tile = (i * 8) / KT;
                    int k_tile = (i * 8) % KT;
                    int m_glob = m_block_start + r_tile;
                    int k_glob = k_step + k_tile;
                    
                    if (m_glob < 64 && k_glob < 288) {
                        int b, ho, wo, r_idx, s_idx_rem, c_idx;
                        int rem_m, rem_k;
                        fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                        fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                        fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c_idx);
                        fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r_idx, s_idx_rem);
                        int hi = ho * STRIDE - PAD + r_idx * DILATION;
                        int wi = wo * STRIDE - PAD + s_idx_rem * DILATION;
                        if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {
                            cp_async_ampere(sA + r_tile * A_STRIDE + k_tile, Input + ((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c_idx, 16);
                        }
                    }
                }
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {
                    int k_tile = (i * 8) / NT;
                    int n_tile = (i * 8) % NT;
                    int k_glob = k_step + k_tile;
                    int n_glob = n_block_start + n_tile;
                    if (k_glob < 288 && n_glob < 32) {
                        cp_async_ampere(sB + k_tile * B_STRIDE + n_tile, Weight + (long long)k_glob * K_OUT + n_glob, 16);
                    }
                }
                cp_async_commit_group();
            }
        }
        cp_async_wait_group<STAGES - 2>();
    }
    __syncthreads();

    // MAIN LOOP
    for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {
        if (!is_producer) {
            int stage = k_tile % STAGES;
            half* sA = (half*)(smem + a_smem_offset + stage * 2048);
            half* sB = (half*)(smem + b_smem_offset + stage * 2048);
            
            // SIMT FMA Compute
            for (int k_idx = 0; k_idx < KT; ++k_idx) {
                int k_glob = k_tile * KT + k_idx;
                if (k_glob >= 288) break;
                
                for (int i = warp_id - PRODUCER_WARPS; i < MT * NT / 32; i += (N_WARPS - PRODUCER_WARPS)) {
                    int m_local = (i * 32 + (tid % 32)) / NT;
                    int n_local = (i * 32 + (tid % 32)) % NT;
                    if (m_local < MT && n_local < NT) {
                         sC[m_local * NT + n_local] += (float)sA[m_local * KT + k_idx] * (float)sB[k_idx * B_STRIDE + n_local];
                    }
                }
            }
        }

        if (is_producer) {
            int next_k = k_tile + STAGES - 1;
            if (next_k < total_k_tiles) {
                int stage = next_k % STAGES;
                half* sA = (half*)(smem + a_smem_offset + stage * 2048);
                half* sB = (half*)(smem + b_smem_offset + stage * 2048);
                int k_step = next_k * KT;

                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {
                    int r_tile = (i * 8) / KT;
                    int k_tile = (i * 8) % KT;
                    int m_glob = m_block_start + r_tile;
                    int k_glob = k_step + k_tile;
                    if (m_glob < 64 && k_glob < 288) {
                        int b, ho, wo, r_idx, s_idx_rem, c_idx;
                        int rem_m, rem_k;
                        fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                        fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                        fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c_idx);
                        fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r_idx, s_idx_rem);
                        int hi = ho * STRIDE - PAD + r_idx * DILATION;
                        int wi = wo * STRIDE - PAD + s_idx_rem * DILATION;
                        if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {
                            cp_async_ampere(sA + r_tile * A_STRIDE + k_tile, Input + ((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c_idx, 16);
                        } else {
                            // Zero padding for robustness
                            *(uint4*)(sA + r_tile * A_STRIDE + k_tile) = make_uint4(0, 0, 0, 0);
                        }
                    }
                }
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {
                    int k_tile = (i * 8) / NT;
                    int n_tile = (i * 8) % NT;
                    int k_glob = k_step + k_tile;
                    int n_glob = n_block_start + n_tile;
                    if (k_glob < 288 && n_glob < 32) {
                        cp_async_ampere(sB + k_tile * B_STRIDE + n_tile, Weight + (long long)k_glob * K_OUT + n_glob, 16);
                    }
                }
                cp_async_commit_group();
                cp_async_wait_group<STAGES - 2>();
            } else {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    // Epilogue
    if (!is_producer) {
        for (int i = tid - 32 * PRODUCER_WARPS; i < MT * NT; i += 32 * (N_WARPS - PRODUCER_WARPS)) {
            int r = i / NT;
            int c = i % NT;
            int m_glob = m_block_start + r;
            int n_glob = n_block_start + c;
            if (m_glob < 64 && n_glob < 32) {
                float val = sC[i];
                
                Output[(long long)m_glob * K_OUT + n_glob] = (half)val;
            }
        }
    }
#else
    // Fallback for unaligned/non-Ampere (SIMT FMA)
    int m_block_start = blockIdx.x * MT;
    int n_block_start = blockIdx.y * NT;
    extern __shared__ char smem[];
    half* sA = (half*)smem;
    half* sB = (half*)(smem + MT * KT * 2);
    float* sC = (float*)(smem + MT * KT * 2 + KT * NT * 2);

    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {
        sC[i] = 0.0f;
    }
    __syncthreads();

    int total_k_tiles = (288 + KT - 1) / KT;
    for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {
        int k_step = k_tile * KT;
        for (int i = tid; i < MT * KT; i += N_WARPS * 32) {
             int row = i / KT; int col = i % KT;
             int m_glob = m_block_start + row; int k_glob = k_step + col;
             half val = 0.0;
             if (m_glob < 64 && k_glob < 288) {
                 int b, ho, wo, r_idx, s_idx_rem, c_idx;
                 int rem_m, rem_k;
                 fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                 fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                 fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c_idx);
                 fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r_idx, s_idx_rem);
                 
                 int hi = ho * STRIDE - PAD + r_idx * DILATION; 
                 int wi = wo * STRIDE - PAD + s_idx_rem * DILATION;
                 if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {
                     val = Input[((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c_idx];
                 }
             }
             sA[row * KT + col] = val;
        }
        for (int i = tid; i < NT * KT; i += N_WARPS * 32) {
             int row = i / NT; int col = i % NT;
             int k_glob = k_step + row; int n_glob = n_block_start + col;
             half val = 0.0;
             if (k_glob < 288 && n_glob < 32) val = Weight[(long long)k_glob * K_OUT + n_glob];
             sB[col * KT + row] = val; // Note: row as col for transposed load
        }
        __syncthreads();
        
        for (int k_idx = 0; k_idx < KT; ++k_idx) {
            int k_glob = k_step + k_idx;
            if (k_glob >= 288) break;
            for (int i = tid; i < MT * NT; i += N_WARPS * 32) {
                int m_local = i / NT;
                int n_local = i % NT;
                sC[i] += (float)sA[m_local * KT + k_idx] * (float)sB[n_local * KT + k_idx];
            }
        }
        __syncthreads();
    }
    
    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {
        int r = i / NT; int c = i % NT;
        int m_glob = m_block_start + r; int n_glob = n_block_start + c;
        if (m_glob < 64 && n_glob < 32) { 
            float val = sC[i];
            
            Output[(long long)m_glob * K_OUT + n_glob] = (half)val;
        }
    }
#endif
}
