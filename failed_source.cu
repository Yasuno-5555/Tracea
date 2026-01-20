
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define MT 128
#define NT 64
#define KT 16
#define N_WARPS 17
#define STAGES 2
#define PRODUCER_WARPS 1
#define USE_CP_ASYNC 1

// Conv Constants
#define BATCH 64
#define H_IN 56
#define W_IN 56
#define C_IN 64
#define K_OUT 64
#define H_OUT 56
#define W_OUT 56
#define R_SZ 3
#define S_SZ 3
#define STRIDE 1
#define PAD 1

#define HW_MAGIC 1
#define HW_SHIFT 0
#define W_MAGIC 1
#define W_SHIFT 0
#define S_MAGIC 1
#define S_SHIFT 0
#define C_MAGIC 1
#define C_SHIFT 0


__device__ __forceinline__ void fast_divmod(int val, int magic, int shift, int divisor, int& div, int& mod) {
    div = (int)(((unsigned int)val * (unsigned int)magic) >> shift);
    mod = val - div * divisor;
}


#define A_STRIDE 24
#define B_STRIDE 72


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
    int m_block_start = blockIdx.x * MT;
    int n_block_start = blockIdx.y * NT;
    extern __shared__ char smem[];
    int a_smem_offset = 128; // Header for barriers if needed
    int b_smem_offset = a_smem_offset + 6144 * STAGES;

    // Fragments
    int cons_warp = warp_id - PRODUCER_WARPS;
    int mt_per_warp = MT / (N_WARPS - PRODUCER_WARPS);
    const int M_FRAGS = MT / (N_WARPS - PRODUCER_WARPS) / 16;
    const int N_FRAGS = NT / 16;

    // Use float accumulator for accuracy and standard MMA paths
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[M_FRAGS][N_FRAGS];
    #pragma unroll
    for(int i=0; i<M_FRAGS; ++i) 
        for(int j=0; j<N_FRAGS; ++j) 
            wmma::fill_fragment(acc[i][j], 0.0f);

    int total_k_tiles = (576 + KT - 1) / KT;

    // PROLOGUE: Load initial stages
    if (is_producer) {
        for (int s_idx = 0; s_idx < STAGES - 1; ++s_idx) {
            if (s_idx < total_k_tiles) {
                half* sA = (half*)(smem + a_smem_offset + s_idx * 6144);
                half* sB = (half*)(smem + b_smem_offset + s_idx * 2304);
                int k_step = s_idx * KT;

                // Load A (Input)
                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {
                    int r_tile = (i * 8) / KT;
                    int k_tile = (i * 8) % KT;
                    int m_glob = m_block_start + r_tile;
                    int k_glob = k_step + k_tile;
                    
                    if (m_glob < 200704 && k_glob < 576) {
                        int b, ho, wo, r, s, c;
                        int rem_m, rem_k;
                        fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                        fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                        fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c);
                        fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r, s);
                        int hi = ho * STRIDE - PAD + r;
                        int wi = wo * STRIDE - PAD + s;
                        if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {
                            cp_async_ampere(sA + r_tile * A_STRIDE + k_tile, Input + ((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c, 16);
                        }
                    }
                }
                // Load B (Weight) - [K, N] row-major
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {
                    int k_tile = (i * 8) / NT;
                    int n_tile = (i * 8) % NT;
                    int k_glob = k_step + k_tile;
                    int n_glob = n_block_start + n_tile;
                    if (k_glob < 576 && n_glob < 64) {
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
            half* sA = (half*)(smem + a_smem_offset + stage * 6144);
            half* sB = (half*)(smem + b_smem_offset + stage * 2304);
            
            for (int k_inner = 0; k_inner < KT; k_inner += 16) {
                // Pre-load B fragments to avoid redundant loads in mi loop
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[N_FRAGS];
                #pragma unroll
                for (int ni = 0; ni < N_FRAGS; ++ni) {
                    wmma::load_matrix_sync(frag_b[ni], sB + k_inner * B_STRIDE + ni * 16, B_STRIDE);
                }

                #pragma unroll
                for (int mi = 0; mi < M_FRAGS; ++mi) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
                    wmma::load_matrix_sync(frag_a, sA + (cons_warp * mt_per_warp + mi * 16) * A_STRIDE + k_inner, A_STRIDE);
                    #pragma unroll
                    for (int ni = 0; ni < N_FRAGS; ++ni) {
                        wmma::mma_sync(acc[mi][ni], frag_a, frag_b[ni], acc[mi][ni]);
                    }
                }
            }
        }

        if (is_producer) {
            int next_k = k_tile + STAGES - 1;
            if (next_k < total_k_tiles) {
                int stage = next_k % STAGES;
                half* sA = (half*)(smem + a_smem_offset + stage * 6144);
                half* sB = (half*)(smem + b_smem_offset + stage * 2304);
                int k_step = next_k * KT;

                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {
                    int r_tile = (i * 8) / KT;
                    int k_tile = (i * 8) % KT;
                    int m_glob = m_block_start + r_tile;
                    int k_glob = k_step + k_tile;
                    if (m_glob < 200704 && k_glob < 576) {
                        int b, ho, wo, r, s, c;
                        int rem_m, rem_k;
                        fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                        fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                        fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c);
                        fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r, s);
                        int hi = ho * STRIDE - PAD + r;
                        int wi = wo * STRIDE - PAD + s;
                        if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) {
                            cp_async_ampere(sA + r_tile * A_STRIDE + k_tile, Input + ((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c, 16);
                        }
                    }
                }
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {
                    int k_tile = (i * 8) / NT;
                    int n_tile = (i * 8) % NT;
                    int k_glob = k_step + k_tile;
                    int n_glob = n_block_start + n_tile;
                    if (k_glob < 576 && n_glob < 64) {
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
        // Use float for Smem epilogue to match acc type
        float* sC = (float*)smem;
        #pragma unroll
        for (int i=0; i<M_FRAGS; ++i) {
            for (int j=0; j<N_FRAGS; ++j) {
                wmma::store_matrix_sync(sC + (cons_warp * mt_per_warp + i*16)*NT + (j*16), acc[i][j], NT, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = tid; i < MT * NT; i += 32 * (N_WARPS - 1)) {
            int r = i / NT;
            int c = i % NT;
            int m_glob = m_block_start + r;
            int n_glob = n_block_start + c;
            if (m_glob < 200704 && n_glob < 64) {
                Output[(long long)m_glob * K_OUT + n_glob] = (half)sC[i];
            }
        }
    }
#else
    // Fallback for unaligned C_IN (e.g. Stem)
    int m_block_start = blockIdx.x * MT;
    int n_block_start = blockIdx.y * NT;
    extern __shared__ char smem[];
    half* sA = (half*)smem;
    half* sB = (half*)(smem + MT * KT * 2);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT/16][NT/16];
    #pragma unroll
    for(int i=0; i<MT/16; ++i) for(int j=0; j<NT/16; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    int total_k_tiles = (576 + KT - 1) / KT;
    for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {
        int k_step = k_tile * KT;
        for (int i = tid; i < MT * KT; i += N_WARPS * 32) {
             int row = i / KT; int col = i % KT;
             int m_glob = m_block_start + row; int k_glob = k_step + col;
             half val = 0.0;
             if (m_glob < 200704 && k_glob < 576) {
                 int b, ho, wo, r, s, c;
                 int rem_m, rem_k;
                 fast_divmod(m_glob, HW_MAGIC, HW_SHIFT, H_OUT * W_OUT, b, rem_m);
                 fast_divmod(rem_m, W_MAGIC, W_SHIFT, W_OUT, ho, wo);
                 fast_divmod(k_glob, C_MAGIC, C_SHIFT, C_IN, rem_k, c);
                 fast_divmod(rem_k, S_MAGIC, S_SHIFT, S_SZ, r, s);
                 
                 int hi = ho * STRIDE - PAD + r; int wi = wo * STRIDE - PAD + s;
                 if (hi >= 0 && hi < H_IN && wi >= 0 && wi < W_IN) val = Input[((long long)b * H_IN * W_IN + hi * W_IN + wi) * C_IN + c];
             }
             sA[row * KT + col] = val;
        }
        for (int i = tid; i < NT * KT; i += N_WARPS * 32) {
             int row = i / NT; int col = i % NT;
             int k_glob = k_step + row; int n_glob = n_block_start + col;
             half val = 0.0;
             if (k_glob < 576 && n_glob < 64) val = Weight[(long long)k_glob * K_OUT + n_glob];
             sB[col * KT + row] = val;
        }
        __syncthreads();
        #pragma unroll
        for (int k=0; k<KT/16; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
            #pragma unroll
            for (int i=0; i<MT/16; ++i) {
                wmma::load_matrix_sync(fA, sA + i*16*KT + k*16, KT);
                #pragma unroll
                for (int j=0; j<NT/16; ++j) {
                    if (i == 0) wmma::load_matrix_sync(fB, sB + j*16*KT + k*16, KT);
                    wmma::mma_sync(acc[i][j], fA, fB, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    float* sC = (float*)smem;
    for (int i=0; i<MT/16; ++i) for (int j=0; j<NT/16; ++j) wmma::store_matrix_sync(sC + (i*16)*NT + (j*16), acc[i][j], NT, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {
        int r = i / NT; int c = i % NT;
        int m_glob = m_block_start + r; int n_glob = n_block_start + c;
        if (m_glob < 200704 && n_glob < 64) Output[(long long)m_glob * K_OUT + n_glob] = (half)sC[i];
    }
#endif
}
