
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define MT 64
#define NT 64
#define KT 32
#define STAGES 3
#define NUM_WARPS 4
#define PRODUCER_WARPS 1
#define A_STRIDE 40
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


extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32, 1) gemm_mma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C_global,
    int M, int N, int K
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    bool is_producer = (warp_id < PRODUCER_WARPS);

    extern __shared__ char smem[];
    int a_smem_offset = 128;
    int b_smem_offset = a_smem_offset + 5120 * STAGES;

    int a_tile_row = blockIdx.y * MT;
    int b_tile_col = blockIdx.x * NT;
    int cons_warp = warp_id - PRODUCER_WARPS;
    int mt_per_warp = MT / (NUM_WARPS - PRODUCER_WARPS);
    const int M_FRAGS = MT / (NUM_WARPS - PRODUCER_WARPS) / 16;
    const int N_FRAGS = NT / 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc[M_FRAGS][N_FRAGS];
    #pragma unroll
    for(int mi=0; mi<M_FRAGS; mi++) {
        for(int ni=0; ni<N_FRAGS; ni++) {
            wmma::fill_fragment(frag_acc[mi][ni], (half)0.0f);
        }
    }

    int total_tiles = (K + KT - 1) / KT;

    // PROLOGUE: Pre-load STAGES - 1 tiles
    if (is_producer) {
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < total_tiles) {
                int stage = s; // Simple direct mapping for prologue
                half* sA = (half*)(smem + a_smem_offset + stage * 5120);
                half* sB = (half*)(smem + b_smem_offset + stage * 4608);
                
                int k_tile = s;
                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {
                    int m = (i * 8) / KT;
                    int k = (i * 8) % KT;
                    if (a_tile_row + m < M && k_tile * KT + k < K)
                        cp_async_ampere(sA + m * A_STRIDE + k, A + (a_tile_row + m) * K + (k_tile * KT + k), 16);
                }
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {
                    int k = (i * 8) / NT;
                    int n = (i * 8) % NT;
                    if (k_tile * KT + k < K && b_tile_col + n < N)
                        cp_async_ampere(sB + k * B_STRIDE + n, B + (k_tile * KT + k) * N + (b_tile_col + n), 16);
                }
                cp_async_commit_group();
            }
        }
        // Ensure Tile 0 is ready. (STAGES-1 committed. Keep STAGES-2 in flight).
        cp_async_wait_group<STAGES - 2>();
    }
    __syncthreads();

    // MAIN LOOP
    for (int k_tile = 0; k_tile < total_tiles; ++k_tile) {
        if (!is_producer) {
            int stage = k_tile % STAGES;
            half* sA = (half*)(smem + a_smem_offset + stage * 5120);
            half* sB = (half*)(smem + b_smem_offset + stage * 4608);
            for (int k_inner = 0; k_inner < KT; k_inner += 16) {
                #pragma unroll
                for (int mi = 0; mi < M_FRAGS; ++mi) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
                    wmma::load_matrix_sync(frag_a, sA + (cons_warp * mt_per_warp + mi * 16) * A_STRIDE + k_inner, A_STRIDE);
                    #pragma unroll
                    for (int ni = 0; ni < N_FRAGS; ++ni) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
                        wmma::load_matrix_sync(frag_b, sB + k_inner * B_STRIDE + ni * 16, B_STRIDE);
                        wmma::mma_sync(frag_acc[mi][ni], frag_a, frag_b, frag_acc[mi][ni]);
                    }
                }
            }
        }
        
        if (is_producer) {
            int next_k = k_tile + STAGES - 1;
            if (next_k < total_tiles) {
                int stage = next_k % STAGES;
                half* sA = (half*)(smem + a_smem_offset + stage * 5120);
                half* sB = (half*)(smem + b_smem_offset + stage * 4608);
                
                int k_next_tile_idx = next_k; // rename to handle capture
                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {
                    int m = (i * 8) / KT;
                    int k = (i * 8) % KT;
                    if (a_tile_row + m < M && k_next_tile_idx * KT + k < K)
                        cp_async_ampere(sA + m * A_STRIDE + k, A + (a_tile_row + m) * K + (k_next_tile_idx * KT + k), 16);
                }
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {
                    int k = (i * 8) / NT;
                    int n = (i * 8) % NT;
                    if (k_next_tile_idx * KT + k < K && b_tile_col + n < N)
                        cp_async_ampere(sB + k * B_STRIDE + n, B + (k_next_tile_idx * KT + k) * N + (b_tile_col + n), 16);
                }
                cp_async_commit_group();
                // Ensure k_tile + 1 is ready for next iter
                cp_async_wait_group<STAGES - 2>();
            } else {
                // Epilogue drain: Ensure remaining tiles ready. Safe fallback.
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (!is_producer) {
        #pragma unroll
        for (int mi = 0; mi < M_FRAGS; ++mi) {
             #pragma unroll
             for (int ni = 0; ni < N_FRAGS; ++ni) {
                 int row = a_tile_row + cons_warp * mt_per_warp + mi * 16;
                 int col = b_tile_col + ni * 16;
                 if (row < M && col < N)
                     wmma::store_matrix_sync((half*)C_global + row * N + col, frag_acc[mi][ni], N, wmma::mem_row_major);
             }
        }
    }
}
