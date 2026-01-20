
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define MT 128
#define NT 128
#define KT 32
#define STAGES 2
#define NUM_WARPS 9
#define PRODUCER_WARPS 1


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

__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;");
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
    int lane_id = tid % 32;
    bool is_producer = (warp_id < PRODUCER_WARPS);

    extern __shared__ char smem[];
    int a_smem_offset = 128; // Small offset for safety
    int b_smem_offset = a_smem_offset + 8192;

    int a_tile_row = blockIdx.y * MT;
    int b_tile_col = blockIdx.x * NT;
    int cons_warp = warp_id - PRODUCER_WARPS;
    int mt_per_warp = MT / 8;
    const int M_FRAGS = MT / 8 / 16;
    const int N_FRAGS = NT / 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc[M_FRAGS][N_FRAGS];
    #pragma unroll
    for(int mi=0; mi<M_FRAGS; mi++)
        for(int ni=0; ni<N_FRAGS; ni++)
            wmma::fill_fragment(frag_acc[mi][ni], (half)0.0f);

    for (int k_tile = 0; k_tile < (K + KT - 1) / KT; ++k_tile) {
        if (is_producer) {
            half* sA = (half*)(smem + a_smem_offset);
            half* sB = (half*)(smem + b_smem_offset);
            #pragma unroll
            for (int i = tid; i < (MT * KT) / 8; i += 32) {
                int m = (i * 8) / KT;
                int k = (i * 8) % KT;
                if (a_tile_row + m < M && k_tile * KT + k < K)
                    cp_async_ampere(sA + m * KT + k, A + (a_tile_row + m) * K + (k_tile * KT + k), 16);
            }
            #pragma unroll
            for (int i = tid; i < (KT * NT) / 8; i += 32) {
                int k = (i * 8) / NT;
                int n = (i * 8) % NT;
                if (k_tile * KT + k < K && b_tile_col + n < N)
                    cp_async_ampere(sB + k * NT + n, B + (k_tile * KT + k) * N + (b_tile_col + n), 16);
            }
            cp_async_commit_group();
            cp_async_wait_group_0();
        }
        __syncthreads();

        if (!is_producer) {
            half* sA = (half*)(smem + a_smem_offset);
            half* sB = (half*)(smem + b_smem_offset);
            for (int k_inner = 0; k_inner < KT; k_inner += 16) {
                #pragma unroll
                for (int mi = 0; mi < M_FRAGS; ++mi) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
                    wmma::load_matrix_sync(frag_a, sA + (cons_warp * mt_per_warp + mi * 16) * KT + k_inner, KT);
                    #pragma unroll
                    for (int ni = 0; ni < N_FRAGS; ++ni) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
                        wmma::load_matrix_sync(frag_b, sB + k_inner * NT + ni * 16, NT);
                        wmma::mma_sync(frag_acc[mi][ni], frag_a, frag_b, frag_acc[mi][ni]);
                    }
                }
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
}
