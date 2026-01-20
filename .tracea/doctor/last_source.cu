
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define MT 64
#define KT 64
#define NUM_WARPS 4
#define STAGES 2
#define D_VAL 64
#define STRIDE 72
#define STRIDE_S (64 + 8)
#define D_OVER_16 4


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


extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32, 1) flash_attention_v2_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    long long B, long long H, long long S, long long D,
    float scale
) {
    int tile_idx = blockIdx.x; int h = blockIdx.y; int b = blockIdx.z;
    int tid = threadIdx.x; int warp_id = tid / 32; int lane_id = tid % 32;
    const int PRODUCER_WARPS = 1; 
    bool is_producer = (warp_id < PRODUCER_WARPS);

    long long offset_base = b * H * S * D + h * S * D;
    Q += offset_base; K += offset_base; V += offset_base; O += offset_base;

    extern __shared__ char smem[];
    
    // Pointers for K/V buffers (Circular for Pipeline)
    half* smem_K_base = (half*)(smem + 26752);
    half* smem_V_base = (half*)(smem + 45184);
    
    // Pointers for S/P/O buffers (Fixed/Aliased for Consumers)
    // S and O share memory generally, P is intermediate
    float* smem_S_ptr = (float*)(smem + 128);
    float* smem_O_ptr = smem_S_ptr; 
    half* smem_P_ptr = (half*)(smem + 18560);

    int cons_warp = warp_id - PRODUCER_WARPS;
    int mt_per_warp = MT / (NUM_WARPS - PRODUCER_WARPS);
    
    // Q Logic: Load Once
    int q_row_start = cons_warp * 16;
    int q_row_glob = tile_idx * MT + q_row_start; // Corrected MT
    
    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q[D_OVER_16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[D_VAL/16]; // Accumulator is float because O is accumulated math
    // Initialize O accumulator
    #pragma unroll
    for(int k=0; k<D_VAL/16; ++k) wmma::fill_fragment(acc_O[k], 0.0f);
    
    // Softmax Stats
    float m_prev[16]; float l_prev[16];
    #pragma unroll
    for(int i=0; i<16; ++i) { m_prev[i] = -50000.0f; l_prev[i] = 0.0f; }

    // 1. Load Q (Consumers Only)
    if (!is_producer && cons_warp < (MT / 16)) {
        // We use P buffer temporarily to load Q tile parts? Or just use register load if D is small?
        // Reuse P buffer for Q loading scratchpad allows coalesced load
        half* my_sq_buf = smem_P_ptr + cons_warp * 16 * KT; // Careful with sizing
        #pragma unroll
        for(int k=0; k<D_OVER_16; ++k) {
            // Coalesced Load Logic... simplified from original
             int r_ld = lane_id / 2; int c_bs = (lane_id % 2) * 8;
             long long sqr = (q_row_glob + r_ld < S) ? (q_row_glob + r_ld) : (S - 1);
             if (sqr < 0) sqr = 0;
             if (q_row_glob + r_ld < S && k*16 + c_bs < D) {
                 *((uint4*)&my_sq_buf[r_ld*16 + c_bs]) = *((uint4*)&Q[sqr*D + k*16 + c_bs]);
             } else {
                 *((uint4*)&my_sq_buf[r_ld*16 + c_bs]) = make_uint4(0,0,0,0);
             }
             __syncwarp();
             wmma::load_matrix_sync(frag_Q[k], my_sq_buf, 16);
             __syncwarp();
        }
    }
    __syncthreads(); // Barrier 1: Q Loaded

    // Pipeline Setup
    int total_tiles = (S + KT - 1) / KT;
    
    // PROLOGUE
    if (is_producer) {
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < total_tiles) {
                int stage = s; 
                half* sK = smem_K_base + stage * KT * STRIDE;
                half* sV = smem_V_base + stage * KT * STRIDE;
                int k_tile_idx = s;
                int k_start = k_tile_idx * KT;
                
                // Load K and V
                for (int idx = tid * 8; idx < KT * D_VAL; idx += 32 * 8) {
                     int r = idx / D_VAL; int c = idx % D_VAL;
                     half* k_ptr_dst = &sK[r*STRIDE + c];
                     half* v_ptr_dst = &sV[r*STRIDE + c];
                     long long safe_r = (k_start + r < S) ? (k_start + r) : (S - 1);
                     if (safe_r < 0) safe_r = 0;
                     // Predicated Load
                     cp_async_ampere(k_ptr_dst, &K[safe_r * D + c], (k_start + r < S) ? 16 : 0);
                     cp_async_ampere(v_ptr_dst, &V[safe_r * D + c], (k_start + r < S) ? 16 : 0);
                     // Zero padding handled implicitly by cp_async 0? No, need explicit zeroing if OOB
                     if (k_start + r >= S) {
                          *((uint4*)k_ptr_dst) = make_uint4(0,0,0,0);
                          *((uint4*)v_ptr_dst) = make_uint4(0,0,0,0);
                     }
                }
                cp_async_commit_group();
            }
        }
        cp_async_wait_group<STAGES - 1>();
    }
    __syncthreads();

    // MAIN LOOP
    for (int k_tile = 0; k_tile < total_tiles; ++k_tile) {
        // CONSUMER
        if (!is_producer && cons_warp < (MT / 16)) {
            int stage = k_tile % STAGES;
            half* sK_base = smem_K_base + stage * KT * STRIDE;
            half* sV_base = smem_V_base + stage * KT * STRIDE;
            
            // 1. Compute S = Q * K^T
            // We iterate over the K tile in 16-step chunks? No, KT is usually 16 or 32 or 64.
            // Assuming KT is multiple of 16.
            
            float* my_sS = smem_S_ptr + cons_warp * 16 * KT; 
            half* my_sP = smem_P_ptr + cons_warp * 16 * KT;

            for (int step = 0; step < KT / 16; ++step) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S;
                wmma::fill_fragment(acc_S, 0.0f);
                #pragma unroll
                for(int k=0; k<D_OVER_16; ++k) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;
                    wmma::load_matrix_sync(frag_K, sK_base + step * 16 * STRIDE + k * 16, STRIDE);
                    wmma::mma_sync(acc_S, frag_Q[k], frag_K, acc_S);
                }
                // Store S to Smem for Softmax
                wmma::store_matrix_sync(my_sS + step * 16, acc_S, STRIDE_S, wmma::mem_row_major);
                __syncwarp(); 

                // 2. Softmax (Inter-warp sync not needed here, intra-warp is enough as we own the row)
                float row_m_curr = -50000.0f;
                // ... (Softmax Logic Copied verbatim mostly) ...
                if (lane_id < 16) { 
                    #pragma unroll 
                    for(int c=0; c<16; ++c) { 
                        int col_glob = k_tile * KT + step * 16 + c;
                        float sv = my_sS[lane_id * STRIDE_S + step * 16 + c] * scale; 
                        if (col_glob >= S || q_row_glob + lane_id >= S) sv = -50000.0f;
                        if (true && col_glob > q_row_glob + lane_id) sv = -50000.0f;
                        if(sv > row_m_curr) row_m_curr = sv; 
                    } 
                }
                
                float m_new_vals[16]; float row_p_sum = 0.0f;
                #pragma unroll
                for(int i=0; i<16; ++i) {
                     float cur_m = __shfl_sync(0xffffffff, (lane_id < 16) ? row_m_curr : -50000.0f, i);
                     m_new_vals[i] = fmaxf(m_prev[i], cur_m);
                }

                if (lane_id < 16) { 
                    #pragma unroll 
                    for(int c=0; c<16; ++c) { 
                        int col_glob = k_tile * KT + step * 16 + c;
                        float sv = my_sS[lane_id * STRIDE_S + step * 16 + c] * scale;
                        if (col_glob >= S || q_row_glob + lane_id >= S) sv = -50000.0f; 
                        if (true && col_glob > q_row_glob + lane_id) sv = -50000.0f;
                        float s = expf(sv - m_new_vals[lane_id]); 
                        my_sP[lane_id * KT + step * 16 + c] = __float2half(s); 
                        row_p_sum += s; 
                    } 
                }
                __syncwarp();

                // Correction of O accumulator using new Max
                #pragma unroll
                for(int k=0; k<D_VAL/16; ++k) {
                    float m_p = __shfl_sync(0xffffffff, (lane_id < 16) ? m_prev[lane_id] : -50000.0f, lane_id/4);
                    float m_n = __shfl_sync(0xffffffff, (lane_id < 16) ? m_new_vals[lane_id] : -50000.0f, lane_id/4);
                    float m_p2 = __shfl_sync(0xffffffff, (lane_id < 16) ? m_prev[lane_id] : -50000.0f, lane_id/4+8);
                    float m_n2 = __shfl_sync(0xffffffff, (lane_id < 16) ? m_new_vals[lane_id] : -50000.0f, lane_id/4+8);
                    float exp_a = expf(m_p - m_n); float exp_b = expf(m_p2 - m_n2);
                    for(int i=0; i<acc_O[k].num_elements; ++i) {
                        float r_exp = (i < 4) ? exp_a : exp_b;
                        acc_O[k].x[i] *= r_exp;
                    }
                }

                // Update L/M
                #pragma unroll
                for(int i=0; i<16; ++i) {
                     float cur_ps = __shfl_sync(0xffffffff, (lane_id < 16) ? row_p_sum : 0.0f, i);
                     float ep = expf(m_prev[i] - m_new_vals[i]);
                     l_prev[i] = l_prev[i] * ep + cur_ps;
                     m_prev[i] = m_new_vals[i];
                }
                
                // 3. Compute O += P * V
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;
                wmma::load_matrix_sync(frag_P, my_sP + step * 16, KT);
                #pragma unroll
                for(int k_v=0; k_v<D_VAL/16; ++k_v) { 
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V;
                    wmma::load_matrix_sync(frag_V, sV_base + step * 16 * STRIDE + k_v * 16, STRIDE);
                    wmma::mma_sync(acc_O[k_v], frag_P, frag_V, acc_O[k_v]);
                }
            }
        }

        // PRODUCER (Prefetch Next)
        if (is_producer) {
            int next_k = k_tile + STAGES - 1;
            if (next_k < total_tiles) {
                int stage = next_k % STAGES;
                half* sK = smem_K_base + stage * KT * STRIDE;
                half* sV = smem_V_base + stage * KT * STRIDE;
                int k_next_idx = next_k;
                int k_start = k_next_idx * KT;
                
                for (int idx = tid * 8; idx < KT * D_VAL; idx += 32 * 8) {
                     int r = idx / D_VAL; int c = idx % D_VAL;
                     half* k_ptr_dst = &sK[r*STRIDE + c];
                     half* v_ptr_dst = &sV[r*STRIDE + c];
                     long long safe_r = (k_start + r < S) ? (k_start + r) : (S - 1);
                     if (safe_r < 0) safe_r = 0;
                     cp_async_ampere(k_ptr_dst, &K[safe_r * D + c], (k_start + r < S) ? 16 : 0);
                     cp_async_ampere(v_ptr_dst, &V[safe_r * D + c], (k_start + r < S) ? 16 : 0);
                     if (k_start + r >= S) {
                          *((uint4*)k_ptr_dst) = make_uint4(0,0,0,0);
                          *((uint4*)v_ptr_dst) = make_uint4(0,0,0,0);
                     }
                }
                cp_async_commit_group();
                cp_async_wait_group<STAGES - 1>();
            } else {
                cp_async_wait_group<0>(); // Drain for safety at end
            }
        }
        __syncthreads();
    }

    // EPILOGUE: Store O
    if (!is_producer && cons_warp < (MT / 16)) {
         if (q_row_glob < S) {
             float* my_sO = smem_O_ptr + cons_warp * 16 * D_VAL;
             // Store un-normalized O to smem for final division
             #pragma unroll
             for(int k=0; k<D_VAL/16; ++k) wmma::store_matrix_sync(my_sO + k * 16, acc_O[k], D_VAL, wmma::mem_row_major);
             __syncwarp();
             
             // Final Division by L and Store to Global
             if (lane_id < 16) {
                 if (q_row_glob + lane_id < S) {
                     float lp = l_prev[lane_id];
                     #pragma unroll
                     for (int k=0; k<D_VAL/16; ++k) {
                         float* s_row = my_sO + lane_id * D_VAL + k * 16;
                         half* g_row = O + (q_row_glob + lane_id) * D + k * 16;
                         #pragma unroll 
                         for(int c=0; c<16; ++c) g_row[c] = __float2half(s_row[c] / (lp + 1e-6f));
                     }
                 }
             }
         }
    }
}
