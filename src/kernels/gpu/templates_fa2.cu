// templates_fa2.cu - FlashAttention-2 Template Kernel
// Tracea v3.1 - Structural Optimization Framework

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>


using namespace nvcuda;

// ============================================================================
// SoftmaxMode - Controls Online Softmax Update Granularity
// ============================================================================
enum SoftmaxMode {
  PER_TILE = 0,      // Update per K/V tile (baseline, most accurate)
  PER_TWO_TILES = 1, // Update every 2 tiles (reduced sync overhead)
  FULL_BR = 2        // Full Br rows at once (reserved for future)
};

// ============================================================================
// Primitives (cp.async for Ampere+)
// ============================================================================
__device__ __forceinline__ void cp_async_fa2(void *dst, const void *src,
                                             int bytes) {
  uint32_t dst_addr = (uint32_t)(uintptr_t)(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" ::"r"(dst_addr),
               "l"(src), "n"(16));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// ============================================================================
// FA2 Template Kernel
// ============================================================================
template <int PIPE_DEPTH,   // Pipeline stages (2, 3, 4)
          int SWIZZLE_MODE, // Smem swizzle (0=None, 1=Xor2, etc.)
          int BARRIER_MODE, // 0=None (syncthreads), 1=ProdCons (mbarrier)
          int SOFTMAX_MODE, // 0=PerTile, 1=PerTwoTiles, 2=FullBr
          int Br,           // Block rows (64, 128)
          int Bc,           // Block cols for K/V (32, 64)
          int Headdim       // Head dimension (64, 128)
          >
__global__ void __launch_bounds__(256, 1)
    fa2_template_kernel(const half *__restrict__ Q, const half *__restrict__ K,
                        const half *__restrict__ V, half *__restrict__ O, int B,
                        int H, int S, int D, float scale) {
  // ========================================================================
  // Rule A: Single Init Responsibility
  // ========================================================================
  extern __shared__ char smem[];

  int tile_idx = blockIdx.x;
  int h = blockIdx.y;
  int b = blockIdx.z;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  // Rule B: Static Role Assignment
  constexpr int PRODUCER_WARPS = 1;
  bool is_producer = (warp_id < PRODUCER_WARPS);
  int cons_warp = warp_id - PRODUCER_WARPS;

  // ========================================================================
  // Shared Memory Layout
  // ========================================================================
  constexpr int STRIDE = Headdim + 8; // Bank padding
  constexpr int STRIDE_S = Bc + 8;

  // K/V circular buffers for pipeline
  half *smem_K = (half *)(smem);
  half *smem_V = (half *)(smem + PIPE_DEPTH * Bc * STRIDE * sizeof(half));
  float *smem_S = (float *)(smem + 2 * PIPE_DEPTH * Bc * STRIDE * sizeof(half));
  half *smem_P = (half *)(smem_S + Br * STRIDE_S);
  float *smem_O = (float *)(smem_P + Br * Bc * sizeof(half));

  // ========================================================================
  // Offset Calculation
  // ========================================================================
  long long offset_base = (long long)b * H * S * D + (long long)h * S * D;
  Q += offset_base;
  K += offset_base;
  V += offset_base;
  O += offset_base;

  int q_row_start = tile_idx * Br + cons_warp * 16;

  // ========================================================================
  // Initialize Accumulators
  // ========================================================================
  constexpr int D_FRAGS = Headdim / 16;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[D_FRAGS];
#pragma unroll
  for (int k = 0; k < D_FRAGS; ++k) {
    wmma::fill_fragment(acc_O[k], 0.0f);
  }

  // Online Softmax State
  float m_prev[16]; // Running max per row
  float l_prev[16]; // Running sum (denominator)
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    m_prev[i] = -50000.0f;
    l_prev[i] = 0.0f;
  }

  // ========================================================================
  // Load Q (Consumers Only)
  // ========================================================================
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      frag_Q[D_FRAGS];
  if (!is_producer && cons_warp < (Br / 16)) {
#pragma unroll
    for (int d = 0; d < D_FRAGS; ++d) {
      // Load Q tile directly into registers
      if (q_row_start < S) {
        // Simplified load - actual implementation uses smem buffer
        // wmma::load_matrix_sync(frag_Q[d], ...);
      }
    }
  }
  __syncthreads();

  // ========================================================================
  // Main Loop: Iterate over K/V tiles
  // ========================================================================
  int total_tiles = (S + Bc - 1) / Bc;

  // --- Prologue: Pre-load first (PIPE_DEPTH - 1) tiles ---
  if (is_producer) {
    for (int s = 0; s < PIPE_DEPTH - 1 && s < total_tiles; ++s) {
      int stage = s;
      half *sK = smem_K + stage * Bc * STRIDE;
      half *sV = smem_V + stage * Bc * STRIDE;
      int k_start = s * Bc;

      // Load K and V tiles
      for (int idx = tid * 8; idx < Bc * Headdim; idx += 32 * 8) {
        int r = idx / Headdim;
        int c = idx % Headdim;
        if (k_start + r < S && c < D) {
          cp_async_fa2(&sK[r * STRIDE + c], &K[(k_start + r) * D + c], 16);
          cp_async_fa2(&sV[r * STRIDE + c], &V[(k_start + r) * D + c], 16);
        }
      }
      cp_async_commit();
    }
    cp_async_wait<PIPE_DEPTH - 2>();
  }
  __syncthreads();

  // Accumulator for PerTwoTiles mode
  float deferred_max[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    deferred_max[i] = -50000.0f;

  // --- Main Loop ---
  for (int k_tile = 0; k_tile < total_tiles; ++k_tile) {

    // ====================================================================
    // Consumer: Compute QK^T, Softmax, O update
    // ====================================================================
    if (!is_producer && cons_warp < (Br / 16)) {
      int stage = k_tile % PIPE_DEPTH;
      half *sK = smem_K + stage * Bc * STRIDE;
      half *sV = smem_V + stage * Bc * STRIDE;

      // Step 1: Compute S = Q * K^T (per 16x16 tile)
      for (int step = 0; step < Bc / 16; ++step) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S;
        wmma::fill_fragment(acc_S, 0.0f);

#pragma unroll
        for (int d = 0; d < D_FRAGS; ++d) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
              frag_K;
          wmma::load_matrix_sync(frag_K, sK + step * 16 * STRIDE + d * 16,
                                 STRIDE);
          wmma::mma_sync(acc_S, frag_Q[d], frag_K, acc_S);
        }

        // Store S to shared memory for softmax
        wmma::store_matrix_sync(smem_S + cons_warp * 16 * STRIDE_S + step * 16,
                                acc_S, STRIDE_S, wmma::mem_row_major);
        __syncwarp();

        // Step 2: Online Softmax with SoftmaxMode branching
        if constexpr (SOFTMAX_MODE == PER_TILE) {
          // === PerTile: Update max/sum every tile ===
          float local_max = -50000.0f;
          if (lane_id < 16) {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
              int col_glob = k_tile * Bc + step * 16 + c;
              float sv = smem_S[cons_warp * 16 * STRIDE_S + lane_id * STRIDE_S +
                                step * 16 + c] *
                         scale;
              if (col_glob >= S || q_row_start + lane_id >= S)
                sv = -50000.0f;
              if (sv > local_max)
                local_max = sv;
            }
          }

          // Compute new max
          float m_new[16];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float cur_m = __shfl_sync(
                0xffffffff, (lane_id < 16) ? local_max : -50000.0f, i);
            m_new[i] = fmaxf(m_prev[i], cur_m);
          }

// Rescale O accumulator
#pragma unroll
          for (int d = 0; d < D_FRAGS; ++d) {
#pragma unroll
            for (int i = 0; i < acc_O[d].num_elements; ++i) {
              float exp_diff = expf(m_prev[i % 16] - m_new[i % 16]);
              acc_O[d].x[i] *= exp_diff;
            }
          }

          // Compute P = exp(S - m_new), update l
          float row_sum = 0.0f;
          if (lane_id < 16) {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
              int col_glob = k_tile * Bc + step * 16 + c;
              float sv = smem_S[cons_warp * 16 * STRIDE_S + lane_id * STRIDE_S +
                                step * 16 + c] *
                         scale;
              if (col_glob >= S || q_row_start + lane_id >= S)
                sv = -50000.0f;
              float p = expf(sv - m_new[lane_id]);
              smem_P[cons_warp * 16 * Bc + lane_id * Bc + step * 16 + c] =
                  __float2half(p);
              row_sum += p;
            }
          }
          __syncwarp();

// Update l_prev
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float sum_i =
                __shfl_sync(0xffffffff, (lane_id < 16) ? row_sum : 0.0f, i);
            float exp_diff = expf(m_prev[i] - m_new[i]);
            l_prev[i] = l_prev[i] * exp_diff + sum_i;
            m_prev[i] = m_new[i];
          }

        } else if constexpr (SOFTMAX_MODE == PER_TWO_TILES) {
          // === PerTwoTiles: Accumulate local max, update every 2 steps ===
          float local_max = -50000.0f;
          if (lane_id < 16) {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
              int col_glob = k_tile * Bc + step * 16 + c;
              float sv = smem_S[cons_warp * 16 * STRIDE_S + lane_id * STRIDE_S +
                                step * 16 + c] *
                         scale;
              if (col_glob >= S || q_row_start + lane_id >= S)
                sv = -50000.0f;
              if (sv > local_max)
                local_max = sv;
            }
          }

// Accumulate deferred max
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float cur_m = __shfl_sync(
                0xffffffff, (lane_id < 16) ? local_max : -50000.0f, i);
            deferred_max[i] = fmaxf(deferred_max[i], cur_m);
          }

          // Only update on odd steps (every 2 tiles)
          if (step % 2 == 1) {
            float m_new[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
              m_new[i] = fmaxf(m_prev[i], deferred_max[i]);
            }

// Rescale O accumulator
#pragma unroll
            for (int d = 0; d < D_FRAGS; ++d) {
#pragma unroll
              for (int i = 0; i < acc_O[d].num_elements; ++i) {
                float exp_diff = expf(m_prev[i % 16] - m_new[i % 16]);
                acc_O[d].x[i] *= exp_diff;
              }
            }

            // TODO: Compute P for both accumulated steps
            // (Simplified for v3.1 - full implementation requires storing 2 S
            // tiles)

#pragma unroll
            for (int i = 0; i < 16; ++i) {
              m_prev[i] = m_new[i];
              deferred_max[i] = -50000.0f; // Reset
            }
          }
        }
        // FULL_BR is reserved for future optimization

        // Step 3: Compute O += P * V
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            frag_P;
        wmma::load_matrix_sync(frag_P, smem_P + cons_warp * 16 * Bc + step * 16,
                               Bc);

#pragma unroll
        for (int d = 0; d < D_FRAGS; ++d) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
              frag_V;
          wmma::load_matrix_sync(frag_V, sV + step * 16 * STRIDE + d * 16,
                                 STRIDE);
          wmma::mma_sync(acc_O[d], frag_P, frag_V, acc_O[d]);
        }
      }
    }

    // ====================================================================
    // Producer: Prefetch Next K/V Tile
    // ====================================================================
    if (is_producer) {
      int next_k = k_tile + PIPE_DEPTH - 1;
      if (next_k < total_tiles) {
        int stage = next_k % PIPE_DEPTH;
        half *sK = smem_K + stage * Bc * STRIDE;
        half *sV = smem_V + stage * Bc * STRIDE;
        int k_start = next_k * Bc;

        for (int idx = tid * 8; idx < Bc * Headdim; idx += 32 * 8) {
          int r = idx / Headdim;
          int c = idx % Headdim;
          if (k_start + r < S && c < D) {
            cp_async_fa2(&sK[r * STRIDE + c], &K[(k_start + r) * D + c], 16);
            cp_async_fa2(&sV[r * STRIDE + c], &V[(k_start + r) * D + c], 16);
          }
        }
        cp_async_commit();
        cp_async_wait<PIPE_DEPTH - 2>();
      } else {
        cp_async_wait<0>();
      }
    }
    __syncthreads();
  }

  // ========================================================================
  // Epilogue: Normalize O by l and store
  // ========================================================================
  if (!is_producer && cons_warp < (Br / 16)) {
    if (q_row_start < S) {
#pragma unroll
      for (int d = 0; d < D_FRAGS; ++d) {
        wmma::store_matrix_sync(smem_O + cons_warp * 16 * Headdim + d * 16,
                                acc_O[d], Headdim, wmma::mem_row_major);
      }
      __syncwarp();

      // Final normalization
      if (lane_id < 16 && q_row_start + lane_id < S) {
        float l = l_prev[lane_id];
#pragma unroll
        for (int d = 0; d < D_FRAGS; ++d) {
          float *src = smem_O + (cons_warp * 16 + lane_id) * Headdim + d * 16;
          half *dst = O + (q_row_start + lane_id) * D + d * 16;
#pragma unroll
          for (int c = 0; c < 16; ++c) {
            dst[c] = __float2half(src[c] / (l + 1e-6f));
          }
        }
      }
    }
  }
}

// ============================================================================
// Explicit Instantiations for HeroScope Dispatch
// ============================================================================
extern "C" {

// Common configurations for FA2
void launch_fa2_p2_s0_b0_sm0_br64_bc32_d64(const half *Q, const half *K,
                                           const half *V, half *O, int B, int H,
                                           int S, int D, float scale, int gridX,
                                           int gridY, int gridZ, int blockX,
                                           int blockY, int blockZ, int smem,
                                           void *stream) {
  dim3 grid(gridX, gridY, gridZ);
  dim3 block(blockX, blockY, blockZ);
  fa2_template_kernel<2, 0, 0, PER_TILE, 64, 32, 64>
      <<<grid, block, smem, (cudaStream_t)stream>>>(Q, K, V, O, B, H, S, D,
                                                    scale);
}

void launch_fa2_p3_s0_b1_sm0_br128_bc64_d64(const half *Q, const half *K,
                                            const half *V, half *O, int B,
                                            int H, int S, int D, float scale,
                                            int gridX, int gridY, int gridZ,
                                            int blockX, int blockY, int blockZ,
                                            int smem, void *stream) {
  dim3 grid(gridX, gridY, gridZ);
  dim3 block(blockX, blockY, blockZ);
  fa2_template_kernel<3, 0, 1, PER_TILE, 128, 64, 64>
      <<<grid, block, smem, (cudaStream_t)stream>>>(Q, K, V, O, B, H, S, D,
                                                    scale);
}

void launch_fa2_p3_s0_b1_sm1_br128_bc64_d64(const half *Q, const half *K,
                                            const half *V, half *O, int B,
                                            int H, int S, int D, float scale,
                                            int gridX, int gridY, int gridZ,
                                            int blockX, int blockY, int blockZ,
                                            int smem, void *stream) {
  dim3 grid(gridX, gridY, gridZ);
  dim3 block(blockX, blockY, blockZ);
  fa2_template_kernel<3, 0, 1, PER_TWO_TILES, 128, 64, 64>
      <<<grid, block, smem, (cudaStream_t)stream>>>(Q, K, V, O, B, H, S, D,
                                                    scale);
}

} // extern "C"
