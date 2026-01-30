use crate::emitter::traits::UnifiedOpIR;
use crate::core::config::PipelineConfig;
use crate::backend::cuda::CudaBackend;

pub fn generate_low_rank_mlp(ir: &UnifiedOpIR) -> String {
    let (m, n, k, r_dim) = if let crate::emitter::traits::UnifiedOpType::LowRankMlp { m, n, k, r } = &ir.op_type {
        (*m, *n, *k, *r)
    } else {
        panic!("Invalid OpType for low_rank_mlp");
    };

    let config = &ir.tiling;
    let mt = config.m_tile;
    let nt = config.n_tile;
    let kt = config.k_tile;
    
    // We assume r fits in smem or is tiled. For simplicity, assume r fits in smem if it's <= 128.
    // We'll use a fixed r_tile for now if it's larger.
    let rt = if r_dim <= 64 { r_dim } else { 32 }; 

    let stages = config.num_stages;
    let num_warps = if let Some(fw) = config.force_num_warps { fw } else { 4 };
    
    let a_stride = kt + 8;
    let b_stride = nt + 8;
    let r_stride = rt + 8;

    // Design:
    // Smem Layout:
    // [Header]
    // [sX: MT x KT x Stages]
    // [sA: KT x RT x Stages]
    // [sT: MT x RT] (Intermediate accumulator)
    // [sB: RT x NT x Stages]

    let smem_x_bytes = mt * a_stride * 2;
    let smem_a_bytes = kt * r_stride * 2;
    let smem_t_bytes = mt * r_stride * 2;
    let smem_b_bytes = rt * b_stride * 2;

    let x_offset = 128;
    let a_offset = (x_offset + smem_x_bytes * stages as u32 + 255) & !255;
    let t_offset = (a_offset + smem_a_bytes * stages as u32 + 255) & !255;
    let b_offset = (t_offset + smem_t_bytes + 255) & !255;

    format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define MT {mt}
#define NT {nt}
#define KT {kt}
#define RT {rt}
#define R_DIM {r_dim}
#define STAGES {stages}
#define NUM_WARPS {num_warps}

{primitives}

extern "C" __global__ void low_rank_mlp_kernel(
    const half* __restrict__ X,    // [M, K]
    const half* __restrict__ A,    // [K, R]
    const half* __restrict__ B,    // [R, N]
    half* __restrict__ C_global,  // [M, N]
    int M, int N, int K,
    // TTG
    const uint* __restrict__ l1_active_tiles,
    const unsigned char* __restrict__ l2_metadata
) {{
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    // --- TTG Indirection ---
    int tile_idx_m, tile_idx_n;
    #if {ttg_enabled}
        uint physical_id = blockIdx.x;
        uint logical_id = l1_active_tiles[physical_id];
        const uint* l2_ptr = (const uint*)(l2_metadata + logical_id * 20);
        tile_idx_m = l2_ptr[0];
        tile_idx_n = l2_ptr[1];
    #else
        tile_idx_m = blockIdx.y;
        tile_idx_n = blockIdx.x;
    #endif

    int x_tile_row = tile_idx_m * MT;
    int b_tile_col = tile_idx_n * NT;

    extern __shared__ char smem[];
    half* sX_base = (half*)(smem + {x_offset});
    half* sA_base = (half*)(smem + {a_offset});
    half* sT = (half*)(smem + {t_offset});
    half* sB_base = (half*)(smem + {b_offset});

    // 1. Initialize sT (Intermediate MT x RT)
    for (int i = tid; i < MT * {r_stride}; i += NUM_WARPS * 32) {{
        sT[i] = __float2half(0.0f);
    }}
    __syncthreads();

    // 2. Stage 1: T = X * A (M x R = (M x K) * (K x R))
    // Simplification: We do this block-by-block. 
    // In a single CTA, we compute the MT x RT part of T.
    // Wait, if R_DIM > RT, we need another loop. 
    // For now, assume R_DIM == RT for simplicity or handle loop.

    for (int r_step = 0; r_step < R_DIM; r_step += RT) {{
        // Reset sT for this r_step if needed, or accumulate.
        // If we want to compute MT x NT, we only need the RELEVANT RT rows of B.
        // But for T = X * A, we need ALL of K but only RT of A.

        for (int k_tile = 0; k_tile < (K + KT - 1) / KT; ++k_tile) {{
            // Load X and A tiles
            #pragma unroll
            for (int i = tid; i < (MT * KT) / 8; i += NUM_WARPS * 32) {{
                int m_in = (i * 8) / KT;
                int k_in = (i * 8) % KT;
                int glob_k = k_tile * KT + k_in;
                if (x_tile_row + m_in < M && glob_k < K)
                    sX_base[(k_tile % STAGES) * (MT * {a_stride}) + m_in * {a_stride} + k_in] = X[(x_tile_row + m_in) * K + glob_k];
                else
                    sX_base[(k_tile % STAGES) * (MT * {a_stride}) + m_in * {a_stride} + k_in] = __float2half(0.0f);
            }}
            #pragma unroll
            for (int i = tid; i < (KT * RT) / 8; i += NUM_WARPS * 32) {{
                int k_in = (i * 8) / RT;
                int r_in = (i * 8) % RT;
                int glob_k = k_tile * KT + k_in;
                int glob_r = r_step + r_in;
                if (glob_k < K && glob_r < R_DIM)
                    sA_base[(k_tile % STAGES) * (KT * {r_stride}) + k_in * {r_stride} + r_in] = A[glob_k * R_DIM + glob_r];
                else
                    sA_base[(k_tile % STAGES) * (KT * {r_stride}) + k_in * {r_stride} + r_in] = __float2half(0.0f);
            }}
            __syncthreads();

            // MMA for T += X * A
            // (Warp tiling for intermediate T)
            // ... (Skipping complex warp tiling for P0, using simple loop)
            if (warp_id < 4) {{ // Assume 4 warps for MMA
                // Very basic load/mma
            }}
            __syncthreads();
        }}

        // 3. Stage 2: C = T * B (M x N = (M x R) * (R x N))
        // Now sT has MT x RT. We load B (RT x NT) and compute.
        for (int i = tid; i < (RT * NT) / 8; i += NUM_WARPS * 32) {{
             int r_in = (i * 8) / NT;
             int n_in = (i * 8) % NT;
             int glob_r = r_step + r_in;
             int glob_n = b_tile_col + n_in;
             if (glob_r < R_DIM && glob_n < N)
                 sB_base[r_in * {b_stride} + n_in] = B[glob_r * N + glob_n];
             else
                 sB_base[r_in * {b_stride} + n_in] = __float2half(0.0f);
        }}
        __syncthreads();

        // Accumulate into frag_acc (C_global)
        // ...
    }}

    // Placeholder for P0: Final Store
    // (Real implementation would use WMMA fragments for efficiency)
    // For now, this is a blueprint.
}}
"#, mt=mt, nt=nt, kt=kt, rt=rt, r_dim=r_dim, stages=stages, num_warps=num_warps,
primitives=CudaBackend::get_primitive_defs(),
x_offset=x_offset, a_offset=a_offset, t_offset=t_offset, b_offset=b_offset,
a_stride=a_stride, b_stride=b_stride, r_stride=r_stride,
ttg_enabled=(if config.ttg_enabled { 1 } else { 0 }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
    use crate::core::config::PipelineConfig;

    #[test]
    fn test_low_rank_mlp_generation() {
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::LowRankMlp { m: 256, n: 256, k: 256, r: 64 },
            precison: "f16".to_string(),
            tiling: PipelineConfig::new(2, 64, 64, 32),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };
        let source = generate_low_rank_mlp(&ir);
        assert!(source.contains("low_rank_mlp_kernel"));
        assert!(source.contains("sX_base"));
        assert!(source.contains("sA_base"));
    }
}
