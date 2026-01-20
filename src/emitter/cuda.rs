use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
pub use crate::kernels::attention::cuda_emitter::FlashAttentionEmitter;
use crate::core::config::PipelineConfig;
use crate::backend::cuda::CudaBackend;
use crate::emitter::layout::LayoutPolicy;

pub struct CUDAEmitter {}

impl CUDAEmitter {
    pub fn new() -> Self {
        Self {}
    }

    fn generate_gemm(&self, m: u32, n: u32, k: u32, config: &PipelineConfig) -> String {
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        let stages = config.num_stages;
        let num_warps = config.force_num_warps.unwrap_or(8);
        let smem_a_bytes = mt * kt * 2;
        let smem_b_bytes = kt * nt * 2;

        format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define MT {mt}
#define NT {nt}
#define KT {kt}
#define STAGES {stages}
#define NUM_WARPS {num_warps}
#define PRODUCER_WARPS 1

{primitives}

extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32, 1) gemm_mma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C_global,
    int M, int N, int K
) {{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    bool is_producer = (warp_id < PRODUCER_WARPS);

    extern __shared__ char smem[];
    int a_smem_offset = 128;
    int b_smem_offset = a_smem_offset + {smem_a_bytes} * STAGES;

    int a_tile_row = blockIdx.y * MT;
    int b_tile_col = blockIdx.x * NT;
    int cons_warp = warp_id - PRODUCER_WARPS;
    int mt_per_warp = MT / 8;
    const int M_FRAGS = MT / 8 / 16;
    const int N_FRAGS = NT / 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc[M_FRAGS][N_FRAGS];
    #pragma unroll
    for(int mi=0; mi<M_FRAGS; mi++) {{
        for(int ni=0; ni<N_FRAGS; ni++) {{
            wmma::fill_fragment(frag_acc[mi][ni], (half)0.0f);
        }}
    }}

    for (int k_tile = 0; k_tile < (K + KT - 1) / KT; ++k_tile) {{
        if (is_producer) {{
            int stage = k_tile % STAGES;
            half* sA = (half*)(smem + a_smem_offset + stage * {smem_a_bytes});
            half* sB = (half*)(smem + b_smem_offset + stage * {smem_b_bytes});
            #pragma unroll
            for (int i = tid; i < (MT * KT) / 8; i += 32) {{
                int m = (i * 8) / KT;
                int k = (i * 8) % KT;
                if (a_tile_row + m < M && k_tile * KT + k < K)
                    cp_async_ampere(sA + m * KT + k, A + (a_tile_row + m) * K + (k_tile * KT + k), 16);
            }}
            #pragma unroll
            for (int i = tid; i < (KT * NT) / 8; i += 32) {{
                int k = (i * 8) / NT;
                int n = (i * 8) % NT;
                if (k_tile * KT + k < K && b_tile_col + n < N)
                    cp_async_ampere(sB + k * NT + n, B + (k_tile * KT + k) * N + (b_tile_col + n), 16);
            }}
            cp_async_commit_group();
            cp_async_wait_group_0();
        }}
        __syncthreads();

        if (!is_producer) {{
            int stage = k_tile % STAGES;
            half* sA = (half*)(smem + a_smem_offset + stage * {smem_a_bytes});
            half* sB = (half*)(smem + b_smem_offset + stage * {smem_b_bytes});
            for (int k_inner = 0; k_inner < KT; k_inner += 16) {{
                #pragma unroll
                for (int mi = 0; mi < M_FRAGS; ++mi) {{
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
                    wmma::load_matrix_sync(frag_a, sA + (cons_warp * mt_per_warp + mi * 16) * KT + k_inner, KT);
                    #pragma unroll
                    for (int ni = 0; ni < N_FRAGS; ++ni) {{
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
                        wmma::load_matrix_sync(frag_b, sB + k_inner * NT + ni * 16, NT);
                        wmma::mma_sync(frag_acc[mi][ni], frag_a, frag_b, frag_acc[mi][ni]);
                    }}
                }}
            }}
        }}
        __syncthreads();
    }}

    if (!is_producer) {{
        #pragma unroll
        for (int mi = 0; mi < M_FRAGS; ++mi) {{
             #pragma unroll
             for (int ni = 0; ni < N_FRAGS; ++ni) {{
                 int row = a_tile_row + cons_warp * mt_per_warp + mi * 16;
                 int col = b_tile_col + ni * 16;
                 if (row < M && col < N)
                     wmma::store_matrix_sync((half*)C_global + row * N + col, frag_acc[mi][ni], N, wmma::mem_row_major);
             }}
        }}
    }}
}}
"# , mt=mt, nt=nt, kt=kt, stages=stages, num_warps=num_warps, smem_a_bytes=smem_a_bytes, smem_b_bytes=smem_b_bytes, primitives=CudaBackend::get_primitive_defs())
    }
}

impl Emitter for CUDAEmitter {
    fn emit_sync(&mut self, _req: crate::semantic::transition::SyncRequirement) -> String {
        "__syncthreads();\n".to_string()
    }
    fn generate_from_ir(&self, ir: &UnifiedOpIR) -> String {
        match &ir.op_type {
            UnifiedOpType::FusedAttention { b: _b, s: _s, d: _d, h, dh, causal } => {
                let emitter = FlashAttentionEmitter::new(ir.tiling.clone());
                emitter.generate_kernel(*h as usize, *dh as usize, *causal)
            }
            UnifiedOpType::Gemm { m, n, k } => {
                self.generate_gemm(*m, *n, *k, &ir.tiling)
            }
        }
    }
}
