use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
pub use crate::kernels::attention::cuda_emitter::FlashAttentionEmitter;
use crate::core::config::{PipelineConfig, LayoutPolicy};
use crate::core::op::EpilogueOp;
use crate::backend::cuda::CudaBackend;

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
        let consumers = if let Some(fw) = config.force_num_warps {
            fw - 1
        } else {
            let max_consumers = mt / 16;
            if max_consumers >= 8 { 8 }
            else if max_consumers >= 4 { 4 }
            else if max_consumers >= 2 { 2 }
            else { 1 }
        };
        let num_warps = consumers + 1;
        
        let a_stride = kt + 8;
        let b_stride = nt + 8;
        let smem_a_bytes = mt * a_stride * 2;
        let smem_b_bytes = kt * b_stride * 2;

        let mut epilogue_args = String::new();
        let mut epilogue_apply = String::new();
        
        for (i, op) in config.epilogue.iter().enumerate() {
            match op {
                EpilogueOp::BiasAdd { .. } => {
                    epilogue_args.push_str(&format!(", const float* __restrict__ bias_{}", i));
                    epilogue_apply.push_str(&format!("tracea::epilogue::BiasAdd op_{}; op_{}.bias = bias_{}; val = op_{}(val, n_glob);\n", i, i, i, i));
                }
                EpilogueOp::ReLU => {
                    epilogue_apply.push_str(&format!("tracea::epilogue::ReLU op_{}; val = op_{}(val);\n", i, i));
                }
                EpilogueOp::Gelu => {
                    epilogue_apply.push_str(&format!("tracea::epilogue::Gelu op_{}; val = op_{}(val);\n", i, i));
                }
                EpilogueOp::SiLU => {
                    epilogue_apply.push_str(&format!("tracea::epilogue::SiLU op_{}; val = op_{}(val);\n", i, i));
                }
                EpilogueOp::ResidualAdd { .. } => {
                    epilogue_args.push_str(&format!(", const float* __restrict__ residual_{}", i));
                    // Residual add usually adds element-wise from distinct tensor, assumed (M, N) layout
                    epilogue_apply.push_str(&format!("tracea::epilogue::ResidualAdd op_{}; op_{}.residual = residual_{}; val = op_{}(val, (long long)m_glob * K_OUT + n_glob);\n", i, i, i, i));
                }
                EpilogueOp::BiasAddSiLU { .. } => {
                     epilogue_args.push_str(&format!(", const float* __restrict__ bias_{}", i));
                     epilogue_apply.push_str(&format!("tracea::epilogue::BiasAddSiLU op_{}; op_{}.bias = bias_{}; val = op_{}(val, n_glob);\n", i, i, i, i));
                }
                _ => {}
            }
        }

        format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

{epilogue_defs}

using namespace nvcuda;

#define MT {mt}
#define NT {nt}
#define KT {kt}
#define STAGES {stages}
#define NUM_WARPS {num_warps}
#define PRODUCER_WARPS 1
#define A_STRIDE {a_stride}
#define B_STRIDE {b_stride}

{primitives}

extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32, 1) gemm_mma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C_global,
    int M, int N, int K{epilogue_args}
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
    int mt_per_warp = MT / (NUM_WARPS - PRODUCER_WARPS);
    const int M_FRAGS = MT / (NUM_WARPS - PRODUCER_WARPS) / 16;
    const int N_FRAGS = NT / 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc[M_FRAGS][N_FRAGS];
    #pragma unroll
    for(int mi=0; mi<M_FRAGS; mi++) {{
        for(int ni=0; ni<N_FRAGS; ni++) {{
            wmma::fill_fragment(frag_acc[mi][ni], 0.0f);
        }}
    }}

    int total_tiles = (K + KT - 1) / KT;

    // PROLOGUE: Pre-load STAGES - 1 tiles
    if (is_producer) {{
        for (int s = 0; s < STAGES - 1; ++s) {{
            if (s < total_tiles) {{
                int stage = s; // Simple direct mapping for prologue
                half* sA = (half*)(smem + a_smem_offset + stage * {smem_a_bytes});
                half* sB = (half*)(smem + b_smem_offset + stage * {smem_b_bytes});
                
                int k_tile = s;
                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {{
                    int m = (i * 8) / KT;
                    int k = (i * 8) % KT;
                    if (a_tile_row + m < M && k_tile * KT + k < K)
                        cp_async_ampere(sA + m * A_STRIDE + k, A + (a_tile_row + m) * K + (k_tile * KT + k), 16);
                }}
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {{
                    int k = (i * 8) / NT;
                    int n = (i * 8) % NT;
                    if (k_tile * KT + k < K && b_tile_col + n < N)
                        cp_async_ampere(sB + k * B_STRIDE + n, B + (k_tile * KT + k) * N + (b_tile_col + n), 16);
                }}
                cp_async_commit_group();
            }}
        }}
        // Ensure Tile 0 is ready. (STAGES-1 committed. Keep STAGES-2 in flight).
        cp_async_wait_group<STAGES - 2>();
    }}
    __syncthreads();

    // MAIN LOOP
    for (int k_tile = 0; k_tile < total_tiles; ++k_tile) {{
        if (!is_producer) {{
            int stage = k_tile % STAGES;
            half* sA = (half*)(smem + a_smem_offset + stage * {smem_a_bytes});
            half* sB = (half*)(smem + b_smem_offset + stage * {smem_b_bytes});
            for (int k_inner = 0; k_inner < KT; k_inner += 16) {{
                #pragma unroll
                for (int mi = 0; mi < M_FRAGS; ++mi) {{
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
                    wmma::load_matrix_sync(frag_a, sA + (cons_warp * mt_per_warp + mi * 16) * A_STRIDE + k_inner, A_STRIDE);
                    #pragma unroll
                    for (int ni = 0; ni < N_FRAGS; ++ni) {{
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
                        wmma::load_matrix_sync(frag_b, sB + k_inner * B_STRIDE + ni * 16, B_STRIDE);
                        wmma::mma_sync(frag_acc[mi][ni], frag_a, frag_b, frag_acc[mi][ni]);
                    }}
                }}
            }}
        }}
        
        if (is_producer) {{
            int next_k = k_tile + STAGES - 1;
            if (next_k < total_tiles) {{
                int stage = next_k % STAGES;
                half* sA = (half*)(smem + a_smem_offset + stage * {smem_a_bytes});
                half* sB = (half*)(smem + b_smem_offset + stage * {smem_b_bytes});
                
                int k_next_tile_idx = next_k; // rename to handle capture
                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {{
                    int m = (i * 8) / KT;
                    int k = (i * 8) % KT;
                    if (a_tile_row + m < M && k_next_tile_idx * KT + k < K)
                        cp_async_ampere(sA + m * A_STRIDE + k, A + (a_tile_row + m) * K + (k_next_tile_idx * KT + k), 16);
                }}
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {{
                    int k = (i * 8) / NT;
                    int n = (i * 8) % NT;
                    if (k_next_tile_idx * KT + k < K && b_tile_col + n < N)
                        cp_async_ampere(sB + k * B_STRIDE + n, B + (k_next_tile_idx * KT + k) * N + (b_tile_col + n), 16);
                }}
                cp_async_commit_group();
                // Ensure k_tile + 1 is ready for next iter
                cp_async_wait_group<STAGES - 2>();
            }} else {{
                // Epilogue drain: Ensure remaining tiles ready. Safe fallback.
                cp_async_wait_group<0>();
            }}
        }}
        __syncthreads();
    }}

    if (!is_producer) {{
        float* sC = (float*)smem;
        #pragma unroll
        for (int mi = 0; mi < M_FRAGS; ++mi) {{
             #pragma unroll
             for (int ni = 0; ni < N_FRAGS; ++ni) {{
                 int row_tile_offset = (cons_warp * mt_per_warp + mi * 16);
                 int col_tile_offset = (ni * 16);
                 // Store to padded smem to avoid bank conflicts if needed, but row major here.
                 wmma::store_matrix_sync(sC + row_tile_offset * NT + col_tile_offset, frag_acc[mi][ni], NT, wmma::mem_row_major);
             }}
        }}
        __syncthreads();

        int epis_tid = tid - PRODUCER_WARPS * 32;
        #pragma unroll
        for (int i = epis_tid; i < MT * NT; i += (NUM_WARPS - PRODUCER_WARPS) * 32) {{
             // Logic to distribute work among consumers
             // Or can use strict mapping. 
             // Simple grid stride over the tile:
             int r = i / NT;
             int c = i % NT;
             
             int row = a_tile_row + r;
             int col = b_tile_col + c;

             if (row < M && col < N) {{
                 int n_glob = col;
                 int m_glob = row;
                 float val = sC[i];
                 {epilogue_apply}
                 C_global[(long long)row * N + col] = (half)val;
             }}
        }}
    }}
}}
"# , mt=mt, nt=nt, kt=kt, stages=stages, num_warps=num_warps, smem_a_bytes=smem_a_bytes, smem_b_bytes=smem_b_bytes, 
primitives=CudaBackend::get_primitive_defs(), 
epilogue_defs=include_str!("../kernels/gpu/epilogue.cuh"),
epilogue_args=epilogue_args,
epilogue_apply=epilogue_apply)
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
            UnifiedOpType::Elementwise { .. } => {
                panic!("Elementwise Ops should be handled by UniversalEmitter.");
            }
            UnifiedOpType::Conv2d { .. } => {
                crate::emitter::conv::generate_conv(ir)
            }
            UnifiedOpType::ConvTranspose2d { .. } => {
                crate::emitter::conv_transpose::generate_conv_transpose(ir)
            }
        }
    }
}

