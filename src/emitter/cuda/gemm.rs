use crate::core::config::PipelineConfig;
use crate::core::op::EpilogueOp;
use crate::backend::cuda::CudaBackend;
use crate::emitter::traits::EmissionError;

pub fn generate_gemm(_m: u32, _n: u32, _k: u32, config: &PipelineConfig) -> Result<String, EmissionError> {
    let mt = config.m_tile;
    let nt = config.n_tile;
    let kt = config.k_tile;
    let stages = config.num_stages.max(2);
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
    
    let padding = config.bank_conflict_padding;
    let a_stride = kt + padding;
    let b_stride = nt + padding;
    let multiplier = if config.double_buffer { 2 } else { 1 };
    let smem_a_bytes = mt * a_stride * multiplier;
    let _smem_b_bytes = kt * b_stride * multiplier;
    
    let a_smem_offset = 128; // Header
    let b_smem_offset = (a_smem_offset + smem_a_bytes * stages as u32 + 1023) & !1023;

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
                epilogue_apply.push_str(&format!("tracea::epilogue::ResidualAdd op_{}; op_{}.residual = residual_{}; val = op_{}(val, (long long)m_glob * N + n_glob);\n", i, i, i, i));
            }
            EpilogueOp::BiasAddSiLU { .. } => {
                 epilogue_args.push_str(&format!(", const float* __restrict__ bias_{}", i));
                 epilogue_apply.push_str(&format!("tracea::epilogue::BiasAddSiLU op_{}; op_{}.bias = bias_{}; val = op_{}(val, n_glob);\n", i, i, i, i));
            }
            _ => {}
        }
    }

    let (warp_m, warp_n) = if consumers == 16 && mt == 128 {
        (8, 2)
    } else {
        (consumers, 1)
    };
    if mt / warp_m < 16 {
         return Err(EmissionError::InvalidTileConfiguration {
             reason: format!("Invalid Warp Partitioning: MT={} / WarpsM={} < 16. M_FRAGS would be 0.", mt, warp_m),
         });
    }

    Ok(format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

// Typedef for NVRTC
typedef unsigned int uint;

using namespace nvcuda;

#define MT {mt}
#define NT {nt}
#define KT {kt}
#define STAGES {stages}

{primitives}

{epilogue_defs}

extern "C" __global__ void gemm_mma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K{epilogue_args}) 
{{
    extern __shared__ char smem[];
    half* sA = (half*)(smem + {a_smem_offset});
    half* sB = (half*)(smem + {b_smem_offset});

    int tx = threadIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    int warp_row = warp_id % {warp_m};
    int warp_col = warp_id / {warp_m};

    int block_row = blockIdx.y * MT;
    int block_col = blockIdx.x * NT;

    const half* A_ptr = A + block_row * K;
    const half* B_ptr = B + block_col; // Assuming Layout is B is Col-major or similar, but simplified NHWC helper
    half* C_global = C + block_row * N + block_col;

    // MMA Fragments
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[{mt} / 16 / {warp_m}][{nt} / 16 / {warp_n}];
    for(int i = 0; i < {mt}/16/{warp_m}; ++i) {{
         for(int j = 0; j < {nt}/16/{warp_n}; ++j) {{
              wmma::fill_fragment(accum[i][j], 0.0f);
         }}
    }}

    int stage = 0;
    // Main loop with dynamic multi-stage pipeline (conceptually double-buffered / software pipelined)
    for (int k_idx = 0; k_idx < K; k_idx += KT) {{
         // Load from Global to Shared Memory (Cooperative load)
         // sA Load
         for (int i = tx; i < MT * KT; i += {num_warps} * 32) {{
              int r = i / KT;
              int c = i % KT;
              int glob_r = block_row + r;
              int glob_c = k_idx + c;
              if (glob_r < M && glob_c < K) {{
                   sA[stage * MT * (KT + 8) + r * (KT + 8) + c] = A_ptr[r * K + c];
              }} else {{
                   sA[stage * MT * (KT + 8) + r * (KT + 8) + c] = __float2half(0.0f);
              }}
         }}
         // sB Load
         for (int i = tx; i < KT * NT; i += {num_warps} * 32) {{
              int r = i / NT;
              int c = i % NT;
              int glob_r = k_idx + r;
              int glob_c = block_col + c;
              if (glob_r < K && glob_c < N) {{
                   sB[stage * KT * (NT + 8) + r * (NT + 8) + c] = B_ptr[r * N + c];
              }} else {{
                   sB[stage * KT * (NT + 8) + r * (NT + 8) + c] = __float2half(0.0f);
              }}
         }}
         __syncthreads();

         // Warp Compute
         for (int kk = 0; kk < KT; kk += 16) {{
              // Load fragments & MMA
              for (int i = 0; i < MT/16/{warp_m}; ++i) {{
                   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
                   int sa_row = warp_row * (MT/{warp_m}) + i * 16;
                   wmma::load_matrix_sync(frag_a, &sA[stage * MT * (KT+8) + sa_row * (KT+8) + kk], KT+8);

                   for (int j = 0; j < NT/16/{warp_n}; ++j) {{
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
                        int sb_col = warp_col * (NT/{warp_n}) + j * 16;
                        wmma::load_matrix_sync(frag_b, &sB[stage * KT * (NT+8) + kk * (NT+8) + sb_col], NT+8);

                        wmma::mma_sync(accum[i][j], frag_a, frag_b, accum[i][j]);
                   }}
              }}
         }}
         __syncthreads();
         stage = (stage + 1) % STAGES;
    }}

    // Write-back to global C with Epilogue evaluation
    // We reuse sA/sB as shared memory workspace for output if necessary, 
    // or write directly from fragments to C global.
    // For general MMA alignment, we can write accumulator fragments to shared memory first to resolve coalescing.
    float* sC = (float*)smem;
    for (int i = 0; i < MT/16/{warp_m}; ++i) {{
         int a_tile_row = warp_row * (MT/{warp_m}) + i * 16;
         for (int j = 0; j < NT/16/{warp_n}; ++j) {{
              int b_tile_col = warp_col * (NT/{warp_n}) + j * 16;
              
              // Store float accumulator to shared
              // Since wmma store expects float type matching accumulator
              wmma::store_matrix_sync(&sC[warp_id * 256], accum[i][j], 16, wmma::mem_row_major);
              __syncthreads();

              // Threads in warp write back to Global C with custom epilogue
              for (int e = tx % 32; e < 256; e += 32) {{
                   int r = e / 16;
                   int c = e % 16;
                   int row = a_tile_row + r;
                   int col = b_tile_col + c;
                   if (row < M && col < N) {{
                       int n_glob = col;
                       int m_glob = row;
                       float val = sC[warp_id * 256 + e];
                       {epilogue_apply}
                       C_global[(long long)row * N + col] = (half)val;
                   }}
              }}
         }}
    }}
}}
"# , mt=mt, nt=nt, kt=kt, stages=stages, num_warps=num_warps, 
primitives=CudaBackend::get_primitive_defs(), 
epilogue_defs=include_str!("../../kernels/gpu/epilogue.cuh"),
epilogue_args=epilogue_args,
epilogue_apply=epilogue_apply,
warp_m=warp_m, warp_n=warp_n, a_smem_offset=a_smem_offset, b_smem_offset=b_smem_offset))
}
