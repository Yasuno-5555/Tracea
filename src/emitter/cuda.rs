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

    fn generate_gemm(&self, _m: u32, _n: u32, _k: u32, config: &PipelineConfig) -> String {
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
        
        let a_stride = kt + 8;
        let b_stride = nt + 8;
        let smem_a_bytes = mt * a_stride * 2;
        let smem_b_bytes = kt * b_stride * 2;
        
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
                    // Residual add usually adds element-wise from distinct tensor, assumed (M, N) layout
                    epilogue_apply.push_str(&format!("tracea::epilogue::ResidualAdd op_{}; op_{}.residual = residual_{}; val = op_{}(val, (long long)m_glob * N + n_glob);\n", i, i, i, i));
                }
                EpilogueOp::BiasAddSiLU { .. } => {
                     epilogue_args.push_str(&format!(", const float* __restrict__ bias_{}", i));
                     epilogue_apply.push_str(&format!("tracea::epilogue::BiasAddSiLU op_{}; op_{}.bias = bias_{}; val = op_{}(val, n_glob);\n", i, i, i, i));
                }
                _ => {}
            }
        }

        // Warp Tiling Logic (2D vs 1D)
        // If consumers=16 and MT=128, 1D split gives 8 rows/warp (Too small). Use 2D (8x2).
        let (warp_m, warp_n) = if consumers == 16 && mt == 128 {
            (8, 2)
        } else {
            (consumers, 1)
        };
        // Verify valid tiling
        if mt / warp_m < 16 {
             // Panic or fallback to safe config to avoid 0 M_FRAGS
             panic!("Invalid Warp Partitioning: MT={} / WarpsM={} < 16. M_FRAGS would be 0.", mt, warp_m);
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

// Typedef for NVRTC
typedef unsigned int uint;

// Warp Tiling
#define WARP_M {warp_m}
#define WARP_N {warp_n}

// Swizzle Macros
#if {swizzle_enabled}
  #define SWIZZLE_PTR(ptr) smem_swizzle_ptr(ptr)
#else
  #define SWIZZLE_PTR(ptr) (ptr)
#endif

{primitives}

    extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32, 1) gemm_mma_kernel(
        const half* __restrict__ A,
        const half* __restrict__ B,
        half* __restrict__ C_global,
        int M, int N, int K,
        // TTG Arguments
        const uint* __restrict__ l1_active_tiles,
        const unsigned char* __restrict__ l2_metadata
        {epilogue_args}
    ) {{
        int tid = threadIdx.x;
        int warp_id = tid / 32;

        // --- TTG Indirection Layer ---
        int tile_idx_m, tile_idx_n;
        uint ttg_role = 0; // 0=Main, 1=Boundary
        bool ttg_active = true;

        #if {ttg_enabled}
            // 1. Hardware Lying: blockIdx.x is the "Physical ID"
            uint physical_id = blockIdx.x;
            // Note: Grid should be 1D: [NumActiveTiles, 1, 1]
            
            // 2. L1 Lookup
            // For now, assume simple array
            uint logical_id = l1_active_tiles[physical_id];
            
            // 3. L2 Lookup
            // struct TileMetadata {{ uint m; uint n; uint k_start; uint k_end; uint role; }}
            // Size = 5 * 4 = 20 bytes. Let's use int* casting for simplicity or struct definition in preamble
            // Using raw offsets for phase A P0
            const uint* l2_ptr = (const uint*)(l2_metadata + logical_id * 20);
            tile_idx_m = l2_ptr[0];
            tile_idx_n = l2_ptr[1];
            // k_start = l2_ptr[2];
            // k_end = l2_ptr[3];
            ttg_role = l2_ptr[4];

        #else
            // Legacy Rectangular Grid
            tile_idx_m = blockIdx.y;
            tile_idx_n = blockIdx.x;
        #endif

        int a_tile_row = tile_idx_m * MT;
        int b_tile_col = tile_idx_n * NT;
        
        bool is_producer = (warp_id < PRODUCER_WARPS);

        extern __shared__ char smem[];

    
    // Consumer Warp Tiling
    int cons_warp = warp_id - PRODUCER_WARPS;
    int warp_row_idx = cons_warp / WARP_N; 
    int warp_col_idx = cons_warp % WARP_N;

    int mt_per_warp = MT / WARP_M;
    int nt_per_warp = NT / WARP_N;
    
    const int M_FRAGS = MT / WARP_M / 16;
    const int N_FRAGS = NT / WARP_N / 16;

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
                half* sA = (half*)(smem + {a_smem_offset} + stage * {smem_a_bytes});
                half* sB = (half*)(smem + {b_smem_offset} + stage * {smem_b_bytes});
                
                int k_tile = s;
                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {{
                    int m = (i * 8) / KT;
                    int k = (i * 8) % KT;
                    if (a_tile_row + m < M && k_tile * KT + k < K)
                        cp_async_ampere(SWIZZLE_PTR(sA + m * A_STRIDE + k), A + (a_tile_row + m) * K + (k_tile * KT + k), 16);
                }}
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {{
                    int k = (i * 8) / NT;
                    int n = (i * 8) % NT;
                    if (k_tile * KT + k < K && b_tile_col + n < N)
                        cp_async_ampere(SWIZZLE_PTR(sB + k * B_STRIDE + n), B + (k_tile * KT + k) * N + (b_tile_col + n), 16);
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
            half* sA = (half*)(smem + {a_smem_offset} + stage * {smem_a_bytes});
            half* sB = (half*)(smem + {b_smem_offset} + stage * {smem_b_bytes});
            for (int k_inner = 0; k_inner < KT; k_inner += 16) {{
                #pragma unroll
                for (int mi = 0; mi < M_FRAGS; ++mi) {{
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
                    // Adjusted for 2D tiling (use warp_row_idx)
                    wmma::load_matrix_sync(frag_a, (half*)SWIZZLE_PTR(sA + (warp_row_idx * mt_per_warp + mi * 16) * A_STRIDE + k_inner), A_STRIDE);
                    #pragma unroll
                    for (int ni = 0; ni < N_FRAGS; ++ni) {{
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
                        // Adjusted for 2D tiling (use warp_col_idx)
                        wmma::load_matrix_sync(frag_b, (half*)SWIZZLE_PTR(sB + k_inner * B_STRIDE + (warp_col_idx * nt_per_warp + ni * 16)), B_STRIDE);
                        wmma::mma_sync(frag_acc[mi][ni], frag_a, frag_b, frag_acc[mi][ni]);
                    }}
                }}
            }}
        }}
        
        if (is_producer) {{
            int next_k = k_tile + STAGES - 1;
            if (next_k < total_tiles) {{
                int stage = next_k % STAGES;
                half* sA = (half*)(smem + {a_smem_offset} + stage * {smem_a_bytes});
                half* sB = (half*)(smem + {b_smem_offset} + stage * {smem_b_bytes});
                
                int k_next_tile_idx = next_k; // rename to handle capture
                #pragma unroll
                for (int i = tid; i < (MT * KT) / 8; i += 32) {{
                    int m = (i * 8) / KT;
                    int k = (i * 8) % KT;
                    if (a_tile_row + m < M && k_next_tile_idx * KT + k < K)
                        cp_async_ampere(SWIZZLE_PTR(sA + m * A_STRIDE + k), A + (a_tile_row + m) * K + (k_next_tile_idx * KT + k), 16);
                }}
                #pragma unroll
                for (int i = tid; i < (KT * NT) / 8; i += 32) {{
                    int k = (i * 8) / NT;
                    int n = (i * 8) % NT;
                    if (k_next_tile_idx * KT + k < K && b_tile_col + n < N)
                        cp_async_ampere(SWIZZLE_PTR(sB + k * B_STRIDE + n), B + (k_next_tile_idx * KT + k) * N + (b_tile_col + n), 16);
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
                 int row_tile_offset = (warp_row_idx * mt_per_warp + mi * 16);
                 int col_tile_offset = (warp_col_idx * nt_per_warp + ni * 16);
                 // Store to padded smem to avoid bank conflicts if needed, but row major here.
                 wmma::store_matrix_sync(sC + row_tile_offset * NT + col_tile_offset, frag_acc[mi][ni], NT, wmma::mem_row_major);
             }}
        }}
        __syncthreads();

        int epis_tid = tid - PRODUCER_WARPS * 32;
        if ({vectorize_epilogue} && N % 8 == 0 && NT % 8 == 0) {{
             // Vectorized Path: float4 stores (8 halves)
             int vec_size = 8;
             // Ensure loop increments by vector size
             #pragma unroll
             for (int i = epis_tid * vec_size; i < MT * NT; i += (NUM_WARPS - PRODUCER_WARPS) * 32 * vec_size) {{
                 int r = i / NT;
                 int c = i % NT;
                 int row = a_tile_row + r;
                 int col = b_tile_col + c;

                 if (row < M && col + 7 < N) {{
                     // Load 8 values from sC (float)
                     float vals[8];
                     #pragma unroll
                     for(int v=0; v<8; ++v) vals[v] = sC[i + v];

                     // Apply Epilogue & Pack
                     half packs[8];
                     int m_glob = row; 
                     #pragma unroll
                     for(int v=0; v<8; ++v) {{
                         int n_glob = col + v;
                         float val = vals[v];
                         {epilogue_apply}
                         packs[v] = (half)val;
                     }}
                     // Store using float4 alias
                     *(float4*)(&C_global[(long long)row * N + col]) = *(float4*)packs;
                 }}
             }}
        }} else {{
             // Scalar Fallback
             #pragma unroll
             for (int i = epis_tid; i < MT * NT; i += (NUM_WARPS - PRODUCER_WARPS) * 32) {{
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
}}
"# , mt=mt, nt=nt, kt=kt, stages=stages, num_warps=num_warps, smem_a_bytes=smem_a_bytes, smem_b_bytes=smem_b_bytes, 
primitives=CudaBackend::get_primitive_defs(), 
epilogue_defs=include_str!("../kernels/gpu/epilogue.cuh"),
epilogue_args=epilogue_args,
epilogue_apply=epilogue_apply,
swizzle_enabled=(if config.swizzle_mode != crate::core::config::SwizzleMode::None { 1 } else { 0 }),
vectorize_epilogue=config.vectorize_epilogue,
ttg_enabled=(if config.ttg_enabled { 1 } else { 0 }),
warp_m=warp_m, warp_n=warp_n, a_smem_offset=a_smem_offset, b_smem_offset=b_smem_offset)
    }

    fn generate_softmax(_dim_size: usize, _total_elements: usize) -> String {
        format!(r#"
typedef unsigned int uint;

extern "C" __global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim_size,
    int num_rows
) {{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= num_rows) return;

    int base = row_idx * dim_size;

    // Find max for numerical stability
    float max_val = -1e38f;
    for (int i = 0; i < dim_size; i++) {{
        max_val = fmaxf(max_val, input[base + i]);
    }}

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < dim_size; i++) {{
        float val = expf(input[base + i] - max_val);
        output[base + i] = val;
        sum += val;
    }}

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim_size; i++) {{
        output[base + i] *= inv_sum;
    }}
}}
"#)
    }

    fn generate_batchnorm(_n: usize, c: usize, h: usize, w: usize, _epsilon: f32) -> String {
        format!(r#"
#include <cuda_fp16.h>
typedef unsigned int uint;

extern "C" __global__ void batchnorm_forward(
    const half* __restrict__ Input,
    const half* __restrict__ Gamma,
    const half* __restrict__ Beta,
    const half* __restrict__ Mean,
    const half* __restrict__ Var,
    half* __restrict__ Output,
    float epsilon,
    int total_elements
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // NCHW layout: idx = n*C*H*W + c*H*W + h*W + w
    int c_idx = (idx / ({h} * {w})) % {c};

    half val = Input[idx];
    half mean = Mean[c_idx];
    half var = Var[c_idx];
    half gamma = Gamma[c_idx];
    half beta = Beta[c_idx];

    float inv_std = rsqrtf(__half2float(var) + epsilon);
    float normalized = (__half2float(val) - __half2float(mean)) * inv_std;
    float result = normalized * __half2float(gamma) + __half2float(beta);

    Output[idx] = __float2half(result);
}}
"#, c=c, h=h, w=w)
    }

    fn generate_global_avg_pool(_n: usize, _c: usize, _h: usize, _w: usize) -> String {
        format!(r#"
typedef unsigned int uint;

extern "C" __global__ void global_avg_pool_kernel(
    const float* __restrict__ Input,
    float* __restrict__ Output,
    int batch_size,
    int channels,
    int spatial_size
) {{
    // One thread per (batch, channel) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels;
    if (idx >= total_outputs) return;

    int b = idx / channels;
    int c = idx % channels;
    int input_offset = b * channels * spatial_size + c * spatial_size;

    float sum = 0.0f;
    for (int i = 0; i < spatial_size; i++) {{
        sum += Input[input_offset + i];
    }}

    Output[idx] = sum / (float)spatial_size;
}}
"#)
    }

    fn generate_linear(_batch: usize, _m: usize, _n: usize, _k: usize) -> String {
        // Simple tiled Linear - for larger sizes, delegate to GEMM
        format!(r#"
#include <cuda_fp16.h>
typedef unsigned int uint;

#define TILE_SIZE 32

extern "C" __global__ void linear_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, int Batch
) {{
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {{
        float a_val = __half2float(A[b * M * K + row * K + k]);
        float b_val = __half2float(B[b * K * N + k * N + col]);
        acc += a_val * b_val;
    }}

    C[b * M * N + row * N + col] = acc;
}}
"#)
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
            UnifiedOpType::Gemm { m, n, k, .. } => self.generate_gemm(*m, *n, *k, &ir.tiling),
            UnifiedOpType::Elementwise { .. } => {
                panic!("Elementwise Ops should be handled by UniversalEmitter.");
            }
            UnifiedOpType::Conv2d { .. } => {
                crate::emitter::conv::generate_conv(ir)
            }
            UnifiedOpType::MatrixCore { m, n, k } => {
                // Low-level MMA call using wmma / MatrixCore primitives
                format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void matrix_core_kernel(const half* a, const half* b, float* c) {{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);
}}
"#)
            }
            UnifiedOpType::ConvTranspose2d { .. } => {
                crate::emitter::conv_transpose::generate_conv_transpose(ir)
            }
            UnifiedOpType::LowRankMlp { .. } => {
                crate::emitter::cuda_low_rank::generate_low_rank_mlp(ir)
            }
            UnifiedOpType::Softmax { dim_size, total_elements, .. } => {
                Self::generate_softmax(*dim_size, *total_elements)
            }
            UnifiedOpType::BatchNorm { n, c, h, w, epsilon, .. } => {
                Self::generate_batchnorm(*n, *c, *h, *w, *epsilon)
            }
            UnifiedOpType::GlobalAveragePool { n, c, h, w } => {
                Self::generate_global_avg_pool(*n, *c, *h, *w)
            }
            UnifiedOpType::Linear { batch, m, n, k, .. } => {
                Self::generate_linear(*batch, *m, *n, *k)
            }
        }
    }
}

