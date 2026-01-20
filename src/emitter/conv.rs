use crate::emitter::traits::UnifiedOpIR;
use crate::backend::cuda::CudaBackend;
use crate::emitter::traits::UnifiedOpType;

// Magic Number Helper
fn magic_u32(n: u32) -> (u32, u32) {
    let nc = n as u64;
    for p in 0..64 {
        let two_p = 1u64 << p;
        let m = (two_p + nc - 1) / nc;
        if m < (1u64 << 32) { 
             return (m as u32, p as u32);
        }
    }
    // Fallback (shouldn't happen for reasonable dims)
    (1, 0)
}

pub fn generate_conv(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::Conv2d { n: batch, h: h_in, w: w_in, c: c_in, k: k_out, r, s, stride, pad, layout: _ } = ir.op_type {
        // Calculate Output Dimensions
        let h_out = (h_in + 2 * pad - r) / stride + 1;
        let w_out = (w_in + 2 * pad - s) / stride + 1;
        
        // Map to GEMM Dimensions
        // M = Batch * H_out * W_out (Output Pixels)
        // N = K_out (Output Channels)
        // K = C_in * R * S (Filter Volume)
        let m_gemm = batch * h_out * w_out;
        let n_gemm = k_out;
        let k_gemm = c_in * r * s;

        let mt = ir.tiling.m_tile;
        let nt = ir.tiling.n_tile;
        let kt = ir.tiling.k_tile;
        let num_warps = ir.tiling.force_num_warps.unwrap_or(4);
        let stages = ir.tiling.num_stages;
        
        let (hw_magic, hw_shift) = magic_u32((h_out * w_out) as u32);
        let (w_magic, w_shift) = magic_u32(w_out as u32);
        
        format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define MT {mt}
#define NT {nt}
#define KT {kt}
#define N_WARPS {num_warps}
#define STAGES {stages}

// Conv Constants
#define BATCH {batch}
#define H_IN {h_in}
#define W_IN {w_in}
#define C_IN {c_in}
#define K_OUT {k_out}
#define H_OUT {h_out}
#define W_OUT {w_out}
#define R_SZ {r}
#define S_SZ {s}
#define STRIDE {stride}
#define PAD {pad}

// Magic Number Helper
// (Computed on host: HW_MAGIC={hw_magic}, HW_SHIFT={hw_shift}, W_MAGIC={w_magic}, W_SHIFT={w_shift})

// Implicit GEMM Helpers
// We calculate magic numbers for W_OUT and (H_OUT * W_OUT).
// div_w: m / W_OUT, mod_w: m % W_OUT
// div_hw: m / (H*W), mod_hw: m % (H*W)

__device__ __forceinline__ void fast_divmod(int val, int magic, int shift, int divisor, int& div, int& mod) {{
    // High Mul: (val * magic) >> 32
    unsigned long long res = (unsigned long long)val * (unsigned long long)magic;
    // For standard magic mul, if shift is 0, we just take high 32 bits.
    // If shift > 0, we shift more.
    div = (int)(res >> (32 + shift));
    mod = val - div * divisor;
}}

__device__ __forceinline__ void get_img_coord(int m_idx, int& b, int& h, int& w) {{
    // Constants injected via format!
    #define HW_MAGIC {hw_magic}
    #define HW_SHIFT {hw_shift}
    #define W_MAGIC  {w_magic}
    #define W_SHIFT  {w_shift}
    
    int hw;
    fast_divmod(m_idx, HW_MAGIC, HW_SHIFT, (H_OUT * W_OUT), b, hw);
    
    fast_divmod(hw, W_MAGIC, W_SHIFT, W_OUT, h, w);
}}

__device__ __forceinline__ void get_filter_coord(int k_idx, int& r, int& s, int& c) {{
    int rs = k_idx / C_IN;
    r = rs / S_SZ;
    s = rs % S_SZ;
    c = k_idx % C_IN;
}}

{primitives}

extern "C" __global__ void __launch_bounds__(N_WARPS * 32, 1) conv2d_implicit_gemm(
    const half* __restrict__ Input,
    const half* __restrict__ Weight,
    half* __restrict__ Output
) {{
    // Block & Thread IDs
    int bx = blockIdx.x; int by = blockIdx.y;
    int tid = threadIdx.x; int warp_id = tid / 32; int lane_id = tid % 32;

    // Tile Offsets
    int m_block_start = bx * MT;
    int n_block_start = by * NT;
    
    // Shared Memory
    extern __shared__ char smem[];
    half* sA = (half*)smem;
    half* sB = (half*)(smem + STAGES * MT * KT * 2);

    // Fragments
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT/16][NT/16];
    #pragma unroll
    for(int i=0; i<MT/16; ++i) 
        for(int j=0; j<NT/16; ++j) 
            wmma::fill_fragment(acc[i][j], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[STAGES][MT/16];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B[STAGES][NT/16];

    // Pipeline Loop
    int total_k_tiles = ({k_gemm} + KT - 1) / KT;
    
    // Prologue (Simplified for brevity, ideally heavily pipelined like GEMM)
    // For implicit GEMM verification, we use a simpler loop first to ensure correctness.
    // Optimizing the pipeline with `cp.async` and index calc is Phase 4.2.
    // Here we do a simpler "Load -> Sync -> Compute" loop to get logic right.
    
    for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {{
        int k_step = k_tile * KT;
        
        // Load Input (A) Tile -> [MT, KT]
        // Threads load cooperatively. Total threads = N_WARPS * 32.
        // Elements to load = MT * KT.
        for (int i = tid; i < MT * KT; i += N_WARPS * 32) {{
             int row_in_tile = i / KT;
             int col_in_tile = i % KT;
             
             int m_glob = m_block_start + row_in_tile;
             int k_glob = k_step + col_in_tile;
             
             half val = 0.0;
             if (m_glob < {m_gemm} && k_glob < {k_gemm}) {{
                 int b, h_out_idx, w_out_idx;
                 get_img_coord(m_glob, b, h_out_idx, w_out_idx);
                 
                 int r_idx, s_idx, c_idx;
                 get_filter_coord(k_glob, r_idx, s_idx, c_idx);
                 
                 int h_in_idx = h_out_idx * STRIDE - PAD + r_idx;
                 int w_in_idx = w_out_idx * STRIDE - PAD + s_idx;
                 
                 if (h_in_idx >= 0 && h_in_idx < H_IN && w_in_idx >= 0 && w_in_idx < W_IN) {{
                     // NHWC Address: ((b * H + h) * W + w) * C + c
                     long long offset = ((long long)b * H_IN * W_IN + h_in_idx * W_IN + w_in_idx) * C_IN + c_idx;
                     val = Input[offset];
                 }}
             }}
             sA[row_in_tile * KT + col_in_tile] = val;
        }}

        // Load Weight (B) Tile -> [KT, NT] (Transposed to [NT, KT] in smem for ColMajor loading?)
        // Standard Tensor Core B is ColMajor [K, N].
        // Weight tensor is usually [K_out, C, R, S] or [K_out, R, S, C] (NHWC style).
        // Let's assume RSCK (K outer, C inner? No, NHWC usually means C is fastest).
        // Standard TF/Pytorch NHWC filter is [K_out, R, S, C] -> NO, Filter is [R, S, C, K_out] usually for NHWC input.
        // Wait, standard conv is O[n, k] += I[n, c] * W[c, k].
        // Tensor Core GEMM C = A * B. A=[M, K], B=[K, N].
        // If B is Weights, it should be [K_gemm, N_gemm] = [C*R*S, K_out].
        // So B is typically stored as [K_out, C, R, S] (NCHW) or [K_out, R, S, C] (NHWC?). 
        // Let's assume input format RSCK [R, S, C, K_out] so K_out is stride-1?
        // Actually, for B matrix to be ColMajor [K, N], it means data is stored column-by-column.
        // Column 0 of B is K elements. That's one filter.
        // So if we want ColMajor B in registers, we want memory to ideally be K-major? 
        // For simplicity, let's assume we load strictly by address and verify layout later.
        
        for (int i = tid; i < NT * KT; i += N_WARPS * 32) {{
             int row_in_tile = i / NT; // k-dim inside tile
             int col_in_tile = i % NT; // n-dim inside tile
             
             int k_glob = k_step + row_in_tile;
             int n_glob = n_block_start + col_in_tile;
             
             half val = 0.0;
             if (k_glob < {k_gemm} && n_glob < {n_gemm}) {{
                 // Flat layout assumption for Weights for now: [K_gemm, N_gemm] row-major in memory
                 // We will fix layout in Adapter.
                 val = Weight[k_glob * {n_gemm} + n_glob];
             }}
             // Store transposed for ColMajor? Or keep RowMajor?
             // wmma::load_matrix_sync for B needs ColMajor or RowMajor.
             // If we use ColMajor in frag_B, we need B to be [K, N] in col major.
             // sB accessed as [col, row].
             sB[col_in_tile * KT + row_in_tile] = val; 
        }}
        
        __syncthreads();
        
        // Compute wmma
        #pragma unroll
        for (int k=0; k<KT/16; ++k) {{
            #pragma unroll
            for (int i=0; i<MT/16; ++i) {{
                wmma::load_matrix_sync(frag_A[0][i], sA + i*16*KT + k*16, KT);
            }}
            #pragma unroll
            for (int j=0; j<NT/16; ++j) {{
                // sB is [NT, KT] (transposed). 
                // We want B sub-matrix [16, 16] starting at (k*16, j*16) in logical B space.
                // In sB, this is (j*16, k*16).
                wmma::load_matrix_sync(frag_B[0][j], sB + j*16*KT + k*16, KT); 
            }}
            
            #pragma unroll
            for (int i=0; i<MT/16; ++i) {{
                #pragma unroll
                for (int j=0; j<NT/16; ++j) {{
                    wmma::mma_sync(acc[i][j], frag_A[0][i], frag_B[0][j], acc[i][j]);
                }}
            }}
        }}
        __syncthreads();
    }}

    // Epilogue
    float* sC = (float*)smem;
    for (int i=0; i<MT/16; ++i) {{
        for (int j=0; j<NT/16; ++j) {{
            // Use sC (float) for storing float accumulators
            wmma::store_matrix_sync(sC + (i*16)*NT + (j*16), acc[i][j], NT, wmma::mem_row_major);
        }}
    }}
    __syncthreads();
    
    // Write out sC to Output (casting to half)
    for (int i = tid; i < MT * NT; i += N_WARPS * 32) {{
        int r = i / NT;
        int c = i % NT;
        int m_glob = m_block_start + r;
        int n_glob = n_block_start + c;
        if (m_glob < {m_gemm} && n_glob < {n_gemm}) {{
            Output[m_glob * {n_gemm} + n_glob] = (half)sC[i];
        }}
    }}
}}
"#
    , primitives=CudaBackend::get_primitive_defs())
    } else {
        panic!("Using conv emitter for non-conv op");
    }
}
