use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use crate::semantic::transition::SyncRequirement;

pub struct MetalEmitter {
    pub device_name: String,
    pub max_threadgroup_memory: usize,
}

impl MetalEmitter {
    pub fn detect() -> Self {
        // Placeholder for Metal discovery
        // On non-macOS, this will just be a dummy
        Self {
            device_name: "Apple M-Series (Simulated)".to_string(),
            max_threadgroup_memory: 32768,
        }
    }

    pub fn generate_gemm(&self, config: crate::PipelineConfig) -> String {
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();
        
        format!(r#"
{primitives}

kernel void gemm_metal_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {{
    // Threadgroup Memory (Shared)
    threadgroup half sA[{mt} * {kt}];
    threadgroup half sB[{kt} * {nt}];

    // simdgroup_matrix is usually 8x8 or 16x16
    // We assume 8x8 for compatibility across early M1/M2
    simdgroup_float8x8 acc;
    #pragma unroll
    for(int i=0; i<1; ++i) acc = simdgroup_float8x8(0.0f);

    for (uint k_step = 0; k_step < K; k_step += {kt}) {{
        // Load data into threadgroup memory
        // (Simplified parallel load)
        uint t_idx = tid.y * 32 + tid.x;
        for (uint i = t_idx; i < {mt} * {kt}; i += 32*4) {{
             uint r = i / {kt}; uint c = i % {kt};
             if (bid.y * {mt} + r < M && k_step + c < K)
                 sA[i] = A[(bid.y * {mt} + r) * K + (k_step + c)];
             else sA[i] = 0;
        }}
        for (uint i = t_idx; i < {kt} * {nt}; i += 32*4) {{
             uint r = i / {nt}; uint c = i % {nt};
             if (k_step + r < K && bid.x * {nt} + c < N)
                 sB[i] = B[(k_step + r) * N + (bid.x * {nt} + c)];
             else sB[i] = 0;
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Matrix Multiply-Accumulate block
        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;
        
        // Simdgroup distribution: Assume 4 simdgroups (128 threads)
        // Each simdgroup handles a 16x16 or 8x8 sub-tile
        uint sg_r = (simd_id / 2) * 8;
        uint sg_c = (simd_id % 2) * 8;

        for (uint ki = 0; ki < {kt}; ki += 8) {{
            simdgroup_load(ma, &sA[sg_r * {kt} + ki], {kt});
            simdgroup_load(mb, &sB[ki * {nt} + sg_c], {nt});
            simdgroup_multiply_accumulate(acc, ma, mb, acc);
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Epilogue: Store results
    uint sg_r = (simd_id / 2) * 8;
    uint sg_c = (simd_id % 2) * 8;
    simdgroup_store(acc, (device float*)&C[(bid.y * {mt} + sg_r) * N + (bid.x * {nt} + sg_c)], N);
}}
"#, mt=mt, nt=nt, kt=kt, primitives=primitives)
    }
}

pub fn generate_metal_conv(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::Conv2d { n: batch, h: h_in, w: w_in, c: c_in, k: k_out, r, s, stride, pad, dilation, .. } = ir.op_type {
        let h_out = (h_in + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * pad - dilation * (s - 1) - 1) / stride + 1;
        
        let m_gemm = batch * h_out * w_out;
        // let k_gemm = c_in * r * s; // Unused variable warning fix

        let mt = ir.tiling.m_tile;
        let nt = ir.tiling.n_tile;

        format!(r#"
#include <metal_stdlib>
using namespace metal;

struct ConvParams {{
    uint batch, h_in, w_in, c_in, k_out;
    uint h_out, w_out, r_sz, s_sz;
    uint stride, pad, dilation;
}};

kernel void conv2d_implicit_gemm(
    device const half* Input [[buffer(0)]],
    device const half* Weight [[buffer(1)]],
    device half* Output [[buffer(2)]],
    constant ConvParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {{
    uint m_glob = bid.y * {mt} + (simd_id / 2) * 8; 
    uint n_glob = bid.x * {nt} + (simd_id % 2) * 8;

    // if (m_glob >= {m_gemm} || n_glob >= p.k_out) return; // Need m_gemm
    if (m_glob >= {m_gemm} || n_glob >= p.k_out) return;

    float acc = 0.0;
    // ...
    Output[m_glob * p.k_out + n_glob] = (half)acc;
}}
"#, mt=mt, nt=nt, m_gemm=m_gemm)
    } else {
        panic!("Metal conv emitter called with invalid op");
    }
}

pub fn generate_metal_attention(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::FusedAttention { .. } = ir.op_type {
        match ir.tiling.attention_variant {
            crate::core::config::AttentionVariant::Naive => generate_naive_attention(ir),
            crate::core::config::AttentionVariant::SimdQK => generate_simd_qk_attention(ir),
            crate::core::config::AttentionVariant::SimdQ => generate_simd_qk_attention(ir), 
            crate::core::config::AttentionVariant::SimdFull => generate_simd_full_attention(ir),
        }
    } else {
         panic!("Generate attention called with wrong op type");
    }
}

fn generate_simd_qk_attention(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::FusedAttention { b: _, h: _, s: _, d: _, dh, causal: _ } = ir.op_type {
        let mt = 16;
        let nt = 32;
        let dk = dh;
        
        let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();

        format!(r#"
{primitives}
#include <metal_simdgroup_matrix>
using namespace metal;

#define BLOCK_M {mt}
#define BLOCK_N {nt}
#define D_HEAD  {dk}

struct FAParams {{
    uint b, h, s, d;
    float scale;
}};

// SimdQK Implementation (Step 1)
// - QK^T: simdgroup_matrix (8x8)
// - Softmax: Naive (Thread-per-element reading from Shared Mem result)
// - PV: Naive (Thread-per-element)
kernel void flash_attention_v2_kernel(
    device const half* Q  [[buffer(0)]],
    device const half* K  [[buffer(1)]],
    device const half* V  [[buffer(2)]],
    device       half* O  [[buffer(3)]],
    constant FAParams& p  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],       
    uint3 bid [[threadgroup_position_in_grid]],  
    uint  tid [[thread_index_in_threadgroup]]
) {{
    uint q_block_idx = bid.x;
    uint head_idx    = bid.y;
    uint batch_idx   = bid.z;

    uint batch_offset = batch_idx * (p.h * p.s * p.d) + head_idx * (p.s * p.d);
    device const half* q_ptr = Q + batch_offset;
    device const half* k_ptr = K + batch_offset;
    device const half* v_ptr = V + batch_offset;
    device       half* o_ptr = O + batch_offset;

    // Shared Memory
    // K Transposed: [D, N] -> [64, 32]
    threadgroup half sK_T[D_HEAD * BLOCK_N];
    // V Normal: [N, D] -> [32, 64]
    threadgroup half sV[BLOCK_N * D_HEAD];
    
    // Intermediate Score Buffer (QK^T)
    // Size [16, 32]. Row Major.
    threadgroup float sS[BLOCK_M * BLOCK_N]; 

    // Simdgroup accumulators
    simdgroup_matrix<float, 8, 8> acc[2][4];
    simdgroup_matrix<half, 8, 8>  matQ[2]; 
    simdgroup_matrix<half, 8, 8>  matK[4]; 
    
    // Loop over K/V blocks
    uint num_steps = (p.s + BLOCK_N - 1) / BLOCK_N;
    
    // Naive State
    float l_i = 1.0f; 
    float m_i = -1e30f;
    float acc_o[D_HEAD];
    for(int i=0; i<D_HEAD; ++i) acc_o[i] = 0.0f;

    for(uint j=0; j<num_steps; ++j) {{
        // 1. Cooperative Load K/V Tile
        // K -> sK_T (Transpose Load)
        // V -> sV (Normal Load)
        for(uint k=0; k<BLOCK_N; ++k) {{
             if (tid == k) {{ 
                 for (uint d=0; d<D_HEAD; ++d) {{
                     uint kv_idx_global = j * BLOCK_N + k;
                     half val_k = 0.0h;
                     half val_v = 0.0h;
                     
                     if (kv_idx_global < p.s) {{
                         val_k = k_ptr[kv_idx_global * p.d + d];
                         val_v = v_ptr[kv_idx_global * p.d + d];
                     }}
                     
                     // Store K Transposed: sK_T[d, k]
                     sK_T[d * BLOCK_N + k] = val_k;
                     
                     // Store V Normal: sV[k, d]
                     sV[k * D_HEAD + d] = val_v;
                 }}
             }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. SIMD Q*K^T Computation
        // Reset Accumulators
        for(int r=0; r<2; ++r) {{
             for(int c=0; c<4; ++c) {{
                 acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
             }}
        }}

        // Loop over D dimension (8 elements at a time)
        for(uint d=0; d<D_HEAD; d+=8) {{
            // Load Q (16x8 slice) from Global
            ulong q_base = (q_block_idx * BLOCK_M) * p.d + d;
            
            simdgroup_load(matQ[0], q_ptr + q_base, p.d);
            simdgroup_load(matQ[1], q_ptr + q_base + 8 * p.d, p.d);

            // Load K^T (32x8 slice) from Shared sK_T
            for(int c=0; c<4; ++c) {{
                simdgroup_load(matK[c], sK_T + d * BLOCK_N + c * 8, BLOCK_N);
            }}

            // Multiply Accumulate
            for(int r=0; r<2; ++r) {{
                for(int c=0; c<4; ++c) {{
                    simdgroup_multiply_accumulate(acc[r][c], matQ[r], matK[c], acc[r][c]);
                }}
            }}
        }}

        // Store QK Result to Shared Memory sS
        for(int r=0; r<2; ++r) {{
            for(int c=0; c<4; ++c) {{
                simdgroup_store(acc[r][c], sS + (r * 8) * BLOCK_N + (c * 8), BLOCK_N);
            }}
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 3. Naive Softmax & V-Mul (reading from sS)
        if (tid < BLOCK_M) {{
            uint q_row = tid;
            uint q_global = q_block_idx * BLOCK_M + q_row;
            if (q_global < p.s) {{
                 for(uint k=0; k<BLOCK_N; ++k) {{
                     uint kv_idx = j * BLOCK_N + k;
                     if (kv_idx >= p.s) break;
                     
                     float score = sS[q_row * BLOCK_N + k];
                     score *= p.scale;
                     
                     float m_new = max(m_i, score);
                     float exp_old = exp(m_i - m_new);
                     float exp_new = exp(score - m_new);
                     float l_new = l_i * exp_old + exp_new;
                     
                     float rescale = exp(m_i - m_new);
                     for(int d=0; d<D_HEAD; ++d) {{
                         acc_o[d] *= rescale;
                         acc_o[d] += (float)sV[k * D_HEAD + d] * exp_new;
                     }}
                     
                     l_i = l_new;
                     m_i = m_new;
                 }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write Output
    if (tid < BLOCK_M) {{
        uint q_global = q_block_idx * BLOCK_M + tid;
        if (q_global < p.s) {{
            for(int d=0; d<D_HEAD; ++d) {{
                acc_o[d] /= l_i;
                o_ptr[q_global * p.d + d] = (half)acc_o[d];
            }}
        }}
    }}
}}
"#, mt=mt, nt=nt, dk=dk)
    } else {
         panic!("Generate attention called with wrong op type");
    }
}

fn generate_naive_attention(ir: &UnifiedOpIR) -> String {
     if let UnifiedOpType::FusedAttention { b: _, h: _, s: _, d: _, dh, causal: _ } = ir.op_type {
        let mt = 16; // Br (Rows of Q per block) - Reduced for naive reg pressure
        let nt = 32; // Bc (Cols of K/V per block)
        let dk = dh; // Head Dim
        
        let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();

        format!(r#"
{primitives}

#define BLOCK_M {mt}
#define BLOCK_N {nt}
#define D_HEAD  {dk}

struct FAParams {{
    uint b, h, s, d;
    float scale;
}};

// Naive Tiled Implementation (Golden Reference)
// - Storage: FP16 (Global/Threadgroup)
// - Compute: FP32 (Registers)
// - Scope: Thread-per-element
    kernel void flash_attention_v2_kernel(
    device const half* Q  [[buffer(0)]],
    device const half* K  [[buffer(1)]],
    device const half* V  [[buffer(2)]],
    device       half* O  [[buffer(3)]],
    constant FAParams& p  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],       
    uint3 bid [[threadgroup_position_in_grid]],  
    uint  tid [[thread_index_in_threadgroup]]
) {{
    // Grid: (M / BLOCK_M, H, B)
    uint q_block_idx = bid.x;
    uint head_idx    = bid.y;
    uint batch_idx   = bid.z;

    // Offsets
    uint batch_offset = batch_idx * (p.h * p.s * p.d) + head_idx * (p.s * p.d);
    device const half* q_ptr = Q + batch_offset;
    device const half* k_ptr = K + batch_offset;
    device const half* v_ptr = V + batch_offset;
    device       half* o_ptr = O + batch_offset;

    // Shared Memory for K and V tiles
    // K tile: [BLOCK_N, D]
    // V tile: [BLOCK_N, D]
    threadgroup half sK[BLOCK_N * D_HEAD];
    threadgroup half sV[BLOCK_N * D_HEAD];

    // Local Accumulators for Output (FP32)
    // Threads = 32. BLOCK_M = 32.
    // Each thread `tid` computes one row of Q (attention for one query token).
    
    uint q_idx_local = tid; // 0..31
    uint q_idx_global = q_block_idx * 32 + q_idx_local;
    
    // Thread-local Storage
    float acc_o[D_HEAD]; // Accumulator for O[q, :]
    for(int i=0; i<D_HEAD; ++i) acc_o[i] = 0.0f;
    
    float l_i = 1.0f; // Denom (sum exp)
    float m_i = -1e30f; // Max score
    
    // Load Q row into registers (FP32)
    float q_reg[D_HEAD];
    if (q_idx_global < p.s) {{
        for(int i=0; i<D_HEAD; ++i) {{
            q_reg[i] = (float)q_ptr[q_idx_global * p.d + i];
        }}
    }} else {{
        for(int i=0; i<D_HEAD; ++i) q_reg[i] = 0.0f;
    }}

    // Loop over K/V blocks
    uint num_steps = (p.s + BLOCK_N - 1) / BLOCK_N;
    
    for(uint j=0; j<num_steps; ++j) {{
        // 1. Load K/V Tile into Shared Memory
        // Simple loop (inefficient but correct)
        
        for(uint k=0; k<BLOCK_N; ++k) {{
             if (tid == k) {{ // One thread loads one ROW of K/V (size D)
                 for (uint d=0; d<D_HEAD; ++d) {{
                     uint kv_idx_global = j * BLOCK_N + k;
                     if (kv_idx_global < p.s) {{
                         sK[k * D_HEAD + d] = k_ptr[kv_idx_global * p.d + d];
                         sV[k * D_HEAD + d] = v_ptr[kv_idx_global * p.d + d];
                     }} else {{
                         sK[k * D_HEAD + d] = 0.0h;
                         sV[k * D_HEAD + d] = 0.0h;
                     }}
                 }}
             }}
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 2. Compute Attention
        // Each thread `tid` (Query `q`) computes scores against all `k` in sK
        if (q_idx_global < p.s) {{
            for(uint k=0; k<BLOCK_N; ++k) {{
                uint kv_idx = j * BLOCK_N + k;
                if (kv_idx >= p.s) break;

                // Dot Q[q_idx] . K[kv_idx]
                float score = 0.0f;
                for(int d=0; d<D_HEAD; ++d) {{
                    score += q_reg[d] * (float)sK[k * D_HEAD + d];
                }}
                score *= p.scale;
                
                // Online Softmax Update
                // m_new = max(m_i, score)
                // l_new = l_i * exp(m_i - m_new) + exp(score - m_new)
                // O_new = O_i * (l_i/l_new * exp(m_i - m_new)) + exp(score - m_new)/l_new * V[k]
                
                float m_new = max(m_i, score);
                float exp_old = exp(m_i - m_new);
                float exp_new = exp(score - m_new);
                
                float l_new = l_i * exp_old + exp_new;
                
                // Update O accumulator
                // float factor_old = (l_i * exp_old) / l_new; // Unused
                
                // Reset/Rescale O accumulators
                float rescale = exp(m_i - m_new);
                
                for(int d=0; d<D_HEAD; ++d) {{
                    acc_o[d] *= rescale;
                    acc_o[d] += (float)sV[k * D_HEAD + d] * exp_new;
                }}
                
                l_i = l_new;
                m_i = m_new;
            }}
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    
    // 3. Write Output
    if (q_idx_global < p.s) {{
        // Normalize O
        for(int d=0; d<D_HEAD; ++d) {{
            acc_o[d] /= l_i; // Final normalization
            o_ptr[q_idx_global * p.d + d] = (half)acc_o[d];
        }}
    }}
}}
"#, mt=mt, nt=nt, dk=dk)
    } else {
         panic!("Generate attention called with wrong op type");
    }
}
fn generate_simd_full_attention(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::FusedAttention { b: _, h: _, s: _, d: _, dh, causal: _ } = ir.op_type {
        let mt = 16;
        let nt = 32;
        let dk = dh;
        
        let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();

        format!(r#"
{primitives}
#include <metal_simdgroup_matrix>
using namespace metal;

#define BLOCK_M {mt}
#define BLOCK_N {nt}
#define D_HEAD  {dk}

struct FAParams {{
    uint b, h, s, d;
    float scale;
}};

// SimdFull Implementation (Step 2)
// - QK^T: simdgroup_matrix (8x8)
// - Softmax: Thread-collaborative (using sS buffer) -> P (Half)
// - PV: simdgroup_matrix (P * V)
kernel void flash_attention_v2_kernel(
    device const half* Q  [[buffer(0)]],
    device const half* K  [[buffer(1)]],
    device const half* V  [[buffer(2)]],
    device       half* O  [[buffer(3)]],
    constant FAParams& p  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],       
    uint3 bid [[threadgroup_position_in_grid]],  
    uint  tid [[thread_index_in_threadgroup]]
) {{
    uint q_block_idx = bid.x;
    uint head_idx    = bid.y;
    uint batch_idx   = bid.z;

    uint batch_offset = batch_idx * (p.h * p.s * p.d) + head_idx * (p.s * p.d);
    device const half* q_ptr = Q + batch_offset;
    device const half* k_ptr = K + batch_offset;
    device const half* v_ptr = V + batch_offset;
    device       half* o_ptr = O + batch_offset;

    // Shared Memory
    threadgroup half sK_T[D_HEAD * BLOCK_N];
    threadgroup half sV[BLOCK_N * D_HEAD];
    
    // Intermediate Score Buffer (QK^T) -> Reused for P (Softmax output)
    // Size [BLOCK_M, BLOCK_N] = 16x32 = 512 elements.
    // Stored as float for Scores, then overwritten as half for P?
    // Float takes 2KB. Half takes 1KB. Safe to overlap if we manage alignment.
    threadgroup float sS[BLOCK_M * BLOCK_N]; 

    // Accumulators for O.
    // O is 16x64.
    // We use D_HEAD=64.
    // Simdgroup O: 16x64.
    // Tiles (8x8):
    // Rows: 16 -> 2 tiles.
    // Cols: 64 -> 8 tiles.
    // Total 16 accumulators.
    simdgroup_matrix<float, 8, 8> acc_o[2][8];
    
    // QK Accumulators (16x32)
    // 2x4 = 8 accumulators.
    simdgroup_matrix<float, 8, 8> acc_qk[2][4];

    // Matrices placeholders
    simdgroup_matrix<half, 8, 8>  matA[2]; 
    simdgroup_matrix<half, 8, 8>  matB[4]; 
    
    // Initialize O accumulators
    for(int r=0; r<2; ++r) {{
        for(int c=0; c<8; ++c) {{
            acc_o[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }}
    }}
    
    uint num_steps = (p.s + BLOCK_N - 1) / BLOCK_N;
    
    // State
    float l_i = 1.0f; 
    float m_i = -1e30f;

    for(uint j=0; j<num_steps; ++j) {{
        // ============================================
        // 1. Load K/V Tile
        // ============================================
        for(uint k=0; k<BLOCK_N; ++k) {{
             if (tid == k) {{
                 for (uint d=0; d<D_HEAD; ++d) {{
                     uint kv_idx = j * BLOCK_N + k;
                     half val_k = (kv_idx < p.s) ? k_ptr[kv_idx * p.d + d] : 0.0h;
                     half val_v = (kv_idx < p.s) ? v_ptr[kv_idx * p.d + d] : 0.0h;
                     
                     // K Transposed: [D, N]
                     sK_T[d * BLOCK_N + k] = val_k;
                     // V Normal: [N, D]
                     sV[k * D_HEAD + d] = val_v;
                 }}
             }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============================================
        // 2. Compute QK^T -> sS
        // ============================================
        for(int r=0; r<2; ++r) {{
             for(int c=0; c<4; ++c) {{
                 acc_qk[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
             }}
        }}

        for(uint d=0; d<D_HEAD; d+=8) {{
            ulong q_base = (q_block_idx * BLOCK_M) * p.d + d;
            simdgroup_load(matA[0], q_ptr + q_base, p.d);
            simdgroup_load(matA[1], q_ptr + q_base + 8 * p.d, p.d);

            for(int c=0; c<4; ++c) {{
                simdgroup_load(matB[c], sK_T + d * BLOCK_N + c * 8, BLOCK_N);
            }}

            for(int r=0; r<2; ++r) {{
                for(int c=0; c<4; ++c) {{
                    simdgroup_multiply_accumulate(acc_qk[r][c], matA[r], matB[c], acc_qk[r][c]);
                }}
            }}
        }}

        // Store QK to sS (Float)
        for(int r=0; r<2; ++r) {{
            for(int c=0; c<4; ++c) {{
                simdgroup_store(acc_qk[r][c], sS + (r * 8) * BLOCK_N + (c * 8), BLOCK_N);
            }}
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // ============================================
        // 3. Softmax & P Calculation
        // ============================================
        // Reading sS (float), updating state, writing P (half) to sS.
        // Parallelize? 
        // 16 rows. 32 threads.
        // Use Threads 0..15.
        if (tid < BLOCK_M) {{
            uint q_row = tid;
            // Row Max
            float row_m = -1e30f;
            for(int k=0; k<BLOCK_N; ++k) {{
                row_m = max(row_m, sS[q_row * BLOCK_N + k]);
            }}
            row_m *= p.scale;
            
            float m_prev = m_i;
            float m_new = max(m_prev, row_m);
            float exp_old = exp(m_prev - m_new); // rescale factor
            
            // Rescale O accumulators
            // We can't access simdgroup registers from single thread easily.
            // BUT each thread participates in simdgroup.
            // Wait, "if (tid < BLOCK_M)" means only partial threads run this.
            // The O rescaling MUST happen across all threads consistently for simdgroup?
            // "simdgroup_matrix" variables are thread-local "handles", but data is distributed.
            // ALL threads in simdgroup must execute operations on `acc_o`.
            // So we CANNOT rescale `acc_o` inside this conditional block.
            // We must calculate factors, sync, then rescale uniformly.
            
            // Calculate Row Sum (l_new)
            float row_sum = 0.0f;
            for(int k=0; k<BLOCK_N; ++k) {{
                float val = sS[q_row * BLOCK_N + k] * p.scale;
                float p_val = exp(val - m_new);
                row_sum += p_val;
                // Write P back to sS as half
                threadgroup half* sP = (threadgroup half*)sS;
                sP[q_row * BLOCK_N + k] = (half)p_val;
            }}

            // Update L
            // l_i = l_i * exp_old + row_sum;
            // But we need to save `exp_old` to rescale O later.
            // Where to save? Shared memory or register?
            // Each thread `tid` updates its own `l_i` and `m_i`.
            // But for `acc_o`, we need `exp_old` accessible to ALL threads?
            // No, `acc_o` is distributed.
            // If `acc_o` holds values for Tile Rows 0..7 and 8..15.
            // Thread 0 participates in computing Row 0..7?
            // How does Simdgroup map threads to rows?
            // "The threads in a simdgroup work together".
            // We can treat `acc_o` as opaque.
            // We can multiply `acc_o` by a vector? No.
            // We can multiply `acc_o` by scalar.
            // BUT each row has DIFFERENT `exp_old`.
            // `simdgroup_multiply` A * B.
            // If we construct a DIAGONAL matrix of rescaling factors? Too complex.
            // Or `acc_o` elementwise multiply? `simdgroup` doesn't support generic elementwise easily.
            
            // Backtrack:
            // "Rescale O_acc" is easy in Naive (per-element).
            // Hard in Simdgroup (opaque tiles).
            // If we cannot rescale `acc_o` rows independently, we cannot use Simd PV accumulation easily across blocks.
            // UNLESS we use `FlashAttention-1` style? (Write O to memory, rescale later).
            // Or `FlashAttention-2`:
            // O = (O * exp_old + P * V).
            // We need to apply `exp_old` to O.
            // O matches Q rows.
            // Rows 0..15 have different `exp_old`.
            
            // Can we effectively multiply `acc_o` (16x64) by Column Vector `E` (16x1)?
            // E[i] = exp_old for row i.
            // `acc_o[i][j] *= E[i]`.
            // Simdgroup doesn't expose this op.
            // Workaround:
            // 1. Store `acc_o` to temporary shared memory `sO`.
            // 2. Rescale `sO` using parallel threads.
            // 3. Load `sO` back to `acc_o`.
            // 4. Accumulate P*V.
            // Efficient enough?
            // Store/Load is high bandwidth but shared mem is fast.
            // 16x64 elements = 1024 halves = 2KB.
            // 2KB Store + 2KB Load per block.
            // It allows us to use Simd PV.
            
            // Implementation:
            // - Compute Softmax Stats (m_new, l_new, rescale factor).
            // - Store Rescale Factor to shared mem `sScale[16]`.
            // - Barrier.
            // - (All threads)
            // - Store `acc_o` to `sO` (Threadgroup memory).
            // - Barrier.
            // - Apply Scale to `sO` (Parallel loop).
            // - Barrier.
            // - Load `acc_o` from `sO`.
            // - Compute P*V -> `acc_pv`.
            // - `acc_o += acc_pv`.
        }}

        // We need shared memory for O stash?
        // We already have `sK_T` (4KB) and `sV` (4KB).
        // `sS` (1KB half, 2KB float).
        // Can we reuse `sK_T` or something?
        // We need `sV` for P*V.
        // We need `sS` (P) for P*V.
        // `sK_T` is unused during PV! (K is done).
        // Reuse `sK_T` for `sO` stash!
        // `sK_T` is 2048 halves (4KB).
        // `acc_o` is 16x64 = 1024 floats (4KB).
        // Perfect fit.
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 4. Rescale O (The Complicated Part)
        // Need to extract rescaling factors from Softmax logic.
        // Let's refine step 3 to just compute P and Factors, store Factors to sScale.
        threadgroup float sScale[BLOCK_M];
        threadgroup float sExpOld[BLOCK_M]; // To store exp_old logic
        
        if (tid < BLOCK_M) {{
            uint q_row = tid;
             
            // Max/Sum Logic
            float row_m = -1e30f;
            for(int k=0; k<BLOCK_N; ++k) row_m = max(row_m, sS[q_row * BLOCK_N + k]);
            row_m *= p.scale;
            
            float m_prev = m_i; // How to get m_i for this row?
            // "Naive State" l_i, m_i are registers.
            // Since tid < BLOCK_M effectively maps thread to row (tid 0 -> row 0), registers work!
            // Thread 0 has m_i for Row 0. Correct.
            
            float m_new = max(m_prev, row_m);
            float exp_old = exp(m_prev - m_new);
            sExpOld[q_row] = exp_old; // Share for rescale of O
            
            float row_sum = 0.0f;
            threadgroup half* sP = (threadgroup half*)sS;
            for(int k=0; k<BLOCK_N; ++k) {{
                float val = sS[q_row * BLOCK_N + k] * p.scale;
                float p_val = exp(val - m_new);
                row_sum += p_val;
                sP[q_row * BLOCK_N + k] = (half)p_val;
            }}
            
            // Update State
            l_i = l_i * exp_old + row_sum;
            m_i = m_new;
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Rescale `acc_o` via sK_T stash
        // 1. Store acc_o
        threadgroup float* sO = (threadgroup float*)sK_T;
        for(int r=0; r<2; ++r) {{
            for(int c=0; c<8; ++c) {{
                simdgroup_store(acc_o[r][c], sO + (r * 8) * (D_HEAD) + (c * 8), D_HEAD);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 2. Apply Scale (Parallel)
        // 16 rows, 64 cols.
        // 32 threads.
        // Each thread processes 16x64 / 32 = 32 elements.
        for(uint i=0; i<32; ++i) {{
            uint linear_idx = tid * 32 + i;
            if (linear_idx < 16 * 64) {{
                uint r = linear_idx / 64;
                // uint c = linear_idx % 64;
                sO[linear_idx] *= sExpOld[r];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 3. Load back
        for(int r=0; r<2; ++r) {{
            for(int c=0; c<8; ++c) {{
                simdgroup_load(acc_o[r][c], sO + (r * 8) * (D_HEAD) + (c * 8), D_HEAD);
            }}
        }}
        
        // ============================================
        // 5. PV Matmul
        // ============================================
        // P: sS (Half, 16x32).
        // V: sV (Half, 32x64). [32 rows, 64 cols].
        // O: 16x64.
        
        // We iterate K_PV = 0..32 (Full blocks of P cols / V rows).
        // Simdgroup P tiles (2 matches 16 rows). 1 tile wide (8 cols).
        // Simdgroup V tiles (4 matches 32 rows). 1 tile wide (8 cols).
        // Wait, V tile usage:
        // acc[r][c] += P[r][k] * V[k][c].
        
        // Loop 'd_outer' (c) 0..64 in steps of 8 (1 tile wide).
        for(int c_out=0; c_out < 8; ++c_out) {{ // 8 horizontal tiles of O
             // We compute tile acc_o[0..1][c_out].
             // Reduce over k (0..32, step 8 -> 4 tiles).
             
             for(int k=0; k<4; ++k) {{
                 // Load P tiles (Column k)
                 // Rows 0..7 (r=0), Rows 8..15 (r=1).
                 simdgroup_load(matA[0], (threadgroup half*)sS + 0 * BLOCK_N + k * 8, BLOCK_N);
                 simdgroup_load(matA[1], (threadgroup half*)sS + 8 * BLOCK_N + k * 8, BLOCK_N);
                 
                 // Load V tiles (Row k, Col c_out)
                 // V in sV [32, 64].
                 // Row k*8. Col c_out*8.
                 // We need 1 tile of V for the multiply?
                 // No, V must match P cols.
                 // P cols is 8 (tile width).
                 // V rows is 8 (tile height).
                 // V cols is 8 (acc width).
                 // So we load V tile at [Row k*8, Col c_out*8].
                 simdgroup_load(matB[0], sV + (k * 8) * D_HEAD + (c_out * 8), D_HEAD);
                 
                 // Multiply
                 simdgroup_multiply_accumulate(acc_o[0][c_out], matA[0], matB[0], acc_o[0][c_out]);
                 simdgroup_multiply_accumulate(acc_o[1][c_out], matA[1], matB[0], acc_o[1][c_out]);
             }}
        }}
    }}
    
    // Write Output
    if (tid < BLOCK_M) {{
        uint q_global = q_block_idx * BLOCK_M + tid;
        if (q_global < p.s) {{
            for(int d=0; d<D_HEAD; ++d) {{
                // Need to extract from acc_o? 
                // We computed acc_o in Simdgroup.
                // We need to store it to memory.
                // Simdgroup Store to sO (sK_T reuse).
                // Then thread read.
            }}
        }}
    }}
    // Store acc_o to sO
    threadgroup float* sO = (threadgroup float*)sK_T;
    for(int r=0; r<2; ++r) {{
        for(int c=0; c<8; ++c) {{
            simdgroup_store(acc_o[r][c], sO + (r * 8) * (D_HEAD) + (c * 8), D_HEAD);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < BLOCK_M) {{
        uint q_global = q_block_idx * BLOCK_M + tid;
        if (q_global < p.s) {{
             for(int d=0; d<D_HEAD; ++d) {{
                 float val = sO[tid * D_HEAD + d];
                 val /= l_i;
                 o_ptr[q_global * p.d + d] = (half)val;
             }}
        }}
    }}
}}
"#)
    } else {
         panic!("Generate attention called with wrong op type");
    }
}
impl Emitter for MetalEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::Barrier => "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string(),
            _ => String::new(),
        }
    }

    fn generate_from_ir(&self, ir: &UnifiedOpIR) -> String {
        match &ir.op_type {
            UnifiedOpType::Gemm { .. } => self.generate_gemm(ir.tiling.clone()),
            UnifiedOpType::FusedAttention { .. } => {
                generate_metal_attention(ir)
            }
            UnifiedOpType::Elementwise { .. } => {
                 panic!("Elementwise Ops should be handled by UniversalEmitter for now.");
            }
            UnifiedOpType::Conv2d { .. } => {
                generate_metal_conv(ir)
            }
            UnifiedOpType::ConvTranspose2d { .. } => {
                "// Metal ConvTranspose2d not yet implemented - fallback to CPU\n".to_string()
            }
            UnifiedOpType::MatrixCore { .. } => {
                panic!("MatrixCore Ops not supported on Metal yet.");
            }
            UnifiedOpType::LowRankMlp { .. } => {
                panic!("LowRankMlp not supported on Metal yet.");
            }
        }
    }
}
