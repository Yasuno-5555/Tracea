use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};

pub fn generate_metal_attention(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::FusedAttention { .. } = ir.op_type {
        eprintln!("[MetalEmitter] Attention Variant: {:?}", ir.tiling.attention_variant);
        match ir.tiling.attention_variant {
            crate::core::config::AttentionVariant::Naive => generate_naive_attention(ir),
            crate::core::config::AttentionVariant::SimdQK => generate_simd_qk_attention(ir),
            crate::core::config::AttentionVariant::SimdQ => generate_simd_qk_attention(ir), 
            crate::core::config::AttentionVariant::SimdFull => generate_simd_full_attention(ir),
            crate::core::config::AttentionVariant::FlashV2 => generate_flash_attention_v2(ir),
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
                
                float m_new = max(m_i, score);
                float exp_old = exp(m_i - m_new);
                float exp_new = exp(score - m_new);
                
                float l_new = l_i * exp_old + exp_new;
                
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

fn generate_flash_attention_v2(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::FusedAttention { b: _, h: _, s: _, d: _, dh, causal: _ } = ir.op_type {
        let block_m = 64; // Queries per block
        let block_n = 64; // KV per block
        let dk = dh;
        
        let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();

        format!(r#"
{primitives}

#define BLOCK_M {block_m}
#define BLOCK_N {block_n}
#define D_HEAD  {dk}
#define THREADS_PER_BLOCK 256  // 8 simdgroups

struct FAParams {{
    uint b, h, s, d;
    float scale;
}};

// FlashAttention V2 - Optimized for Apple Silicon
kernel void flash_attention_v2_kernel(
    device const half* Q  [[buffer(0)]],
    device const half* K  [[buffer(1)]],
    device const half* V  [[buffer(2)]],
    device       half* O  [[buffer(3)]],
    constant FAParams& p  [[buffer(4)]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_group [[simdgroup_index_in_threadgroup]]
) {{
    uint q_block_idx = bid.x;
    uint head_idx    = bid.y;
    uint batch_idx   = bid.z;

    uint batch_head_offset = batch_idx * (p.h * p.s * p.d) + head_idx * (p.s * p.d);
    device const half* q_base = Q + batch_head_offset;
    device const half* k_base = K + batch_head_offset;
    device const half* v_base = V + batch_head_offset;
    device       half* o_base = O + batch_head_offset;

    threadgroup half sK[2][BLOCK_N * D_HEAD];
    threadgroup half sV[2][BLOCK_N * D_HEAD];

    uint q_local_idx = tid; 
    uint q_global_idx = q_block_idx * BLOCK_M + q_local_idx;
    bool q_valid = q_local_idx < BLOCK_M && q_global_idx < p.s;

    float q_reg[D_HEAD];
    float acc_o[D_HEAD];
    for(int i = 0; i < D_HEAD; ++i) {{
        acc_o[i] = 0.0f;
        q_reg[i] = q_valid ? (float)q_base[q_global_idx * p.d + i] : 0.0f;
    }}
    
    float l_i = 1.0f;
    float m_i = -1e30f;

    uint num_kv_blocks = (p.s + BLOCK_N - 1) / BLOCK_N;
    uint write_idx = 0;

    uint elems_per_thread = (BLOCK_N * D_HEAD + 255) / 256;
    for(uint e = 0; e < elems_per_thread; ++e) {{
        uint idx = tid + e * 256;
        if (idx < BLOCK_N * D_HEAD) {{
            uint kv_row = idx / D_HEAD, kv_col = idx % D_HEAD;
            if (kv_row < p.s) {{
                sK[write_idx][idx] = k_base[kv_row * p.d + kv_col];
                sV[write_idx][idx] = v_base[kv_row * p.d + kv_col];
            }} else {{
                sK[write_idx][idx] = 0.0h; sV[write_idx][idx] = 0.0h;
            }}
        }}
    }}
    
    for(uint kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint read_idx = write_idx;
        write_idx = 1 - write_idx;

        if (kv_block + 1 < num_kv_blocks) {{
            uint next_kv_block = kv_block + 1;
            for(uint e = 0; e < elems_per_thread; ++e) {{
                uint idx = tid + e * 256;
                if (idx < BLOCK_N * D_HEAD) {{
                    uint kv_row = idx / D_HEAD, kv_col = idx % D_HEAD;
                    uint kv_global = next_kv_block * BLOCK_N + kv_row;
                    if (kv_global < p.s) {{
                        sK[write_idx][idx] = k_base[kv_global * p.d + kv_col];
                        sV[write_idx][idx] = v_base[kv_global * p.d + kv_col];
                    }} else {{
                        sK[write_idx][idx] = 0.0h; sV[write_idx][idx] = 0.0h;
                    }}
                }}
            }}
        }}

        if (q_valid) {{
            for(uint kv_local = 0; kv_local < BLOCK_N; ++kv_local) {{
                uint kv_global = kv_block * BLOCK_N + kv_local;
                if (kv_global >= p.s) break;
                
                float score = 0.0f;
                for(int d = 0; d < D_HEAD; ++d) {{
                    score += q_reg[d] * (float)sK[read_idx][kv_local * D_HEAD + d];
                }}
                score *= p.scale;
                
                float m_new = max(m_i, score);
                float exp_old = exp(m_i - m_new);
                float exp_new = exp(score - m_new);
                float l_new = l_i * exp_old + exp_new;
                
                for(int d = 0; d < D_HEAD; ++d) {{
                    acc_o[d] = acc_o[d] * exp_old + (float)sV[read_idx][kv_local * D_HEAD + d] * exp_new;
                }}
                
                l_i = l_new;
                m_i = m_new;
            }}
        }}
    }}
    
    if (q_valid) {{
        for(int d = 0; d < D_HEAD; ++d) {{
            o_base[q_global_idx * p.d + d] = (half)(acc_o[d] / l_i);
        }}
    }}
}}
"#, block_m=block_m, block_n=block_n, dk=dk)
    } else {
        panic!("Generate attention called with wrong op type");
    }
}

pub fn generate_simd_full_attention(ir: &UnifiedOpIR) -> String {
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

    threadgroup half sK_T[D_HEAD * BLOCK_N];
    threadgroup half sV[BLOCK_N * D_HEAD];
    threadgroup float sS[BLOCK_M * BLOCK_N]; 

    simdgroup_matrix<float, 8, 8> acc_o[2][8];
    simdgroup_matrix<float, 8, 8> acc_qk[2][4];
    simdgroup_matrix<half, 8, 8>  matA[2]; 
    simdgroup_matrix<half, 8, 8>  matB[4]; 
    
    for(int r=0; r<2; ++r) {{
        for(int c=0; c<8; ++c) {{
            acc_o[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }}
    }}
    
    uint num_steps = (p.s + BLOCK_N - 1) / BLOCK_N;
    float l_i = 1.0f; 
    float m_i = -1e30f;

    for(uint j=0; j<num_steps; ++j) {{
        for(uint k=0; k<BLOCK_N; ++k) {{
             if (tid == k) {{
                 for (uint d=0; d<D_HEAD; ++d) {{
                     uint kv_idx = j * BLOCK_N + k;
                     half val_k = (kv_idx < p.s) ? k_ptr[kv_idx * p.d + d] : 0.0h;
                     half val_v = (kv_idx < p.s) ? v_ptr[kv_idx * p.d + d] : 0.0h;
                     sK_T[d * BLOCK_N + k] = val_k;
                     sV[k * D_HEAD + d] = val_v;
                 }}
             }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for(int r=0; r<2; ++r) {{
             for(int c=0; c<4; ++c) {{
                 acc_qk[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
             }}
        }}

        for(uint d=0; d<D_HEAD; d+=8) {{
            ulong q_offset = (q_block_idx * BLOCK_M) * p.d + d;
            simdgroup_load(matA[0], q_ptr + q_offset, p.d);
            simdgroup_load(matA[1], q_ptr + q_offset + 8 * p.d, p.d);

            for(int c=0; c<4; ++c) {{
                simdgroup_load(matB[c], sK_T + d * BLOCK_N + c * 8, BLOCK_N);
            }}

            for(int r=0; r<2; ++r) {{
                for(int c=0; c<4; ++c) {{
                    simdgroup_multiply_accumulate(acc_qk[r][c], matA[r], matB[c], acc_qk[r][c]);
                }}
            }}
        }}

        for(int r=0; r<2; ++r) {{
            for(int c=0; c<4; ++c) {{
                simdgroup_store(acc_qk[r][c], sS + (r * 8) * BLOCK_N + (c * 8), BLOCK_N);
            }}
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        threadgroup float sExpOld[BLOCK_M];
        if (tid < BLOCK_M) {{
            uint q_row = tid;
            float row_m = -1e30f;
            for(int k=0; k<BLOCK_N; ++k) row_m = max(row_m, sS[q_row * BLOCK_N + k]);
            row_m *= p.scale;
            
            float m_prev = m_i;
            float m_new = max(m_prev, row_m);
            float exp_old = exp(m_prev - m_new);
            sExpOld[q_row] = exp_old;
            
            float row_sum = 0.0f;
            threadgroup half* sP = (threadgroup half*)sS;
            for(int k=0; k<BLOCK_N; ++k) {{
                float val = sS[q_row * BLOCK_N + k] * p.scale;
                float p_val = exp(val - m_new);
                row_sum += p_val;
                sP[q_row * BLOCK_N + k] = (half)p_val;
            }}
            l_i = l_i * exp_old + row_sum;
            m_i = m_new;
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        threadgroup float* sO = (threadgroup float*)sK_T;
        for(int r=0; r<2; ++r) {{
            for(int c=0; c<8; ++c) {{
                simdgroup_store(acc_o[r][c], sO + (r * 8) * (D_HEAD) + (c * 8), D_HEAD);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for(uint i=0; i<32; ++i) {{
            uint linear_idx = tid * 32 + i;
            if (linear_idx < 16 * 64) {{
                uint r = linear_idx / 64;
                sO[linear_idx] *= sExpOld[r];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for(int r=0; r<2; ++r) {{
            for(int c=0; c<8; ++c) {{
                simdgroup_load(acc_o[r][c], sO + (r * 8) * (D_HEAD) + (c * 8), D_HEAD);
            }}
        }}
        
        for(int c_out=0; c_out < 8; ++c_out) {{
             for(int k=0; k<4; ++k) {{
                 simdgroup_load(matA[0], (threadgroup half*)sS + 0 * BLOCK_N + k * 8, BLOCK_N);
                 simdgroup_load(matA[1], (threadgroup half*)sS + 8 * BLOCK_N + k * 8, BLOCK_N);
                 simdgroup_load(matB[0], sV + (k * 8) * D_HEAD + (c_out * 8), D_HEAD);
                 simdgroup_multiply_accumulate(acc_o[0][c_out], matA[0], matB[0], acc_o[0][c_out]);
                 simdgroup_multiply_accumulate(acc_o[1][c_out], matA[1], matB[0], acc_o[1][c_out]);
             }}
        }}
    }}
    
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
