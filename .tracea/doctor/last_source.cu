
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Compile target: sm_80+
// Tr=64 (Block size), processed as 4 warps x 16 rows.

extern "C" __global__ void flash_attention_v2_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    long long B, long long H, long long S, long long D, float scale
) {
    // Assumptions: D=64, BlockDim=(128,1,1) -> 4 Warps
    // Grid: (S/64, H, B)
    
    int tile_idx = blockIdx.x; 
    int h = blockIdx.y;
    int b = blockIdx.z;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (warp_id >= 4) return; 

    // Global Offsets
    long long offset_base = b * H * S * D + h * S * D;
    Q += offset_base;
    K += offset_base;
    V += offset_base;
    O += offset_base;

    // This warp's Q row range (global)
    int q_row_start_local = warp_id * 16;
    int q_row_global = tile_idx * 64 + q_row_start_local;
    
    // Bounds check
    if (q_row_global >= S) return;

    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q[4];
    
    // Load Q
    for(int k=0; k<4; ++k) { 
        wmma::load_matrix_sync(frag_Q[k], Q + q_row_global * D + k * 16, D);
    }
    
    // O accumulator: 16x64 -> 4 chunks of 16x16 (Float32)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[4];
    for(int k=0; k<4; ++k) wmma::fill_fragment(acc_O[k], 0.0f);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;

    // Dynamic Shared Memory
    extern __shared__ char smem[];
    
    // Pointers
    float* smem_S_ptr = (float*)smem;
    float* smem_O_ptr = (float*)&smem_S_ptr[4*16*16];
    half* smem_P_half_ptr = (half*)&smem_O_ptr[4*16*64];
    
    // Warp-local pointers
    float* my_sS = smem_S_ptr + warp_id * 256;         // 16*16 floats
    float* my_sO = smem_O_ptr + warp_id * 1024;        // 16*64 floats
    half* my_P_half = smem_P_half_ptr + warp_id * 256; // 16*16 halves

    float m_prev = -50000.0f;
    float l_prev = 0.0f;

    // Loop over K tiles
    for(int j=0; j<S; j+=16) {
        
        // 1. Compute S_tile = Q * K_tile^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S;
        wmma::fill_fragment(acc_S, 0.0f);
        
        for(int k=0; k<4; ++k) {
             if (j < S) 
                 wmma::load_matrix_sync(frag_K, K + j * D + k*16, D); 
             else 
                 wmma::fill_fragment(frag_K, 0.0f);
             wmma::mma_sync(acc_S, frag_Q[k], frag_K, acc_S);
        }
        
        // 2. Softmax Logic via SMEM
        wmma::store_matrix_sync(my_sS, acc_S, 16, wmma::mem_row_major);
        __syncwarp(); 

        float e_correction = 1.0f;
        
        if (lane_id < 16) {
            int r = lane_id;
            float row_max = -50000.0f;
            for(int c=0; c<16; ++c) {
                row_max = max(row_max, my_sS[r * 16 + c] * scale);
            }
            
            float m_new = max(m_prev, row_max);
            float e_glo = expf(m_prev - m_new);
            float row_sum = 0.0f;
            
            for(int c=0; c<16; ++c) {
                float p = expf(my_sS[r * 16 + c] * scale - m_new);
                my_sS[r * 16 + c] = p; // P
                my_P_half[r * 16 + c] = (__half)p; // Convert to half for next MMA
                row_sum += p;
            }
            
            l_prev = l_prev * e_glo + row_sum;
            m_prev = m_new;
            e_correction = e_glo;
        }
        __syncwarp();
        
        // 3. Rescale O (SMEM roundtrip for safety)
        // Store current accumulated O to SMEM
        for(int k=0; k<4; ++k) 
             wmma::store_matrix_sync(&my_sO[k*16], acc_O[k], 64, wmma::mem_row_major);
        __syncwarp();
        
        // Apply correction
        if (lane_id < 16) {
             int r = lane_id;
             for(int c=0; c<64; ++c) my_sO[r * 64 + c] *= e_correction;
        }
        __syncwarp();
        
        // Load rescaled O back to accumulators
        for(int k=0; k<4; ++k) 
             wmma::load_matrix_sync(acc_O[k], &my_sO[k*16], 64, wmma::mem_row_major);
             
        // 4. Compute O += P * V
        wmma::load_matrix_sync(frag_P, my_P_half, 16);
        
        for(int k=0; k<4; ++k) {
             if (j < S) wmma::load_matrix_sync(frag_V, V + j * D + k*16, D);
             else wmma::fill_fragment(frag_V, 0.0f);
             wmma::mma_sync(acc_O[k], frag_P, frag_V, acc_O[k]);
        }
    }
    
    // Store O to Global
    // Store to SMEM first for division
    for(int k=0; k<4; ++k) 
         wmma::store_matrix_sync(&my_sO[k*16], acc_O[k], 64, wmma::mem_row_major);
    __syncwarp();
    
    if (lane_id < 16) {
        int r = lane_id; 
        for(int c=0; c<64; ++c) {
             float val = my_sO[r * 64 + c] / l_prev;
             O[q_row_global * D + r * D + c] = __float2half(val);
        }
    }
}
