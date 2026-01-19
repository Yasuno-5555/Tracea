
#include <cuda_fp16.h>

#define INFINITY 1e9f

__device__ __forceinline__ unsigned int get_smem_ptr(const void *ptr) {
    unsigned int ret;
    asm volatile("{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void ldmatrix_x4(unsigned int* reg, void* smem_ptr) {
    unsigned int addr = get_smem_ptr(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned int* reg, void* smem_ptr) {
    unsigned int addr = get_smem_ptr(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
}

extern "C" __global__ void flash_attention_v2_kernel(
    const half* __restrict__ Q, 
    const half* __restrict__ K, 
    const half* __restrict__ V, 
    half* __restrict__ O, 
    int B, int H, int S, int D,
    float output_scale
) {
    // Single-buffer layout to save smem
    // sQ: [br, D]
    // sK: [bc, D] 
    // sV: [bc, D] 
    // sP: [br, bc] 
    extern __shared__ __align__(16) uint4 smem_u4[];
    half* smem = (half*)smem_u4;

    half* sQ = &smem[0];
    half* sK = &smem[64 * D];
    half* sV = &smem[64 * D + 64 * D];
    half* sP = &smem[64 * D + 2 * 64 * D];

    int tx = threadIdx.x;
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z; 
    int num_threads = blockDim.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    long long batch_head_offset = (long long)bz * H * S * D + (long long)by * S * D;
    const half* Q_base = Q + batch_head_offset + (long long)bx * 64 * D;
    const half* K_base = K + batch_head_offset;
    const half* V_base = V + batch_head_offset;
    half* O_base = O + batch_head_offset + (long long)bx * 64 * D;
    
    // Initial Load Q
    {
        int items = (64 * D) / 8; 
        for (int i = tx; i < items; i += num_threads) {
            int row = i / (D/8);
            int col = (i % (D/8)) * 8;
            if (bx * 64 + row < S) {
                *(float4*)&sQ[row * D + col] = *(float4*)&Q_base[row * D + col];
            } else {
                *(float4*)&sQ[row * D + col] = make_float4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    int tiles_d = D / 16;
    int tiles_bc = 64 / 8;
    int tiles_v = D / 8;
    int num_k_tiles = (S + 64 - 1) / 64;

    float O_reg[16][4]; 
    float m_curr = -INFINITY;
    float l_curr = 0.0f;
    #pragma unroll
    for(int j=0; j<16; ++j) for(int k=0; k<4; ++k) O_reg[j][k] = 0.0f;

    unsigned int Q_frag[8][4]; 
    bool active = (warp_id * 16 < 64);
    if (active) {
        #pragma unroll
        for(int k_sep=0; k_sep < tiles_d; ++k_sep) {
            ldmatrix_x4(Q_frag[k_sep], &sQ[(warp_id * 16) * D + k_sep * 16]);
        }
    }

    for(int kb = 0; kb < num_k_tiles; ++kb) {
        // Load K, V (Synchronous Load inside loop for now to be safe)
        {
            int items = (64 * D) / 8;
            for (int i = tx; i < items; i += num_threads) {
                int row = i / (D/8);
                int col = (i % (D/8)) * 8;
                if (row < 64) {
                    if (kb * 64 + row < S) {
                        *(float4*)&sK[row * D + col] = *(float4*)&K_base[kb * 64 * D + row * D + col];
                        *(float4*)&sV[row * D + col] = *(float4*)&V_base[kb * 64 * D + row * D + col];
                    } else {
                        *(float4*)&sK[row * D + col] = make_float4(0,0,0,0);
                        *(float4*)&sV[row * D + col] = make_float4(0,0,0,0);
                    }
                }
            }
        }
        __syncthreads();

        if (active) {
            float S_frag[16][4]; 
            #pragma unroll
            for(int j=0; j<16; ++j) for(int r=0; r<4; ++r) S_frag[j][r] = 0.0f;
            
            #pragma unroll
            for(int d_step=0; d_step < tiles_d; ++d_step) {
                #pragma unroll
                for(int k_blk=0; k_blk < (tiles_bc / 2); ++k_blk) { 
                     unsigned int K_reg[4];
                     ldmatrix_x4(K_reg, &sK[(k_blk * 16) * D + d_step * 16]);
                     asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                        : "+f"(S_frag[k_blk*2][0]), "+f"(S_frag[k_blk*2][1]), "+f"(S_frag[k_blk*2][2]), "+f"(S_frag[k_blk*2][3])
                        : "r"(Q_frag[d_step][0]), "r"(Q_frag[d_step][1]), "r"(Q_frag[d_step][2]), "r"(Q_frag[d_step][3]),
                        "r"(K_reg[0]), "r"(K_reg[1]), 
                        "f"(S_frag[k_blk*2][0]), "f"(S_frag[k_blk*2][1]), "f"(S_frag[k_blk*2][2]), "f"(S_frag[k_blk*2][3]));
                     asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                        : "+f"(S_frag[k_blk*2+1][0]), "+f"(S_frag[k_blk*2+1][1]), "+f"(S_frag[k_blk*2+1][2]), "+f"(S_frag[k_blk*2+1][3])
                        : "r"(Q_frag[d_step][0]), "r"(Q_frag[d_step][1]), "r"(Q_frag[d_step][2]), "r"(Q_frag[d_step][3]),
                        "r"(K_reg[2]), "r"(K_reg[3]), 
                        "f"(S_frag[k_blk*2+1][0]), "f"(S_frag[k_blk*2+1][1]), "f"(S_frag[k_blk*2+1][2]), "f"(S_frag[k_blk*2+1][3]));
                }
            }

            float m_new = m_curr;
            #pragma unroll
            for(int j=0; j < tiles_bc; ++j) {
                #pragma unroll
                for(int k=0; k<4; ++k) {
                     int r = (lane_id % 4) + (lane_id / 16) * 4 + (k / 2) * 8;
                     int c = (lane_id / 4) % 4 + (j / 4) * 0; // mapping is complex, let's just use j
                     int gr = bx * 64 + warp_id * 16 + r;
                     int gc = kb * 64 + j * 8 + (lane_id / 4);
                     float val = S_frag[j][k] * output_scale;
                     if (gc >= S || gr >= S) val = -INFINITY;
                     
                     S_frag[j][k] = val; 
                     m_new = fmaxf(m_new, val);
                }
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask /= 2) m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, mask));
            
            float alpha = __expf(m_curr - m_new);
            float row_sum_P = 0.0f;
            #pragma unroll
            for(int j=0; j < tiles_bc; ++j) for(int k=0; k<4; ++k) {
                 S_frag[j][k] = __expf(S_frag[j][k] - m_new); 
                 row_sum_P += S_frag[j][k];
            }
            l_curr = l_curr * alpha + row_sum_P;
            m_curr = m_new;
            #pragma unroll
            for(int j=0; j < 16; ++j) for(int k=0; k<4; ++k) O_reg[j][k] *= alpha;
            
            #pragma unroll
            for(int j=0; j < tiles_bc; ++j) { 
                 int r = (lane_id % 4) + (lane_id / 16) * 4;
                 int c = (lane_id / 4);
                 sP[(warp_id * 16 + r) * 64 + j * 8 + c] = __float2half(S_frag[j][0]);
                 sP[(warp_id * 16 + r + 8) * 64 + j * 8 + c] = __float2half(S_frag[j][2]);
            }
        }
        __syncthreads(); 

        if (active) {
            #pragma unroll
            for(int k_sep=0; k_sep < (tiles_bc / 2); ++k_sep) { 
                 unsigned int P_A[4];
                 ldmatrix_x4(P_A, &sP[(warp_id * 16) * 64 + k_sep * 16]); 
                 #pragma unroll
                 for(int d_blk=0; d_blk < tiles_v; ++d_blk) { 
                     unsigned int V_B[2];
                     ldmatrix_x2(V_B, &sV[(k_sep*16) * D + d_blk * 8]);
                     asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                        : "+f"(O_reg[d_blk][0]), "+f"(O_reg[d_blk][1]), "+f"(O_reg[d_blk][2]), "+f"(O_reg[d_blk][3])
                        : "r"(P_A[0]), "r"(P_A[1]), "r"(P_A[2]), "r"(P_A[3]), 
                        "r"(V_B[0]), "r"(V_B[1]), 
                        "f"(O_reg[d_blk][0]), "f"(O_reg[d_blk][1]), "f"(O_reg[d_blk][2]), "f"(O_reg[d_blk][3]));
                 }
            }
        }
        __syncthreads();
    }
    
    if (active) {
        #pragma unroll
        for(int j=0; j < tiles_v; ++j) { 
             for(int k=0; k<4; ++k) O_reg[j][k] /= l_curr;
             int r = (lane_id % 4) + (lane_id / 16) * 4;
             int c = (lane_id / 4);
             sQ[(warp_id * 16 + r) * D + j * 8 + c] = __float2half(O_reg[j][0]);
             sQ[(warp_id * 16 + r + 8) * D + j * 8 + c] = __float2half(O_reg[j][2]);
        }
    }
    __syncthreads();
    
    {
        int items = (64 * D) / 8;
        for (int i = tx; i < items; i += num_threads) { 
             int row = i / (D/8);
             int col = (i % (D/8)) * 8;
             if (bx * 64 + row < S) {
                 *(float4*)&O_base[row * D + col] = *(float4*)&sQ[row * D + col];
             }
        }
    }
}
