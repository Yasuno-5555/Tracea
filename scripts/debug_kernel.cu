
__global__ void gemm_pipelined_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {
    // Pipeline Stages: 3
    // Tile size: 128x64x32
    extern __shared__ float smem[];
    float* As_base = &smem[0];
    float* Bs_base = &smem[128 * 32 * 3];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) 
        for (int j = 0; j < 8; ++j) 
            acc[i][j] = 0.0f;

    // --- PROLOGUE ---
    for (int s = 0; s < 2; ++s) {
        int k_offset = s * 32;
        if (k_offset < K) {
            // Load A tile
            for (int i = 0; i < 8; ++i) {
                int m_idx = by * 128 + ty * 8 + i;
                int k_idx = k_offset + tx;
                float* dest_a = &As_base[s * (128 * 32) + (ty * 8 + i) * 32 + tx];
                if (m_idx < M && k_idx < K) 
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "l"((unsigned long long)dest_a), "l"((unsigned long long)&A[m_idx * K + k_idx]));
            }
            // Load B tile
            for (int i = 0; i < 8; ++i) {
                int n_idx = bx * 64 + tx * 8 + i;
                int k_idx = k_offset + ty;
                float* dest_b = &Bs_base[s * (64 * 32) + (tx * 8 + i) * 32 + ty];
                if (n_idx < N && k_idx < K)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "l"((unsigned long long)dest_b), "l"((unsigned long long)&B[k_idx * N + n_idx]));
            }
            asm volatile("cp.async.commit_group;");
        }
    }

    // --- MAIN LOOP ---
    for (int k_outer = 2 * 32; k_outer < K + 2 * 32; k_outer += 32) {
        // WAIT for the stage we are about to compute
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        int compute_phase = (k_outer / 32 - 2) % 3;
        int load_phase = (k_outer / 32) % 3;

        // 1. Issue load for load_phase
        if (k_outer < K) {
            for (int i = 0; i < 8; ++i) {
                int m_idx = by * 128 + ty * 8 + i;
                int k_idx = k_outer + tx;
                float* dest_a = &As_base[load_phase * (128 * 32) + (ty * 8 + i) * 32 + tx];
                if (m_idx < M && k_idx < K)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "l"((unsigned long long)dest_a), "l"((unsigned long long)&A[m_idx * K + k_idx]));
                
                int n_idx = bx * 64 + tx * 8 + i;
                int k_idx_b = k_outer + ty;
                float* dest_b = &Bs_base[load_phase * (64 * 32) + (tx * 8 + i) * 32 + ty];
                if (n_idx < N && k_idx_b < K)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "l"((unsigned long long)dest_b), "l"((unsigned long long)&B[k_idx_b * N + n_idx]));
            }
            asm volatile("cp.async.commit_group;");
        }

        // 2. Compute compute_phase
        float* cur_As = &As_base[compute_phase * (128 * 32)];
        float* cur_Bs = &Bs_base[compute_phase * (64 * 32)];
        
        // M3 optimization: Register Double Buffering
        float frag_a[2][8];
        float frag_b[2][8];
        
        #pragma unroll
        for(int i=0; i<8; ++i) frag_a[0][i] = cur_As[(ty * 8 + i) * 32];
        #pragma unroll
        for(int i=0; i<8; ++i) frag_b[0][i] = cur_Bs[(tx * 8 + i) * 32];

        #pragma unroll
        for (int k_inner = 0; k_inner < 32; ++k_inner) {
            int next_k = (k_inner + 1) % 32;
            int cur_reg = k_inner % 2;
            int next_reg = (k_inner + 1) % 2;

            if (k_inner < 31) {
                #pragma unroll
                for(int i=0; i<8; ++i) frag_a[next_reg][i] = cur_As[(ty * 8 + i) * 32 + next_k];
                #pragma unroll
                for(int i=0; i<8; ++i) frag_b[next_reg][i] = cur_Bs[(tx * 8 + i) * 32 + next_k];
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] += frag_a[cur_reg][i] * frag_b[cur_reg][j];
                }
            }
        }
    }

    // Write back with Fusion
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int global_m = by * 128 + ty * 8 + i;
            int global_n = bx * 64 + tx * 8 + j;
            if (global_m < M && global_n < N) {
                float val = acc[i][j];
                
                C[global_m * N + global_n] = val;
            }
        }
    }
}
