use crate::emitter::traits::Emitter;
use crate::semantic::transition::SyncRequirement;
use crate::semantic::swizzle::SwizzleMode;
use crate::semantic::fusion::EpilogueOp;

pub struct CUDAEmitter;

impl CUDAEmitter {
    pub fn new() -> Self {
        Self
    }

    pub fn emit_swizzled_addr(&self, row: &str, col: &str, mode: SwizzleMode) -> String {
        match mode {
            SwizzleMode::Xor128 => {
                format!("(({row} ^ ({col} >> 3)) << 4) | ({col} & 0xF)", row = row, col = col)
            }
            SwizzleMode::None => {
                format!("({row} * 128 + {col})", row = row, col = col) // Default stride
            }
        }
    }

    pub fn generate_pipelined_gemm(&self, config: crate::PipelineConfig) -> String {
        let n = config.num_stages;
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;

        let mut kernel = String::new();
        kernel.push_str(&format!(r#"
__global__ void gemm_pipelined_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {{
    // Pipeline Stages: {n}
    // Tile size: {mt}x{nt}x{kt}
    extern __shared__ float smem[];
    float* As_base = &smem[0];
    float* Bs_base = &smem[{mt} * {kt} * {n}];

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
    for (int s = 0; s < {n_minus_1}; ++s) {{
        int k_offset = s * {kt};
        if (k_offset < K) {{
            // Load A tile
            for (int i = 0; i < 8; ++i) {{
                int m_idx = by * {mt} + ty * 8 + i;
                int k_idx = k_offset + tx;
                float* dest_a = &As_base[s * ({mt} * {kt}) + (ty * 8 + i) * {kt} + tx];
                if (m_idx < M && k_idx < K) 
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dest_a), "l"(&A[m_idx * K + k_idx]));
            }}
            // Load B tile
            for (int i = 0; i < 8; ++i) {{
                int n_idx = bx * {nt} + tx * 8 + i;
                int k_idx = k_offset + ty;
                float* dest_b = &Bs_base[s * ({nt} * {kt}) + (tx * 8 + i) * {kt} + ty];
                if (n_idx < N && k_idx < K)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dest_b), "l"(&B[k_idx * N + n_idx]));
            }}
            asm volatile("cp.async.commit_group;");
        }}
    }}
"#, n = n, mt = mt, nt = nt, kt = kt, n_minus_1 = n - 1));

        kernel.push_str(&format!(r#"
    // --- MAIN LOOP ---
    for (int k_outer = {n_minus_1} * {kt}; k_outer < K + {n_minus_1} * {kt}; k_outer += {kt}) {{
        // WAIT for the stage we are about to compute
        asm volatile("cp.async.wait_group {wait_stages};");
        __syncthreads();

        int compute_phase = (k_outer / {kt} - {n_minus_1}) % {n};
        int load_phase = (k_outer / {kt}) % {n};

        // 1. Issue load for load_phase
        if (k_outer < K) {{
            for (int i = 0; i < 8; ++i) {{
                int m_idx = by * {mt} + ty * 8 + i;
                int k_idx = k_outer + tx;
                float* dest_a = &As_base[load_phase * ({mt} * {kt}) + (ty * 8 + i) * {kt} + tx];
                if (m_idx < M && k_idx < K)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dest_a), "l"(&A[m_idx * K + k_idx]));
                
                int n_idx = bx * {nt} + tx * 8 + i;
                int k_idx_b = k_outer + ty;
                float* dest_b = &Bs_base[load_phase * ({nt} * {kt}) + (tx * 8 + i) * {kt} + ty];
                if (n_idx < N && k_idx_b < K)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dest_b), "l"(&B[k_idx_b * N + n_idx]));
            }}
            asm volatile("cp.async.commit_group;");
        }}

        // 2. Compute compute_phase
        float* cur_As = &As_base[compute_phase * ({mt} * {kt})];
        float* cur_Bs = &Bs_base[compute_phase * ({nt} * {kt})];
        
        // M3 optimization: Register Double Buffering
        float frag_a[2][8];
        float frag_b[2][8];
        
        #pragma unroll
        for(int i=0; i<8; ++i) frag_a[0][i] = cur_As[(ty * 8 + i) * {kt}];
        #pragma unroll
        for(int i=0; i<8; ++i) frag_b[0][i] = cur_Bs[(tx * 8 + i) * {kt}];

        #pragma unroll
        for (int k_inner = 0; k_inner < {kt}; ++k_inner) {{
            int next_k = (k_inner + 1) % {kt};
            int cur_reg = k_inner % 2;
            int next_reg = (k_inner + 1) % 2;

            if (k_inner < {kt_minus_1}) {{
                #pragma unroll
                for(int i=0; i<8; ++i) frag_a[next_reg][i] = cur_As[(ty * 8 + i) * {kt} + next_k];
                #pragma unroll
                for(int i=0; i<8; ++i) frag_b[next_reg][i] = cur_Bs[(tx * 8 + i) * {kt} + next_k];
            }}

            #pragma unroll
            for (int i = 0; i < 8; ++i) {{
                for (int j = 0; j < 8; ++j) {{
                    acc[i][j] += frag_a[cur_reg][i] * frag_b[cur_reg][j];
                }}
            }}
        }}
    }}

    // Write back with Fusion
    for (int i = 0; i < 8; ++i) {{
        for (int j = 0; j < 8; ++j) {{
            int global_m = by * {mt} + ty * 8 + i;
            int global_n = bx * {nt} + tx * 8 + j;
            if (global_m < M && global_n < N) {{
                float val = acc[i][j];
                {epilogue}
                C[global_m * N + global_n] = val;
            }}
        }}
    }}
}}
"#, n_minus_1 = n - 1, kt = kt, kt_minus_1 = kt - 1, wait_stages = n - 2, mt = mt, nt = nt, n = n, epilogue = self.emit_epilogue(config.epilogue.as_slice(), "val")));
        kernel
    }
}

impl Emitter for CUDAEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::WaitAsyncLoad { stages_behind } => {
                format!("asm volatile(\"cp.async.wait_group %0;\" : : \"n\"({}));", stages_behind)
            }
            SyncRequirement::Barrier => "__syncthreads();".to_string(),
            SyncRequirement::None => "".to_string(),
        }
    }

    fn emit_epilogue(&self, ops: &[EpilogueOp], acc_name: &str) -> String {
        let mut code = String::new();
        for op in ops {
            match op {
                EpilogueOp::BiasAdd { bias_ptr } => {
                    code.push_str(&format!("  {acc} += ((float*){ptr})[global_n];\n", acc = acc_name, ptr = bias_ptr));
                }
                EpilogueOp::ReLU => {
                    code.push_str(&format!("  {acc} = ({acc} > 0.0f) ? {acc} : 0.0f;\n", acc = acc_name));
                }
                EpilogueOp::Gelu => {
                    code.push_str(&format!("  {acc} *= 0.5f * (1.0f + tanhf(0.79788456f * ({acc} + 0.044715f * {acc} * {acc} * {acc})));\n", acc = acc_name));
                }
                EpilogueOp::None => {}
            }
        }
        code
    }
}
