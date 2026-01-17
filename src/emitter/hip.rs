use crate::emitter::traits::Emitter;
use crate::semantic::transition::SyncRequirement;
use crate::semantic::fusion::EpilogueOp;

pub struct HIPEmitter;

impl HIPEmitter {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_pipelined_gemm(&self, config: crate::PipelineConfig) -> String {
        let n = config.num_stages;
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;

        let mut kernel = String::new();
        kernel.push_str(&format!(r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void gemm_hip_pipelined_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {{
    // AMD Matrix Core Pipeline: {n} stages
    // Tile size: {mt}x{nt}x{kt}
    __shared__ float smem[{total_smem}];
    float* As_base = &smem[0];
    float* Bs_base = &smem[{mt} * {kt} * {n}];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc[32]; // MFMA f32 16x16 accumulates into 4-16 registers
    for(int i=0; i<32; ++i) acc[i] = 0.0f;

    // --- PROLOGUE ---
    for (int s = 0; s < {n_minus_1}; ++s) {{
        int k_offset = s * {kt};
        if (k_offset < K) {{
            // Load A to LDS (AMD MI-series usually via Registers)
            for (int i = 0; i < 4; ++i) {{ // Simplified load
                int m_idx = by * {mt} + ty * 4 + i;
                int k_idx = k_offset + tx;
                if (m_idx < M && k_idx < K)
                    As_base[s * ({mt} * {kt}) + (ty * 4 + i) * {kt} + tx] = A[m_idx * K + k_idx];
            }}
            // Wait for memory and sync LDS
            __builtin_amdgcn_s_waitcnt(0);
            __syncthreads();
        }}
    }}
"#, total_smem = (mt * kt + nt * kt) * n, n = n, mt = mt, nt = nt, kt = kt, n_minus_1 = n - 1));

        kernel.push_str(&format!(r#"
    // --- MAIN LOOP ---
    for (int k_outer = {n_minus_1} * {kt}; k_outer < K + {n_minus_1} * {kt}; k_outer += {kt}) {{
        int compute_phase = (k_outer / {kt} - {n_minus_1}) % {n};
        int load_phase = (k_outer / {kt}) % {n};

        // 1. Issue load for next-next stage
        if (k_outer < K) {{
            // ... Load more elements to As_base[load_phase ...] ...
        }}

        // 2. Compute current stage
        float* cur_As = &As_base[compute_phase * ({mt} * {kt})];
        float* cur_Bs = &Bs_base[compute_phase * ({nt} * {kt})];

        // MFMA Intrinsic call
        // v_mfma_f32_16x16x16f32(...);
        {mfma_call}
    }}

    // Write back logic...
    {epilogue}
}}
"#, n_minus_1 = n - 1, kt = kt, n = n, mt = mt, nt = nt, mfma_call = self.emit_v_mfma(mt, nt, kt), epilogue = self.emit_epilogue(config.epilogue.as_slice(), "val")));
        kernel
    }

    pub fn emit_v_mfma(&self, mt: u32, nt: u32, kt: u32) -> String {
        format!("v_mfma_f32_{mt}x{nt}x{kt}f32(acc, frag_a, frag_b, acc);")
    }

    pub fn emit_epilogue(&self, ops: &[EpilogueOp], acc_name: &str) -> String {
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
                _ => {}
            }
        }
        code
    }
}

impl Emitter for HIPEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::WaitAsyncLoad { stages_behind: _ } => {
                "__builtin_amdgcn_s_waitcnt(0);".to_string()
            }
            SyncRequirement::Barrier => "__syncthreads();".to_string(),
            SyncRequirement::None => "".to_string(),
        }
    }

    fn emit_epilogue(&self, ops: &[EpilogueOp], acc_name: &str) -> String {
        self.emit_epilogue(ops, acc_name)
    }
}
