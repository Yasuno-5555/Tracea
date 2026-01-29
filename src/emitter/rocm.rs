use crate::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use crate::semantic::transition::SyncRequirement;
use crate::emitter::rocm_driver::*;

pub struct ROCMEmitter {
    pub isa: String,
    pub wavefront_size: i32,
    pub max_lds: i32,
}

impl ROCMEmitter {
    pub fn detect() -> Self {
        let mut isa = "gfx90a".to_string(); // Default fallback
        let mut wavefront_size = 64; // Default CDNA
        let mut max_lds = 65536;

        if let Some(api) = RocmDriverApi::get() {
            let mut count = 0;
            unsafe { (api.hipGetDeviceCount)(&mut count); }
            if count > 0 {
                let mut major = 0;
                let mut minor = 0;
                let mut wf = 64;
                let mut lds = 0;
                unsafe {
                    (api.hipDeviceGetAttribute)(&mut major, HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
                    (api.hipDeviceGetAttribute)(&mut minor, HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
                    (api.hipDeviceGetAttribute)(&mut wf, HIP_DEVICE_ATTRIBUTE_WAVEFRONT_SIZE, 0);
                    (api.hipDeviceGetAttribute)(&mut lds, HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 0);
                }
                
                // Construct approximation of ISA name
                isa = format!("gfx{}{:02}", major, minor);
                wavefront_size = wf;
                max_lds = lds;
            }
        }

        Self { isa, wavefront_size, max_lds }
    }

    pub fn generate_gemm(&self, config: crate::PipelineConfig) -> String {
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        let n_stages = config.num_stages;
        
        let mfma_intrinsic = self.get_mfma_intrinsic(mt, nt, kt);
        let wf_size = self.wavefront_size;
        let primitives = crate::backend::rocm::RocmBackend::get_primitive_defs();

        // Tiling logic:
        // For a 128-thread block (Cuda default), it's 2 waves on CDNA.
        // We divide mt/nt among waves.
        let waves_per_block = 128 / wf_size;
        let mt_per_wave = mt / if waves_per_block > 1 { 2 } else { 1 };
        let _nt_per_wave = nt / if waves_per_block == 1 { 1 } else { 1 }; // Simple 1D wave split for now

        format!(r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

{primitives}

extern "C" __global__ void gemm_rocm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {{
    // LDS Allocation (Pipelined)
    __shared__ half lds_A[{n_stages} * {mt} * {kt}];
    __shared__ half lds_B[{n_stages} * {kt} * {nt}];

    int tid = hipThreadIdx_x;
    int bx = hipBlockIdx_x;
    int by = hipBlockIdx_y;
    
    int wave_id = tid / {wf_size};
    int lane_id = tid % {wf_size};

    // Accumulators for MFMA (32 floats for 32x32x8f16)
    float acc[32];
    #pragma unroll
    for(int i=0; i<32; ++i) acc[i] = 0.0f;

    // Mapping: Wave 0 handles top half, Wave 1 bottom half (if mt=128)
    int row_start = by * {mt} + wave_id * {mt_per_wave};
    int col_start = bx * {nt};

    for (int k_step = 0; k_step < K; k_step += {kt}) {{
        // Load A into LDS
        #pragma unroll
        for(int i=0; i < ({mt}*{kt} + 127)/128; ++i) {{
            int idx = i * 128 + tid;
            if(idx < {mt}*{kt}) {{
                int r = idx / {kt};
                int c = idx % {kt};
                if(by*{mt}+r < M && k_step+c < K)
                    lds_A[idx] = A[(by*{mt}+r)*K + (k_step+c)];
                else
                    lds_A[idx] = 0;
            }}
        }}

        // Load B into LDS
        #pragma unroll
        for(int i=0; i < ({nt}*{kt} + 127)/128; ++i) {{
            int idx = i * 128 + tid;
            if(idx < {nt}*{kt}) {{
                int r = idx / {nt};
                int c = idx % {nt};
                if(k_step+r < K && bx*{nt}+c < N)
                    lds_B[idx] = B[(k_step+r)*N + (bx*{nt}+c)];
                else
                    lds_B[idx] = 0;
            }}
        }}

        __syncthreads();

        // MFMA Lowering
        // For 32x32x8f16: m=32, n=32, k=8.
        // LDS A is [mt][kt], lds B is [kt][nt].
        // Each wave computes its block.
        #pragma unroll
        for(int ki=0; ki < {kt}; ki += 8) {{
            // Construct input vectors for MFMA
            // These are architecture specific builtins
            half4 val_a = *(half4*)&lds_A[wave_id * {mt_per_wave} * {kt} + (lane_id % 32) * {kt} + ki];
            half4 val_b = *(half4*)&lds_B[ki * {nt} + (lane_id / 32) * {nt}]; // Simplified B-load
            
            acc = {mfma_intrinsic}(val_a, val_b, acc, 0, 0, 0);
        }}
        
        __syncthreads();
    }}

    // Epilogue: Store results to C
    #pragma unroll
    for(int i=0; i<32; ++i) {{
        int r = i / 8; // Simplified offset logic for demo
        int c = i % 8;
        if(row_start + r < M && col_start + c < N)
            C[(row_start + r)*N + (col_start + c)] = acc[i];
    }}
}}
"#, mt=mt, nt=nt, kt=kt, n_stages=n_stages, mfma_intrinsic=mfma_intrinsic, wf_size=wf_size, mt_per_wave=mt_per_wave)
    }

    fn get_mfma_intrinsic(&self, m: u32, n: u32, k: u32) -> String {
        // Generational and Shape-based selection
        match self.isa.as_str() {
            i if i.contains("gfx90a") || i.contains("gfx940") => {
                // CDNA2/3: Advanced MFMA
                if m == 32 && n == 32 && k == 8 { "__builtin_amdgcn_mfma_f32_32x32x8f16".to_string() }
                else if m == 32 && n == 32 && k == 16 { "__builtin_amdgcn_mfma_f32_32x32x16f16".to_string() }
                else { "__builtin_amdgcn_mfma_f32_16x16x16f16".to_string() }
            }
            i if i.contains("gfx9") => {
                // First gen CDNA
                "__builtin_amdgcn_mfma_f32_32x32x8f16".to_string()
            }
            i if i.contains("gfx11") => {
                // RDNA3: WMMA or MFMA (depending on specific ROCm version/intrinsic availability)
                "__builtin_amdgcn_mfma_f32_16x16x32_f16".to_string()
            }
            _ => "// No matching MFMA found for ISA".to_string()
        }
    }
}

impl Emitter for ROCMEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::Barrier => "__syncthreads();".to_string(),
            _ => String::new(),
        }
    }

    fn generate_from_ir(&self, ir: &UnifiedOpIR) -> String {
        match &ir.op_type {
            UnifiedOpType::Gemm { .. } => self.generate_gemm(ir.tiling.clone()),
            UnifiedOpType::FusedAttention { .. } => {
                "// ROCm FA2 not yet implemented in Unified Emitter\n".to_string()
            }
            UnifiedOpType::Elementwise { .. } => {
                panic!("Elementwise Ops should be handled by UniversalEmitter.");
            }
            UnifiedOpType::Conv2d { .. } => {
                panic!("Conv2d Ops should be handled by UniversalEmitter.");
            }
            UnifiedOpType::ConvTranspose2d { .. } => {
                "// ROCm ConvTranspose2d not yet implemented - fallback to CPU\n".to_string()
            }
            UnifiedOpType::MatrixCore { .. } => {
                panic!("MatrixCore Ops not supported on ROCm yet.");
            }
            UnifiedOpType::LowRankMlp { .. } => {
                panic!("LowRankMlp not supported on ROCm yet.");
            }
            UnifiedOpType::Softmax { .. } => {
                "// ROCm Softmax not yet implemented in Unified Emitter\n".to_string()
            }
            UnifiedOpType::BatchNorm { .. } => {
                "// ROCm BatchNorm not yet implemented in Unified Emitter\n".to_string()
            }
            UnifiedOpType::GlobalAveragePool { .. } | UnifiedOpType::Linear { .. } => {
                "// ROCm GlobalAveragePool/Linear not yet implemented\n".to_string()
            }
        }
    }
}
