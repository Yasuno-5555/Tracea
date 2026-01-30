use crate::core::tuning::{TunableKernel, SearchSpace};
use crate::core::config::{PipelineConfig, SpecializedInstruction};
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RocmGemmProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl RocmGemmProblem {
    pub fn signature(&self) -> String {
        format!("m{}_n{}_k{}", self.m, self.n, self.k)
    }
}

pub struct RocmGemmAdapter {
    pub runtime: Arc<RuntimeManager>,
    pub problem: RocmGemmProblem,
    // Buffers
    a_buf: crate::runtime::manager::BufferId,
    b_buf: crate::runtime::manager::BufferId,
    c_buf: crate::runtime::manager::BufferId,
}

impl RocmGemmAdapter {
    pub fn new(runtime: Arc<RuntimeManager>, problem: RocmGemmProblem) -> Self {
        let size_a = problem.m * problem.k * 2; // f16
        let size_b = problem.k * problem.n * 2; // f16
        let size_c = problem.m * problem.n * 4; // f32 accumulation

        // Allocate on ROCm device
        // Note: this might fail if no ROCm device is present, user must check beforehand or we mock.
        let a = runtime.alloc(size_a, DeviceBackend::Rocm).unwrap_or(crate::runtime::manager::BufferId(0));
        let b = runtime.alloc(size_b, DeviceBackend::Rocm).unwrap_or(crate::runtime::manager::BufferId(0));
        let c = runtime.alloc(size_c, DeviceBackend::Rocm).unwrap_or(crate::runtime::manager::BufferId(0));

        Self { runtime, problem, a_buf: a, b_buf: b, c_buf: c }
    }
}

impl TunableKernel for RocmGemmAdapter {
    type Config = PipelineConfig;

    fn name(&self) -> &'static str {
        "rocm_gemm_mfma"
    }

    fn search_space(&self) -> SearchSpace<Self::Config> {
        let mut candidates = Vec::new();
        // Matrix Core configs for MFMA: 32x32 tiles are common
        let patterns = [
            (128, 128, 32),
            (64, 64, 32),
            (32, 32, 16), // Smaller tiles for MFMA32
        ];

        for &(m, n, k) in &patterns {
            for &stages in &[2] {
                let mut cfg = PipelineConfig::new(stages, m, n, k as u32);
                cfg.instruction = SpecializedInstruction::RocmMFMA;
                candidates.push(cfg);
            }
        }
        SearchSpace::new(candidates)
    }

    fn is_feasible(&self, cfg: &Self::Config) -> bool {
        if cfg.m_tile % 32 != 0 || cfg.n_tile % 32 != 0 {
            return false;
        }
        true
    }

    fn benchmark(&self, cfg: &Self::Config) -> Option<f32> {
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm { 
                m: self.problem.m as u32,
                n: self.problem.n as u32,
                k: self.problem.k as u32,
                batch: 1,
                epilogue: vec![],
            },
            precison: "f16".to_string(),
            tiling: cfg.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };

        let emitter = UniversalEmitter::new(DeviceBackend::Rocm);
        let source = emitter.generate(ir);
        let kernel_name = "gemm_rocm_kernel";

        let kernel_id = match self.runtime.compile(&source, kernel_name, DeviceBackend::Rocm) {
            Ok(id) => id,
            Err(e) => {
                // If on non-ROCm machine this fails gracefully
                eprintln!("ROCm Compile Failed: {}", e);
                return None;
            }
        };

        // Grid/Block for ROCm
        // Emitter assumes Wave64 (64 threads per wave).
        // 128 threads/block = 2 waves.
        let block_size = 128; // Fixed in emitter logic?
        // Emitter: loops over "waves_per_block = 128 / wf_size".
        // So we must launch with 128 threads per block to match logic.
        
        let grid = (
            (self.problem.n as u32 + cfg.n_tile - 1) / cfg.n_tile, // bx maps to N in emitter? 
             // Emitter: "int col_start = bx * {nt}" -> YES.
            (self.problem.m as u32 + cfg.m_tile - 1) / cfg.m_tile, // by maps to M
            1
        );
        let block = (block_size, 1, 1);

        // Smem logic in emitter is:
        // lds_A[{n_stages} * {mt} * {kt}] (half)
        // lds_B[{n_stages} * {kt} * {nt}] (half)
        let lds_bytes = (cfg.num_stages * (cfg.m_tile * cfg.k_tile + cfg.k_tile * cfg.n_tile)) * 2;

        let args = vec![
            KernelArg::Buffer(self.a_buf),
            KernelArg::Buffer(self.b_buf),
            KernelArg::Buffer(self.c_buf),
            KernelArg::Int(self.problem.m as i32),
            KernelArg::Int(self.problem.n as i32),
            KernelArg::Int(self.problem.k as i32),
        ];

        let start = std::time::Instant::now();
        if let Err(_) = self.runtime.launch(kernel_id, grid, block, lds_bytes, args) {
             return None;
        }
        // Sync? Rocm sync method? runtime.synchronize() currently only checks CUDA.
        // TODO: Implement ROCm sync in RuntimeManager.
        // For now, assume blocking or minimal support.
        
        let nanos = start.elapsed().as_nanos() as f32;

        let gflops = (2.0 * self.problem.m as f32 * self.problem.n as f32 * self.problem.k as f32) / nanos;
        Some(gflops)
    }

    fn cache_key(&self) -> String {
        format!("rocm_gemm_{}", self.problem.signature())
    }
}
