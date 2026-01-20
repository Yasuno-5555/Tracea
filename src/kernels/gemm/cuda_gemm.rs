use crate::core::tuning::{TunableKernel, SearchSpace};
use crate::core::config::{PipelineConfig, SpecializedInstruction};
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CudaGemmProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl CudaGemmProblem {
    pub fn signature(&self) -> String {
        format!("m{}_n{}_k{}", self.m, self.n, self.k)
    }
}

pub struct CudaGemmAdapter {
    pub runtime: Arc<RuntimeManager>,
    pub problem: CudaGemmProblem,
    // Buffers
    pub a_buf: crate::runtime::manager::BufferId,
    pub b_buf: crate::runtime::manager::BufferId,
    pub c_buf: crate::runtime::manager::BufferId,
}

impl CudaGemmAdapter {
    pub fn new(runtime: Arc<RuntimeManager>, problem: CudaGemmProblem) -> Self {
        let size_a = problem.m * problem.k * 2; // f16
        let size_b = problem.k * problem.n * 2; // f16
        let size_c = problem.m * problem.n * 4; // f32 accumulation

        let a = runtime.alloc(size_a, DeviceBackend::Cuda).unwrap();
        let b = runtime.alloc(size_b, DeviceBackend::Cuda).unwrap();
        let c = runtime.alloc(size_c, DeviceBackend::Cuda).unwrap();

        Self { runtime, problem, a_buf: a, b_buf: b, c_buf: c }
    }
}

impl TunableKernel for CudaGemmAdapter {
    type Config = PipelineConfig;

    fn name(&self) -> &'static str {
        "cuda_gemm_mma"
    }

    fn search_space(&self) -> SearchSpace<Self::Config> {
        let mut candidates = Vec::new();
        // Tile Sizes
        for mt in [128, 256] {
            for nt in [128, 256] {
                for kt in [32, 64] {
                    for stages in [2] {
                        for num_warps in [9, 13] {
                            let mut cfg = PipelineConfig::new(stages, mt, nt, kt);
                            cfg.instruction = SpecializedInstruction::CudaMMA;
                            cfg.force_num_warps = Some(num_warps);
                            candidates.push(cfg);
                        }
                    }
                }
            }
        }
        SearchSpace::new(candidates)
    }

    fn is_feasible(&self, cfg: &Self::Config) -> bool {
        // MMA/PTX requirements
        if cfg.m_tile % 16 != 0 || cfg.n_tile % 8 != 0 || cfg.k_tile % 16 != 0 {
            return false;
        }

        // Shared Memory Limit check
        let smem_a_bytes = (cfg.m_tile * cfg.k_tile * 2) as usize;
        let smem_b_bytes = (cfg.k_tile * cfg.n_tile * 2) as usize;
        let required_smem = (smem_a_bytes + smem_b_bytes) * cfg.num_stages as usize + 512;
        
        let limit = self.runtime.get_max_shared_memory(DeviceBackend::Cuda);
        if required_smem > limit {
            return false;
        }

        true
    }

    fn benchmark(&self, cfg: &Self::Config) -> Option<f32> {
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm { 
                m: self.problem.m as u32,
                n: self.problem.n as u32,
                k: self.problem.k as u32 
            },
            precison: "f16".to_string(),
            tiling: cfg.clone(),
        };

        let emitter = UniversalEmitter::new(DeviceBackend::Cuda);
        let source = emitter.generate(ir);
        let kernel_name = "gemm_mma_kernel";

        let kernel_id = match self.runtime.compile(&source, kernel_name, DeviceBackend::Cuda) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("Compile Failed: {}", e);
                return None;
            }
        };

        // Grid/Block
        let num_warps = cfg.force_num_warps.unwrap_or(8);
        let block = (num_warps * 32, 1, 1); 
        
        let grid = (
            (self.problem.m as u32 + cfg.m_tile - 1) / cfg.m_tile,
            (self.problem.n as u32 + cfg.n_tile - 1) / cfg.n_tile,
            1
        );
        
        // Smem calculation: (mt*kt + kt*nt) * 2 bytes * stages + padding + barrier space
        let smem_a_bytes = cfg.m_tile * cfg.k_tile * 2;
        let smem_b_bytes = cfg.k_tile * cfg.n_tile * 2;
        let smem_bytes = (smem_a_bytes + smem_b_bytes) * cfg.num_stages + 512; // Extra for barriers/alignment

        let args = vec![
            KernelArg::Buffer(self.a_buf),
            KernelArg::Buffer(self.b_buf),
            KernelArg::Buffer(self.c_buf),
            KernelArg::Int(self.problem.m as i32),
            KernelArg::Int(self.problem.n as i32),
            KernelArg::Int(self.problem.k as i32),
        ];

        let start = std::time::Instant::now();
        if let Err(e) = self.runtime.launch(kernel_id, grid, block, smem_bytes, args) {
             eprintln!("Launch Failed: {}", e);
             return None;
        }
        self.runtime.synchronize();
        let nanos = start.elapsed().as_nanos() as f32;

        let gflops = (2.0 * self.problem.m as f32 * self.problem.n as f32 * self.problem.k as f32) / nanos;
        Some(gflops)
    }

    fn cache_key(&self) -> String {
        format!("cuda_gemm_{}", self.problem.signature())
    }
}
