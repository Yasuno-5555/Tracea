use crate::core::tuning::{TunableKernel, SearchSpace};
use crate::core::config::{PipelineConfig, SpecializedInstruction};
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetalGemmProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl MetalGemmProblem {
    pub fn signature(&self) -> String {
        format!("m{}_n{}_k{}", self.m, self.n, self.k)
    }
}

pub struct MetalGemmAdapter {
    pub runtime: Arc<RuntimeManager>,
    pub problem: MetalGemmProblem,
    // Buffers
    a_buf: crate::runtime::manager::BufferId,
    b_buf: crate::runtime::manager::BufferId,
    c_buf: crate::runtime::manager::BufferId,
}

impl MetalGemmAdapter {
    pub fn new(runtime: Arc<RuntimeManager>, problem: MetalGemmProblem) -> Self {
        let size_a = problem.m * problem.k * 2; // f16
        let size_b = problem.k * problem.n * 2; // f16
        let size_c = problem.m * problem.n * 4; // f32

        // Allocate on Metal device
        let a = runtime.alloc(size_a, DeviceBackend::Metal).unwrap_or(crate::runtime::manager::BufferId(0));
        let b = runtime.alloc(size_b, DeviceBackend::Metal).unwrap_or(crate::runtime::manager::BufferId(0));
        let c = runtime.alloc(size_c, DeviceBackend::Metal).unwrap_or(crate::runtime::manager::BufferId(0));

        Self { runtime, problem, a_buf: a, b_buf: b, c_buf: c }
    }
}

impl TunableKernel for MetalGemmAdapter {
    type Config = PipelineConfig;

    fn name(&self) -> &'static str {
        "metal_gemm_simd"
    }

    fn search_space(&self) -> SearchSpace<Self::Config> {
        let mut candidates = Vec::new();
        // Apple SIMD groups are 32 wide. 
        // 8x8 matrix Ops are standard.
        // Tiles usually multiple of 32.
        let patterns = [
            (64, 64, 32),
            (32, 32, 32),
        ];

        for &(m, n, k) in &patterns {
            for &stages in &[1, 2] { // Metal maybe less benefit from stages if TG mem is fast?
                let mut cfg = PipelineConfig::new(stages, m, n, k as u32);
                cfg.instruction = SpecializedInstruction::MetalSimdGroup;
                candidates.push(cfg);
            }
        }
        SearchSpace::new(candidates)
    }

    fn is_feasible(&self, cfg: &Self::Config) -> bool {
        // Alignment checks
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

        let emitter = UniversalEmitter::new(DeviceBackend::Metal);
        let source = emitter.generate(ir);
        let kernel_name = "gemm_metal_kernel";

        let kernel_id = match self.runtime.compile(&source, kernel_name, DeviceBackend::Metal) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("Metal Compile Failed: {}", e);
                return None;
            }
        };

        // Grid/Block for Metal
        // Emitter uses [[thread_position_in_grid]] etc.
        // Launch logic needs to map grid/block to Metal threadgroups.
        // RuntimeManager dispatch for Metal? 
        // Need to check what argument structure RuntimeManager expects for Metal launch.
        // It likely passes grid/block as-is to [computEncoder dispatchThreadgroups:grid threadsPerThreadgroup:block]
        
        // Emitter assumes:
        // mt per threadgroup Y (bid.y)
        // nt per threadgroup X (bid.x)
        // threadgroup size?
        // Emitter: "uint t_idx = tid.y * 32 + tid.x;" -> Standard flattened ID.
        // "uint simd_id [[simdgroup_index_in_threadgroup]]"
        // "Simdgroup distribution: Assume 4 simdgroups (128 threads)"
        // So we must launch 128 threads per threadgroup.
        
        let block_threads = 128;
        // Metal uses (width, height, depth) for threads per threadgroup.
        // Let's use (32, 4, 1) -> 128 threads.
        let block = (32, 4, 1);
        
        let grid = (
            (self.problem.n as u32 + cfg.n_tile - 1) / cfg.n_tile,
            (self.problem.m as u32 + cfg.m_tile - 1) / cfg.m_tile,
            1
        );

        // TG Mem: (mt*kt + kt*nt) * sizeof(half)
        let tg_mem_bytes = (cfg.m_tile * cfg.k_tile + cfg.k_tile * cfg.n_tile) * 2;

        let args = vec![
            KernelArg::Buffer(self.a_buf),
            KernelArg::Buffer(self.b_buf),
            KernelArg::Buffer(self.c_buf),
            KernelArg::Int(self.problem.m as i32), // Passed as buffer ref in Metal? 
            // MetalEmitter expects [[buffer(3)]] etc for scalars?
            // "constant uint& M [[buffer(3)]]" -> This implies valid buffer binding OR bytes set directly.
            // RuntimeManager::launch for Metal needs to handle Scalar->setBytes.
            KernelArg::Int(self.problem.n as i32),
            KernelArg::Int(self.problem.k as i32),
        ];

        let start = std::time::Instant::now();
        if let Err(_) = self.runtime.launch(kernel_id, grid, block, tg_mem_bytes, args) {
             return None;
        }
        // self.runtime.synchronize(); // Metal sync needed
        
        let nanos = start.elapsed().as_nanos() as f32;
        let gflops = (2.0 * self.problem.m as f32 * self.problem.n as f32 * self.problem.k as f32) / nanos;
        Some(gflops)
    }

    fn cache_key(&self) -> String {
        format!("metal_gemm_{}", self.problem.signature())
    }
}
