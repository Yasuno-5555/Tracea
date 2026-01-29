use crate::core::tuning::{TunableKernel, SearchSpace};
use crate::core::config::{PipelineConfig, SwizzleMode, SpecializedInstruction};
use crate::emitter::rocm::ROCMEmitter;
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RocmFa2Problem {
    pub b: usize,
    pub s: usize,
    pub h: usize,
    pub d: usize,
    pub is_causal: bool,
}

pub struct RocmFa2Adapter {
    pub runtime: Arc<RuntimeManager>,
    pub problem: RocmFa2Problem,
    // Buffers
    q_buf: crate::runtime::manager::BufferId,
    k_buf: crate::runtime::manager::BufferId,
    v_buf: crate::runtime::manager::BufferId,
    o_buf: crate::runtime::manager::BufferId,
}

impl RocmFa2Adapter {
    pub fn new(runtime: Arc<RuntimeManager>, problem: RocmFa2Problem) -> Self {
        let bytes = problem.b * problem.h * problem.s * problem.d * 2;
        let q = runtime.alloc(bytes, DeviceBackend::Rocm).unwrap();
        let k = runtime.alloc(bytes, DeviceBackend::Rocm).unwrap();
        let v = runtime.alloc(bytes, DeviceBackend::Rocm).unwrap();
        let o = runtime.alloc(bytes, DeviceBackend::Rocm).unwrap();

        Self { runtime, problem, q_buf: q, k_buf: k, v_buf: v, o_buf: o }
    }
}

impl TunableKernel for RocmFa2Adapter {
    type Config = PipelineConfig;

    fn name(&self) -> &'static str {
        "rocm_fa2_kernel"
    }

    fn search_space(&self) -> SearchSpace<Self::Config> {
        let mut candidates = Vec::new();
        // ROCm/AMD often prefers slightly different tiles due to Wave64
        let tile_sizes = [
            (64, 64, 32),
            (128, 64, 32),
            (64, 128, 32),
        ];

        for &(m, n, k) in &tile_sizes {
            for &stages in &[1, 2] { // ROCm often doesn't need many stages due to large LDS
                let mut config = PipelineConfig::new(stages, m, n, k);
                config.force_num_warps = Some(2); // 2 waves for 128 threads
                config.swizzle_mode = SwizzleMode::Xor2;
                candidates.push(config);
            }
        }
        SearchSpace::new(candidates)
    }

    fn is_feasible(&self, cfg: &Self::Config) -> bool {
        // LDS check
        let lds_usage = (cfg.m_tile * cfg.k_tile + cfg.k_tile * cfg.n_tile) * 2 * cfg.num_stages;
        lds_usage <= 65536
    }

    fn benchmark(&self, cfg: &Self::Config) -> Option<f32> {
        // ROCm JIT / benchmark call stub
        // Implementation will call hiprtc / rocm runtime
        Some(1.0) 
    }

    fn cache_key(&self) -> String {
        format!("rocm_fa2_{}_{}", self.problem.d, if self.problem.is_causal { "c" } else { "d" })
    }
}
