use crate::core::tuning::{TunableKernel, SearchSpace, ParameterRange};
use crate::core::config::{PipelineConfig, SwizzleMode, SpecializedInstruction, QuantizationMode};
use super::cuda_emitter::FlashAttentionEmitter;
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fa2Problem {
    pub b: usize,
    pub s: usize,
    pub h: usize,
    pub d: usize,
    pub is_causal: bool,
}

impl Fa2Problem {
    pub fn signature(&self) -> String {
        format!("b{}_s{}_h{}_d{}_{}", self.b, self.s, self.h, self.d, if self.is_causal { "causal" } else { "dense" })
    }
}

pub struct Fa2Adapter {
    pub runtime: Arc<RuntimeManager>,
    pub problem: Fa2Problem,
    // Pre-allocated buffers for benchmarking
    q_buf: crate::runtime::manager::BufferId,
    k_buf: crate::runtime::manager::BufferId,
    v_buf: crate::runtime::manager::BufferId,
    o_buf: crate::runtime::manager::BufferId,
}

impl Fa2Adapter {
    pub fn new(runtime: Arc<RuntimeManager>, problem: Fa2Problem) -> Self {
        // Allocate buffers once: B * H * S * D * sizeof(half)
        let total_elems = problem.b * problem.h * problem.s * problem.d;
        let bytes = total_elems * 2; // f16
        
        let q = runtime.alloc(bytes, DeviceBackend::Cuda).unwrap();
        let k = runtime.alloc(bytes, DeviceBackend::Cuda).unwrap();
        let v = runtime.alloc(bytes, DeviceBackend::Cuda).unwrap();
        let o = runtime.alloc(bytes, DeviceBackend::Cuda).unwrap();

        Self { runtime, problem, q_buf: q, k_buf: k, v_buf: v, o_buf: o }
    }
}

impl TunableKernel for Fa2Adapter {
    type Config = PipelineConfig;

    fn name(&self) -> &'static str {
        "fa2_kernel"
    }

    fn search_space(&self) -> SearchSpace<Self::Config> {
        // Define a grid of candidates
        let mut candidates = Vec::new();
        
        let tile_sizes = [
            (128, 64, 32),
            (64, 128, 32),
            (64, 64, 32),
            (128, 128, 32),
            (32, 64, 32),
            (64, 32, 32),
            (256, 64, 32),
            (64, 256, 32),
        ];

        let num_stages_list = [2, 3]; // 4 stages might hit smem limits for 128x128
        let num_warps_list = [4, 8, 9, 12];

        for &(m, n, k) in &tile_sizes {
            for &stages in &num_stages_list {
                for &warps in &num_warps_list {
                    let mut config = PipelineConfig::new(stages, m, n, k);
                    // Match the emitter's rule: 1 Producer + MT/16 Consumers
                    // But allow tuner to explore variations if forced.
                    config.force_num_warps = Some(warps);
                    config.instruction = SpecializedInstruction::CudaMMA;
                    config.swizzle_mode = SwizzleMode::Xor2;
                    candidates.push(config);
                }
            }
        }

        SearchSpace::new(candidates)
    }

    fn is_feasible(&self, cfg: &Self::Config) -> bool {
        // 1. Tiling constraints
        if cfg.m_tile % 16 != 0 || cfg.n_tile % 16 != 0 {
            eprintln!("[Fa2Adapter] Infeasible: mt {} or nt {} not % 16", cfg.m_tile, cfg.n_tile);
            return false;
        }
        
        // 2. Shared Memory Limits (Dynamic Check)
        let (required_smem, _, _, _, _, _) = FlashAttentionEmitter::calculate_smem_layout(cfg, self.problem.d);
        
        let limit = self.runtime.get_max_shared_memory(DeviceBackend::Cuda);
        
        if required_smem > limit {
            eprintln!("[Fa2Adapter] Infeasible: smem {} > limit {}", required_smem, limit);
            return false;
        }

        // 3. Thread Limits
        let num_warps = cfg.force_num_warps.unwrap_or(4);
        let threads = num_warps * 32;
        let thread_limit = self.runtime.get_max_threads_per_block(DeviceBackend::Cuda);
        if threads > thread_limit as u32 {
            eprintln!("[Fa2Adapter] Infeasible: threads {} > limit {}", threads, thread_limit);
            return false;
        }

        true
    }

    fn benchmark(&self, cfg: &Self::Config) -> Option<f32> {
        // 1. Generate Kernel Source
        let emitter = FlashAttentionEmitter::new(cfg.clone());
        let source = emitter.generate_kernel(self.problem.h, self.problem.d, self.problem.is_causal);
        let kernel_name = "flash_attention_v2_kernel";

        // 2. Compile
        let kernel_id = match self.runtime.compile(&source, kernel_name, DeviceBackend::Cuda) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("Benchmark Compilation Failed: {}", e);
                return None;
            }
        };

        // 3. Setup Launch Params
        let mt = cfg.m_tile;
        let _nt = cfg.n_tile;
        let _dh = self.problem.d as u32;
        let num_warps = cfg.force_num_warps.unwrap_or(1 + (mt / 16));
        
        let block = (num_warps * 32, 1, 1);
        let grid = ( (self.problem.s as u32 + mt - 1) / mt, self.problem.h as u32, self.problem.b as u32 );

        // Shared Memory Calculation (Matches py_bindings logic)
        // Smem = (Warps * 1536) + (Warps * 64 * D) + (8 * N * (D+8))
        // This is a heuristic approximation used in bindings, check emitter for exact?
        // Let's use the explicit logic from emitter if possible, or this safe upper bound.
        // Emitter: ~32KB (S/O) + ~18KB (K) + ~18KB (V) for 128x64.
        // Let's use a robust calculation:
        // Centralized Shared Memory Calculation
        let (smem_bytes, _, _, _, _, _) = FlashAttentionEmitter::calculate_smem_layout(cfg, self.problem.d);

        // 4. Launch & Measure
        // Warn: This includes overhead. For pure kernel time we need events, but Instant is okay for coarse tuning.
        let start = std::time::Instant::now();
        
        let args = vec![
            KernelArg::Buffer(self.q_buf),
            KernelArg::Buffer(self.k_buf),
            KernelArg::Buffer(self.v_buf),
            KernelArg::Buffer(self.o_buf),
            KernelArg::Usize(self.problem.b),
            KernelArg::Usize(self.problem.h),
            KernelArg::Usize(self.problem.s),
            KernelArg::Usize(self.problem.d),
            KernelArg::Float(1.0), // scale
        ];

        if let Err(e) = self.runtime.launch(kernel_id, grid, block, smem_bytes as u32, args) {
             eprintln!("Benchmark Launch Failed: {}", e);
             return None;
        }
        self.runtime.synchronize(); // Wait for completion
        
        let elapsed = start.elapsed();
        let nanos = elapsed.as_nanos() as f32;
        
        // Calculate TFLOPS
        // Flops = 4 * B * H * S * S * D
        let flops = 4.0 * (self.problem.b as f32) * (self.problem.h as f32) * (self.problem.s as f32).powi(2) * (self.problem.d as f32);
        let tflops = (flops / 1e12) / (nanos / 1e9);

        Some(tflops)
    }

    fn cache_key(&self) -> String {
        // In theory include Driver version etc.
        format!("{}:{}", self.name(), self.problem.signature())
    }
}
