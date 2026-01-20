use crate::PipelineConfig;
use crate::optimizer::GPUInfo;
use crate::optimizer::problem::{ProblemDescriptor, LayerType, HeroConfig, ArchHint, Fa2Variant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingPlan {
    Scout, // Wide exploration
    Sniper, // Focused exploration
    Lightweight, // Minimal exploration for small problems
    Balanced, // Standard
}

#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub tile_m: Vec<u32>,
    pub tile_n: Vec<u32>,
    pub tile_k: Vec<u32>,
    pub warps: Vec<u32>,
    pub use_tensor_core: bool,
    pub enable_swizzle: bool,
}

impl SearchSpace {
    pub fn new() -> Self {
        Self {
            tile_m: vec![],
            tile_n: vec![],
            tile_k: vec![],
            warps: vec![],
            use_tensor_core: true,
            enable_swizzle: false,
        }
    }

    pub fn tile_m(mut self, vals: &[u32]) -> Self { self.tile_m = vals.to_vec(); self }
    pub fn tile_n(mut self, vals: &[u32]) -> Self { self.tile_n = vals.to_vec(); self }
    pub fn tile_k(mut self, vals: &[u32]) -> Self { self.tile_k = vals.to_vec(); self }
    pub fn warps(mut self, vals: &[u32]) -> Self { self.warps = vals.to_vec(); self }
    pub fn use_tensor_core(mut self, val: bool) -> Self { self.use_tensor_core = val; self }
    pub fn enable_swizzle(mut self, val: bool) -> Self { self.enable_swizzle = val; self }
}

#[derive(Debug, Clone)]
pub struct TuningContext {
    pub trials_so_far: usize,
    pub best_score: f32,
    pub variance: f32, // GP variance or similar metric
    // Budget, history, etc. can be added later
}

pub trait TuningPolicy: Send + Sync {
    fn search_space(&self) -> SearchSpace;
    fn is_feasible(&self, config: &PipelineConfig, device: &GPUInfo) -> bool;
    fn hero_configs(&self) -> Vec<HeroConfig>;
    fn sampling_plan(&self, ctx: &TuningContext) -> SamplingPlan; // Renamed from sampling_strategy
    
    fn estimate_smem_usage(&self, config: &PipelineConfig) -> usize {
        // Base estimation, can be overridden. 
        // Simple double buffering assumption: 2 * (M*K + K*N) * sizeof(dtype)
        // Assuming FP16 (2 bytes)
        let tile_bytes = (config.m_tile * config.k_tile + config.k_tile * config.n_tile) * 2;
        (tile_bytes * config.num_stages as u32) as usize
    }
}

// --- Conv2d Policy ---
pub struct Conv2dPolicy {
    problem: ProblemDescriptor,
}

impl Conv2dPolicy {
    pub fn new(problem: &ProblemDescriptor) -> Self {
        Self { problem: problem.clone() }
    }
}

use crate::optimizer::problem::HeroScope;

impl TuningPolicy for Conv2dPolicy {
    fn search_space(&self) -> SearchSpace {
        let mut space = SearchSpace::new()
            .tile_k(&[16, 32])
            .warps(&[5, 9, 17])
            .use_tensor_core(true)
            .enable_swizzle(true);

        // Batch dependent M-tile selection
        // M in Conv2d usually maps to Batch * H * W or similar
        // For small batch, we might want smaller tiles if M is small
        if self.problem.batch <= 32 {
            space = space.tile_m(&[64]);
        } else {
            space = space.tile_m(&[64, 128]);
        }
        
        space = space.tile_n(&[32, 64, 128]); // N maps to Output Channels / K filters

        space
    }

    fn hero_configs(&self) -> Vec<HeroConfig> {
        use crate::core::config::{SwizzleMode, SpecializedInstruction};
        use crate::optimizer::problem::Layout;
        let mut configs = vec![];
        let layout = match self.problem.layer_type {
            LayerType::Conv2d(l) => l,
            _ => Layout::NHWC,
        };
        let is_small_batch = self.problem.batch <= 32;

        match layout {
            Layout::NHWC => {
                if is_small_batch {
                    // 🏆 The "God Config" (29.33 TFLOPS Winner)
                    let mut cfg = PipelineConfig::new(2, 64, 64, 32).with_warps(5);
                    cfg.instruction = SpecializedInstruction::CudaMMA;
                    cfg.swizzle_mode = SwizzleMode::Xor4;
                    configs.push(HeroConfig {
                        config: cfg,
                        note: "Ampere-3070 Conv2d-B32 God Config",
                        arch_hint: ArchHint::NvidiaAmpere,
                        scope: HeroScope::Exact,
                    });
                } else {
                    // 🏗️ B=64+ : "M=128 Standard" (25 TFLOPS)
                    let mut cfg = PipelineConfig::new(2, 128, 64, 32).with_warps(9);
                    cfg.instruction = SpecializedInstruction::CudaMMA;
                    cfg.swizzle_mode = SwizzleMode::Xor4;
                    configs.push(HeroConfig {
                        config: cfg,
                        note: "Ampere-3070 Conv2d-B64 Standard",
                        arch_hint: ArchHint::NvidiaAmpere,
                        scope: HeroScope::Exact,
                    });
                }
            }
            Layout::NCHW => {
                // 🐢 NCHW: Safe Driving
                let mut cfg = PipelineConfig::new(2, 64, 64, 16).with_warps(5);
                cfg.instruction = SpecializedInstruction::CudaMMA;
                cfg.swizzle_mode = SwizzleMode::Xor4;
                configs.push(HeroConfig {
                    config: cfg,
                    note: "Ampere-3070 Conv2d NCHW Baseline",
                    arch_hint: ArchHint::NvidiaAmpere,
                    scope: HeroScope::Layer,
                });
            }
            _ => {}
        }

        configs
    }

    fn is_feasible(&self, cfg: &PipelineConfig, dev: &GPUInfo) -> bool {
        let nw = cfg.force_num_warps.unwrap_or(4);
        
        // Alignment check: M_FRAGS must be > 0
        // M_FRAGS = MT / (N_WARPS - PRODUCER_WARPS) / 16
        // PRODUCER_WARPS is 1 if aligned, 0 otherwise
        let is_aligned = true; // Conv2d NHWC usually aligned for modern cases
        let producer_warps = if is_aligned { 1 } else { 0 };
        let consumers = nw - producer_warps;
        
        if consumers == 0 || (cfg.m_tile / consumers) < 16 {
            return false;
        }
        
        // N_FRAGS = NT / 16
        if cfg.n_tile < 16 {
            return false;
        }

        if self.estimate_smem_usage(cfg) > dev.shared_memory_per_block {
            return false;
        }

        true
    }

    fn sampling_plan(&self, ctx: &TuningContext) -> SamplingPlan {
         if self.problem.batch <= 32 {
            SamplingPlan::Scout
        } else {
            SamplingPlan::Sniper
        }
    }
}

// --- Gemm Policy ---
pub struct GemmPolicy {
    _problem: ProblemDescriptor,
}

impl GemmPolicy {
    pub fn new(problem: &ProblemDescriptor) -> Self {
        Self { _problem: problem.clone() }
    }
}

impl TuningPolicy for GemmPolicy {
    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .tile_m(&[64, 128])
            .tile_n(&[64, 128])
            .tile_k(&[32])
            .warps(&[4, 8, 16])
            .use_tensor_core(true)
            .enable_swizzle(false) // GEMM: Off by default
    }

    fn hero_configs(&self) -> Vec<HeroConfig> {
        use crate::core::config::{SpecializedInstruction};
        let mut configs = vec![];
        let output_size = self._problem.m * self._problem.n;

        if output_size < 1024 * 1024 {
            // 🐣 Small GEMM: Occupancy Focus
            let mut cfg = PipelineConfig::new(2, 64, 64, 32).with_warps(4);
            cfg.instruction = SpecializedInstruction::CudaMMA;
            configs.push(HeroConfig {
                config: cfg,
                note: "Small GEMM Hero (Occupancy Focus)",
                arch_hint: ArchHint::NvidiaAmpere,
                scope: HeroScope::Layer,
            });
        } else {
            // 🦖 Large GEMM: Throughput Focus (The 36 TFLOPS Legend)
            let mut cfg_legend = PipelineConfig::new(3, 128, 128, 32).with_warps(8);
            cfg_legend.instruction = SpecializedInstruction::CudaMMA;
            configs.push(HeroConfig {
                config: cfg_legend,
                note: "Large GEMM Champion (36 TFLOPS Legend)",
                arch_hint: ArchHint::NvidiaAmpere,
                scope: HeroScope::Exact,
            });

            // Backup Hero
            let mut cfg_backup = PipelineConfig::new(2, 128, 64, 32).with_warps(8);
            cfg_backup.instruction = SpecializedInstruction::CudaMMA;
            configs.push(HeroConfig {
                config: cfg_backup,
                note: "Large GEMM Backup (Smem Saver)",
                arch_hint: ArchHint::NvidiaAmpere,
                scope: HeroScope::Layer,
            });
        }
        configs
    }

    fn is_feasible(&self, cfg: &PipelineConfig, dev: &GPUInfo) -> bool {
         if self.estimate_smem_usage(cfg) > dev.shared_memory_per_block {
            return false;
        }
        if dev.max_registers_per_thread < 32 { // Dummy check
             return false;
        }
        true
    }

    fn sampling_plan(&self, ctx: &TuningContext) -> SamplingPlan {
        let size = self._problem.m * self._problem.n * self._problem.k;
        if size < 1_000_000 {
            SamplingPlan::Lightweight
        } else {
            SamplingPlan::Balanced
        }
    }
}


// --- FA2 Policy ---
pub struct Fa2Policy {
    _problem: ProblemDescriptor,
    _variant: Fa2Variant,
}

impl Fa2Policy {
    pub fn new(problem: &ProblemDescriptor, variant: Fa2Variant) -> Self {
        Self { _problem: problem.clone(), _variant: variant }
    }
    
    fn estimate_qk_smem(&self, cfg: &PipelineConfig) -> usize {
        // Tile Q * Tile K * 2 bytes
        // Approx: tile_m * tile_k * 2
        (cfg.m_tile * cfg.k_tile * 2) as usize
    }

    fn estimate_softmax_smem(&self, cfg: &PipelineConfig) -> usize {
        // Row max/sum buffer: tile_m * 4 bytes (float)
        (cfg.m_tile * 4) as usize
    }

     fn estimate_pv_smem(&self, cfg: &PipelineConfig) -> usize {
         // Tile P * Tile V
         // Approx: tile_m * tile_n * 2
         (cfg.m_tile * cfg.n_tile * 2) as usize
    }
}

impl TuningPolicy for Fa2Policy {
    fn search_space(&self) -> SearchSpace {
         SearchSpace::new()
            .tile_m(&[64, 128])   
            .tile_n(&[64])
            .tile_k(&[32])
            .warps(&[4, 8])
            .use_tensor_core(true)
            .enable_swizzle(true)
    }

    fn hero_configs(&self) -> Vec<HeroConfig> {
        use crate::core::config::{SpecializedInstruction, SwizzleMode};
        let s_len = self._problem.n; // KV Seq Len
        let mut heroes = Vec::new();

        // 🏆 Hero 1: The "Baseline Champion" (7.63 TFLOPS Proven)
        let mut cfg1 = PipelineConfig::new(2, 128, 64, 32).with_warps(9);
        cfg1.instruction = SpecializedInstruction::CudaMMA;
        cfg1.swizzle_mode = SwizzleMode::Xor4;
        heroes.push(HeroConfig {
            config: cfg1,
            note: "FA2 Baseline Champion (7.63 TFLOPS)",
            arch_hint: ArchHint::NvidiaAmpere,
            scope: HeroScope::Exact,
        });

        // 🛡️ Hero 2: The "Occupancy Saver" (For Large S)
        if s_len >= 2048 {
            let mut cfg2 = PipelineConfig::new(2, 64, 64, 32).with_warps(5);
            cfg2.instruction = SpecializedInstruction::CudaMMA;
            cfg2.swizzle_mode = SwizzleMode::Xor4;
            heroes.push(HeroConfig {
                config: cfg2,
                note: "FA2 Large S Hero (Occupancy Saver)",
                arch_hint: ArchHint::NvidiaAmpere,
                scope: HeroScope::Exact,
            });
        }

        heroes
    }

    fn is_feasible(&self, cfg: &PipelineConfig, dev: &GPUInfo) -> bool {
        let smem_qk = self.estimate_qk_smem(cfg);
        let smem_softmax = self.estimate_softmax_smem(cfg);
        let smem_pv = self.estimate_pv_smem(cfg);

        let total = smem_qk + smem_softmax + smem_pv;

        if total > dev.shared_memory_per_block {
            return false;
        }
        true
    }

    fn sampling_plan(&self, ctx: &TuningContext) -> SamplingPlan {
         let size = self._problem.m * self._problem.n * self._problem.k;
         if size < 2_000_000 {
            SamplingPlan::Lightweight
        } else {
            SamplingPlan::Sniper
        }
    }
}


pub struct PolicyFactory;

impl PolicyFactory {
    pub fn derive(problem: &ProblemDescriptor) -> Box<dyn TuningPolicy> {
        match problem.layer_type {
            LayerType::Gemm => Box::new(GemmPolicy::new(problem)),
            LayerType::Conv2d(_) => Box::new(Conv2dPolicy::new(problem)),
            LayerType::FlashAttention(v) => Box::new(Fa2Policy::new(problem, v)),
        }
    }
}
