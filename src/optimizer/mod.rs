use crate::PipelineConfig;

#[derive(Debug, Clone)]
pub struct GPUInfo {
    pub name: String,
    pub shared_memory_per_block: usize,
    pub max_registers_per_thread: u32,
    pub max_warps_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub shared_memory_per_sm: usize,
    pub has_tensor_cores: bool,
}

impl GPUInfo {
    pub fn rtx3070() -> Self {
        Self {
            name: "NVIDIA GeForce RTX 3070".to_string(),
            shared_memory_per_block: 48 * 1024, // Standard limit, can be increased to 100KB
            max_registers_per_thread: 255,
            max_warps_per_sm: 48,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 100 * 1024,
            has_tensor_cores: true,
        }
    }

    pub fn a100() -> Self {
        Self {
            name: "NVIDIA A100".to_string(),
            shared_memory_per_block: 164 * 1024,
            max_registers_per_thread: 255,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            shared_memory_per_sm: 164 * 1024,
            has_tensor_cores: true,
        }
    }
}

pub enum PruneReason {
    SharedMemoryOverflow { required: usize, available: usize },
    RegisterPressure { required: u32, available: u32 },
    InvalidAlignment,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationGoal {
    MaximizeTFLOPS,
    MinimizeLatency,
    Balanced { tflops_weight: f32 },
}

pub mod benchmark;
pub mod cache;

use benchmark::{MicroBenchmark, Observation, BenchmarkResult};
use cache::{TuningCache, CacheKey};

#[derive(Debug, Clone)]
pub struct GaussianProcess {
    pub observations: Vec<Observation>,
    pub length_scale: f32, // RBF kernel length scale
    pub noise_sigma: f32,
}

impl GaussianProcess {
    pub fn new() -> Self {
        Self { 
            observations: Vec::new(),
            length_scale: 1.0,
            noise_sigma: 0.1,
        }
    }

    pub fn observe(&mut self, obs: Observation) {
        self.observations.push(obs);
    }

    /// Simplified RBF kernel based prediction
    pub fn predict(&self, vec: &[f32]) -> (f32, f32) {
        if self.observations.is_empty() {
            return (0.0, 1.0); // Prior: Mean 0, Var 1
        }

        let mut mean = 0.0;
        let mut total_weight = 0.0;
        
        for obs in &self.observations {
            let obs_vec = obs.config.to_vector();
            let dist_sq: f32 = vec.iter().zip(obs_vec.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum();
            
            let weight = ( -dist_sq / (2.0 * self.length_scale.powi(2)) ).exp();
            mean += weight * obs.score;
            total_weight += weight;
        }

        if total_weight > 1e-6 {
            mean /= total_weight;
        }

        // Variance decreases as we get closer to observed points
        let min_dist_sq = self.observations.iter()
            .map(|obs| {
                let obs_vec = obs.config.to_vector();
                vec.iter().zip(obs_vec.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
            })
            .fold(f32::INFINITY, f32::min);
        
        let variance = 1.0 - ( -min_dist_sq / (2.0 * self.length_scale.powi(2)) ).exp();
        
        (mean, variance.max(self.noise_sigma))
    }

    /// Expected Improvement (EI) Acquisition Function
    pub fn expected_improvement(&self, vec: &[f32], current_best_y: f32) -> f32 {
        let (mu, sigma) = self.predict(vec);
        if sigma < 1e-6 { return 0.0; }

        let z = (mu - current_best_y) / sigma;
        
        // Simplified EI: Using a normal distribution approximation
        // In a real impl, we'd use PDF/CDF of Normal dist.
        // EI = (mu - best)*CDF(z) + sigma*PDF(z)
        let phi_z = ( -0.5 * z.powi(2) ).exp() / (2.0 * std::f32::consts::PI).sqrt();
        let tau_z = if z > 0.0 { 1.0 } else { 0.0 }; // Very crude CDF jump

        (mu - current_best_y) * tau_z + sigma * phi_z
    }
}

#[derive(Debug, Clone)]
pub struct AutoTuner {
    pub gpu: GPUInfo,
    pub gp: GaussianProcess,
    pub best_config: Option<PipelineConfig>,
}

impl AutoTuner {
    pub fn new(gpu: GPUInfo) -> Self {
        Self {
            gpu,
            gp: GaussianProcess::new(),
            best_config: None,
        }
    }

    pub fn optimize<B: MicroBenchmark>(&mut self, benchmark: &B, iterations: usize, goal: OptimizationGoal) -> PipelineConfig {
        // 0. Detect Environment
        let env = benchmark.device_info();

        // 1. Check Cache
        let key = CacheKey {
            gpu: self.gpu.name.clone(),
            m: benchmark.m(),
            n: benchmark.n(),
            k: benchmark.k(),
            dtype: "f16".to_string(), // TODO: Detect dtype
            cuda_version: env.cuda_version,
            driver_version: env.driver_version,
            sm_arch: env.sm_arch,
        };
        let mut cache = TuningCache::new();
        if let Some(config) = cache.get(&key) {
            println!("[Tracea] 💎 Cache Hit! Using optimized config for {}x{}x{}", key.m, key.n, key.k);
            return config;
        }

        // 1. Initial random feasible config
        let mut initial_config = PipelineConfig::new(2, 128, 128, 32);
        initial_config.use_tensor_cores = true;
        let res = benchmark.measure(&initial_config);
        let score = self.calculate_score(&res, goal);
        
        self.gp.observe(Observation { config: initial_config.clone(), score });

        let mut current_best_score = score;
        if score > 0.0 {
            self.best_config = Some(initial_config.clone());
        }
        for _ in 0..iterations {
            // 2. Propose next candidate by maximizing EI
            let candidate = self.propose_candidate(current_best_score);

            // 3. Measure performance
            let res = benchmark.measure(&candidate);
            let score = self.calculate_score(&res, goal);

            println!("[Tracea] Testing {:>3}x{:>3}x{:>2} ({} stages) -> {:.2} TFLOPS, {:.3} ms (Score: {:.2})", 
                candidate.m_tile, candidate.n_tile, candidate.k_tile, candidate.num_stages, 
                res.tflops, res.latency_ms, score);

            // 4. Update model
            let obs = Observation { config: candidate, score };
            if obs.score > current_best_score {
                current_best_score = obs.score;
                self.best_config = Some(obs.config.clone());
            }
            self.gp.observe(obs);
        }

           // 5. Finalize
        let best_config = if let Some(best) = self.best_config.clone() {
            best
        } else {
            println!("[Tracea] ⚠️  Warning: No optimal config found. Using Safe Fallback.");
            let fallback = PipelineConfig {
                num_stages: 2,
                m_tile: 128,
                n_tile: 128,
                k_tile: 32,
                use_tensor_cores: true,
                epilogue: Vec::new(),
            };
            cache.set(key, fallback.clone());
            fallback
        };
        
        println!("[Tracea] 💾 Tuning complete. Optimal config saved for {}x{}x{}", benchmark.m(), benchmark.n(), benchmark.k());
        best_config
    }
    fn calculate_score(&self, res: &BenchmarkResult, goal: OptimizationGoal) -> f32 {
        match goal {
            OptimizationGoal::MaximizeTFLOPS => res.tflops,
            OptimizationGoal::MinimizeLatency => 1000.0 / res.latency_ms.max(1e-6), // Score: Inv Latency
            OptimizationGoal::Balanced { tflops_weight } => {
                let latency_score = 1000.0 / res.latency_ms.max(1e-6);
                res.tflops * tflops_weight + latency_score * (1.0 - tflops_weight)
            }
        }
    }

    fn propose_candidate(&self, current_best_y: f32) -> PipelineConfig {
        // Search for config that maximizes GP Expected Improvement
        // For simplicity, we sample a few feasible configs and pick the best EI
        let mut best_ei = -1.0;
        let mut best_config = PipelineConfig::new(2, 64, 64, 8);

        // Dynamic Search Space Generation
        let mut search_space = Vec::new();
        let tile_sizes = [64, 128]; // 32 might be too small for effective TC mapping
        let k_sizes = [16, 32, 64];
        let stage_counts = [2, 3, 4];
        
        for &mt in &tile_sizes {
            for &nt in &tile_sizes {
                for &kt in &k_sizes {
                    for &stages in &stage_counts {
                        let mut cfg = PipelineConfig::new(stages, mt, nt, kt);
                        cfg.use_tensor_cores = true;
                        search_space.push(cfg);
                    }
                }
            }
        }

        for config in search_space {
            if self.is_feasible(&config).is_ok() {
                let ei = self.gp.expected_improvement(&config.to_vector(), current_best_y);
                if ei > best_ei {
                    best_ei = ei;
                    best_config = config;
                }
            }
        }
        best_config
    }

    pub fn is_feasible(&self, config: &PipelineConfig) -> Result<(), PruneReason> {
        // 1. Shared Memory Check
        let (element_size, s_a, s_b) = if config.use_tensor_cores {
            (2, config.k_tile, config.n_tile) // FP16, no skew for now
        } else {
            (4, config.k_tile + 4, config.n_tile + 4) // FP32, skewed
        };

        let smem_a = config.m_tile * s_a * element_size;
        let smem_b = s_b * config.k_tile * element_size;
        let total_smem = (smem_a + smem_b) * config.num_stages;

        if total_smem as usize > self.gpu.shared_memory_per_block {
            return Err(PruneReason::SharedMemoryOverflow {
                required: total_smem as usize,
                available: self.gpu.shared_memory_per_block,
            });
        }

        // 2. Register Pressure Estimation
        let regs_per_thread = if config.use_tensor_cores {
            // Mapping for m16n8k16: 
            // 256 threads / 4 warps = 128 threads? No, 2 warps? 
            // Let's assume 128 threads (4 warps) for 128x128 tile.
            // Each warp handles 64x64.
            // 4 tiles in M, 8 tiles in N = 32 MMA fragments.
            // 32 * 4 (acc) + (4 * 4 A-frag + 8 * 2 B-frag) = 128 + 16 + 16 = 160 regs.
            let warp_m = 64;
            let warp_n = 64;
            let tiles_m = warp_m / 16;
            let tiles_n = warp_n / 8;
            let acc_regs = tiles_m * tiles_n * 4;
            let frag_regs = (tiles_m * 4) + (tiles_n * 2);
            acc_regs + frag_regs + 32 // + overhead
        } else {
            (config.m_tile / 16) * (config.n_tile / 16) * 8 + 16
        };

        if regs_per_thread > self.gpu.max_registers_per_thread {
            return Err(PruneReason::RegisterPressure {
                required: regs_per_thread,
                available: self.gpu.max_registers_per_thread,
            });
        }

        // 3. Alignment Check (Ampere 128-byte)
        if total_smem % 128 != 0 {
             return Err(PruneReason::InvalidAlignment);
        }

        Ok(())
    }
}
