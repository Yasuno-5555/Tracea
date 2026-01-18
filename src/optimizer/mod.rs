use crate::PipelineConfig;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

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
            shared_memory_per_block: 99 * 1024, // Real limit with dynamic SMEM config
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

    fn shape_features(&self, m: u32, n: u32, k: u32) -> Vec<f32> {
        vec![
            (m as f32).log2(),
            (n as f32).log2(),
            (k as f32).log2(),
            (m as f32) / (n as f32).max(1.0),
            (n as f32) / (k as f32).max(1.0),
        ]
    }

    fn config_features(&self, config: &PipelineConfig, m: u32, n: u32, k: u32) -> Vec<f32> {
        let mut v = config.to_vector();
        // Add relative ratios: tile size compared to total shape
        v.push(config.m_tile as f32 / m as f32);
        v.push(config.n_tile as f32 / n as f32);
        v.push(config.k_tile as f32 / k as f32);
        v
    }

    fn roofline_prior(&self, m: u32, n: u32, k: u32, config: &PipelineConfig) -> f32 {
        // RTX 3070 Constants
        let peak_tflops = if config.use_tensor_cores { 160.0 } else { 20.0 };
        let mem_bw = 448.0; // GB/s
        
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let bytes = (m as f64 * k as f64 + k as f64 * n as f64 + m as f64 * n as f64) * 2.0;
        let intensity = (ops / bytes) as f32; // FLOPs/Byte
        
        let bw_limit = (mem_bw * intensity) / 1000.0; // TFLOPS
        
        f32::min(peak_tflops, bw_limit)
    }

    /// Shape-Aware Product Kernel Prediction
    pub fn predict(&self, m: u32, n: u32, k: u32, config: &PipelineConfig) -> (f32, f32) {
        let prior_mean = self.roofline_prior(m, n, k, config);
        
        if self.observations.is_empty() {
            return (prior_mean, 1.0); 
        }

        let mut mean_diff = 0.0;
        let mut total_weight = 0.0;
        
        let cur_shape = self.shape_features(m, n, k);
        let cur_config = self.config_features(config, m, n, k);
        
        for obs in &self.observations {
            let obs_shape = self.shape_features(obs.m, obs.n, obs.k);
            let obs_config = self.config_features(&obs.config, obs.m, obs.n, obs.k);
            
            // 1. K_shape
            let d_shape: f32 = cur_shape.iter().zip(obs_shape.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum();
            let k_shape = ( -d_shape / 2.0 ).exp();
            
            // 2. K_config
            let d_config: f32 = cur_config.iter().zip(obs_config.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum();
            let k_config = ( -d_config / 2.0 ).exp();
            
            // Product Kernel: K = K_shape * K_config
            let weight = k_shape * k_config;
            
            // Learn the residual from the prior
            let obs_prior = self.roofline_prior(obs.m, obs.n, obs.k, &obs.config);
            mean_diff += weight * (obs.score - obs_prior);
            total_weight += weight;
        }

        let final_mean = if total_weight > 1e-6 {
            prior_mean + (mean_diff / total_weight)
        } else {
            prior_mean
        };

        // Variance decreases as we get closer to observed points
        let min_dist_prod = self.observations.iter()
            .map(|obs| {
                 let obs_shape = self.shape_features(obs.m, obs.n, obs.k);
                 let obs_config = self.config_features(&obs.config, obs.m, obs.n, obs.k);
                 let ds = cur_shape.iter().zip(obs_shape.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>();
                 let dc = cur_config.iter().zip(obs_config.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>();
                 ( -ds/2.0 ).exp() * ( -dc/2.0 ).exp()
            })
            .fold(0.0, f32::max);
        
        (final_mean, (1.0 - min_dist_prod).max(self.noise_sigma))
    }

    /// Expected Improvement (EI) Acquisition Function
    pub fn expected_improvement(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, current_best_y: f32) -> f32 {
        let (mu, sigma) = self.predict(m, n, k, config);
        if sigma < 1e-6 { return 0.0; }

        let z = (mu - current_best_y) / sigma;
        
        let phi_z = ( -0.5 * z.powi(2) ).exp() / (2.0 * std::f32::consts::PI).sqrt();
        let tau_z = 0.5 * (1.0 + (z / 2.0f32.sqrt()).erf()); // Better CDF approx if available, or just keeping it simple

        (mu - current_best_y) * tau_z + sigma * phi_z
    }
}

// Minimal erf implementation for CDF
trait Erf { fn erf(self) -> f32; }
impl Erf for f32 {
    fn erf(self) -> f32 {
        let x = self;
        let a1 =  0.254829592;
        let a2 = -0.284496736;
        let a3 =  1.421413741;
        let a4 = -1.453152027;
        let a5 =  1.061405429;
        let p  =  0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
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

    pub fn optimize<B: MicroBenchmark>(&mut self, benchmark: &B, iterations: usize, goal: OptimizationGoal, epilogue: Vec<crate::core::op::EpilogueOp>) -> PipelineConfig {
        eprintln!("[SR] >>> optimize entry with epilogue count: {} >>>", epilogue.len());
        // 0. Detect Environment
        let env = benchmark.device_info();

        // 1. Check Cache
        let key = CacheKey {
            gpu: self.gpu.name.clone(),
            m: benchmark.m(),
            n: benchmark.n(),
            k: benchmark.k(),
            dtype: "f16".to_string(), // TODO: Detect dtype
            epilogue: epilogue.clone(),
            cuda_version: env.cuda_version,
            driver_version: env.driver_version,
            sm_arch: env.sm_arch,
        };
        let mut cache = TuningCache::new();
        if let Some(config) = cache.get(&key) {
            println!("[Tracea] 💎 Cache Hit! Using optimized config for {}x{}x{} (Epilogue: {:?})", key.m, key.n, key.k, epilogue);
            return config;
        }

        // 1. Initial random feasible config
        let m = benchmark.m();
        let n = benchmark.n();
        let k = benchmark.k();

        eprintln!("[Tracea] 🔍 Starting Tuning for {}x{}x{}", m, n, k);

        let mut initial_config = PipelineConfig::new(2, 128, 128, 32);
        initial_config.use_tensor_cores = true;
        initial_config.epilogue = epilogue.clone();

        eprintln!("[Tracea] 🧪 Measuring Initial Config...");
        let res = benchmark.measure(&initial_config);
        let score = self.calculate_score(&res, goal);
        eprintln!("[Tracea] 📈 Initial Score: {:.2}", score);
        
        self.gp.observe(Observation { m, n, k, config: initial_config.clone(), score });

        let mut current_best_score = score;
        self.best_config = Some(initial_config.clone());

        for i in 0..iterations {
            eprintln!("[Tracea] 🔄 Tuning Iteration {}/{}...", i + 1, iterations);
            // 2. Propose next candidate by maximizing EI with shape context
            let mut candidate = self.propose_candidate(m, n, k, current_best_score);
            candidate.epilogue = epilogue.clone();

            // 3. Measure performance
            eprintln!("[Tracea] 🧪 Measuring Candidate: {}x{}x{}", candidate.m_tile, candidate.n_tile, candidate.k_tile);
            let res = benchmark.measure(&candidate);
            let score = self.calculate_score(&res, goal);

            println!("[Tracea] Testing {:>3}x{:>3}x{:>2} ({} stages) -> {:.2} TFLOPS, {:.3} ms (Score: {:.2})", 
                candidate.m_tile, candidate.n_tile, candidate.k_tile, candidate.num_stages, 
                res.tflops, res.latency_ms, score);

            // 4. Update model
            let obs = Observation { m, n, k, config: candidate.clone(), score };
            if score > current_best_score {
                current_best_score = score;
                self.best_config = Some(candidate);
            }
            self.gp.observe(obs);
        }
        
        // 5. Finalize
        let best_config = self.best_config.clone().unwrap_or_else(|| {
            println!("[Tracea] ⚠️  Warning: No optimal config found. Using Safe Fallback.");
            PipelineConfig {
                num_stages: 2,
                m_tile: 128,
                n_tile: 128,
                k_tile: 32,
                use_tensor_cores: true,
                swizzle_mode: crate::core::config::SwizzleMode::None,
                quantization: crate::core::config::QuantizationMode::None,
                epilogue: epilogue.clone(),
            }
        });
        
        cache.set(key, best_config.clone());
        println!("[Tracea] 💾 Tuning complete. Optimal config saved for {}x{}x{}", m, n, k);
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

    fn propose_candidate(&self, m: u32, n: u32, k: u32, current_best_y: f32) -> PipelineConfig {
        // Search for config that maximizes GP Expected Improvement
        let mut best_ei = -1.0;
        let mut best_config = PipelineConfig::new(2, 128, 128, 32);

        // Dynamic Search Space Generation
        let mut search_space = Vec::new();
        let tile_sizes = [64, 128, 256]; 
        let k_sizes = [16, 32, 64];
        let stage_counts = [2, 3, 4, 5];
        
        let swizzles = [crate::core::config::SwizzleMode::None, crate::core::config::SwizzleMode::Xor2, crate::core::config::SwizzleMode::Xor4, crate::core::config::SwizzleMode::Xor8];
        
        for &mt in &tile_sizes {
            for &nt in &tile_sizes {
                for &kt in &k_sizes {
                    for &stages in &stage_counts {
                        for &swizzle in &swizzles {
                            let mut cfg = PipelineConfig::new(stages, mt, nt, kt);
                            cfg.use_tensor_cores = true;
                            cfg.swizzle_mode = swizzle;
                            search_space.push(cfg);
                        }
                    }
                }
            }
        }

        for config in search_space {
            if self.is_feasible(&config).is_ok() {
                let ei = self.gp.expected_improvement(m, n, k, &config, current_best_y);
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

    pub fn get_env_info(device: &Arc<CudaDevice>) -> crate::optimizer::benchmark::EnvironmentInfo {
        eprintln!("[Tracea] 🔍 Detecting Environment...");
        
        // Use safer wrappers if possible, but for now just add checks
        let mut driver_ver: i32 = 0;
        unsafe { 
            let lib = cudarc::driver::sys::lib();
            lib.cuDriverGetVersion(&mut driver_ver);
        };
        eprintln!("[Tracea]  - Driver Version: {}", driver_ver);
        
        let mut nvrtc_major = 0;
        let mut nvrtc_minor = 0;
        unsafe { 
            let lib = cudarc::nvrtc::sys::lib();
            lib.nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor);
        };
        eprintln!("[Tracea]  - NVRTC Version: {}.{}", nvrtc_major, nvrtc_minor);

        let sm_major = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(8);
        let sm_minor = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(6);
        let sm = sm_major * 10 + sm_minor;
        eprintln!("[Tracea]  - SM Arch: {}", sm);
        
        crate::optimizer::benchmark::EnvironmentInfo {
            cuda_version: format!("{}.{}", nvrtc_major, nvrtc_minor),
            driver_version: format!("{}.{}", driver_ver / 1000, (driver_ver % 1000) / 10),
            sm_arch: sm as u32,
        }
    }
}
