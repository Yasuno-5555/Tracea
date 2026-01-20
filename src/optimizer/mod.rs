use crate::PipelineConfig;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

use crate::runtime::manager::DeviceBackend;

#[derive(Debug, Clone)]
pub struct GPUInfo {
    pub name: String,
    pub backend: DeviceBackend,
    pub shared_memory_per_block: usize,
    pub max_registers_per_thread: u32,
    pub max_warps_per_sm: u32, // CUDA: Warps, ROCm: Waves, Metal: Simdgroups
    pub wavefront_size: u32,  // CUDA: 32, ROCm: 64 or 32, Metal: 32
    pub max_blocks_per_sm: u32,
    pub shared_memory_per_sm: usize,
    pub has_specialized_units: bool,
}

impl GPUInfo {
    pub fn rtx3070() -> Self {
        Self {
            name: "NVIDIA GeForce RTX 3070".to_string(),
            backend: DeviceBackend::Cuda,
            shared_memory_per_block: 99 * 1024, // Real limit with dynamic SMEM config
            max_registers_per_thread: 255,
            max_warps_per_sm: 48,
            wavefront_size: 32,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 100 * 1024,
            has_specialized_units: true,
        }
    }

    pub fn mi250() -> Self {
        Self {
            name: "AMD Instinct MI250X".to_string(),
            backend: DeviceBackend::Rocm,
            shared_memory_per_block: 64 * 1024,
            max_registers_per_thread: 256,
            max_warps_per_sm: 32,
            wavefront_size: 64,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 64 * 1024,
            has_specialized_units: true,
        }
    }

    pub fn a100() -> Self {
        Self {
            name: "NVIDIA A100-SXM4-40GB".to_string(),
            backend: DeviceBackend::Cuda,
            shared_memory_per_block: 164 * 1024,
            max_registers_per_thread: 255,
            max_warps_per_sm: 64,
            wavefront_size: 32,
            max_blocks_per_sm: 32,
            shared_memory_per_sm: 164 * 1024,
            has_specialized_units: true,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcquisitionFunction {
    /// Expected Improvement (standard)
    EI,
    /// Upper Confidence Bound (more exploration)
    UCB,
    /// Thompson Sampling (parallel exploration)
    Thompson,
}

pub mod benchmark;
pub mod cache;
pub mod policy;

use benchmark::{MicroBenchmark, Observation, BenchmarkResult, Conv2dBenchmark, ConvConfig, Conv2dProblem};
use cache::{TuningCache, CacheKey};
use crate::core::config::MagicNumberStrategy;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

#[derive(Debug, Clone)]
pub struct GaussianProcess {
    pub observations: Vec<Observation>,
    /// ARD (Automatic Relevance Determination) length scales - one per feature dimension
    pub length_scales: Vec<f32>,
    pub noise_sigma: f32,
    /// Exploration weight for UCB acquisition function (higher = more exploration)
    pub exploration_beta: f32,
    /// Number of feature dimensions (set on first observation)
    num_features: usize,
}

impl GaussianProcess {
    pub fn new() -> Self {
        Self { 
            observations: Vec::new(),
            length_scales: vec![1.0; 10], // Default 10 dims, will resize
            noise_sigma: 0.1,
            exploration_beta: 2.0,
            num_features: 10,
        }
    }

    pub fn observe(&mut self, obs: Observation) {
        // On first observation, set feature dimension count
        if self.observations.is_empty() {
            let features = self.config_features(&obs.config, obs.m, obs.n, obs.k);
            self.num_features = features.len();
            self.length_scales = vec![1.0; self.num_features];
        }
        self.observations.push(obs);
        
        // Adaptive: Re-optimize hyperparameters after every 5 observations
        if self.observations.len() % 5 == 0 && self.observations.len() > 5 {
            self.optimize_hyperparams();
        }
    }

    /// Optimize ARD length scales via marginal log-likelihood maximization
    /// Uses simple gradient-free approach: perturb each length scale and check improvement
    pub fn optimize_hyperparams(&mut self) {
        if self.observations.len() < 3 {
            return;
        }
        
        let lr = 0.1;
        let num_iters = 10;
        
        for _ in 0..num_iters {
            let current_ll = self.marginal_log_likelihood();
            
            for dim in 0..self.length_scales.len() {
                // Try increasing length scale
                let orig = self.length_scales[dim];
                self.length_scales[dim] = orig * (1.0 + lr);
                let ll_up = self.marginal_log_likelihood();
                
                // Try decreasing length scale
                self.length_scales[dim] = orig * (1.0 - lr);
                let ll_down = self.marginal_log_likelihood();
                
                // Pick the best
                if ll_up > current_ll && ll_up > ll_down {
                    self.length_scales[dim] = orig * (1.0 + lr);
                } else if ll_down > current_ll {
                    self.length_scales[dim] = orig * (1.0 - lr);
                } else {
                    self.length_scales[dim] = orig;
                }
                
                // Clamp to reasonable range
                self.length_scales[dim] = self.length_scales[dim].clamp(0.1, 10.0);
            }
        }
    }

    /// Approximate marginal log-likelihood (simplified for efficiency)
    fn marginal_log_likelihood(&self) -> f32 {
        if self.observations.is_empty() {
            return 0.0;
        }
        
        let n = self.observations.len();
        let mut sum_sq_error = 0.0;
        let mut sum_variance = 0.0;
        
        // Leave-one-out cross-validation approximation
        for i in 0..n {
            let obs = &self.observations[i];
            let (mu, sigma) = self.predict_excluding(obs.m, obs.n, obs.k, &obs.config, i);
            let error = obs.score - mu;
            sum_sq_error += error.powi(2);
            sum_variance += sigma.powi(2) + self.noise_sigma.powi(2);
        }
        
        // Negative sum of squared errors (higher is better)
        -sum_sq_error / sum_variance.max(1e-6)
    }

    /// Predict excluding a specific observation (for LOO-CV)
    fn predict_excluding(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, exclude_idx: usize) -> (f32, f32) {
        let prior_mean = self.roofline_prior(m, n, k, config, &GPUInfo::rtx3070());
        
        if self.observations.len() <= 1 {
            return (prior_mean, 1.0);
        }

        let mut mean_diff = 0.0;
        let mut total_weight = 0.0;
        
        let cur_config = self.config_features(config, m, n, k);
        
        for (i, obs) in self.observations.iter().enumerate() {
            if i == exclude_idx {
                continue;
            }
            
            let obs_config = self.config_features(&obs.config, obs.m, obs.n, obs.k);
            let dist_sq = self.ard_distance_sq(&cur_config, &obs_config);
            let weight = (-dist_sq / 2.0).exp();
            
            let obs_prior = self.roofline_prior(obs.m, obs.n, obs.k, &obs.config, &GPUInfo::rtx3070());
            mean_diff += weight * (obs.score - obs_prior);
            total_weight += weight;
        }

        let mu = prior_mean + mean_diff / (total_weight + self.noise_sigma);
        let sigma = 1.0 / (total_weight + 1.0).sqrt();
        
        (mu, sigma)
    }

    /// ARD kernel distance: weighted squared distance with per-dimension length scales
    fn ard_distance_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dist_sq = 0.0;
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            let ls = self.length_scales.get(i).copied().unwrap_or(1.0);
            dist_sq += (ai - bi).powi(2) / ls.powi(2);
        }
        dist_sq
    }

    /// UCB (Upper Confidence Bound) acquisition function
    /// Provides more aggressive exploration than EI
    pub fn ucb(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &GPUInfo) -> f32 {
        let (mu, sigma) = self.predict(m, n, k, config, gpu);
        mu + self.exploration_beta * sigma
    }

    /// Thompson Sampling: Samples a score from the posterior distribution
    /// Used for exploration by picking the candidate with the highest sampled score.
    pub fn thompson_sample(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &GPUInfo) -> f32 {
        let (mu, sigma) = self.predict(m, n, k, config, gpu);
        
        let mut rng = thread_rng();
        let dist = Normal::new(mu, sigma.max(1e-6)).unwrap_or(Normal::new(mu, 1e-6).unwrap());
        dist.sample(&mut rng)
    }

    /// Latin Hypercube Sampling (LHS)
    /// Generates n points in d dimensions such that each dimension is evenly sampled.
    pub fn latin_hypercube_sample(n: usize, dims: usize) -> Vec<Vec<f32>> {
        let mut rng = thread_rng();
        let mut samples = vec![vec![0.0; dims]; n];
        
        for d in 0..dims {
            let mut perms: Vec<usize> = (0..n).collect();
            perms.shuffle(&mut rng);
            
            for i in 0..n {
                let lower = perms[i] as f32 / n as f32;
                let upper = (perms[i] + 1) as f32 / n as f32;
                let val: f32 = rng.gen_range(lower..upper);
                samples[i][d] = val;
            }
        }
        samples
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

    fn roofline_prior(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, _gpu: &GPUInfo) -> f32 {
        let is_tc = config.instruction != crate::core::config::SpecializedInstruction::None;
        let peak_tflops = if is_tc { 160.0 } else { 20.0 };
        let mem_bw = 448.0; // GB/s
        
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let bytes = (m as f64 * k as f64 + k as f64 * n as f64 + m as f64 * n as f64) * 2.0;
        let intensity = (ops / bytes) as f32; // FLOPs/Byte
        
        let bw_limit = (mem_bw * intensity) / 1000.0; // TFLOPS
        
        f32::min(peak_tflops, bw_limit)
    }

    /// Shape-Aware Product Kernel Prediction with ARD
    pub fn predict(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &GPUInfo) -> (f32, f32) {
        let prior_mean = self.roofline_prior(m, n, k, config, gpu);
        
        if self.observations.is_empty() {
            return (prior_mean, 1.0); 
        }

        let mut mean_diff = 0.0;
        let mut total_weight = 0.0;
        
        // Combine features: Shape features + Config features
        let cur_shape = self.shape_features(m, n, k);
        let cur_config = self.config_features(config, m, n, k);
        let cur_features: Vec<f32> = cur_shape.iter().chain(cur_config.iter()).cloned().collect();

        // Ensure length_scales matches feature count (resize if needed, though usually fixed after first obs)
        if self.length_scales.len() != cur_features.len() {
             // This is a "const" method so we can't mutate length_scales. 
             // Ideally this is handled in observe(). For now we just use 1.0 if size mismatch.
        }
        
        for obs in &self.observations {
            // Recompute obs features (could be cached, but cheap enough for now)
            let obs_shape = self.shape_features(obs.m, obs.n, obs.k);
            let obs_config = self.config_features(&obs.config, obs.m, obs.n, obs.k);
            let obs_features: Vec<f32> = obs_shape.iter().chain(obs_config.iter()).cloned().collect();
            
            // Check if feature counts match
            if cur_features.len() != obs_features.len() { continue; }

            // ARD Distance
            let dist_sq = self.ard_distance_sq(&cur_features, &obs_features);
            
            // RBF Kernel
            let weight = (-dist_sq / 2.0).exp();
            
            let obs_prior = self.roofline_prior(obs.m, obs.n, obs.k, &obs.config, gpu);
            mean_diff += weight * (obs.score - obs_prior);
            total_weight += weight;
        }

        let mu = prior_mean + mean_diff / (total_weight + self.noise_sigma);
        let sigma = 1.0 / (total_weight + 1.0).sqrt();
        
        (mu, sigma)
    }

    /// Expected Improvement (EI) Acquisition Function
    pub fn expected_improvement(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, current_best_y: f32, gpu: &GPUInfo) -> f32 {
        let (mu, sigma) = self.predict(m, n, k, config, gpu);
        if sigma < 1e-6 { return 0.0; }

        let z = (mu - current_best_y) / sigma;
        
        let phi_z = ( -0.5 * z.powi(2) ).exp() / (2.0 * std::f32::consts::PI).sqrt();
        let tau_z = 0.5 * (1.0 + (z / 2.0f32.sqrt()).mu_erf()); 

        (mu - current_best_y) * tau_z + sigma * phi_z
    }
}

// Minimal erf implementation for CDF
trait MuErf { fn mu_erf(self) -> f32; }
impl MuErf for f32 {
    fn mu_erf(self) -> f32 {
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
    pub blacklist: std::collections::HashSet<Vec<u32>>,
    pub runtime: Option<Arc<crate::runtime::RuntimeManager>>, // Doctor access via Runtime
}

impl AutoTuner {
    pub fn new(gpu: GPUInfo) -> Self {
        Self {
            gpu,
            gp: GaussianProcess::new(),
            best_config: None,
            blacklist: std::collections::HashSet::new(),
            runtime: None,
        }
    }
    
    pub fn with_runtime(mut self, runtime: Arc<crate::runtime::RuntimeManager>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub fn optimize<B: MicroBenchmark>(&mut self, benchmark: &B, iterations: usize, goal: OptimizationGoal, epilogue: Vec<crate::core::op::EpilogueOp>) -> PipelineConfig {
        eprintln!("[SR] >>> optimize entry with epilogue count: {} >>>", epilogue.len());
        // 0. Detect Environment
        let env = benchmark.device_info();

        // 1. Check Cache
        let key = CacheKey {
            backend: self.gpu.backend,
            gpu: self.gpu.name.clone(),
            m: benchmark.m(),
            n: benchmark.n(),
            k: benchmark.k(),
            dtype: "f16".to_string(), 
            epilogue: epilogue.clone(),
            env_version: env.api_version.clone(),
            arch: env.arch.clone(),
            op_fingerprint: Some("gemm".to_string()),
        };
        let mut cache = TuningCache::new();
        if let Some(config) = cache.get(&key) {
            println!("[Tracea] 💎 Cache Hit! Using optimized config for {}x{}x{} (Backend: {:?})", key.m, key.n, key.k, key.backend);
            return config;
        }

        // 1. Initial random feasible config
        let m = benchmark.m();
        let n = benchmark.n();
        let k = benchmark.k();

        eprintln!("[Tracea] 🔍 Starting Tuning for {}x{}x{}", m, n, k);

        let mut initial_config = PipelineConfig::new(2, 128, 128, 32);
        initial_config.instruction = if self.gpu.backend == DeviceBackend::Cuda { 
            crate::core::config::SpecializedInstruction::CudaMMA 
        } else { 
            crate::core::config::SpecializedInstruction::RocmMFMA 
        };
        initial_config.epilogue = epilogue.clone();

        eprintln!("[Tracea] 🧪 Measurig Initial Config...");
        // Validation for initial config
        if !benchmark.validate_config(&initial_config) {
             eprintln!("[Tracea] ⚠️ Initial Config Failed Validation!");
             // Fallback?
        }
        
        let res = benchmark.measure(&initial_config);
        let default_policy = policy::TuningPolicy::sniper();
        let score = self.calculate_score(&res, goal, &default_policy);
        eprintln!("[Tracea] 📈 Initial Score: {:.2}", score);
        
        self.gp.observe(Observation { m, n, k, config: initial_config.clone(), score });

        let mut current_best_score = score;
        self.best_config = Some(initial_config.clone());

        for i in 0..iterations {
            eprintln!("[Tracea] 🔄 Tuning Iteration {}/{}...", i + 1, iterations);
            
            // 2. Propose next candidate: Rotate acquisition functions for balance
            let acq = if i < iterations / 2 {
                if i % 2 == 0 { AcquisitionFunction::UCB } else { AcquisitionFunction::Thompson }
            } else {
                AcquisitionFunction::EI
            };

            let mut candidate = self.propose_candidate(m, n, k, current_best_score, acq);
            candidate.epilogue = epilogue.clone();
            
            // Check Blacklist
            // let candidate_vec = candidate.to_vector(); 
            // let vec_key: Vec<u32> = candidate.to_vector().iter().map(|&x| x as u32).collect(); 

            // config.to_vector() returns Vec<f32>.
            // We need a stable key.
            // Let's rely on simple string or just use loop filter.
            // Actually propose_candidate might return same candidate if we don't exclude it?
            // GP optimization is stochastic if we explore?
            
            // 3. Validation Probe
            eprintln!("[Tracea] 🕵️ Probing Candidate: {}x{}x{}", candidate.m_tile, candidate.n_tile, candidate.k_tile);
            if !benchmark.validate_config(&candidate) {
                eprintln!("[Tracea] ❌ Candidate Failed Validation. Blacklisting.");
                
                // Doctor Consultation
                if let Some(rt) = &self.runtime {
                     // Check if Doctor has a critical diagnosis from the last attempt
                     if let Some(error_report) = rt.doctor.last_error() {
                         eprintln!("[Tracea Doctor] 🩺 Diagnosis on Validation Failure: {}", error_report.message);
                         if error_report.message.contains("CUDA_ERROR_NO_BINARY_FOR_GPU") || 
                            error_report.suggestion.contains("architecture mismatch") {
                             eprintln!("[Tracea] 🛑 Critical Error Detected by Doctor. Aborting Tuning.");
                             break;
                         }
                     }
                }

                self.gp.observe(Observation { m, n, k, config: candidate.clone(), score: -1.0 });
                continue;
            }

            // 4. Measure performance
            eprintln!("[Tracea] 🧪 Measuring Candidate: {}x{}x{}", candidate.m_tile, candidate.n_tile, candidate.k_tile);
            let res = benchmark.measure(&candidate);
            // Default policy for generic optimize
            let default_policy = policy::TuningPolicy::sniper(); 
            let score = self.calculate_score(&res, goal, &default_policy);

            println!("[Tracea] Testing {:>3}x{:>3}x{:>2} ({} stages) -> {:.2} TFLOPS, {:.3} ms (Score: {:.2})", 
                candidate.m_tile, candidate.n_tile, candidate.k_tile, candidate.num_stages, 
                res.tflops, res.latency_ms, score);

            // 5. Update model
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
                m_tile: 64,
                n_tile: 64,
                k_tile: 32,
                instruction: crate::core::config::SpecializedInstruction::None,
                swizzle_mode: crate::core::config::SwizzleMode::None,
                quantization: crate::core::config::QuantizationMode::None,
                layout_policy: Some(crate::core::config::LayoutPolicy::RowMajor),
                epilogue: epilogue.clone(),
                force_num_warps: None,
            }
        });
        
        cache.set(key, best_config.clone());
        println!("[Tracea] 💾 Tuning complete. Optimal config saved for {}x{}x{}", m, n, k);
        best_config
    }

    /// Optimized Tuning for Conv2d Kernels
    pub fn optimize_conv<B: Conv2dBenchmark>(&mut self, benchmark: &B, iterations_usize: usize, goal: OptimizationGoal) -> ConvConfig {
        let problem = benchmark.problem();
        let (m, n, k) = problem.gemm_dims();
        let env = benchmark.device_info();

        // 0. Derive Meta Policy
        let policy = policy::TuningPolicy::derive(problem.batch as u32);
        eprintln!("[MetaTuner] 🛰 Selected Policy: {} (Trials: {}, Constraints: {:?})", 
            policy.strategy_name, policy.max_trials, policy.constraints);

        // 1. Check Cache
        let key = CacheKey {
            backend: self.gpu.backend,
            gpu: self.gpu.name.clone(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            dtype: "f16".to_string(),
            epilogue: vec![], // Conv usually has fixed or separate epilogue for now
            env_version: env.api_version.clone(),
            arch: env.arch.clone(),
            op_fingerprint: Some(format!("conv:r={},s={},stride={},pad={}", problem.kernel_h, problem.kernel_w, problem.stride, problem.pad)),
        };
        
        let mut cache = TuningCache::new();
        if let Some(config) = cache.get(&key) {
            // Mapping back from PipelineConfig to ConvConfig
            return ConvConfig {
                base: config,
                use_tensor_core: true,
                use_nhwc: true,
                magic_strategy: MagicNumberStrategy::select_for(problem.h_out() * problem.w_out()),
            };
        }

        eprintln!("[Tracea] 🔍 Starting Conv2d Tuning for {}", problem.name);

        // 2. Initial Config
        let initial_conv_config = ConvConfig::default_for_problem(&problem);
        if !benchmark.validate_config(&initial_conv_config) {
            eprintln!("[Tracea] ⚠️ Default Conv Config failed validation!");
        }

        let res = benchmark.measure(&initial_conv_config);
        let score = self.calculate_score(&res, goal, &policy);
        self.gp.observe(Observation { m: m as u32, n: n as u32, k: k as u32, config: initial_conv_config.base.clone(), score });

        let mut current_best_score = score;
        let mut best_conv_config = initial_conv_config;

        let max_trials = policy.max_trials;
        for i in 0..max_trials {
            eprintln!("[Tracea] 🔄 Conv2d Iteration {}/{}...", i + 1, max_trials);
            
            // Step 1: Sanity Check - Inject Hero Config for B=32 ResNet50-Conv3x3 (Iter 0)
            if i == 0 && problem.name.contains("B32") && problem.name.contains("3x3") {
                let hero_base = PipelineConfig::new(2, 64, 64, 16);
                let hero = ConvConfig {
                    base: PipelineConfig {
                        force_num_warps: Some(5), // 1P + 4C (Aligned for M=64)
                        ..hero_base
                    },
                    use_tensor_core: true,
                    use_nhwc: true,
                    magic_strategy: MagicNumberStrategy::select_for(problem.h_out() * problem.w_out()),
                };
                println!("[Hero] Injecting known-good config for B32: {:?}", hero);
                let res = benchmark.measure(&hero);
                let score = self.calculate_score(&res, goal, &policy);
                println!("[Hero] Result: {:.2} TFLOPS (Score: {:.2})", res.tflops, score);
                self.gp.observe(Observation { m: m as u32, n: n as u32, k: k as u32, config: hero.base.clone(), score });
                if score > current_best_score {
                    current_best_score = score;
                    best_conv_config = hero;
                }
                continue; // Move to next iteration after hero injection
            }

            // Hero Injection for B=64 (ResNet50 Conv3x3)
            if i == 1 && problem.name.contains("B64") && problem.name.contains("3x3") {
                 let hero_base = PipelineConfig::new(2, 128, 64, 32);
                 let hero = ConvConfig {
                    base: PipelineConfig {
                        force_num_warps: Some(9), // 1P + 8C (Optimal for M=128)
                        instruction: crate::core::config::SpecializedInstruction::CudaMMA,
                        ..hero_base
                    },
                    use_tensor_core: true,
                    use_nhwc: true,
                    magic_strategy: MagicNumberStrategy::select_for(problem.h_out() * problem.w_out()),
                };
                println!("[Hero] Injecting known-good config for B64: {:?}", hero);
                let res = benchmark.measure(&hero);
                let score = self.calculate_score(&res, goal, &policy);
                println!("[Hero] Result: {:.2} TFLOPS (Score: {:.2})", res.tflops, score);
                self.gp.observe(Observation { m: m as u32, n: n as u32, k: k as u32, config: hero.base.clone(), score });
                if score > current_best_score {
                    current_best_score = score;
                    best_conv_config = hero;
                }
                continue;
            }

            // Step 2: Early Signal Kill (Space Reset)
            if i == 5 && current_best_score < 1.0 { // Arbitrary threshold for "hopeless"
                eprintln!("[MetaTuner] 🛑 Early Signal Kill! Tuning is not progressing. Resetting space...");
                // Resetting GP or changing policy? For now, just break and use fallback
                break;
            }

            let acq = if i < max_trials / 2 { policy.acq_function } else { AcquisitionFunction::EI };
            let candidate_base = self.propose_candidate_with_policy(m as u32, n as u32, k as u32, current_best_score, acq, &policy);
            
            let candidate = ConvConfig {
                base: candidate_base,
                use_tensor_core: true,
                use_nhwc: true,
                magic_strategy: MagicNumberStrategy::select_for(problem.h_out() * problem.w_out()),
            };

            if !benchmark.validate_config(&candidate) {
                continue;
            }

            let res = benchmark.measure(&candidate);
            let score = self.calculate_score(&res, goal, &policy);

            println!("[Tracea] Testing Conv {:>3}x{:>3}x{:>2} -> {:.2} TFLOPS (Mean: {:.2}, StdDev: {:.2}, Score: {:.2})", 
                candidate.base.m_tile, candidate.base.n_tile, candidate.base.k_tile, res.tflops, res.mean_tflops, res.std_dev, score);

            self.gp.observe(Observation { m: m as u32, n: n as u32, k: k as u32, config: candidate.base.clone(), score });
            if score > current_best_score {
                current_best_score = score;
                best_conv_config = candidate;
            }
        }

        cache.set(key, best_conv_config.base.clone());
        best_conv_config
    }
    fn calculate_score(&self, res: &BenchmarkResult, goal: OptimizationGoal, policy: &policy::TuningPolicy) -> f32 {
        let tflops = res.mean_tflops - policy.noise_penalty_k * res.std_dev;
        match goal {
            OptimizationGoal::MaximizeTFLOPS => tflops,
            OptimizationGoal::MinimizeLatency => {
                let robust_latency = (1000.0 / tflops.max(1e-6)).max(res.latency_ms);
                1000.0 / robust_latency
            },
            OptimizationGoal::Balanced { tflops_weight } => {
                let latency_score = 1000.0 / res.latency_ms.max(1e-6);
                tflops * tflops_weight + latency_score * (1.0 - tflops_weight)
            }
        }
    }
    
    // Legacy support wrapper or can be deprecated
    fn propose_candidate(&self, m: u32, n: u32, k: u32, current_best_y: f32, acq: AcquisitionFunction) -> PipelineConfig {
        let dummy_policy = policy::TuningPolicy::derive(32); // fallback
        self.propose_candidate_with_policy(m, n, k, current_best_y, acq, &dummy_policy)
    }

    fn propose_candidate_with_policy(&self, m: u32, n: u32, k: u32, current_best_y: f32, acq: AcquisitionFunction, policy: &policy::TuningPolicy) -> PipelineConfig {
        let mut best_score = -1e9;
        let mut best_config = PipelineConfig::new(2, 64, 64, 32);

        // Dynamic Search Space Generation based on Backend
        let (m_sizes, n_sizes) = match self.gpu.backend {
            DeviceBackend::Cuda => {
                // Heuristic: Segregate M Tile based on Total M (Workload Size)
                // Threshold 150,000 separates B=32 (100k) from B=64 (200k).
                let m_candidates = if m >= 150_000 {
                    vec![64, 128] // Include 64 as fallback, favor 128
                } else {
                    vec![64] // Keep it agile for small batch (B<=32)
                };
                let n_candidates = vec![32, 64, 128]; // Flexible N
                (m_candidates, n_candidates)
            },
            DeviceBackend::Rocm => (vec![32, 64, 128], vec![32, 64, 128]),
            DeviceBackend::Metal => (vec![16, 32, 64], vec![16, 32, 64]),
        };
        
        let k_sizes = [16, 32, 64]; // Extended K search space
        let stage_counts = [2, 3];
        let warp_counts = match self.gpu.backend {
            DeviceBackend::Cuda => vec![5, 9, 17], // 1P + 4C, 1P + 8C, 1P + 16C (Aligned)
            _ => vec![1],
        };
        
        let instructions = match self.gpu.backend {
            DeviceBackend::Cuda => vec![crate::core::config::SpecializedInstruction::CudaMMA],
            DeviceBackend::Rocm => vec![crate::core::config::SpecializedInstruction::RocmMFMA],
            DeviceBackend::Metal => vec![crate::core::config::SpecializedInstruction::MetalSimdGroup],
        };

        for &mt in &m_sizes {
            for &nt in &n_sizes {
                for &kt in &k_sizes {
                    for &stages in &stage_counts {
                        for &nw in &warp_counts {
                            for &inst in &instructions {
                                let mut cfg = PipelineConfig::new(stages, mt, nt, kt);
                                cfg.instruction = inst;
                                cfg.force_num_warps = Some(nw);

                            if self.is_feasible(&cfg, Some(policy)).is_ok() {
                                let score = match acq {
                                    AcquisitionFunction::EI => self.gp.expected_improvement(m, n, k, &cfg, current_best_y, &self.gpu),
                                    AcquisitionFunction::UCB => self.gp.ucb(m, n, k, &cfg, &self.gpu),
                                    AcquisitionFunction::Thompson => self.gp.thompson_sample(m, n, k, &cfg, &self.gpu),
                                };

                                    if score > best_score {
                                        best_score = score;
                                        best_config = cfg;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        best_config
    }

    pub fn is_feasible(&self, config: &PipelineConfig, policy_opt: Option<&policy::TuningPolicy>) -> Result<(), PruneReason> {
        // 1. General Constraints
        if config.m_tile == 0 || config.n_tile == 0 || config.k_tile == 0 {
            return Err(PruneReason::InvalidAlignment);
        }

        // Policy-based warp constraints
        if let Some(policy) = policy_opt {
             let nw = config.force_num_warps.unwrap_or(4);
             if policy.constraints == policy::ConstraintLevel::Strict {
                 // Sniper: Strict on warp counts and efficiency
                 if nw > 16 { return Err(PruneReason::InvalidAlignment); }
             } else {
                 // Scout: Relaxed
                 if nw > 32 { return Err(PruneReason::InvalidAlignment); }
             }
        }
        let is_special = config.instruction != crate::core::config::SpecializedInstruction::None;
        
        // 1. Backend-Specific Alignment
        match self.gpu.backend {
            DeviceBackend::Cuda if is_special => {
                if config.m_tile % 16 != 0 || config.n_tile % 8 != 0 || config.k_tile % 16 != 0 {
                    return Err(PruneReason::InvalidAlignment);
                }
            }
            DeviceBackend::Rocm if is_special => {
                if config.m_tile % 32 != 0 || config.n_tile % 32 != 0 || config.k_tile % 8 != 0 {
                    return Err(PruneReason::InvalidAlignment);
                }
            }
            _ => {}
        }

        // 3. Tensor Core Warp Split Check
        if config.instruction == crate::core::config::SpecializedInstruction::CudaMMA {
            let nw = config.force_num_warps.unwrap_or(4);
            let num_consumers = if nw > 1 { nw - 1 } else { nw };
            if config.m_tile < num_consumers * 16 || config.m_tile % (num_consumers * 16) != 0 {
                return Err(PruneReason::InvalidAlignment);
            }
        }

        // 2. Shared Memory Check (LDS for AMD)
        let element_size = 2; // Assuming FP16 for now
        let a_stride = config.k_tile + 8;
        let b_stride = config.n_tile + 8;
        let smem_a_bytes = config.m_tile * a_stride * element_size;
        let smem_b_bytes = config.k_tile * b_stride * element_size;
        
        // Header for barriers and async management (matches emitter)
        let header_bytes = 128;
        let smem_compute = header_bytes + (smem_a_bytes + smem_b_bytes) * config.num_stages;
        
        // Epilogue Shared Memory (matches benchmark.rs and emitter)
        // Stores M*N accumulators in float precision (4 bytes)
        let smem_epilogue = config.m_tile * config.n_tile * 4;
        
        let total_smem = std::cmp::max(smem_compute, smem_epilogue);

        if total_smem as usize > self.gpu.shared_memory_per_block {
            println!("[Reject] {:?} SharedMemoryOverflow: req={} avail={}", config, total_smem, self.gpu.shared_memory_per_block);
            return Err(PruneReason::SharedMemoryOverflow {
                required: total_smem as usize,
                available: self.gpu.shared_memory_per_block,
            });
        }

        // 3. Register Pressure Estimation (Heuristic)
        let nw = config.force_num_warps.unwrap_or(4);
        let num_threads = nw * 32;

        let est_regs = if config.use_tensor_cores() {
            let num_consumers = if nw > 1 { nw - 1 } else { nw };
            
            // Alignment check: Each consumer must handle at least one 16x16 tile
            if config.m_tile % (num_consumers * 16) != 0 {
                println!("[Reject] {:?} InvalidAlignment: M={} consumers={}", config, config.m_tile, num_consumers);
                return Err(PruneReason::InvalidAlignment);
            }

            let m_frags = config.m_tile / (num_consumers * 16);
            let n_frags = config.n_tile / 16;
            
            // Accumulators: Each fragment holds 8 float values (32 bits).
            let regs_acc = m_frags * n_frags * 8;
            
            // Shared memory fragments (matrix_a and matrix_b)
            // frag_a is reused, frag_b is cached in an array in the consumer loop
            let regs_a = 8; 
            let regs_b = n_frags * 8; 
            
            // Indexing, pipeline state, and function overhead
            let overhead = 16;
            
            regs_acc + regs_a + regs_b + overhead
        } else {
            let elements_per_thread = (config.m_tile * config.n_tile) / num_threads;
            elements_per_thread + 32
        };

        if est_regs > 255 {
            println!("[Reject] {:?} RegisterPressure: est={} max=255", config, est_regs);
            return Err(PruneReason::RegisterPressure {
                required: est_regs as u32,
                available: 255,
            });
        }
        
        Ok(())
    }

    pub fn get_env_info(device_opt: &Option<Arc<CudaDevice>>) -> crate::optimizer::benchmark::EnvironmentInfo {
        if let Some(device) = device_opt {
            let mut driver_ver: i32 = 0;
            unsafe { let _ = cudarc::driver::sys::lib().cuDriverGetVersion(&mut driver_ver); };
            let mut nvrtc_v = (0, 0);
            unsafe { let _ = cudarc::nvrtc::sys::lib().nvrtcVersion(&mut nvrtc_v.0, &mut nvrtc_v.1); };
            let sm_major = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(8);
            let sm_minor = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(6);
            
            crate::optimizer::benchmark::EnvironmentInfo {
                backend: DeviceBackend::Cuda,
                api_version: format!("{}.{}", nvrtc_v.0, nvrtc_v.1),
                driver_version: format!("{}.{}", driver_ver / 1000, (driver_ver % 1000) / 10),
                arch: format!("sm_{}{}", sm_major, sm_minor),
            }
        } else {
            // Check ROCm
            crate::optimizer::benchmark::EnvironmentInfo {
                backend: DeviceBackend::Rocm,
                api_version: "6.0".to_string(),
                driver_version: "6.0".to_string(),
                arch: "gfx90a".to_string(),
            }
        }
    }
}

