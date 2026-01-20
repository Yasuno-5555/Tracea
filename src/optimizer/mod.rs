use crate::PipelineConfig;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

use crate::runtime::manager::DeviceBackend;
use crate::optimizer::problem::{ProblemDescriptor, LayerType, HeroConfig, ArchHint, HeroScope, Layout, Fa2Variant};
use crate::optimizer::policy::{TuningPolicy, PolicyFactory, SamplingPlan, TuningContext, SearchSpace};

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
    // ... (Other GPU implementations omitted for brevity but should typically be here) ...
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationGoal {
    MaximizeTFLOPS,
    MinimizeLatency,
    Balanced { tflops_weight: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcquisitionFunction {
    EI,
    UCB,
    Thompson,
}

pub mod benchmark;
pub mod cache;
pub mod policy;
pub mod problem;

use benchmark::{MicroBenchmark, Observation, BenchmarkResult, Conv2dBenchmark, ConvConfig, Conv2dProblem};
use cache::{TuningCache, CacheKey};
use crate::core::config::MagicNumberStrategy;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

#[derive(Debug, Clone)]
pub struct GaussianProcess {
    pub observations: Vec<Observation>,
    pub length_scales: Vec<f32>,
    pub noise_sigma: f32,
    pub exploration_beta: f32,
    num_features: usize,
}

impl GaussianProcess {
    pub fn new() -> Self {
        Self { 
             observations: Vec::new(),
             length_scales: vec![1.0; 10], 
             noise_sigma: 0.1,
             exploration_beta: 2.0,
             num_features: 10,
        }
    }

    pub fn observe(&mut self, obs: Observation, gpu: &GPUInfo) {
        if self.observations.is_empty() {
             let features = self.config_features(&obs.config, obs.m, obs.n, obs.k);
             self.num_features = features.len();
             self.length_scales = vec![1.0; self.num_features];
        }
        self.observations.push(obs);
        if self.observations.len() % 5 == 0 && self.observations.len() > 5 {
            self.optimize_hyperparams(gpu);
        }
    }
    
    pub fn optimize_hyperparams(&mut self, gpu: &GPUInfo) {
        if self.observations.len() < 3 { return; }
        let lr = 0.1;
        let num_iters = 10;
        for _ in 0..num_iters {
            let current_ll = self.marginal_log_likelihood(gpu);
            for dim in 0..self.length_scales.len() {
                let orig = self.length_scales[dim];
                self.length_scales[dim] = orig * (1.0 + lr);
                let ll_up = self.marginal_log_likelihood(gpu);
                self.length_scales[dim] = orig * (1.0 - lr);
                let ll_down = self.marginal_log_likelihood(gpu);
                if ll_up > current_ll && ll_up > ll_down {
                    self.length_scales[dim] = orig * (1.0 + lr);
                } else if ll_down > current_ll {
                    self.length_scales[dim] = orig * (1.0 - lr);
                } else {
                    self.length_scales[dim] = orig;
                }
                self.length_scales[dim] = self.length_scales[dim].clamp(0.1, 10.0);
            }
        }
    }

    fn marginal_log_likelihood(&self, gpu: &GPUInfo) -> f32 {
        if self.observations.is_empty() { return 0.0; }
        let n = self.observations.len();
        let mut sum_sq_error = 0.0;
        let mut sum_variance = 0.0;
        for i in 0..n {
            let obs = &self.observations[i];
            let (mu, sigma) = self.predict_excluding(obs.m, obs.n, obs.k, &obs.config, i, gpu);
            let error = obs.score - mu;
            sum_sq_error += error.powi(2);
            sum_variance += sigma.powi(2) + self.noise_sigma.powi(2);
        }
        -sum_sq_error / sum_variance.max(1e-6)
    }
    
    fn predict_excluding(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, exclude_idx: usize, gpu: &GPUInfo) -> (f32, f32) {
        let prior_mean = self.roofline_prior(m, n, k, config, gpu);
        if self.observations.len() <= 1 { return (prior_mean, 1.0); }
        let mut mean_diff = 0.0;
        let mut total_weight = 0.0;
        let cur_features: Vec<f32> = self.shape_features(m, n, k).into_iter().chain(self.config_features(config, m, n, k)).collect();
        for (i, obs) in self.observations.iter().enumerate() {
            if i == exclude_idx { continue; }
            let obs_features: Vec<f32> = self.shape_features(obs.m, obs.n, obs.k).into_iter().chain(self.config_features(&obs.config, obs.m, obs.n, obs.k)).collect();
            let dist_sq = self.ard_distance_sq(&cur_features, &obs_features);
            let weight = (-dist_sq / 2.0).exp();
            let obs_prior = self.roofline_prior(obs.m, obs.n, obs.k, &obs.config, gpu);
            mean_diff += weight * (obs.score - obs_prior);
            total_weight += weight;
        }
        let mu = prior_mean + mean_diff / (total_weight + self.noise_sigma);
        let sigma = 1.0 / (total_weight + 1.0).sqrt();
        (mu, sigma)
    }
    
    fn ard_distance_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dist_sq = 0.0;
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            let ls = self.length_scales.get(i).copied().unwrap_or(1.0);
            dist_sq += (ai - bi).powi(2) / ls.powi(2);
        }
        dist_sq
    }
    
    fn config_features(&self, config: &PipelineConfig, m: u32, n: u32, k: u32) -> Vec<f32> {
        let mut v = config.to_vector();
        v.push(config.m_tile as f32 / m as f32);
        v.push(config.n_tile as f32 / n as f32);
        v.push(config.k_tile as f32 / k as f32);
        v
    }
    
    fn roofline_prior(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, _gpu: &GPUInfo) -> f32 {
         let is_tc = config.instruction != crate::core::config::SpecializedInstruction::None;
         let peak_tflops = if is_tc { 160.0 } else { 20.0 };
         let mem_bw = 448.0; 
         let ops = 2.0 * m as f64 * n as f64 * k as f64;
         let bytes = (m as f64 * k as f64 + k as f64 * n as f64 + m as f64 * n as f64) * 2.0;
         let intensity = (ops / bytes) as f32;
         let bw_limit = (mem_bw * intensity) / 1000.0;
         f32::min(peak_tflops, bw_limit)
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

    pub fn predict(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &GPUInfo) -> (f32, f32) {
         let prior_mean = self.roofline_prior(m, n, k, config, gpu);
         if self.observations.is_empty() { return (prior_mean, 1.0); }
         
         let cur_shape = self.shape_features(m, n, k);
         let cur_config = self.config_features(config, m, n, k);
         let cur_features: Vec<f32> = cur_shape.iter().chain(cur_config.iter()).cloned().collect();

         let mut mean_diff = 0.0;
         let mut total_weight = 0.0;
         
         for obs in &self.observations {
             let obs_shape = self.shape_features(obs.m, obs.n, obs.k);
             let obs_config = self.config_features(&obs.config, obs.m, obs.n, obs.k);
             let obs_features: Vec<f32> = obs_shape.iter().chain(obs_config.iter()).cloned().collect();
             
             if cur_features.len() != obs_features.len() { continue; }
             let dist_sq = self.ard_distance_sq(&cur_features, &obs_features);
             let weight = (-dist_sq / 2.0).exp();
             let obs_prior = self.roofline_prior(obs.m, obs.n, obs.k, &obs.config, gpu);
             mean_diff += weight * (obs.score - obs_prior);
             total_weight += weight;
         }
         
         let mu = prior_mean + mean_diff / (total_weight + self.noise_sigma);
         let sigma = 1.0 / (total_weight + 1.0).sqrt();
         (mu, sigma)
    }

    pub fn expected_improvement(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, current_best_y: f32, gpu: &GPUInfo) -> f32 {
         let (mu, sigma) = self.predict(m, n, k, config, gpu);
         if sigma < 1e-6 { return 0.0; }
         let z = (mu - current_best_y) / sigma;
         // Minimal erf replacement
         let phi_z = ( -0.5 * z.powi(2) ).exp() / (2.0 * std::f32::consts::PI).sqrt();
         let pt = 0.5 * (1.0 + (z / 2.0f32.sqrt()).tanh()); // Approx for compilation speed
         (mu - current_best_y) * pt + sigma * phi_z
    }
    
    pub fn ucb(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &GPUInfo) -> f32 {
        let (mu, sigma) = self.predict(m, n, k, config, gpu);
        mu + self.exploration_beta * sigma
    }
    
    pub fn thompson_sample(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &GPUInfo) -> f32 {
        let (mu, sigma) = self.predict(m, n, k, config, gpu);
        let mut rng = thread_rng();
        let dist = Normal::new(mu, sigma.max(1e-6)).unwrap_or(Normal::new(mu, 1e-6).unwrap());
        dist.sample(&mut rng)
    }
}

#[derive(Debug, Clone)]
pub struct AutoTuner {
    pub gpu: GPUInfo,
    pub gp: GaussianProcess,
    pub best_config: Option<PipelineConfig>,
    pub runtime: Option<Arc<crate::runtime::RuntimeManager>>, // Doctor access via Runtime
}

impl AutoTuner {
    pub fn new(gpu: GPUInfo) -> Self {
        Self {
            gpu,
            gp: GaussianProcess::new(),
            best_config: None,
            runtime: None,
        }
    }
    
    pub fn with_runtime(mut self, runtime: Arc<crate::runtime::RuntimeManager>) -> Self {
        self.runtime = Some(runtime);
        self
    }
    
    // --- Legacy Adapters ---
    
    // OLD: optimize_legacy (replaces original optimize)
    pub fn optimize<B: MicroBenchmark>(&mut self, benchmark: &B, iterations: usize, goal: OptimizationGoal, epilogue: Vec<crate::core::op::EpilogueOp>) -> PipelineConfig {
        // Adapt to ProblemDescriptor
        let problem = ProblemDescriptor::new_gemm(benchmark.m() as usize, benchmark.n() as usize, benchmark.k() as usize);
        eprintln!("[Adapter] optimize -> optimize_v2 for GEMM {}x{}x{}", problem.m, problem.n, problem.k);
        let config = self.optimize_v2(benchmark, &problem, iterations, goal);
        // Epilogue injection currently manual in legacy path, v2 should handle it better or we inject here
        let mut final_config = config;
        final_config.epilogue = epilogue;
        final_config
    }
    
    // OLD: optimize_conv
    pub fn optimize_conv<B: Conv2dBenchmark>(&mut self, benchmark: &B, iterations: usize, goal: OptimizationGoal) -> ConvConfig {
        let p = benchmark.problem();
        let layout = Layout::NHWC; // Assume NHWC
        
        let problem = ProblemDescriptor::new_conv2d(p.batch, p.h_in, p.w_in, p.c_in, p.c_out, layout);
        eprintln!("[Adapter] optimize_conv -> optimize_v2 for Conv2d {}", problem.name);
        
        let magic_strategy = MagicNumberStrategy::select_for(p.h_out() * p.w_out());
        let adapter = ConvBenchmarkAdapter {
            inner: benchmark,
            magic_strategy,
        };
        
        let base_config = self.optimize_v2(&adapter, &problem, iterations, goal);
        
        ConvConfig {
            base: base_config,
            use_tensor_core: true,
            use_nhwc: true,
            magic_strategy,
        }
    }
}

struct ConvBenchmarkAdapter<'a, B: Conv2dBenchmark> {
    inner: &'a B,
    magic_strategy: MagicNumberStrategy,
}

impl<'a, B: Conv2dBenchmark> MicroBenchmark for ConvBenchmarkAdapter<'a, B> {
    fn measure(&self, config: &PipelineConfig) -> BenchmarkResult {
        self.inner.measure(&ConvConfig {
            base: config.clone(),
            use_tensor_core: true,
            use_nhwc: true,
            magic_strategy: self.magic_strategy,
        })
    }
    fn validate_config(&self, config: &PipelineConfig) -> bool {
        self.inner.validate_config(&ConvConfig {
            base: config.clone(),
            use_tensor_core: true,
            use_nhwc: true,
            magic_strategy: self.magic_strategy,
        })
    }
    fn m(&self) -> u32 { self.inner.problem().gemm_dims().0 as u32 }
    fn n(&self) -> u32 { self.inner.problem().gemm_dims().1 as u32 }
    fn k(&self) -> u32 { self.inner.problem().gemm_dims().2 as u32 }
    fn device_info(&self) -> benchmark::EnvironmentInfo { self.inner.device_info() }
}


    // --- Core V2 Logic ---

impl AutoTuner {
    
    pub fn optimize_v2<B: MicroBenchmark>(&mut self, benchmark: &B, problem: &ProblemDescriptor, iterations: usize, goal: OptimizationGoal) -> PipelineConfig {
        let policy = PolicyFactory::derive(problem);
        eprintln!("[Tracea] 🚀 Starting V2 Tuning for {} [layer_type={:?}]", problem.name, problem.layer_type);
        let key = CacheKey {
            backend: self.gpu.backend,
            gpu: self.gpu.name.clone(),
            m: problem.m as u32,
            n: problem.n as u32,
            k: problem.k as u32,
            dtype: "f16".to_string(), 
            epilogue: vec![], // To be refined
            env_version: "v2".to_string(),
            arch: match self.gpu.name.as_str() {
                n if n.contains("RTX 30") => "Ampere".to_string(),
                n if n.contains("RTX 40") => "Ada".to_string(),
                _ => "Unknown".to_string(),
            },
            op_fingerprint: Some(problem.name.clone()),
        };
        let mut cache = TuningCache::new();
        if let Some(config) = cache.get(&key) {
            println!("[Tracea] 💎 Cache Hit! {:?}", key);
            return config;
        }

        eprintln!("[Tracea] 🚀 Starting V2 Tuning for {} (Policy: {})", problem.name, "Derived");

        let mut current_best_score = -1e9;
        let mut best_config = None;
        let mut trials_so_far = 0;
        
        // 1. Hero Injection (Priority 0)
        let heroes = policy.hero_configs();
        eprintln!("[Tracea] 🦸 Found {} Hero Configs", heroes.len());
        
        for (idx, hero) in heroes.iter().enumerate() {
            eprintln!("[Tracea] 🦸 Injecting Hero #{}: {} ({:?})", idx+1, hero.note, hero.scope);
            if !benchmark.validate_config(&hero.config) {
                eprintln!("[Tracea] ⚠️ Hero config failed validation! Skipping.");
                continue;
            }
            
            let res = benchmark.measure(&hero.config);
            let score = self.calculate_score(&res, goal);
            eprintln!("[Tracea] 🦸 Hero Result [hero=true]: {:.2} TFLOPS (Score: {:.2})", res.tflops, score);
            
            self.gp.observe(Observation { 
                m: problem.m as u32, n: problem.n as u32, k: problem.k as u32, 
                config: hero.config.clone(), score 
            }, &self.gpu);
            
            if score > current_best_score {
                current_best_score = score;
                best_config = Some(hero.config.clone());
            }
        }
        
        if heroes.is_empty() {
            // Initial Random if no heroes
             let init = PipelineConfig::new(2, 128, 128, 32); 
             // ...
        }

        // 2. Optimization Loop
        for i in 0..iterations {
            trials_so_far += 1;
            eprintln!("[Tracea] 🔄 Iteration {}/{}", i + 1, iterations);
            
            // Context update
            let ctx = TuningContext {
                trials_so_far,
                best_score: current_best_score,
                variance: 0.0, // TODO: Get from GP
            };
            
            let plan = policy.sampling_plan(&ctx);
            
            let acq = match plan {
                SamplingPlan::Scout | SamplingPlan::Lightweight => AcquisitionFunction::Thompson,
                SamplingPlan::Sniper | SamplingPlan::Balanced => AcquisitionFunction::UCB, // Or EI
            };
            
            let candidate = self.propose_candidate(problem, &policy.search_space(), acq, &policy, current_best_score);
            
             if !benchmark.validate_config(&candidate) {
                // ... Blacklist logic ...
                continue;
            }

            let res = benchmark.measure(&candidate);
            let score = self.calculate_score(&res, goal);
            
            println!("[Tracea] Testing {:>3}x{:>3}x{:>2} -> {:.2} TFLOPS (Score: {:.2})", 
                candidate.m_tile, candidate.n_tile, candidate.k_tile, res.tflops, score);

            self.gp.observe(Observation { 
                m: problem.m as u32, n: problem.n as u32, k: problem.k as u32, 
                config: candidate.clone(), score 
            }, &self.gpu);

            if score > current_best_score {
                current_best_score = score;
                best_config = Some(candidate);
            }
        }
        
        let final_config = best_config.unwrap_or_else(|| PipelineConfig::new(2, 64, 64, 32));
        cache.set(key, final_config.clone());
        final_config
    }
    
    fn calculate_score(&self, res: &BenchmarkResult, goal: OptimizationGoal) -> f32 {
         let tflops = res.mean_tflops; // Simplified from previous noise penalty
         match goal {
            OptimizationGoal::MaximizeTFLOPS => tflops,
            OptimizationGoal::MinimizeLatency => 1000.0 / res.latency_ms.max(1e-6),
            OptimizationGoal::Balanced { tflops_weight } => {
                let latency_score = 1000.0 / res.latency_ms.max(1e-6);
                tflops * tflops_weight + latency_score * (1.0 - tflops_weight)
            }
        }
    }
    
    fn propose_candidate(&self, problem: &ProblemDescriptor, space: &SearchSpace, acq: AcquisitionFunction, policy: &Box<dyn TuningPolicy>, current_best: f32) -> PipelineConfig {
        let mut best_acq = -1e9;
        let mut best_cfg = PipelineConfig::new(2, 64, 64, 32);
        
        // Exhaustive search over specified space (in real impl, use random sampling if space is huge)
        for &mt in &space.tile_m {
            for &nt in &space.tile_n {
                for &kt in &space.tile_k {
                    for &w in &space.warps {
                         let mut cfg = PipelineConfig::new(2, mt, nt, kt).with_warps(w);
                         cfg.instruction = if space.use_tensor_core { crate::core::config::SpecializedInstruction::CudaMMA } else { crate::core::config::SpecializedInstruction::None };
                         cfg.swizzle_mode = if space.enable_swizzle { crate::core::config::SwizzleMode::Xor4 } else { crate::core::config::SwizzleMode::None };
                         
                         if !policy.is_feasible(&cfg, &self.gpu) { continue; }
                         
                         let score = match acq {
                             AcquisitionFunction::UCB => self.gp.ucb(problem.m as u32, problem.n as u32, problem.k as u32, &cfg, &self.gpu),
                             AcquisitionFunction::EI => self.gp.expected_improvement(problem.m as u32, problem.n as u32, problem.k as u32, &cfg, current_best, &self.gpu),
                             AcquisitionFunction::Thompson => self.gp.thompson_sample(problem.m as u32, problem.n as u32, problem.k as u32, &cfg, &self.gpu),
                         };
                         
                         if score > best_acq {
                             best_acq = score;
                             best_cfg = cfg;
                         }
                    }
                }
            }
        }
        best_cfg
    }
}
