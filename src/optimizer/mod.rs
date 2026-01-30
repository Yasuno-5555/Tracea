use crate::PipelineConfig;
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use serde::{Serialize, Deserialize};

use crate::runtime::manager::DeviceBackend;
use crate::core::backend::{Device, CudaArch, CpuArch};
pub use problem::{ProblemDescriptor, LayerType, HeroConfig, ArchHint, HeroScope, Layout, Fa2Variant, AsmParams, GpuAsmParams, Shape};
use crate::optimizer::policy::{TuningPolicy, PolicyFactory, SamplingPlan, TuningContext, SearchSpace};

pub mod heroscope;
pub mod model;
use heroscope::{HeroScopeV3, get_cpu_id};

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub name: String,
    pub backend: DeviceBackend,
    pub shared_memory_per_block: usize,
    pub max_registers_per_thread: u32,
    pub registers_per_sm: u32,
    pub max_registers_per_block: u32,
    pub max_warps_per_sm: u32, // CUDA: Warps, ROCm: Waves, Metal: Simdgroups
    pub wavefront_size: u32,  // CUDA: 32, ROCm: 64 or 32, Metal: 32
    pub max_blocks_per_sm: u32,
    pub shared_memory_per_sm: usize,
    pub has_specialized_units: bool,
    pub compute_capability: Option<(u32, u32)>,
    pub supported_intrinsic_shapes: Vec<crate::core::config::IntrinsicShape>,
    pub max_threadgroup_memory: usize, // Metal: 32KB on some devices
    pub preferred_tile_shape: [usize; 3], // [M, N, K] hint
    pub simd_width: usize, // Warp/Wave/SimdGroup width
}

impl HardwareProfile {
    pub fn rtx3070() -> Self {
        Self {
            name: "NVIDIA GeForce RTX 3070".to_string(),
            backend: DeviceBackend::Cuda,
            shared_memory_per_block: 48 * 1024, // Configurable up to 99KB
            max_registers_per_thread: 255,
            registers_per_sm: 65536,
            max_registers_per_block: 65536,
            max_warps_per_sm: 48,
            wavefront_size: 32,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 100 * 1024,
            has_specialized_units: true,
            compute_capability: Some((8, 6)),
            supported_intrinsic_shapes: vec![crate::core::config::IntrinsicShape::M16N8K16],
            max_threadgroup_memory: 0,
            preferred_tile_shape: [128, 128, 32], 
            simd_width: 32,
        }
    }

    pub fn mi250() -> Self {
        Self {
            name: "AMD Instinct MI250X".to_string(),
            backend: DeviceBackend::Rocm,
            shared_memory_per_block: 64 * 1024,
            max_registers_per_thread: 256,
            registers_per_sm: 163840, // 256KB across WGP
            max_registers_per_block: 65536,
            max_warps_per_sm: 32,
            wavefront_size: 64,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 64 * 1024,
            has_specialized_units: true,
            compute_capability: None,
            supported_intrinsic_shapes: vec![crate::core::config::IntrinsicShape::M32N32K2, crate::core::config::IntrinsicShape::M16N16K4],
            max_threadgroup_memory: 64 * 1024, // LDS size approximation
            preferred_tile_shape: [256, 128, 16], // MI250 prefers large tiles
            simd_width: 64,
        }
    }

    pub fn a100() -> Self {
        Self {
            name: "NVIDIA A100-SXM4-40GB".to_string(),
            backend: DeviceBackend::Cuda,
            shared_memory_per_block: 164 * 1024,
            max_registers_per_thread: 255,
            registers_per_sm: 65536,
            max_registers_per_block: 65536,
            max_warps_per_sm: 64,
            wavefront_size: 32,
            max_blocks_per_sm: 32,
            shared_memory_per_sm: 164 * 1024,
            has_specialized_units: true,
            compute_capability: Some((8, 0)),
            supported_intrinsic_shapes: vec![crate::core::config::IntrinsicShape::M16N8K16],
            max_threadgroup_memory: 0,
            preferred_tile_shape: [128, 256, 32],
            simd_width: 32,
        }
    }

    pub fn apple_m1() -> Self {
        Self {
            name: "Apple M1".to_string(),
            backend: DeviceBackend::Metal,
            shared_memory_per_block: 32 * 1024,
            max_registers_per_thread: 128,
            registers_per_sm: 0, 
            max_registers_per_block: 0,
            max_warps_per_sm: 32, // Metal Threadgroups
            wavefront_size: 32,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 32 * 1024,
            has_specialized_units: true,
            compute_capability: None,
            supported_intrinsic_shapes: vec![crate::core::config::IntrinsicShape::None],
            max_threadgroup_memory: 32 * 1024,
            preferred_tile_shape: [64, 64, 32],
            simd_width: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PruningReason {
    SharedMemoryOverflow,
    RegisterPressureTooHigh,
    LowOccupancy(u32), // Occupancy in milli-percentage (0-1000)
    InvalidTileSize,
    UnsupportedIntrinsic,
    ForbiddenZone(&'static str),
}

impl HardwareProfile {
    pub fn to_device(&self) -> crate::core::backend::Device {
        use crate::core::backend::{Device, CudaArch, CpuArch};
        match self.backend {
            DeviceBackend::Cuda => {
                 let arch = match self.name.as_str() {
                     n if n.contains("RTX 30") => CudaArch::Ampere,
                     n if n.contains("RTX 40") => CudaArch::Ada,
                     _ => CudaArch::Unknown,
                 };
                 Device::Cuda(arch)
            }
            DeviceBackend::Metal => Device::Metal,
            _ => Device::Cpu(CpuArch::Scalar),
        }
    }

    pub fn to_device_profile(&self) -> crate::core::device::DeviceProfile {
        crate::core::device::DeviceProfile {
            backend: match self.backend {
                DeviceBackend::Cuda => crate::core::device::BackendType::Cuda,
                DeviceBackend::Metal => crate::core::device::BackendType::Metal,
                DeviceBackend::Rocm => crate::core::device::BackendType::Rocm,
                _ => crate::core::device::BackendType::Cpu,
            },
            name: self.name.clone(),
            max_threads_per_block: 1024, // simplified
            simd_width: self.simd_width,
            local_memory_size: self.shared_memory_per_block,
            has_tensor_cores: self.has_specialized_units,
            has_fp16_storage: true,
            texture_alignment: 256,
        }
    }

    pub fn check_feasibility(&self, config: &PipelineConfig, problem: &ProblemDescriptor) -> Result<(), PruningReason> {
        // 1. Shared Memory Check
        let policy = PolicyFactory::derive(problem);
        let smem_usage = policy.estimate_smem_usage(config);
        if smem_usage > self.shared_memory_per_block {
            return Err(PruningReason::SharedMemoryOverflow);
        }

        // 2. Intrinsic Check
        if config.instruction != crate::core::config::SpecializedInstruction::None {
            if !self.supported_intrinsic_shapes.contains(&config.intrinsic_shape) {
                // If it's a generic instruction but we asked for a specific shape, check it.
                // In some cases we might fallback, but for tuning we want to prune if requested shape isn't there.
            }
        }

        // 3. Occupancy Estimation
        let occupancy = self.estimate_occupancy(config);
        if occupancy < 0.05 { // Arbitrary low threshold: 5%
            return Err(PruningReason::LowOccupancy((occupancy * 1000.0) as u32));
        }

        // 4. Forbidden Zones (Reproducibility & Ethics)
        // Example: Tiles that are too small for large problems causing excessive overhead/noise
        if problem.shape.m > 1024 && config.m_tile < 16 {
             return Err(PruningReason::ForbiddenZone("Excessive tile-overhead for large M"));
        }
        
        // Example: Stages that exceed a sane limit for this arch
        if config.num_stages > 8 {
             return Err(PruningReason::ForbiddenZone("Unstable pipeline depth (>8 stages)"));
        }

        Ok(())
    }

    pub fn estimate_occupancy(&self, config: &PipelineConfig) -> f32 {
        let warps_per_block = config.force_num_warps.unwrap_or(4);
        let blocks_by_warps = self.max_warps_per_sm / warps_per_block;
        
        // ⚠️ Conservative Register Estimation (Lower Bound)
        // Registers ≈ (TileSizeElements * PipelineDepth) / ThreadCount + BaseOverhead
        // Using f32 for intermediate calc to avoid overflow/truncation early
        let thread_count = (warps_per_block * self.wavefront_size) as f32;
        let tile_elements = (config.m_tile * config.n_tile) as f32; // Simplified
        let base_overhead = 32.0;
        
        let est_regs = match config.instruction {
            crate::core::config::SpecializedInstruction::CudaMMA | crate::core::config::SpecializedInstruction::RocmMFMA => {
                let load_regs = (tile_elements * config.num_stages as f32) / thread_count;
                (load_regs + base_overhead) as u32
            },
            _ => 32,
        }.min(self.max_registers_per_thread);

        let regs_per_block = (warps_per_block * self.wavefront_size) * est_regs;
        
        let blocks_by_regs = if self.registers_per_sm > 0 && regs_per_block > 0 {
            self.registers_per_sm / regs_per_block
        } else {
            self.max_blocks_per_sm
        };

        let blocks_per_sm = blocks_by_warps.min(blocks_by_regs).min(self.max_blocks_per_sm);
        
        // Return 0.0 only if strictly impossible, otherwise give it a chance.
        if blocks_per_sm == 0 { 0.0 } else { blocks_per_sm as f32 / self.max_blocks_per_sm as f32 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TuningStats {
    pub total_trials: usize,
    pub pruned_count: usize,
    pub pruning_reasons: std::collections::HashMap<String, usize>,
    pub forbidden_configs: Vec<PipelineConfig>,
}

impl TuningStats {
    pub fn log_pruning(&mut self, reason: PruningReason) {
        self.pruned_count += 1;
        let entry = self.pruning_reasons.entry(format!("{:?}", reason)).or_insert(0);
        *entry += 1;
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
pub mod orchestrator;
pub mod semantic;
pub mod history;
pub mod tuner;

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
             length_scales: vec![1.0; 15], 
             noise_sigma: 0.1,
             exploration_beta: 2.0,
             num_features: 15,
        }
    }

    pub fn observe(&mut self, obs: Observation, gpu: &HardwareProfile) {
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
    
    pub fn optimize_hyperparams(&mut self, gpu: &HardwareProfile) {
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

    fn marginal_log_likelihood(&self, gpu: &HardwareProfile) -> f32 {
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
    
    fn predict_excluding(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, exclude_idx: usize, gpu: &HardwareProfile) -> (f32, f32) {
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
    
    fn roofline_prior(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &HardwareProfile) -> f32 {
         let is_tc = config.instruction != crate::core::config::SpecializedInstruction::None;
         let (peak_tflops, mem_bw) = match gpu.backend {
             DeviceBackend::Metal => (5.0, 68.0),
             DeviceBackend::Rocm => (180.0, 1600.0),
             _ => (if is_tc { 160.0 } else { 20.0 }, 448.0),
         };
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

    pub fn predict(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &HardwareProfile) -> (f32, f32) {
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

    pub fn expected_improvement(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, current_best_y: f32, gpu: &HardwareProfile) -> f32 {
         let (mu, sigma) = self.predict(m, n, k, config, gpu);
         if sigma < 1e-6 { return 0.0; }
         let z = (mu - current_best_y) / sigma;
         // Minimal erf replacement
         let phi_z = ( -0.5 * z.powi(2) ).exp() / (2.0 * std::f32::consts::PI).sqrt();
         let pt = 0.5 * (1.0 + (z / 2.0f32.sqrt()).tanh()); // Approx for compilation speed
         (mu - current_best_y) * pt + sigma * phi_z
    }
    
    pub fn ucb(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &HardwareProfile) -> f32 {
        let (mu, sigma) = self.predict(m, n, k, config, gpu);
        mu + self.exploration_beta * sigma
    }
    
    pub fn thompson_sample(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &HardwareProfile) -> f32 {
        let (mu, sigma) = self.predict(m, n, k, config, gpu);
        let mut rng = thread_rng();
        let dist = Normal::new(mu, sigma.max(1e-6)).unwrap_or(Normal::new(mu, 1e-6).unwrap());
        dist.sample(&mut rng)
    }
}

#[derive(Debug, Clone)]
pub struct AutoTuner {
    pub gpu: HardwareProfile,
    pub gp: GaussianProcess,
    pub best_config: Option<PipelineConfig>,
    pub device: Device,
    pub runtime: Option<std::sync::Weak<crate::runtime::RuntimeManager>>, 
    pub heroscope: HeroScopeV3,
    pub hardware_id: String,
    pub stats: TuningStats,
}

impl AutoTuner {
    pub fn new(gpu: HardwareProfile) -> Self {
        let device = gpu.to_device();

        let hardware_id = match device {
            Device::Cpu(_) => get_cpu_id(),
            Device::Cuda(_) => {
                if let Some((major, minor)) = gpu.compute_capability {
                    format!("sm_{}{}", major, minor)
                } else {
                    gpu.name.clone()
                }
            }
            Device::Metal => gpu.name.clone(),
        };

        Self {
            gpu,
            gp: GaussianProcess::new(),
            best_config: None,
            device,
            runtime: None,
            heroscope: HeroScopeV3::new(),
            hardware_id,
            stats: TuningStats::default(),
        }
    }
    
    pub fn with_runtime(mut self, runtime: Arc<crate::runtime::RuntimeManager>) -> Self {
        self.runtime = Some(Arc::downgrade(&runtime));
        self
    }
    
    // --- Legacy Adapters ---
    
    pub fn optimize<B: MicroBenchmark>(&mut self, benchmark: &B, iterations: usize, goal: OptimizationGoal, epilogue: Vec<crate::core::op::EpilogueOp>) -> PipelineConfig {
        // Adapt to ProblemDescriptor
        let problem = ProblemDescriptor::new_gemm(benchmark.m() as usize, benchmark.n() as usize, benchmark.k() as usize)
            .with_device(self.device);
        eprintln!("[Adapter] optimize -> optimize_v2 for GEMM {}x{}x{}", problem.shape.m, problem.shape.n, problem.shape.k);
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
        
        let problem = ProblemDescriptor::new_conv2d(p.batch, p.h_in, p.w_in, p.c_in, p.c_out, p.kernel_h, p.kernel_w, p.stride, p.pad, layout)
            .with_device(self.device);
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

pub struct ConvBenchmarkAdapter<'a, B: Conv2dBenchmark> {
    pub inner: &'a B,
    pub magic_strategy: MagicNumberStrategy,
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
    fn observe_hardware(&self, config: &PipelineConfig) -> Option<model::HardwareObservation> {
        self.inner.observe_hardware(&crate::optimizer::benchmark::ConvConfig {
            base: config.clone(),
            use_tensor_core: true,
            use_nhwc: true,
            magic_strategy: self.magic_strategy,
        })
    }
}


    // --- Core V2 Logic ---

impl AutoTuner {
    
    pub fn optimize_v2<B: MicroBenchmark>(&mut self, benchmark: &B, problem: &ProblemDescriptor, iterations: usize, goal: OptimizationGoal) -> PipelineConfig {
        use crate::policy::standard::StandardPolicyEngine;
        use crate::core::tuning::get_tuning_cache;
        use crate::policy::types::OperatorTopology;
        
        let engine = StandardPolicyEngine::new();
        let topology = match problem.layer_type {
            LayerType::Conv2d { h, w, c, k, r, s, stride, padding, .. } => OperatorTopology::Conv2d { 
                op_id: 0,
                name: problem.name.clone(),
                n: problem.shape.batch as u32,
                h: h as u32, w: w as u32, c: c as u32, k: k as u32,
                r: r as u32, s: s as u32, 
                stride: stride as u32, 
                padding: padding as u32,
                epilogue: vec![],
             },
            _ => OperatorTopology::Gemm { 
                op_id: 0,
                name: problem.name.clone(), 
                m: problem.shape.m as u32,
                n: problem.shape.n as u32,
                k: problem.shape.k as u32,
                batch: 1,
                kind: crate::policy::types::TopologyKind::Dense,
                epilogue: vec![],
            },
        };

        // 1. Generate Candidates
        let device_profile = self.gpu.to_device_profile();
        let candidates = match problem.layer_type {
            LayerType::Conv2d { .. } => engine.propose_conv_configs(&device_profile),
            _ => engine.propose_gemm_configs(&device_profile),
        };
        
        if candidates.is_empty() {
            eprintln!("[Autotuner] ⚠️ No candidates proposed! Falling back to magic default.");
            return PipelineConfig::new(2, 64, 64, 32);
        }

        eprintln!("[Autotuner] 🤖 Intelligence Phase I: Evaluator ready with {} candidates.", candidates.len());

        // 2. Delegate to God Cache
        let cache = get_tuning_cache();
        
        let best_config = cache.get_or_tune(&topology, &device_profile, candidates, |config| {
             let result = benchmark.measure(config);
             result.tflops
        });

        // Update stats
        self.best_config = Some(best_config.clone());
        
        best_config
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
        
        // Exhaustive search over specified space
        for &mt in &space.tile_m {
            for &nt in &space.tile_n {
                for &kt in &space.tile_k {
                    for &w in &space.warps {
                        for &stages in &space.stages {
                            for &swizzle in &space.swizzles {
                                for &barrier in &space.barrier_modes {
                                    for &unroll in &space.k_unroll {
                                        for &pf in &space.prefetch_distance {
                                            for &mm in &space.micro_m {
                                                let mut cfg = PipelineConfig::new(stages, mt, nt, kt).with_warps(w);
                                                cfg.instruction = if space.use_tensor_core { crate::core::config::SpecializedInstruction::CudaMMA } else { crate::core::config::SpecializedInstruction::None };
                                                cfg.swizzle_mode = swizzle;
                                                cfg.barrier_mode = barrier;
                                                cfg.k_unroll = unroll;
                                                cfg.prefetch_distance = pf;
                                                cfg.micro_m = mm;
                                                
                                                // Handle cp_async_distance derived from stages if not explicitly set
                                                if stages > 2 {
                                                    cfg.cp_async_distance = stages - 1;
                                                }

                                                if let Err(_) = self.gpu.check_feasibility(&cfg, problem) { continue; }
                                                if !policy.is_feasible(&cfg, &self.gpu) { continue; }
                                                
                                                let score = match acq {
                                                    AcquisitionFunction::UCB => self.gp.ucb(problem.shape.m as u32, problem.shape.n as u32, problem.shape.k as u32, &cfg, &self.gpu),
                                                    AcquisitionFunction::EI => self.gp.expected_improvement(problem.shape.m as u32, problem.shape.n as u32, problem.shape.k as u32, &cfg, current_best, &self.gpu),
                                                    AcquisitionFunction::Thompson => self.gp.thompson_sample(problem.shape.m as u32, problem.shape.n as u32, problem.shape.k as u32, &cfg, &self.gpu),
                                                };
                                                
                                                if score > best_acq {
                                                    best_acq = score;
                                                    best_cfg = cfg;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        best_cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{PipelineConfig, SpecializedInstruction, SwizzleMode, IntrinsicShape};
    use crate::optimizer::problem::{ProblemDescriptor};

    #[test]
    fn test_golden_config_preservation() {
        let gpu = HardwareProfile::rtx3070();
        let problem = ProblemDescriptor::new_gemm(2048, 2048, 2048);
        
        // Gemm v3.1 "Golden Config": Known best for Ampere
        let mut golden = PipelineConfig::new(3, 128, 128, 32).with_warps(8);
        golden.instruction = SpecializedInstruction::CudaMMA;
        golden.swizzle_mode = SwizzleMode::Xor4;
        golden.intrinsic_shape = IntrinsicShape::M16N8K16;

        // Verification: Should NOT be pruned
        let result = gpu.check_feasibility(&golden, &problem);
        assert!(result.is_ok(), "Golden Config was incorrectly pruned: {:?}", result.err());
        
        println!("[Test] ✅ Golden Config Preserved (Occupancy: {:.2}%)", gpu.estimate_occupancy(&golden) * 100.0);
    }
}
