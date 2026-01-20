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

    /// Shape-Aware Product Kernel Prediction
    pub fn predict(&self, m: u32, n: u32, k: u32, config: &PipelineConfig, gpu: &GPUInfo) -> (f32, f32) {
        let prior_mean = self.roofline_prior(m, n, k, config, gpu);
        
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
            
            let mut dist_sq = 0.0;
            for (a, b) in cur_shape.iter().zip(obs_shape.iter()) { dist_sq += (a - b).powi(2); }
            for (a, b) in cur_config.iter().zip(obs_config.iter()) { dist_sq += (a - b).powi(2); }
            
            let weight = (-dist_sq / (2.0 * self.length_scale.powi(2))).exp();
            
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
            let score = self.calculate_score(&res, goal);

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
        let mut best_ei = -1.0;
        let mut best_config = PipelineConfig::new(2, 64, 64, 32);

        // Dynamic Search Space Generation based on Backend
        let tile_sizes = match self.gpu.backend {
            DeviceBackend::Cuda => [64, 128, 256],
            DeviceBackend::Rocm => [32, 64, 128], // AMD prefers smaller/aligned tiles for MFMA
            DeviceBackend::Metal => [16, 32, 64],
        };
        
        let k_sizes = [16, 32, 64];
        let stage_counts = [2, 3, 4];
        
        let instructions = match self.gpu.backend {
            DeviceBackend::Cuda => [crate::core::config::SpecializedInstruction::None, crate::core::config::SpecializedInstruction::CudaMMA],
            DeviceBackend::Rocm => [crate::core::config::SpecializedInstruction::None, crate::core::config::SpecializedInstruction::RocmMFMA],
            DeviceBackend::Metal => [crate::core::config::SpecializedInstruction::None, crate::core::config::SpecializedInstruction::MetalSimdGroup],
        };

        for &mt in &tile_sizes {
            for &nt in &tile_sizes {
                for &kt in &k_sizes {
                    for &stages in &stage_counts {
                        for &inst in &instructions {
                            let mut cfg = PipelineConfig::new(stages, mt, nt, kt);
                            cfg.instruction = inst;
                            cfg.force_num_warps = Some(match self.gpu.backend {
                                DeviceBackend::Cuda => 4,
                                DeviceBackend::Rocm => 1, // Single wave per block for simple kernels
                                DeviceBackend::Metal => 1,
                            });

                            if self.is_feasible(&cfg).is_ok() {
                                let ei = self.gp.expected_improvement(m, n, k, &cfg, current_best_y, &self.gpu);
                                if ei > best_ei {
                                    best_ei = ei;
                                    best_config = cfg;
                                }
                            }
                        }
                    }
                }
            }
        }
        best_config
    }

    pub fn is_feasible(&self, config: &PipelineConfig) -> Result<(), PruneReason> {
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

        // 2. Shared Memory Check (LDS for AMD)
        let element_size = 2; // Assuming FP16 for now
        let buf_a = config.m_tile * config.k_tile * element_size;
        let buf_b = config.n_tile * config.k_tile * element_size;
        let total_smem = (buf_a + buf_b) * config.num_stages;

        if total_smem as usize > self.gpu.shared_memory_per_block {
            return Err(PruneReason::SharedMemoryOverflow {
                required: total_smem as usize,
                available: self.gpu.shared_memory_per_block,
            });
        }

        // 3. Register Pressure Estimation (Heuristic)
        let _wf_size = self.gpu.wavefront_size;
        let num_threads = 128; // Standard block size

        let est_regs = if config.use_tensor_cores() {
            let elements_per_thread = (config.m_tile * config.n_tile) / num_threads;
            elements_per_thread + 32 + 16
        } else {
            let elements_per_thread = (config.m_tile * config.n_tile) / num_threads;
             elements_per_thread + 16
        };

        if est_regs > self.gpu.max_registers_per_thread {
             if est_regs > 255 {
                  return Err(PruneReason::RegisterPressure {
                      required: est_regs,
                      available: self.gpu.max_registers_per_thread,
                  });
             }
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
