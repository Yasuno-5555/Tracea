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

pub mod benchmark;
use benchmark::{MicroBenchmark, Observation};

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
            mean += weight * obs.tflops;
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
}

impl AutoTuner {
    pub fn new(gpu: GPUInfo) -> Self {
        Self {
            gpu,
            gp: GaussianProcess::new(),
        }
    }

    pub fn optimize<B: MicroBenchmark>(&mut self, benchmark: &B, iterations: usize) -> PipelineConfig {
        // Initial random feasible config
        let initial_config = PipelineConfig::new(2, 64, 64, 8);
        let mut current_best_y = benchmark.measure(&initial_config);
        self.gp.observe(Observation { config: initial_config.clone(), tflops: current_best_y });

        for _ in 0..iterations {
            // 1. Propose next candidate by maximizing EI
            let candidate = self.propose_candidate(current_best_y);

            // 2. Measure performance
            let tflops = benchmark.measure(&candidate);

            // 3. Update model
            let obs = Observation { config: candidate, tflops };
            if obs.tflops > current_best_y {
                current_best_y = obs.tflops;
            }
            self.gp.observe(obs);
        }

        self.gp.observations.iter()
            .max_by(|a, b| a.tflops.partial_cmp(&b.tflops).unwrap())
            .unwrap().config.clone()
    }

    fn propose_candidate(&self, current_best_y: f32) -> PipelineConfig {
        // Search for config that maximizes GP Expected Improvement
        // For simplicity, we sample a few feasible configs and pick the best EI
        let mut best_ei = -1.0;
        let mut best_config = PipelineConfig::new(2, 64, 64, 8);

        let search_space = vec![
            PipelineConfig::new(2, 64, 64, 8),
            PipelineConfig::new(4, 128, 128, 16),
            PipelineConfig::new(8, 64, 64, 32),
            PipelineConfig::new(3, 128, 64, 16),
        ];

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
        let smem_a = config.m_tile * config.k_tile * 4;
        let smem_b = config.n_tile * config.k_tile * 4;
        let total_smem = (smem_a + smem_b) * config.num_stages;

        if total_smem as usize > self.gpu.shared_memory_per_block {
            return Err(PruneReason::SharedMemoryOverflow {
                required: total_smem as usize,
                available: self.gpu.shared_memory_per_block,
            });
        }

        // 2. Register Pressure Estimation (Simplified)
        let regs_per_thread = (config.m_tile / 16) * (config.n_tile / 16) * 8 + 16;
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
