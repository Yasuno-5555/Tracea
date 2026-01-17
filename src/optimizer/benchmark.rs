use crate::PipelineConfig;

/// Trait for measuring the actual performance of a generated kernel
pub trait MicroBenchmark {
    fn measure(&self, config: &PipelineConfig) -> f32; // Returns TFLOPS
}

/// Simulated benchmark for testing the Auto-tuner without a physical GPU
pub struct SimulatedBenchmark;

impl MicroBenchmark for SimulatedBenchmark {
    fn measure(&self, config: &PipelineConfig) -> f32 {
        // Simple heuristic: M*N*K / (Latency + Overhead)
        // More stages and larger tiles (up to a point) give better performance
        let base = 10.0;
        let scale = (config.num_stages as f32 * config.m_tile as f32 * config.n_tile as f32).log2();
        base + scale * 0.5
    }
}
#[derive(Debug, Clone)]
pub struct Observation {
    pub config: PipelineConfig,
    pub tflops: f32,
}
