use crate::optimizer::AcquisitionFunction;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintLevel {
    Strict,
    Relaxed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingMode {
    Fast,      // 1 initial + 2 refinement
    Sensitive, // 1 initial + 4 refinement
}

#[derive(Debug, Clone)]
pub struct TuningPolicy {
    pub strategy_name: &'static str,
    pub max_trials: usize,
    pub acq_function: AcquisitionFunction,
    pub constraints: ConstraintLevel,
    pub sampling_mode: SamplingMode,
    pub noise_penalty_k: f32,
}

impl TuningPolicy {
    pub fn scout() -> Self {
        Self {
            strategy_name: "Scout",
            max_trials: 60,
            acq_function: AcquisitionFunction::Thompson,
            constraints: ConstraintLevel::Relaxed,
            sampling_mode: SamplingMode::Sensitive,
            noise_penalty_k: 1.5,
        }
    }

    pub fn sniper() -> Self {
        Self {
            strategy_name: "Sniper",
            max_trials: 30,
            acq_function: AcquisitionFunction::UCB,
            constraints: ConstraintLevel::Strict,
            sampling_mode: SamplingMode::Fast,
            noise_penalty_k: 0.5,
        }
    }

    pub fn derive(batch_size: u32) -> Self {
        if batch_size <= 32 {
            Self::scout()
        } else {
            Self::sniper()
        }
    }
}
