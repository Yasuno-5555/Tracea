use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::fmt::Debug;

/// Represents the possible values for a configuration parameter.
/// Used to define the search space for Bayesian optimization and Grid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    /// A discrete set of options (e.g., block sizes: [32, 64, 128])
    Discrete(Vec<i32>),
    /// A categorical set of options (e.g., pipeline stages: "1", "2") - stored as indices or strings?
    /// For simplicity, let's interpret generic parameters as derived within the kernel adapter
    /// or keep it simple for now.
    Categorical(Vec<String>),
    /// A continuous range (min, max)
    Continuous { min: f64, max: f64 },
}

/// A collection of configurations to explore.
/// For the initial implementation, this will primarily support Grid Search (explicit list of Configs).
pub struct SearchSpace<C> {
    pub candidates: Vec<C>,
}

impl<C> SearchSpace<C> {
    pub fn new(candidates: Vec<C>) -> Self {
        Self { candidates }
    }
}

/// The core trait that any tunable kernel (FA2, GEMM, Conv) must implement.
/// This decouples the "Search Logic" from the "Execution Logic".
pub trait TunableKernel {
    /// The configuration struct (e.g., `Fa2Config`) that defines *how* the kernel runs.
    type Config: Clone + Serialize + DeserializeOwned + Debug + PartialEq;

    /// Unique name of the kernel (e.g., "fa2")
    fn name(&self) -> &'static str;

    /// Defines the search space for this kernel given the current problem instance.
    /// For Grid Search, this returns a list of all candidates to try.
    fn search_space(&self) -> SearchSpace<Self::Config>;

    /// Fast feasibility check. Returns true if the config is valid for the hardware/problem.
    /// Should check register limits, shared memory size, etc. BEFORE attempting execution.
    fn is_feasible(&self, cfg: &Self::Config) -> bool;

    /// Runs the kernel and returns a score (lower is usually better for latency, higher for FLOPS).
    /// But let's standarize: Standard return is Option<f32>.
    /// By convention in Tracea, let's say higher score = better (FLOPS), or use a metric enum.
    /// The design doc says "Option<f32>". Let's assume this is the performance metric (e.g. TFLOPS or throughput).
    /// Returns None if runtime failure occurs (compile error, etc) even if feasible check passed.
    fn benchmark(&self, cfg: &Self::Config) -> Option<f32>;

    /// Generates a unique cache key for (Device + Driver + Kernel + Problem + Runtime).
    fn cache_key(&self) -> String;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    GridSearch,
    Bayesian, // Future work
}

use crate::core::cache::{load_cache, save_cache};

/// Generic Tuning Entry Point
pub fn tune_kernel<K: TunableKernel>(kernel: &K, mode: SearchMode) -> K::Config {
    // 1. Check Cache
    let key = kernel.cache_key();
    if let Some(cfg) = load_cache(&key) { 
        return cfg; 
    }

    // 2. Search
    let best_config = match mode {
        SearchMode::GridSearch => grid_search(kernel),
        SearchMode::Bayesian => {
            // Fallback to grid for now until Bayesian is implemented
            println!("Bayesian search not yet implemented, falling back to GridSearch");
            grid_search(kernel)
        }
    };

    // 3. Save Cache
    save_cache(&key, &best_config);

    best_config
}

use std::io::Write;

fn grid_search<K: TunableKernel>(kernel: &K) -> K::Config {
    let space = kernel.search_space();
    let total = space.candidates.len();
    eprintln!("[Tuner] 🔍 Starting Grid Search ({} candidates)...", total);

    let mut best_score = -1.0;
    let mut best_config = space.candidates.first().expect("Search space cannot be empty").clone();

    for (i, cfg) in space.candidates.into_iter().enumerate() {
        if !kernel.is_feasible(&cfg) {
            eprintln!("[Tuner] [{}/{}] Skipping Infeasible: {:?}", i+1, total, cfg);
            continue;
        }

        eprint!("[Tuner] [{}/{}] Testing: {:?} ... ", i+1, total, cfg);
        let _ = std::io::stderr().flush();
        
        if let Some(score) = kernel.benchmark(&cfg) {
            eprintln!("{:.2} TFLOPS", score);
            if score > best_score {
                best_score = score;
                best_config = cfg;
            }
        } else {
            eprintln!("FAILED");
        }
    }
    
    eprintln!("[Tuner] 🏆 Best Config: {:?} ({:.2} TFLOPS)", best_config, best_score);
    best_config
}
