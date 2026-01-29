use crate::policy::types::OperatorTopology;
use crate::core::device::DeviceProfile;
use crate::core::config::{GemmVariant, PipelineConfig};
use std::collections::HashMap;
use std::sync::{Mutex, Arc, OnceLock};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Serialize, Deserialize};

// Key: (OpName, M, N, K, DeviceName)
pub type TuningKey = (String, u32, u32, u32, String);

// Persistent Cache Entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub config: PipelineConfig,
    pub score: f32, // TFLOPS
    pub timestamp: u64,
}

// Global Singleton Cache
static GLOBAL_CACHE: OnceLock<Arc<TuningCache>> = OnceLock::new();

pub fn get_tuning_cache() -> Arc<TuningCache> {
    GLOBAL_CACHE.get_or_init(|| {
        Arc::new(TuningCache::new())
    }).clone()
}

pub struct TuningCache {
    cache: Mutex<HashMap<String, CacheEntry>>, // Keyed by stringified TuningKey for JSON compat
    file_path: String,
}

impl TuningCache {
    pub fn new() -> Self {
        let file_path = ".tracea/tuning_cache_v2.json".to_string();
        let cache = Self::load_cache(&file_path).unwrap_or_else(HashMap::new);
        
        Self {
            cache: Mutex::new(cache),
            file_path,
        }
    }

    fn load_cache(path: &str) -> Option<HashMap<String, CacheEntry>> {
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            serde_json::from_reader(reader).ok()
        } else {
            None
        }
    }

    fn save_cache(&self) {
        let cache = self.cache.lock().unwrap();
        if let Some(parent) = std::path::Path::new(&self.file_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(file) = File::create(&self.file_path) {
            let writer = BufWriter::new(file);
            let _ = serde_json::to_writer_pretty(writer, &*cache);
        }
    }

    fn make_key_str(op: &OperatorTopology, device: &DeviceProfile) -> Option<String> {
        match op {
            OperatorTopology::Gemm { m, n, k, .. } => {
                Some(format!("gemm_{}_{}_{}_{}", m, n, k, device.name))
            },
            OperatorTopology::Conv2d { name, n: batch, h, w, c, k, r, s, stride, .. } => {
                // Conv key: conv_NAME_B_H_W_C_K_R_S_STRIDE_device
                Some(format!("conv_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}", name, batch, h, w, c, k, r, s, stride, device.name))
            },
            _ => None
        }
    }

    /// The "God" method: Get best config or run extensive tuning
    pub fn get_or_tune<F>(&self, op: &OperatorTopology, device: &DeviceProfile, candidates: Vec<PipelineConfig>, run_benchmark: F) -> PipelineConfig 
    where F: Fn(&PipelineConfig) -> f32 // Returns TFLOPS
    {
        let key_str = match Self::make_key_str(op, device) {
            Some(k) => k,
            None => return candidates.first().cloned().unwrap_or_default(), // Fallback
        };

        // 1. Check Cache
        {
            let cache = self.cache.lock().unwrap();
            if let Some(entry) = cache.get(&key_str) {
                // Return cached best
                return entry.config.clone();
            }
        }

        // 2. Tune (Grid Search)
        eprintln!("[Autotuner] 🧠 Tuning cache miss for {}. Evaluation {} candidates...", key_str, candidates.len());
        
        let mut best_config = candidates.first().cloned().unwrap_or_default();
        let mut max_score = 0.0;

        for (i, config) in candidates.iter().enumerate() {
            eprint!("\r[Autotuner] 🔄 Checking candidate {}/{}...", i+1, candidates.len());
            let score = run_benchmark(config);
            if score > max_score {
                max_score = score;
                best_config = config.clone();
            }
        }
        eprintln!("\n[Autotuner] 🏆 Best Score: {:.2} TFLOPS", max_score);

        // 3. Update Cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key_str, CacheEntry {
                config: best_config.clone(),
                score: max_score,
                timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            });
        }
        
        // Persist immediately for safety
        self.save_cache();

        best_config
    }
}

// Backward compatibility methods for existing code
impl TuningCache {
    pub fn get_best_variant(op: &OperatorTopology, device: &DeviceProfile) -> Option<GemmVariant> {
        let instance = get_tuning_cache();
        let key_str = Self::make_key_str(op, device)?;
        let cache = instance.cache.lock().unwrap();
        
        cache.get(&key_str).map(|entry| entry.config.gemm_variant)
    }

    pub fn make_key(op: &OperatorTopology, device: &DeviceProfile) -> Option<TuningKey> {
        match op {
            OperatorTopology::Gemm { name: _, m, n, k, .. } => {
                Some(("gemm".to_string(), *m, *n, *k, device.name.clone()))
            },
            _ => None
        }
    }

    pub fn update_entry(_key: TuningKey, _variant: GemmVariant) {
        // No-op for now as we transitioned to PipelineConfig based cache
        // Or we could map GemmVariant to a simple config and store it?
        // Let's leave it no-op to avoid polluting the new high-quality cache with legacy variants.
    }
}

// ============================================
//  Restored Autotuner Traits (Backward Compatibility)
// ============================================

pub struct ParameterRange; // Placeholder

pub struct SearchSpace<T> {
    pub candidates: Vec<T>,
}

impl<T> SearchSpace<T> {
    pub fn new(candidates: Vec<T>) -> Self {
        Self { candidates }
    }
}

pub trait TunableKernel {
    type Config: Clone;
    fn name(&self) -> &'static str;
    fn search_space(&self) -> SearchSpace<Self::Config>;
    fn is_feasible(&self, cfg: &Self::Config) -> bool;
    fn benchmark(&self, cfg: &Self::Config) -> Option<f32>;
    fn cache_key(&self) -> String;
}

pub enum SearchMode {
    GridSearch,
    RandomSearch(usize),
}

pub fn tune_kernel<K: TunableKernel>(kernel: &K, _mode: SearchMode) -> K::Config {
    let space = kernel.search_space();
    let candidates = space.candidates;

    if candidates.is_empty() {
        panic!("Search space is empty for kernel: {}", kernel.name());
    }

    let mut best_config = candidates[0].clone();
    let mut best_score = f32::NEG_INFINITY;

    for cfg in &candidates {
        if !kernel.is_feasible(cfg) { continue; }
        
        // Simple sequential benchmark
        if let Some(score) = kernel.benchmark(cfg) {
            if score > best_score {
                best_score = score;
                best_config = cfg.clone();
            }
        }
    }
    
    // Fallback
    best_config
}
