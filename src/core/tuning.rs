use crate::policy::types::OperatorTopology;
use crate::core::device::DeviceProfile;
use crate::core::config::GemmVariant;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::LazyLock; // Or OnceLock if on newer Rust

// Simple global cache for now. In real system, this would be persistent.
// Key: (OpName, M, N, K, DeviceName)
// Value: Best Variant
pub type TuningKey = (String, u32, u32, u32, String);

static mut GLOBAL_CACHE: Option<Mutex<HashMap<TuningKey, GemmVariant>>> = None;
static INIT: std::sync::Once = std::sync::Once::new();

fn get_cache() -> &'static Mutex<HashMap<TuningKey, GemmVariant>> {
    unsafe {
        INIT.call_once(|| {
            GLOBAL_CACHE = Some(Mutex::new(HashMap::new()));
        });
        GLOBAL_CACHE.as_ref().unwrap()
    }
}

pub struct TuningCache; // Namespace struct

impl TuningCache {
    pub fn get_best_variant(op: &OperatorTopology, device: &DeviceProfile) -> Option<GemmVariant> {
        let key = Self::make_key(op, device)?;
        let cache = get_cache().lock().unwrap();
        cache.get(&key).cloned()
    }

    pub fn update(op: &OperatorTopology, device: &DeviceProfile, variant: GemmVariant) {
        if let Some(key) = Self::make_key(op, device) {
            Self::update_entry(key, variant);
        }
    }

    pub fn update_entry(key: TuningKey, variant: GemmVariant) {
        let mut cache = get_cache().lock().unwrap();
        cache.insert(key, variant);
    }

    pub fn make_key(op: &OperatorTopology, device: &DeviceProfile) -> Option<TuningKey> {
        match op {
            OperatorTopology::Gemm { name: _, m, n, k, .. } => {
                Some(("gemm".to_string(), *m, *n, *k, device.name.clone()))
            },
            _ => None
        }
    }
}

// ============================================
//  Restored Autotuner Traits (Was Overwritten)
// ============================================

pub struct ParameterRange; // Placeholder if not used extensively

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
    
    // Fallback if nothing feasible found (should behave better, but for now return first)
    best_config
}
