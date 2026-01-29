use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use crate::core::config::PipelineConfig;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CacheKey {
    pub backend: crate::runtime::manager::DeviceBackend,
    pub gpu: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub dtype: String,
    pub epilogue: Vec<crate::core::op::EpilogueOp>,
    // Environment Invalidation
    pub env_version: String, // CUDA/ROCm/Metal API version
    pub arch: String,        // sm_86, gfx90a, etc.
    // Op-specific fingerprint (e.g., conv parameters)
    pub op_fingerprint: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct TuningCache {
    entries: HashMap<String, PipelineConfig>,
    #[serde(skip)]
    file_path: PathBuf,
}

impl TuningCache {
    pub fn new() -> Self {
        eprintln!("[Tracea] 📁 Initializing Tuning Cache...");
        let mut path = PathBuf::from(".tracea");
        if !path.exists() {
            eprintln!("[Tracea] 📁 Creating .tracea directory...");
            let _ = fs::create_dir_all(&path);
        }
        path.push("tuning_cache.json");
        eprintln!("[Tracea] 📁 Cache Path: {:?}", path);
        
        if path.exists() {
            eprintln!("[Tracea] 📁 Loading existing cache...");
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(entries) = serde_json::from_str::<HashMap<String, PipelineConfig>>(&content) {
                   eprintln!("[Tracea] 📁 Cache loaded with {} entries.", entries.len());
                   return Self { entries, file_path: path };
                } else {
                    eprintln!("[Tracea] ⚠️  Warning: Failed to deserialize cache. JSON might be corrupt.");
                }
            } else {
                eprintln!("[Tracea] ⚠️  Warning: Failed to read cache file.");
            }
        }
        
        eprintln!("[Tracea] 📁 Initializing new empty cache.");
        Self {
            entries: HashMap::new(),
            file_path: path,
        }
    }

    fn make_key(key: &CacheKey) -> String {
        eprintln!("[Tracea] 🔑 Generating Cache Key...");
        // Normalize epilogue: remove pointers for caching
        let normalized_epilogue: Vec<String> = key.epilogue.iter().map(|op| {
            match op {
                crate::core::op::EpilogueOp::None => "none".to_string(),
                crate::core::op::EpilogueOp::BiasAdd { .. } => "bias".to_string(),
                crate::core::op::EpilogueOp::ReLU => "relu".to_string(),
                crate::core::op::EpilogueOp::Gelu => "gelu".to_string(),
                crate::core::op::EpilogueOp::SiLU => "silu".to_string(),
                crate::core::op::EpilogueOp::ResidualAdd { .. } => "residual".to_string(),
                crate::core::op::EpilogueOp::BiasAddSiLU { .. } => "bias_silu".to_string(),
                crate::core::op::EpilogueOp::BatchNorm { .. } => "batchnorm".to_string(),
            }
        }).collect();
        let epi_str = normalized_epilogue.join(",");

        format!("{:?}:{}:{}:{}:{}:{}:{}:{}:{}:{}", 
            key.backend, key.gpu, key.m, key.n, key.k, key.dtype, epi_str,
            key.env_version, key.arch, key.op_fingerprint.as_deref().unwrap_or("none"))
    }

    pub fn get(&self, key: &CacheKey) -> Option<PipelineConfig> {
        let s_key = Self::make_key(key);
        self.entries.get(&s_key).cloned()
    }

    pub fn set(&mut self, key: CacheKey, config: PipelineConfig) {
        let s_key = Self::make_key(&key);
        self.entries.insert(s_key, config);
        self.save();
    }

    fn save(&self) {
        if let Ok(content) = serde_json::to_string_pretty(&self.entries) {
            if let Err(e) = fs::write(&self.file_path, content) {
                eprintln!("[Tracea] ⚠️ Failed to save cache: {:?}", e);
            }
        }
    }
}
