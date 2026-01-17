use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use crate::core::config::PipelineConfig;

#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Debug)]
pub struct CacheKey {
    pub gpu: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub dtype: String,
    // Environment Invalidation
    pub cuda_version: String,
    pub driver_version: String,
    pub sm_arch: u32,
}

#[derive(Serialize, Deserialize)]
pub struct TuningCache {
    entries: HashMap<String, PipelineConfig>,
    #[serde(skip)]
    file_path: PathBuf,
}

impl TuningCache {
    pub fn new() -> Self {
        let mut path = PathBuf::from(".tracea");
        if !path.exists() {
            let _ = fs::create_dir_all(&path);
        }
        path.push("tuning_cache.json");
        
        if path.exists() {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(entries) = serde_json::from_str::<HashMap<String, PipelineConfig>>(&content) {
                   return Self { entries, file_path: path };
                }
            }
        }
        
        Self {
            entries: HashMap::new(),
            file_path: path,
        }
    }

    fn make_key(key: &CacheKey) -> String {
        format!("{}:{}:{}:{}:{}:{}:{}:{}", key.gpu, key.m, key.n, key.k, key.dtype, key.cuda_version, key.driver_version, key.sm_arch)
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
