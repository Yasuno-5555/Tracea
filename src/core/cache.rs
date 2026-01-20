use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn get_cache_path() -> PathBuf {
    let mut path = if let Ok(home) = std::env::var("USERPROFILE") {
        PathBuf::from(home)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
    } else {
        PathBuf::from(".")
    };
    
    path.push(".tracea");
    if !path.exists() {
        let _ = fs::create_dir_all(&path);
    }
    path.push("tuning_cache.json");
    path
}

#[derive(Serialize, Deserialize, Default)]
struct TuningCacheFile {
    entries: HashMap<String, Value>,
}

pub fn load_cache<T: serde::de::DeserializeOwned>(key: &str) -> Option<T> {
    let path = get_cache_path();
    if !path.exists() {
        return None;
    }

    let content = fs::read_to_string(path).ok()?;
    let cache: TuningCacheFile = serde_json::from_str(&content).ok()?;
    
    if let Some(val) = cache.entries.get(key) {
        serde_json::from_value(val.clone()).ok()
    } else {
        None
    }
}

pub fn save_cache<T: Serialize>(key: &str, config: &T) {
    let path = get_cache_path();
    
    // Read existing
    let mut cache = if path.exists() {
         fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<TuningCacheFile>(&s).ok())
            .unwrap_or_default()
    } else {
        TuningCacheFile::default()
    };

    if let Ok(val) = serde_json::to_value(config) {
         cache.entries.insert(key.to_string(), val);
         if let Ok(s) = serde_json::to_string_pretty(&cache) {
             let _ = fs::write(path, s);
         }
    }
}
