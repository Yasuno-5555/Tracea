use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use serde::{Serialize, Deserialize};
use crate::PipelineConfig;

#[derive(Serialize, Deserialize)]
pub struct ConfigCache {
    // Key: "DeviceName:M:N:K"
    pub entries: HashMap<String, PipelineConfig>,
}

impl ConfigCache {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    pub fn load(path: &str) -> Self {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Self::new(),
        };
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap_or(0);
        serde_json::from_str(&content).unwrap_or_else(|_| Self::new())
    }

    pub fn save(&self, path: &str) {
        let content = serde_json::to_string_pretty(self).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    pub fn get(&self, device: &str, m: u32, n: u32, k: u32) -> Option<&PipelineConfig> {
        let key = format!("{}:{}:{}:{}", device, m, n, k);
        self.entries.get(&key)
    }

    pub fn insert(&mut self, device: &str, m: u32, n: u32, k: u32, config: PipelineConfig) {
        let key = format!("{}:{}:{}:{}", device, m, n, k);
        self.entries.insert(key, config);
    }
}
