use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::PipelineConfig;
use crate::runtime::manager::DeviceBackend;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HistoryEntry {
    pub backend: DeviceBackend,
    pub config: PipelineConfig,
    pub tflops: f32,
    pub timestamp: u64,
    pub explanation: String,
}

#[derive(Debug)]
pub struct TuningHistory {
    pub entries: HashMap<String, HistoryEntry>, // Key: "HardwareId:OpFingerprint:Precision"
    path: String,
}

impl TuningHistory {
    pub fn load_or_create(path: &str) -> Self {
        if let Ok(data) = std::fs::read_to_string(path) {
            if let Ok(entries) = serde_json::from_str(&data) {
                return Self { entries, path: path.to_string() };
            }
        }
        Self { entries: HashMap::new(), path: path.to_string() }
    }

    pub fn save(&self) -> Result<(), String> {
        let data = serde_json::to_string_pretty(&self.entries).map_err(|e| e.to_string())?;
        std::fs::write(&self.path, data).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn record(&mut self, key: String, entry: HistoryEntry) {
        self.entries.insert(key, entry);
        let _ = self.save();
    }

    pub fn query(&self, key: &str) -> Option<&HistoryEntry> {
        self.entries.get(key)
    }

    pub fn generate_explanation(backend: DeviceBackend, tflops: f32, reason: &str) -> String {
        format!(
            "Selected {:?} because it achieved {:.2} TFLOPS. Reason: {}",
            backend, tflops, reason
        )
    }
}
