use crate::policy::types::GraphTopology;
use std::fs;
use std::path::Path;

pub struct ModelLoader;

impl ModelLoader {
    /// Loads a graph topology from a JSON file.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<GraphTopology, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))
    }

    /// Saves a graph topology to a JSON file (useful for debugging/exporting).
    pub fn save_json<P: AsRef<Path>>(graph: &GraphTopology, path: P) -> Result<(), String> {
        let content = serde_json::to_string_pretty(graph).map_err(|e| format!("Failed to serialize JSON: {}", e))?;
        fs::write(path, content).map_err(|e| format!("Failed to write file: {}", e))
    }
}
