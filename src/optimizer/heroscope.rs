use std::collections::HashMap;
use crate::PipelineConfig;
use crate::optimizer::problem::LayerType;

#[derive(Debug, Clone)]
pub struct HeroScopeV3 {
    entries: HashMap<String, PipelineConfig>,
}

impl HeroScopeV3 {
    pub fn new() -> Self {
        let mut entries = HashMap::new();
        
        // Initial "Known Heros" from v3 spec
        // Format: "LayerType:Arch"
        
        // CPU GEMM Heroes
        let mut cpu_gemm = PipelineConfig::new(1, 128, 128, 256);
        cpu_gemm.k_unroll = 4;
        cpu_gemm.prefetch_distance = 128;
        cpu_gemm.micro_m = 6;
        
        entries.insert("Gemm:GenericCPU".to_string(), cpu_gemm);
        
        // GPU GEMM Heroes (Ampere sm_80/sm_86)
        let mut gpu_gemm = PipelineConfig::new(3, 128, 128, 32);
        gpu_gemm.cp_async_distance = 2; // pipeline_depth = 3
        gpu_gemm.swizzle_mode = crate::core::config::SwizzleMode::Xor4;
        gpu_gemm.k_unroll = 1; // Handled by template usually
        
        entries.insert("Gemm:sm_86".to_string(), gpu_gemm.clone());
        entries.insert("Gemm:sm_80".to_string(), gpu_gemm);
        
        // GPU GEMM fallback for older/unknown architectures
        let mut gpu_default = PipelineConfig::new(2, 128, 128, 32);
        gpu_default.swizzle_mode = crate::core::config::SwizzleMode::None;
        entries.insert("Gemm:sm_70".to_string(), gpu_default);

        Self { entries }
    }

    pub fn get_hero(&self, hw_id: &str, layer: LayerType) -> Option<PipelineConfig> {
        let key = format!("{:?}:{}", layer, hw_id);
        if let Some(hero) = self.entries.get(&key) {
            return Some(hero.clone());
        }

        // Tiered fallback logic
        if hw_id.starts_with("sm_") {
             // Fallback to sm_80 if sm_8X/90/etc missing (optimistic)
             let base_key = format!("{:?}:sm_80", layer);
             if let Some(hero) = self.entries.get(&base_key) {
                 return Some(hero.clone());
             }
        }

        None
    }
}

pub fn get_cpu_id() -> String {
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        let output = Command::new("wmic")
            .args(&["cpu", "get", "name"])
            .output();
        if let Ok(out) = output {
            let s = String::from_utf8_lossy(&out.stdout);
            let lines: Vec<&str> = s.lines().collect();
            if lines.len() >= 2 {
                return lines[1].trim().to_string();
            }
        }
    }
    "GenericCPU".to_string()
}
