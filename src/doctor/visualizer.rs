use crate::runtime::manager::{RuntimeManager, BufferId};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ErrorReport {
    pub max_abs_error: f32,
    pub mean_squared_error: f32,
    pub sample_size: usize,
    pub heatmap: Option<String>,
    pub causal_analysis: Option<String>,
}

use crate::policy::types::GraphTopology;
use crate::runtime::plan::{ExecutionPlan, ExecutionStep};

pub struct Visualizer {
    runtime: std::sync::Arc<RuntimeManager>,
}

impl Visualizer {
    pub fn new(runtime: std::sync::Arc<RuntimeManager>) -> Self {
        Self { runtime }
    }

    pub fn compare_tensors(&self, b1: BufferId, b2: BufferId, size: usize) -> Result<ErrorReport, String> {

        // Need to read as f32. Current runtime might read as u8. 
        // I'll assume f32 for now and cast.
        let mut u8_data1 = vec![0u8; size * 4];
        let mut u8_data2 = vec![0u8; size * 4];

        self.runtime.copy_from_device(b1, &mut u8_data1)?;
        self.runtime.copy_from_device(b2, &mut u8_data2)?;

        let d1: &[f32] = unsafe { std::slice::from_raw_parts(u8_data1.as_ptr() as *const f32, size) };
        let d2: &[f32] = unsafe { std::slice::from_raw_parts(u8_data2.as_ptr() as *const f32, size) };

        let mut max_abs_err = 0.0f32;
        let mut sum_sq_err = 0.0f64;

        for i in 0..size {
            let diff = (d1[i] - d2[i]).abs();
            if diff > max_abs_err { max_abs_err = diff; }
            sum_sq_err += (diff as f64) * (diff as f64);
        }

        let mse = (sum_sq_err / (size as f64)) as f32;

        let causal_analysis = self.analyze_causes(max_abs_err, mse);

        let heatmap = if size <= 1024 {
            Some(self.generate_ascii_heatmap(d1, d2, size))
        } else {
            None
        };

        Ok(ErrorReport {
            max_abs_error: max_abs_err,
            mean_squared_error: mse,
            sample_size: size,
            heatmap,
            causal_analysis: Some(causal_analysis),
        })
    }

    fn analyze_causes(&self, mae: f32, mse: f32) -> String {
        if mae < 1e-6 {
            "No significant error detected. Bit-wise or epsilon-level parity confirmed.".to_string()
        } else if mae < 1e-3 {
            "Minor accumulation error. Likely due to different FMA ordering or atomic float performance optimizations.".to_string()
        } else if mae < 1e-1 {
            "Moderated discrepancy. Possible precision mismatch (e.g. TF32 vs FP32) or non-deterministic rounding in hardware.".to_string()
        } else {
            "CRITICAL: Structural divergence. Checkerboard patterns or high bias suggest incorrect indexing or logic mismatch in the emitter.".to_string()
        }
    }

    fn generate_ascii_heatmap(&self, d1: &[f32], d2: &[f32], size: usize) -> String {
        let mut s = String::new();
        let width = 32;
        for i in 0..size {
            if i > 0 && i % width == 0 { s.push('\n'); }
            let diff = (d1[i] - d2[i]).abs();
            let char = if diff < 1e-6 { '.' }
                       else if diff < 1e-4 { '*' }
                       else if diff < 1e-2 { 'x' }
                       else { '#' };
            s.push(char);
        }
        s
    }

    pub fn export_mermaid(&self, topology: &GraphTopology, path: &str) {
        let mut output = String::from("graph TD\n");
        for op in &topology.operators {
            let label = format!("node_{}[\"{}: {}\"]", op.op_id(), op.op_id(), op.name());
            output.push_str(&format!("  {}\n", label));
        }
        for (prod, cons) in &topology.dependencies {
            output.push_str(&format!("  node_{} --> node_{}\n", prod, cons));
        }
        let _ = std::fs::write(path, output);
        println!("[Doctor] 🗺 Saved graph visualization to {}", path);
    }

    pub fn export_execution_plan_mermaid(&self, plan: &ExecutionPlan, path: &str) {
        let mut output = String::from("graph LR\n");
        output.push_str("  subgraph Arena\n");
        output.push_str(&format!("    size[\"Total Size: {} bytes\"]\n", plan.arena_size));
        output.push_str("  end\n");

        for (i, step) in plan.steps.iter().enumerate() {
            match step {
                ExecutionStep::LaunchKernel { kernel_id, .. } => {
                    output.push_str(&format!("  step_{}[\"Step {}: Kernel {:?}\"]\n", i, i, kernel_id));
                }
                ExecutionStep::Memcpy { src_offset, dst_offset, size } => {
                    output.push_str(&format!("  step_{}[\"Step {}: Memcpy {}->{} ({} bytes)\"]\n", i, i, src_offset, dst_offset, size));
                }
            }
            if i > 0 {
                output.push_str(&format!("  step_{} --> step_{}\n", i - 1, i));
            }
        }
        let _ = std::fs::write(path, output);
        println!("[Doctor] 🗺 Saved execution plan visualization to {}", path);
    }
}
