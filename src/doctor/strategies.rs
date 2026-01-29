use super::diagnosis::{JitResultInfo, KernelLaunchInfo};

pub trait DiagnosticStrategy {
    fn analyze_jit(&self, info: &JitResultInfo) -> Option<String>;
    fn analyze_launch(&self, info: &KernelLaunchInfo) -> Option<String>;
}

pub struct GenericStrategy;
pub struct GemmStrategy;
pub struct ElementwiseStrategy;

impl DiagnosticStrategy for GenericStrategy {
    fn analyze_jit(&self, info: &JitResultInfo) -> Option<String> {
        let mut reasons = Vec::new();
        if info.stderr.contains("syntax error") {
            reasons.push("Syntax error in generated code.");
        }
        if info.stderr.contains("undefined") {
            reasons.push("Undefined identifier (missing header?");
        }
        if reasons.is_empty() { None } else { Some(reasons.join(" ")) }
    }

    fn analyze_launch(&self, info: &KernelLaunchInfo) -> Option<String> {
        if let Some(err) = &info.last_runtime_error {
            if err.contains("INVALID_VALUE") {
                return Some("Check grid/block dims or argument alignment.".to_string());
            }
        }
        None
    }
}

impl DiagnosticStrategy for GemmStrategy {
    fn analyze_jit(&self, info: &JitResultInfo) -> Option<String> {
        if info.stderr.contains("cp.async") {
             return Some("Pipeline intrinsics failed. Check Compute Capability >= 8.0.".to_string());
        }
        None
    }

    fn analyze_launch(&self, info: &KernelLaunchInfo) -> Option<String> {
        if let Some(err) = &info.last_runtime_error {
            if err.contains("ILLEGAL_ADDRESS") {
                return Some("Shared Memory Padding likely insufficient (+8 required).".to_string());
            }
        }
        None
    }
}

impl DiagnosticStrategy for ElementwiseStrategy {
    fn analyze_jit(&self, info: &JitResultInfo) -> Option<String> {
        if info.stderr.contains("math.h") {
             return Some("NVRTC cannot find math.h. Remove include.".to_string());
        }
        None
    }

    fn analyze_launch(&self, info: &KernelLaunchInfo) -> Option<String> {
        // Elementwise is usually simple, but check for alignment
         if let Some(err) = &info.last_runtime_error {
            if err.contains("MISALIGNED") {
                return Some("Vectorized load/store misalignment.".to_string());
            }
        }
        None
    }
}

pub fn get_strategy(kernel_name: &str) -> Box<dyn DiagnosticStrategy> {
    if kernel_name.contains("gemm") {
        Box::new(GemmStrategy)
    } else if kernel_name.contains("elementwise") {
        Box::new(ElementwiseStrategy)
    } else {
        Box::new(GenericStrategy)
    }
}
