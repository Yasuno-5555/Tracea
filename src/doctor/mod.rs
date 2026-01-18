pub mod profiler;
pub mod capabilities;
pub mod registry;
pub mod engine;
pub mod telemetry;

pub use profiler::get_capabilities;
pub use capabilities::{TraceaCapabilities, BackendCapabilities};
pub use registry::{KernelVariant, Requirement, Preference, BackendKind};
pub use engine::{KernelRequestContext, Decision, DecisionReason, PrecisionPolicy, CompileStrategy, FallbackEntry};

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::ffi::c_void;

pub fn plan_kernel(kernel_id: &str, request: KernelRequestContext) -> Decision {
    let caps = get_capabilities();
    engine::select_variant(&caps, kernel_id, request)
}

pub fn load_kernel(device: Arc<CudaDevice>, decision: &Decision) -> Result<CudaFunction, String> {
    let variant_id = decision.selected_variant.ok_or("No variant selected in decision")?;
    let variants = registry::get_variants_for_id(variant_id); 
    let variant = variants.get(0).ok_or_else(|| format!("Variant {} not found in registry", variant_id))?;

    match variant.backend {
        BackendKind::Cuda => {
            // Logic for loading CUDA kernels (PTX or Cubin)
            // Simplified check: usually we decide between JIT or prebuilt based on compile_strategy from decision
            match decision.compile_strategy {
                 CompileStrategy::Precompiled => {
                     let path = format!("E:/Projects/Tracea/prebuilt/{}.cubin", variant_id);
                     if std::path::Path::new(&path).exists() {
                         let module_name = format!("doctor_mod_{}", variant_id);
                         let module_name_static = Box::leak(module_name.into_boxed_str());
                         let kernel_name_static = Box::leak(variant.kernel_id.to_string().into_boxed_str());
                         
                         device.load_ptx(Ptx::from_file(path), module_name_static, &[kernel_name_static])
                             .map_err(|e| format!("Load CUBIN via Driver error: {:?}", e))?;
                         
                         device.get_func(module_name_static, kernel_name_static)
                             .ok_or_else(|| format!("Function {} not found in loaded module", variant.kernel_id))
                     } else {
                         Err(format!("CUBIN file not found at {}", path))
                     }
                 },
                 CompileStrategy::JIT | CompileStrategy::AOT => {
                     Err("JIT compilation not yet implemented in Doctor for CUDA".to_string())
                 }
            }
        },
        BackendKind::Rocm | BackendKind::Metal | BackendKind::Cpu => {
            Err(format!("Backend {:?} not supported on CUDA device loader", variant.backend))
        }
    }
}

pub fn explain_decision(decision: &Decision) -> String {
    format!("{:?}", decision.reason)
}

pub fn list_available_variants(kernel_id: &str) -> Vec<&'static str> {
    registry::get_variants_for(kernel_id)
        .iter()
        .map(|v| v.id)
        .collect()
}
