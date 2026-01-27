use serde::{Serialize, Deserialize};

pub mod registry;
pub mod engine;
pub mod capabilities;
pub mod diagnosis;
pub mod profiler;
pub mod telemetry;
pub mod strategies;
pub mod visualizer;

// Re-exports for Diagnostics
pub use diagnosis::{
    Doctor, DoctorConfig, DoctorState,
    DoctorErrorReport, DoctorErrorKind, DoctorArtifacts,
    EnvironmentReport, EnvStatus,
    JitResultInfo, AssemblerResultInfo, ModuleLoadInfo, KernelLaunchInfo
};

// Re-exports for Intelligence/Engine
pub use registry::BackendKind;
pub use engine::{KernelRequestContext, PrecisionPolicy, Decision as PlanDecision};

// Legacy/Facade functions

pub fn plan_kernel(kernel_id: &str, ctx: engine::KernelRequestContext) -> engine::Decision {
    // Determine capabilities (cached or fresh)
    let caps = get_capabilities();
    engine::select_variant(&caps, kernel_id, ctx)
}

pub fn get_capabilities() -> capabilities::TraceaCapabilities {
    profiler::get_capabilities()
}
