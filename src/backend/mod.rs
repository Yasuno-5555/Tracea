/// Abstraction for a compute backend (CUDA, ROCm, Metal, CPU).
/// Provides hardware identification and generic constraints checking.
pub trait Backend {
    /// Returns the unique device identifier (e.g., "sm_86_RTX3070", "gfx1100", "cpu_ryzen9").
    fn device_id(&self) -> String;

    /// Returns the driver version (e.g., "cuda_12.4", "llvm_16").
    fn driver_version(&self) -> String;

    /// Returns the runtime version (e.g., "cuda_rt_12.3").
    fn runtime_version(&self) -> String;

    /// Returns the maximum shared memory per block in bytes.
    fn max_shared_memory(&self) -> usize;

    /// Returns the maximum threads per block.
    fn max_threads_per_block(&self) -> usize;
}

/// Marker trait for Backends that support executing generated CUDA/PTX code.
pub trait CudaBackendExt: Backend {
    // Methods specific to CUDA interaction could go here, 
    // or just be generic methods in the struct implementation.
}

// #[cfg(feature = "cuda")] // Removed
pub mod cuda;

pub mod cpu;
pub mod rocm;
pub mod metal;
#[cfg(feature = "vulkan")]
pub mod vulkan;
