use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CudaArch {
    Ampere,
    Ada,
    Hopper,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuArch {
    Avx2,
    Avx512,
    Neon,
    Scalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cuda(CudaArch),
    Cpu(CpuArch),
    Metal,
}

impl Default for Device {
    fn default() -> Self {
        // Default to CPU Scalar if no other info, but usually provided by Runtime
        Device::Cpu(CpuArch::Scalar)
    }
}
