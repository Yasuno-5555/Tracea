use crate::core::device::DeviceProfile;
use crate::policy::types::OperatorTopology;

use crate::core::config::GemmVariant;

pub struct CostModel;

impl CostModel {
    /// Estimates execution latency in microseconds for GEMM with specific variant
    pub fn estimate_gemm_latency(op: &OperatorTopology, variant: GemmVariant, device: &DeviceProfile) -> f32 {
        let (flops, bytes) = Self::get_op_stats(op);
        
        // Efficiency Factor based on Variant
        let efficiency = match variant {
            GemmVariant::Naive => 0.05, // Very slow
            GemmVariant::Tiled => 0.40, // Decent
            GemmVariant::Simd => 0.85,  // Near peak
        };
        
        // 1. Compute Latency: FLOPS / (Peak_FLOPS * efficiency)
        let num_cores = device.max_threads_per_block * 4; 
        let clock_mhz = 1500.0;
        let ops_per_clock = if device.has_tensor_cores { 256.0 } else { 32.0 };
        // If variant is Naive, we don't use Tensor Cores even if available
        let effective_ops_per_clock = if variant == GemmVariant::Simd { ops_per_clock } else { 32.0 };
        
        let peak_flops_per_sec = (num_cores as f32) * clock_mhz * 1e6 * effective_ops_per_clock;
        
        let compute_time_s = if peak_flops_per_sec > 0.0 {
            flops / (peak_flops_per_sec * efficiency)
        } else {
            0.0
        };

        // 2. Memory Latency
        let bandwidth_bytes_sec = if device.has_tensor_cores { 1.5e12 } else { 200e9 };
        let mem_time_s = bytes / bandwidth_bytes_sec;

        let latency_s = f32::max(compute_time_s, mem_time_s);
        latency_s * 1e6 
    }

    /// Estimates execution latency in microseconds (Generic)
    pub fn estimate_latency_us(op: &OperatorTopology, device: &DeviceProfile) -> f32 {
         // Fallback to "best effort" or average
         let (flops, bytes) = Self::get_op_stats(op);
         // ... reuse logic ...
         0.0 // Placeholder
    }

    pub fn estimate_memory_bytes(op: &OperatorTopology) -> usize {
        let (_, bytes) = Self::get_op_stats(op);
        bytes as usize
    }

    fn get_op_stats(op: &OperatorTopology) -> (f32, f32) {
        match op {
            OperatorTopology::Gemm { m, n, k, .. } => {
                let flops = 2.0 * (*m as f32) * (*n as f32) * (*k as f32);
                let bytes = 4.0 * ((*m * *k) + (*k * *n) + (*m * *n)) as f32; // F32
                (flops, bytes)
            },
            OperatorTopology::Attention { b, s, h, d, .. } => {
                // Simplified Attention: 4 * b * h * s^2 * d
                let flops = 4.0 * (*b as f32) * (*h as f32) * (*s as f32).powi(2) * (*d as f32);
                // IO: Q,K,V + Output
                let bytes = 4.0 * 4.0 * (*b * *s * *h * *d) as f32; 
                (flops, bytes)
            },
             OperatorTopology::Conv2d { n, c, h, w, k, .. } => {
                 // Forward pass flops
                 // Output size roughly H*W
                 // Flops: 2 * N * K * C * 3*3 * H * W (assuming 3x3 kernel)
                 let flops = 2.0 * (*n as f32) * (*k as f32) * (*c as f32) * 9.0 * (*h as f32) * (*w as f32);
                 let bytes = 4.0 * (*n * *c * *h * *w + *n * *k * *h * *w) as f32; 
                 (flops, bytes)
             },
             _ => (0.0, 0.0), // Elementwise negligible
        }
    }
}
