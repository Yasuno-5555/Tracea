use crate::core::device::DeviceProfile;
use crate::policy::types::OperatorTopology;

use crate::core::config::GemmVariant;

pub struct CostModel;

impl CostModel {
    /// Estimates execution latency in microseconds for GEMM with specific variant
    pub fn estimate_gemm_latency(op: &OperatorTopology, variant: GemmVariant, device: &DeviceProfile) -> f32 {
        let (flops, bytes) = Self::get_op_stats(op);
        
        // Efficiency Factor based on Variant (Empirical for M1/Tracea)
        let (compute_efficiency, mem_efficiency) = match variant {
            GemmVariant::Naive => (0.10, 0.40),
            GemmVariant::Tiled => (0.50, 0.70),
            GemmVariant::Simd => (0.85, 0.85),
        };
        
        let peak_flops = if variant == GemmVariant::Simd && device.has_tensor_cores {
            (device.max_threads_per_block as f32) * 1500.0 * 1e6 * 256.0
        } else {
            (device.max_threads_per_block as f32) * 1500.0 * 1e6 * 32.0
        };
        
        let compute_time_s = flops / (peak_flops * compute_efficiency);
        let bandwidth = if device.has_tensor_cores { 1.5e12 } else { 200e9 };
        let mem_time_s = bytes / (bandwidth * mem_efficiency);

        f32::max(compute_time_s, mem_time_s) * 1e6 
    }

    /// Estimates execution latency in microseconds for Attention with specific variant
    pub fn estimate_attention_latency(op: &OperatorTopology, variant: crate::core::config::AttentionVariant, device: &DeviceProfile) -> f32 {
        let (flops, bytes) = Self::get_op_stats(op);
        
        let efficiency = match variant {
            crate::core::config::AttentionVariant::Naive => 0.05,
            crate::core::config::AttentionVariant::SimdQK => 0.15,
            crate::core::config::AttentionVariant::FlashV2 => 0.40,
            _ => 0.05,
        };

        let peak_flops = (device.max_threads_per_block as f32) * 1500.0 * 1e6 * 32.0;
        let compute_time_s = flops / (peak_flops * efficiency);
        let bandwidth = if device.has_tensor_cores { 1.5e12 } else { 200e9 };
        let mem_time_s = bytes / bandwidth;

        f32::max(compute_time_s, mem_time_s) * 1e6
    }

    /// Estimates execution latency in microseconds (Generic)
    pub fn estimate_latency_us(op: &OperatorTopology, device: &DeviceProfile) -> f32 {
         // Fallback to "best effort" or average
         let (flops, _) = Self::get_op_stats(op);
         let peak_flops = (device.max_threads_per_block as f32) * 1500.0 * 1e6 * 32.0;
         (flops / (peak_flops * 0.4)) * 1e6
    }

    pub fn estimate_memory_bytes(op: &OperatorTopology) -> usize {
        let (_, bytes) = Self::get_op_stats(op);
        bytes as usize
    }

    fn get_op_stats(op: &OperatorTopology) -> (f32, f32) {
        match op {
            OperatorTopology::Gemm { m, n, k, batch, .. } => {
                let flops = 2.0 * (*m as f32) * (*n as f32) * (*k as f32) * (*batch as f32);
                let bytes = 4.0 * ((*m * *k) + (*k * *n) + (*m * *n)) as f32 * (*batch as f32); // F32
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
             OperatorTopology::Relu { n, .. } => {
                 ( *n as f32, 4.0 * 2.0 * (*n as f32) )
             },
             OperatorTopology::Elementwise { n, .. } => {
                 ( *n as f32, 4.0 * 3.0 * (*n as f32) ) // Assuming binary op
             },
             OperatorTopology::BatchNorm { n, c, h, w, .. } => {
                 let count = (*n as f32) * (*c as f32) * (*h as f32) * (*w as f32);
                 ( 5.0 * count, 4.0 * 2.0 * count ) // Very rough: mean/var/gamma/beta etc handled per channel
             },
             OperatorTopology::GlobalAveragePool { n, c, h, w, .. } => {
                 let count = (*n as f32) * (*c as f32) * (*h as f32) * (*w as f32);
                 ( count, 4.0 * (count + (*n * *c) as f32) )
             },
             OperatorTopology::Linear { batch, m, n, k, .. } => {
                 let flops = 2.0 * (*m as f32) * (*n as f32) * (*k as f32) * (*batch as f32);
                 let bytes = 4.0 * ((*m * *k) + (*k * *n) + (*m * *n)) as f32 * (*batch as f32);
                 (flops, bytes)
             },
             _ => (0.0, 0.0),
         }
     }
 }
