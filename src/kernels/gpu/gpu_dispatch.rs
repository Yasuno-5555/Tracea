use crate::core::config::PipelineConfig;
use crate::runtime::manager::{DeviceBackend, KernelArg};
use std::ffi::c_void;

pub fn dispatch_gpu_gemm(
    config: &PipelineConfig,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: i32, n: i32, k: i32,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    stream: *mut c_void,
) -> bool {
    #[cfg(feature = "cpp")]
    {
        use super::cuda_bindings::*;
        use half::f16;

        // Map configuration to pre-compiled templates
        return match (config.num_stages, config.swizzle_mode, config.barrier_mode, config.m_tile, config.n_tile, config.k_tile) {
            (2, crate::core::config::SwizzleMode::None, crate::core::config::BarrierMode::None, 128, 128, 32) => {
                unsafe {
                    launch_gemm_v3_s2_sw0_b0_m128_n128_k32(
                        a_ptr as *const f16,
                        b_ptr as *const f16,
                        c_ptr as *mut f32,
                        m, n, k,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem,
                        stream
                    );
                }
                true
            }
            (3, crate::core::config::SwizzleMode::Xor4, crate::core::config::BarrierMode::ProducerConsumer, 128, 128, 32) => {
                unsafe {
                    launch_gemm_v3_s3_sw1_b1_m128_n128_k32(
                        a_ptr as *const f16,
                        b_ptr as *const f16,
                        c_ptr as *mut f32,
                        m, n, k,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem,
                        stream
                    );
                }
                true
            }
            _ => false, // No specialized template found, fallback to NVRTC
        };
    }
    #[cfg(not(feature = "cpp"))]
    {
        let _ = (config, a_ptr, b_ptr, c_ptr, m, n, k, grid, block, smem, stream);
        false
    }
}
