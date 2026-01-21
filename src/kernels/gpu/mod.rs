pub mod gpu_dispatch;

#[cfg(feature = "cpp")]
pub mod cuda_bindings {
    use std::ffi::c_void;
    use half::f16;

    extern "C" {
        pub fn launch_gemm_v3_s2_sw0_b0_m128_n128_k32(
            a: *const f16, b: *const f16, c: *mut f32,
            m: i32, n: i32, k: i32,
            gridX: u32, gridY: u32, gridZ: u32,
            blockX: u32, blockY: u32, blockZ: u32,
            smem: u32, stream: *mut c_void
        );

        pub fn launch_gemm_v3_s3_sw1_b1_m128_n128_k32(
            a: *const f16, b: *const f16, c: *mut f32,
            m: i32, n: i32, k: i32,
            gridX: u32, gridY: u32, gridZ: u32,
            blockX: u32, blockY: u32, blockZ: u32,
            smem: u32, stream: *mut c_void
        );
    }
}
