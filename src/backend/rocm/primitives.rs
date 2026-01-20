
pub struct RocmPrimitives;

impl RocmPrimitives {
    pub fn all_definitions() -> String {
        let mut code = String::new();
        code.push_str(Self::mfma_defs().as_str());
        code.push_str(Self::lds_swizzle_defs().as_str());
        code
    }

    fn mfma_defs() -> String {
        r#"
typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef float v16f32 __attribute__((ext_vector_type(16)));
typedef float v32f32 __attribute__((ext_vector_type(32)));
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v32f32 mfma_f32_32x32x8f16(v4f16 a, v4f16 b, v32f32 acc) {
    return __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, acc, 0, 0, 0);
}

__device__ __forceinline__ v16f32 mfma_f32_16x16x16f16(v4f16 a, v4f16 b, v16f32 acc) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
}
"#.to_string()
    }

    fn lds_swizzle_defs() -> String {
        r#"
__device__ __forceinline__ float ds_swizzle(float src, int pattern) {
    return __builtin_amdgcn_ds_swizzle(src, pattern);
}
"#.to_string()
    }
}
