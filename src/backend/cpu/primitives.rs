
pub struct CpuPrimitives;

impl CpuPrimitives {
    pub fn all_definitions() -> String {
        let mut code = String::new();
        code.push_str(Self::simd_defs().as_str());
        code
    }

    fn simd_defs() -> String {
        r#"
#if defined(__AVX512F__)
#include <immintrin.h>
// AVX512 Helpers
#elif defined(__ARM_NEON)
#include <arm_neon.h>
// NEON Helpers
#endif
"#.to_string()
    }
}
