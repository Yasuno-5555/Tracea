
pub struct MetalPrimitives;

impl MetalPrimitives {
    pub fn all_definitions() -> String {
        let mut code = String::new();
        code.push_str(Self::simd_group_defs().as_str());
        code.push_str(Self::threadgroup_memory_defs().as_str());
        code
    }

    fn simd_group_defs() -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

template<typename T>
inline T tracea_simd_shuffle(T val, ushort lane) {
    return simd_shuffle(val, lane);
}

template<typename T>
inline T tracea_simd_broadcast(T val, ushort lane) {
    return simd_broadcast(val, lane);
}
"#.to_string()
    }

    fn threadgroup_memory_defs() -> String {
        r#"
inline void tracea_barrier() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"#.to_string()
    }
}
