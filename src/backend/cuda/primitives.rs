
pub struct CudaPrimitives;

impl CudaPrimitives {
    /// Generates cp.async instruction (Ampere/Hopper compatible)
    /// Currently defaults to Ampere cp.async.ca.shared.global
    pub fn cp_async(_dst: &str, _src: &str, _size: usize) -> String {
        // We return the C++ wrapper code or the inline asm? 
        // The Emitter currently emits a C++ kernel. 
        // So this should probably return the C++ helper function DEFINITION, 
        // or the call itself?
        // The implementation plan says "Handles architecture-specific syntax".
        // Let's provide the C++ helper definitions here.
        
        format!(r#"
__device__ __forceinline__ void cp_async_ampere(void* smem_ptr, const void* global_ptr, uint32_t size) {{
    // In 64-bit mode, use 64-bit generic pointer (or shared window pointer)
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;"
        : 
        : "l"(smem_ptr), "l"(global_ptr)
    );
}}
"#)
    }
    
    // Actually, simply returning strings might be messy if we want to call them with different args.
    // Maybe we should return the DEFINITIONS as a block, and then have helper methods for calls?
    // Let's stick to generating the DEFINITIONS first, as fa2.rs embeds them at the top.
    
    pub fn all_definitions() -> String {
        let mut code = String::new();
        code.push_str(Self::ldmatrix_def().as_str());
        code.push_str(Self::mma_def().as_str());
        code.push_str(Self::mbarrier_defs().as_str());
        code.push_str(Self::cp_async_def().as_str());
        code
    }

    fn ldmatrix_def() -> String {
        r#"
__device__ __forceinline__ void ldmatrix_m8n8_x4(uint32_t* regs, void* smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "l"(smem_ptr)
    );
}
"#.to_string()
    }

    fn mma_def() -> String {
        r#"
__device__ __forceinline__ void mma_m16n8k16_f16(float* acc, uint32_t* a, uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}
"#.to_string()
    }

    fn mbarrier_defs() -> String {
        r#"
__device__ __forceinline__ void mbarrier_init(uint64_t* mbarrier_ptr, uint32_t expected_count) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(smem_addr), "r"(expected_count));
}

__device__ __forceinline__ void mbarrier_invalidate(uint64_t* mbarrier_ptr) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.inval.shared.b64 [%0];" : : "r"(smem_addr));
}

__device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t* mbarrier_ptr) {
    uint64_t state;
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(smem_addr));
    return state;
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* mbarrier_ptr, uint32_t tx_bytes) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;" : : "r"(smem_addr), "r"(tx_bytes));
}

// Hopper-only primitive (sm_90+). Commented for sm_80 compatibility.
/*
__device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(uint64_t* mbarrier_ptr, uint32_t tx_bytes) {
    uint64_t state;
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;" : "=l"(state) : "r"(smem_addr), "r"(tx_bytes));
    return state;
}
*/

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbarrier_ptr, uint64_t phase) {
    uint32_t mbarrier_addr = (uint32_t)__cvta_generic_to_shared(mbarrier_ptr);
    uint64_t state = (phase << 63);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  wait_loop:\n\t"
        "  mbarrier.test_wait.shared.b64 p, [%0], %1;\n\t"
        "  @!p bra wait_loop;\n\t"
        "}\n\t"
        : 
        : "r"(mbarrier_addr), "l"(state)
    );
}
// Note: PTX 'mbarrier.wait' is Hopper+. For Ampere we must use test_wait loop or similar.
// I will use a tighter assembly loop.
"#.to_string()
    }

    fn cp_async_def() -> String {
        r#"
__device__ __forceinline__ void cp_async_ampere(void* dst, const void* src, bool p) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(dst);
    if (p) {
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p, %2, 0; @p cp.async.ca.shared.global [%0], [%1], 16; }\n"
            : : "r"(smem_addr), "l"(src), "r"((int)p)
        );
    } else {
        *((uint4*)dst) = make_uint4(0, 0, 0, 0);
    }
}

// Pipeline Management
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// Helper for XOR Swizzling (128B aligned safe)
__device__ __forceinline__ uint32_t smem_swizzle(uint32_t addr) {
    // bits 4,5,6 XORed with bits 7,8,9
    uint32_t sw = (addr >> 4) & 0x7;
    return addr ^ (sw << 7);
}

__device__ __forceinline__ void* smem_swizzle_ptr(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t sw_addr = smem_swizzle(addr);
    return __cvta_shared_to_generic((size_t)sw_addr);
}
"#.to_string()
    }
}
