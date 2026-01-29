pub struct VulkanPrimitives;

impl VulkanPrimitives {
    pub fn get_subgroup_defs() -> String {
        r#"
// Vulkan Subgroup Primitives
// Requires GL_KHR_shader_subgroup_arithmetic and shuffle

float warpShuffleXor(float val, uint mask) {
    return subgroupShuffleXor(val, mask);
}

float warpAdd(float val) {
    return subgroupAdd(val);
}

bool warpAll(bool val) {
    return subgroupAll(val);
}

// Memory Layout Primitives
uint smem_swizzle(uint addr) {
    uint sw = (addr >> 4) & 0x7u;
    return addr ^ (sw << 7);
}
"#.to_string()
    }
}
