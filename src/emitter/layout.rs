use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutPolicy {
    RowMajor,
    ColumnMajor,
    XorSwizzled,
}

impl LayoutPolicy {
    /// Generates the C++ expression for calculating the shared memory offset
    /// with the given row, column, and stride.
    pub fn get_offset_expr(&self, row: &str, col: &str, stride: &str) -> String {
        match self {
            LayoutPolicy::RowMajor => format!("(({}) * ({}) + ({}))", row, stride, col),
            LayoutPolicy::ColumnMajor => format!("(({}) * ({}) + ({}))", col, stride, row),
            LayoutPolicy::XorSwizzled => {
                // Typical XOR swizzling for 128B (8x float16) aligned access
                // (row * stride + col) ^ (row % 8)
                // This is a simplified version often used in Tensor Core kernels.
                format!("((({}) * ({}) + ({})) ^ (({})) % 8))", row, stride, col, row)
            }
        }
    }

    /// Generates the swizzle function call if needed
    pub fn wrap_swizzle(&self, addr_expr: &str) -> String {
        match self {
            LayoutPolicy::XorSwizzled => format!("smem_swizzle({})", addr_expr),
            _ => addr_expr.to_string(),
        }
    }
}
