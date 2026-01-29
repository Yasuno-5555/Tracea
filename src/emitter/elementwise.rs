use crate::core::op::ElementwiseType;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};

pub fn generate_elementwise(ir: &UnifiedOpIR) -> String {
    let (op_type, n) = match &ir.op_type {
        UnifiedOpType::Elementwise { op_type, n } => (op_type, *n),
        _ => panic!("Invalid OpType for Elementwise Generator"),
    };

    let kernel_name = match op_type {
        ElementwiseType::Add => "elementwise_add",
        ElementwiseType::Mul => "elementwise_mul",
        ElementwiseType::Relu => "elementwise_relu",
        ElementwiseType::Gelu => "elementwise_gelu",
        ElementwiseType::Sigmoid => "elementwise_sigmoid",
        ElementwiseType::Tanh => "elementwise_tanh",
    };

    let params = match op_type {
        ElementwiseType::Add | ElementwiseType::Mul => 
            "const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n",
        _ => "const float* __restrict__ inp, float* __restrict__ out, int n",
    };

    let core_logic = match op_type {
        ElementwiseType::Add => "c[idx] = a[idx] + b[idx];",
        ElementwiseType::Mul => "c[idx] = a[idx] * b[idx];",
        ElementwiseType::Relu => "out[idx] = fmaxf(inp[idx], 0.0f);",
        ElementwiseType::Gelu => {
            "float x = inp[idx];
             out[idx] = 0.5f * x * (1.0f + erff(x * 0.70710678f));"
        },
        ElementwiseType::Sigmoid => "out[idx] = 1.0f / (1.0f + expf(-inp[idx]));",
        ElementwiseType::Tanh => "out[idx] = tanhf(inp[idx]);",
    };

    format!(r#"

extern "C" __global__ void {kernel_name}({params}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        {core_logic}
    }}
}}
"#, kernel_name=kernel_name, params=params, core_logic=core_logic)
}
