use crate::core::op::ElementwiseType;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use crate::runtime::manager::DeviceBackend;

pub fn generate_elementwise(ir: &UnifiedOpIR, backend: DeviceBackend) -> String {
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

    match backend {
        DeviceBackend::Cuda | DeviceBackend::Rocm => generate_cuda_elementwise(kernel_name, op_type, n),
        DeviceBackend::Metal => generate_metal_elementwise(kernel_name, op_type, n),
        _ => format!("// Elementwise not implemented for {:?}", backend),
    }
}

fn generate_cuda_elementwise(kernel_name: &str, op_type: &ElementwiseType, n: usize) -> String {
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

fn generate_metal_elementwise(kernel_name: &str, op_type: &ElementwiseType, n: usize) -> String {
    let params = match op_type {
        ElementwiseType::Add | ElementwiseType::Mul => 
            "device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], constant int& n_val [[buffer(3)]], device const uint* l1_map [[buffer(4)]], device const TileMetadata* l2_table [[buffer(5)]]",
        _ => "device const float* inp [[buffer(0)]], device float* out [[buffer(1)]], constant int& n_val [[buffer(2)]], device const uint* l1_map [[buffer(3)]], device const TileMetadata* l2_table [[buffer(4)]]",
    };

    let core_logic = match op_type {
        ElementwiseType::Add => "c[idx] = a[idx] + b[idx];",
        ElementwiseType::Mul => "c[idx] = a[idx] * b[idx];",
        ElementwiseType::Relu => "out[idx] = max(inp[idx], 0.0f);",
        ElementwiseType::Gelu => {
            "float x = inp[idx];
             out[idx] = 0.5f * x * (1.0f + erf(x * 0.70710678f));"
        },
        ElementwiseType::Sigmoid => "out[idx] = 1.0f / (1.0f + exp(-inp[idx]));",
        ElementwiseType::Tanh => "out[idx] = tanh(inp[idx]);",
    };

    format!(r#"
#include <metal_stdlib>
using namespace metal;

struct TileMetadata {{
    uint region_m, region_n, k_start, k_end, role;
}};

kernel void {kernel_name}(
    {params},
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {{
    uint logical_id = l1_map[bid];
    TileMetadata tile = l2_table[logical_id];
    
    // For 1D elementwise, region_m/n represents the element offset
    // Each threadgroup handles 1024 elements (consistent with TTGBuilder)
    uint group_offset = tile.region_m * 1024; 
    uint idx = group_offset + tid;

    if (idx < (uint)n_val) {{
        {core_logic}
    }}
}}
"#, kernel_name=kernel_name, params=params, core_logic=core_logic)
}
