use crate::emitter::traits::UnifiedOpType;

pub fn generate_metal_softmax(dim_size: usize, stride: usize, total_elements: usize) -> String {
    let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();
    format!(r#"
{primitives}
#include <metal_stdlib>
using namespace metal;

struct TileMetadata {{
    uint region_m, region_n, k_start, k_end, role;
}};

kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* l1_map [[buffer(2)]],
    device const TileMetadata* l2_table [[buffer(3)]],
    uint g_idx [[thread_position_in_grid]]
) {{
    uint total_elements = {total_elements};
    uint dim_size = {dim_size};
    
    uint row_idx = g_idx;
    uint base = row_idx * dim_size; 
    if (base >= total_elements) return;

    float max_val = -1e38;
    for (uint i = 0; i < dim_size; i++) {{
        max_val = max(max_val, input[base + i]);
    }}

    float sum = 0.0;
    for (uint i = 0; i < dim_size; i++) {{
        float val = exp(input[base + i] - max_val);
        output[base + i] = val;
        sum += val;
    }}

    for (uint i = 0; i < dim_size; i++) {{
        output[base + i] /= sum;
    }}
}}
"#, total_elements=total_elements, dim_size=dim_size, primitives=primitives)
}

pub fn generate_epilogue_code(ops: &[crate::core::op::EpilogueOp], val_name: &str, channel_idx_name: &str, global_idx_name: &str) -> (String, String) {
    let mut code = String::new();
    let mut args = String::new();
    let mut buffer_idx = 6; // Starting buffer index for epilogue after standard Conv buffers

    for (i, op) in ops.iter().enumerate() {
        match op {
            crate::core::op::EpilogueOp::ReLU => {
                code.push_str(&format!("    {} = max({}, 0.0f);\n", val_name, val_name));
            }
            crate::core::op::EpilogueOp::BatchNorm { epsilon, .. } => {
                let suffix = if ops.len() > 1 { format!("_{}", i) } else { "".to_string() };
                args.push_str(&format!(", device const float* Gamma{} [[buffer({})]]", suffix, buffer_idx));
                args.push_str(&format!(", device const float* Beta{} [[buffer({})]]", suffix, buffer_idx + 1));
                args.push_str(&format!(", device const float* Mean{} [[buffer({})]]", suffix, buffer_idx + 2));
                args.push_str(&format!(", device const float* Var{} [[buffer({})]]", suffix, buffer_idx + 3));
                buffer_idx += 4;

                code.push_str(&format!(r#"
    {{
        float gamma = Gamma{}[{}];
        float beta = Beta{}[{}];
        float mean = Mean{}[{}];
        float var = Var{}[{}];
        {} = ({} - mean) * rsqrt(var + {}f) * gamma + beta;
    }}
"#, suffix, channel_idx_name, suffix, channel_idx_name, suffix, channel_idx_name, suffix, channel_idx_name, val_name, val_name, epsilon));
            }
            crate::core::op::EpilogueOp::ResidualAdd { .. } => {
                let suffix = if ops.len() > 1 { format!("_{}", i) } else { "".to_string() };
                args.push_str(&format!(", device const float* Residual{} [[buffer({})]]", suffix, buffer_idx));
                buffer_idx += 1;
                code.push_str(&format!("    {} += (float)Residual{}[{}];\n", val_name, suffix, global_idx_name));
            }
            crate::core::op::EpilogueOp::BiasAdd { .. } => {
                 let suffix = if ops.len() > 1 { format!("_{}", i) } else { "".to_string() };
                 args.push_str(&format!(", device const float* Bias{} [[buffer({})]]", suffix, buffer_idx));
                 buffer_idx += 1;
                 code.push_str(&format!("    {} += (float)Bias{}[{}];\n", val_name, suffix, channel_idx_name));
            }
            _ => {}
        }
    }
    (args, code)
}
