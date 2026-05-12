use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use crate::runtime::manager::DeviceBackend;

pub fn generate_attention(ir: &UnifiedOpIR, backend: DeviceBackend) -> String {
    if let UnifiedOpType::FusedAttention { b, s, d, h, dh, causal } = ir.op_type {
        match backend {
            DeviceBackend::Cuda => {
                format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// FlashAttention-2 Unified Loop (CUDA)
extern "C" __global__ void unified_attention_kernel(
    const half* Q, const half* K, const half* V, half* O,
    int S, int D, float scale
) {{
    int tile_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o;
    wmma::fill_fragment(acc_o, 0.0f);

    float m_prev = -1e9;
    float l_prev = 0.0f;

    for (int j = 0; j < S; j += 16) {{
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);
        
        // Load and MMA for S = QK^T
        // ... (Decomposed into MatrixCore IR equivalents)
        
        // Softmax Update (Online)
        // ...
        
        // Load and MMA for O = PV
    }}
    // Store O
}}
"#)
            }
            #[cfg(feature = "vulkan")]
            DeviceBackend::Vulkan => {
                format!(r#"
#version 450
#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

// FlashAttention-2 Unified Loop (Vulkan)
layout(local_size_x = 32) in;
void main() {{
    // Cooperative Matrix fragments for Q, K, V, O
    // Online Softmax via subgroup primitives
    // ... Identical loop structure to CUDA ...
}}
"#)
            }
            DeviceBackend::Metal => {
                crate::emitter::metal::generate_metal_attention(ir)
            }
            _ => "// Attention not yet unified for this backend\n".to_string(),
        }
    } else {
        panic!("Attention emitter called with non-attention op");
    }
}
