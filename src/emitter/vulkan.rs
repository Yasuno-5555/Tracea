use crate::emitter::traits::UnifiedOpIR;
use crate::emitter::traits::UnifiedOpType;

pub fn generate_vulkan_conv(ir: &UnifiedOpIR) -> String {
    let primitive_defs = crate::backend::vulkan::VulkanBackend::get_primitive_defs();
    if let UnifiedOpType::Conv2d { n: batch, h: h_in, w: w_in, c: c_in, k: k_out, r, s, stride, pad, dilation, .. } = ir.op_type {
        let h_out = (h_in + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * pad - dilation * (s - 1) - 1) / stride + 1;
        
        let m_gemm = batch * h_out * w_out;
        let n_gemm = k_out;
        let k_gemm = c_in * r * s;

        let local_sz_x = 16;
        let local_sz_y = 16;

        format!(r#"#version 450
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

layout(local_size_x = {lx}, local_size_y = {ly}, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InputBuffer {{ float input_data[]; }};
layout(set = 0, binding = 1) buffer WeightBuffer {{ float weight_data[]; }};
layout(set = 0, binding = 2) buffer OutputBuffer {{ float output_data[]; }};

{primitives}

layout(push_constant) uniform Params {{
    int batch, h_in, w_in, c_in, k_out;
    int h_out, w_out, r_sz, s_sz;
    int stride, pad, dilation;
}} p;

void main() {{
    int g_idx = int(gl_GlobalInvocationID.x);
    int g_idy = int(gl_GlobalInvocationID.y);

    if (g_idx >= {m_gemm} || g_idy >= {n_gemm}) return;

    // Implicit GEMM indices
    int bn = g_idx / ({h_out} * {w_out});
    int rem_m = g_idx % ({h_out} * {w_out});
    int ho = rem_m / {w_out};
    int wo = rem_m % {w_out};
    
    int ci_out = g_idy;

    float acc = 0.0;
    for (int r = 0; r < {r_sz}; ++r) {{
        for (int s = 0; s < {s_sz}; ++s) {{
            for (int c = 0; c < {c_in}; ++c) {{
                int hi = ho * p.stride - p.pad + r * p.dilation;
                int wi = wo * p.stride - p.pad + s * p.dilation;
                
                if (hi >= 0 && hi < p.h_in && wi >= 0 && wi < p.w_in) {{
                    int input_off = ((bn * p.h_in + hi) * p.w_in + wi) * p.c_in + c;
                    int weight_off = ((r * {s_sz} + s) * p.c_in + c) * p.k_out + ci_out;
                    acc += input_data[input_off] * weight_data[weight_off];
                }}
            }}
        }}
    }}

    output_data[g_idx * p.k_out + g_idy] = acc;
}}
"#, 
        lx=local_sz_x, ly=local_sz_y,
        m_gemm=m_gemm, n_gemm=n_gemm,
        r_sz=r, s_sz=s, c_in=c_in,
        h_out=h_out, w_out=w_out,
        primitives=primitive_defs
        )
    } else {
        panic!("Vulkan emitter called with non-conv op");
    }
}

pub fn generate_vulkan_mma(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::MatrixCore { m, n, k } = ir.op_type {
        let primitive_defs = crate::backend::vulkan::VulkanBackend::get_primitive_defs();
        format!(r#"#version 450
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer A {{ float16_t data_a[]; }};
layout(set = 0, binding = 1) buffer B {{ float16_t data_b[]; }};
layout(set = 0, binding = 2) buffer C {{ float data_c[]; }};

{primitives}

void main() {{
    cooperativeMatrixKHR<float16_t, gl_ScopeSubgroup, {m}, {k}, gl_MatrixUseA> matA;
    cooperativeMatrixKHR<float16_t, gl_ScopeSubgroup, {k}, {n}, gl_MatrixUseB> matB;
    cooperativeMatrixKHR<float, gl_ScopeSubgroup, {m}, {n}, gl_MatrixUseAccumulator> matC;

    matC = cooperativeMatrixKHR<float, gl_ScopeSubgroup, {m}, {n}, gl_MatrixUseAccumulator>(0.0);

    // Simplified load for demo/baseline
    cooperativeMatrixLoadKHR(matA, data_a, 0, {k}, gl_CooperativeMatrixLayoutRowMajor);
    cooperativeMatrixLoadKHR(matB, data_b, 0, {n}, gl_CooperativeMatrixLayoutRowMajor);
    
    matC = cooperativeMatrixMulAddKHR(matA, matB, matC);

    cooperativeMatrixStoreKHR(matC, data_c, 0, {n}, gl_CooperativeMatrixLayoutRowMajor);
}}
"#,
        m=m, n=n, k=k, primitives=primitive_defs
        )
    } else {
        panic!("Vulkan mma emitter called with invalid op");
    }
}
