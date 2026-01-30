pub fn generate_batchnorm(n: usize, c: usize, h: usize, w: usize, epsilon: f32) -> String {
    format!(r#"
#include <metal_stdlib>
using namespace metal;

struct TileMetadata {{
    uint region_m, region_n, k_start, k_end, role;
}};

kernel void batchnorm_forward(
    device const half* Input [[buffer(0)]],
    device const half* Gamma [[buffer(1)]],
    device const half* Beta [[buffer(2)]],
    device const half* Mean [[buffer(3)]],
    device const half* Var [[buffer(4)]],
    device half* Output [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    device const uint* l1_map [[buffer(7)]],
    device const TileMetadata* l2_table [[buffer(8)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    uint logical_id = l1_map[bid];
    TileMetadata tile = l2_table[logical_id];
    uint group_offset = tile.region_m * 1024;
    uint idx = group_offset + tid;
    
    if (idx >= {n} * {c} * {h} * {w}) return;
    
    uint c_idx = idx % {c}; 
    
    half val = Input[idx];
    half mean = Mean[c_idx];
    half var = Var[c_idx];
    half gamma = Gamma[c_idx];
    half beta = Beta[c_idx];
    
    float inv_std = rsqrt((float)var + epsilon);
    half out = (val - mean) * (half)inv_std * gamma + beta;
    
    Output[idx] = out;
}}
"#, n=n, c=c, h=h, w=w)
}

pub fn generate_global_avg_pool(_n: usize, c: usize, h: usize, w: usize) -> String {
    format!(r#"
#include <metal_stdlib>
using namespace metal;

struct TileMetadata {{
    uint region_m, region_n, k_start, k_end, role;
}};

kernel void global_avg_pool_kernel(
    device const float* Input [[buffer(0)]],
    device float* Output [[buffer(1)]],
    device const uint* l1_map [[buffer(2)]],
    device const TileMetadata* l2_table [[buffer(3)]],
    uint bid [[threadgroup_position_in_grid]]
) {{
    uint logical_id = l1_map[bid];
    TileMetadata tile = l2_table[logical_id];
    uint c_idx = tile.region_m; 
    
    uint hw = {h} * {w};
    float sum = 0.0f;
    for (uint i = 0; i < hw; i++) {{
        sum += Input[c_idx * hw + i];
    }}
    Output[c_idx] = sum / (float)hw;
}}
"#, h=h, w=w)
}

pub fn generate_linear(batch: usize, m: usize, n: usize, k: usize) -> String {
    format!(r#"
#include <metal_stdlib>
using namespace metal;

struct TileMetadata {{
    uint region_m, region_n, k_start, k_end, role;
}};

kernel void linear_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& Batch [[buffer(6)]],
    device const uint* l1_map [[buffer(7)]],
    device const TileMetadata* l2_table [[buffer(8)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    uint logical_id = l1_map[bid];
    TileMetadata tile = l2_table[logical_id];
    uint b_idx = tile.k_start;
    
    // Simple tiling for Linear
    uint m_idx = tile.region_m * 32 + (tid / 32);
    uint n_idx = tile.region_n * 32 + (tid % 32);

    if (m_idx >= (uint)M || n_idx >= (uint)N) return;

    float acc = 0.0f;
    for (uint k = 0; k < (uint)K; k++) {{
        acc += (float)A[b_idx * M * K + m_idx * K + k] * (float)B[b_idx * K * N + k * N + n_idx];
    }}
    C[b_idx * M * N + m_idx * N + n_idx] = acc;
}}
"#)
}
