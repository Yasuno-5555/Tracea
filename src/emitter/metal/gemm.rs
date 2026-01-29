use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use super::utils::{generate_epilogue_code};

pub fn generate_metal_gemm(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::Gemm { .. } = ir.op_type {
        if ir.tiling.gemm_variant == crate::core::config::GemmVariant::Tiled {
            generate_gemm_tiled(ir)
        } else if ir.tiling.double_buffer {
            generate_gemm_double_buffer(ir)
        } else {
            generate_gemm_single_buffer(ir)
        }
    } else {
        panic!("Metal gemm emitter called with invalid op");
    }
}

fn generate_gemm_single_buffer(ir: &UnifiedOpIR) -> String {
    let mt = ir.tiling.m_tile;
    let nt = ir.tiling.n_tile;
    let kt = ir.tiling.k_tile;
    let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();
    
    let m_subtiles = mt / 8;
    let n_subtiles = nt / 8;

    let epilogue = match &ir.op_type {
        UnifiedOpType::Gemm { epilogue, .. } => epilogue,
        _ => &vec![],
    };
    let (epi_args, epi_code) = generate_epilogue_code(epilogue, "val", "channel_idx", "global_out_idx");
    
    format!(r#"
{primitives}

// Single Buffer GEMM with Simdgroup Matrix Operations
kernel void unified_gemm_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]]{epi_args},
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint  tid [[thread_index_in_threadgroup]]
) {{
    threadgroup half sA[{mt} * {kt}];
    threadgroup half sB[{kt} * {nt}];

    uint sg_base_row = (simd_id / ({n_subtiles}/2)) * 16;
    uint sg_base_col = (simd_id % ({n_subtiles}/2)) * 16;

    simdgroup_float8x8 acc[{m_subtiles}/2][{n_subtiles}/2];
    for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
        for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
            acc[mi][ni] = simdgroup_float8x8(0.0f);
        }}
    }}

    for (uint k_step = 0; k_step < K; k_step += {kt}) {{
        for (uint i = tid; i < {mt} * {kt}; i += 128) {{
            uint r = i / {kt}; uint c = i % {kt};
            uint gr = bid.y * {mt} + r;
            uint gc = k_step + c;
            sA[i] = (gr < M && gc < K) ? A[gr * K + gc] : half(0);
        }}
        for (uint i = tid; i < {kt} * {nt}; i += 128) {{
            uint r = i / {nt}; uint c = i % {nt};
            uint gr = k_step + r;
            uint gc = bid.x * {nt} + c;
            sB[i] = (gr < K && gc < N) ? B[gr * N + gc] : half(0);
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;
        
        for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
            for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
                uint local_row = sg_base_row + mi * 16;
                uint local_col = sg_base_col + ni * 16;
                
                for (uint ki = 0; ki < {kt}; ki += 8) {{
                    simdgroup_load(ma, &sA[local_row * {kt} + ki], {kt});
                    simdgroup_load(mb, &sB[ki * {nt} + local_col], {nt});
                    simdgroup_multiply_accumulate(acc[mi][ni], ma, mb, acc[mi][ni]);
                }}
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Store with Epilogue
    threadgroup float sStore[{mt} * {nt}];
    for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
        for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
            simdgroup_store(acc[mi][ni], &sStore[(sg_base_row + mi * 16) * {nt} + (sg_base_col + ni * 16)], {nt});
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {mt} * {nt}; i += 128) {{
        uint out_row = bid.y * {mt} + i / {nt};
        uint out_col = bid.x * {nt} + i % {nt};
        if (out_row < M && out_col < N) {{
            float val = sStore[i];
            uint channel_idx = out_col;
            uint global_out_idx = out_row * N + out_col;
{epi_code}
            C[global_out_idx] = val;
        }}
    }}
}}
"#, mt=mt, nt=nt, kt=kt, primitives=primitives, m_subtiles=m_subtiles, n_subtiles=n_subtiles, epi_args=epi_args, epi_code=epi_code)
}

fn generate_gemm_double_buffer(ir: &UnifiedOpIR) -> String {
    let mt = ir.tiling.m_tile;
    let nt = ir.tiling.n_tile;
    let kt = ir.tiling.k_tile;
    let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();
    
    let m_subtiles = mt / 8;
    let n_subtiles = nt / 8;

    let epilogue = match &ir.op_type {
        UnifiedOpType::Gemm { epilogue, .. } => epilogue,
        _ => &vec![],
    };
    let (epi_args, epi_code) = generate_epilogue_code(epilogue, "val", "channel_idx", "global_out_idx");
    
    format!(r#"
{primitives}

// Double Buffer GEMM - Ping-Pong pattern for memory latency hiding
kernel void unified_gemm_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]]{epi_args},
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint  tid [[thread_index_in_threadgroup]]
) {{
    // Double buffers
    threadgroup half sA[2][{mt} * {kt}];
    threadgroup half sB[2][{kt} * {nt}];

    uint sg_base_row = (simd_id / ({n_subtiles}/2)) * 16;
    uint sg_base_col = (simd_id % ({n_subtiles}/2)) * 16;

    simdgroup_float8x8 acc[{m_subtiles}/2][{n_subtiles}/2];
    for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
        for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
            acc[mi][ni] = simdgroup_float8x8(0.0f);
        }}
    }}

    uint num_k_tiles = (K + {kt} - 1) / {kt};

    // PROLOGUE: Load first tile into buffer 0
    for (uint i = tid; i < {mt} * {kt}; i += 128) {{
        uint r = i / {kt}; uint c = i % {kt};
        uint gr = bid.y * {mt} + r;
        sA[0][i] = (gr < M && c < K) ? A[gr * K + c] : half(0);
    }}
    for (uint i = tid; i < {kt} * {nt}; i += 128) {{
        uint r = i / {nt}; uint c = i % {nt};
        uint gc = bid.x * {nt} + c;
        sB[0][i] = (r < K && gc < N) ? B[r * N + gc] : half(0);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MAIN LOOP: Load next, compute current
    uint curr_buf = 0;
    for (uint k_tile = 0; k_tile < num_k_tiles; ++k_tile) {{
        uint next_buf = 1 - curr_buf;
        uint k_off_next = (k_tile + 1) * {kt};

        // Load next tile (if not last iteration)
        if (k_tile + 1 < num_k_tiles) {{
            for (uint i = tid; i < {mt} * {kt}; i += 128) {{
                uint r = i / {kt}; uint c = i % {kt};
                uint gr = bid.y * {mt} + r;
                uint gc = k_off_next + c;
                sA[next_buf][i] = (gr < M && gc < K) ? A[gr * K + gc] : half(0);
            }}
            for (uint i = tid; i < {kt} * {nt}; i += 128) {{
                uint r = i / {nt}; uint c = i % {nt};
                uint gr = k_off_next + r;
                uint gc = bid.x * {nt} + c;
                sB[next_buf][i] = (gr < K && gc < N) ? B[gr * N + gc] : half(0);
            }}
        }}

        // Compute current tile
        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;
        
        for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
            for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
                uint local_row = sg_base_row + mi * 16;
                uint local_col = sg_base_col + ni * 16;
                
                for (uint ki = 0; ki < {kt}; ki += 8) {{
                    simdgroup_load(ma, &sA[curr_buf][local_row * {kt} + ki], {kt});
                    simdgroup_load(mb, &sB[curr_buf][ki * {nt} + local_col], {nt});
                    simdgroup_multiply_accumulate(acc[mi][ni], ma, mb, acc[mi][ni]);
                }}
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
        curr_buf = next_buf;
    }}

    // STORE RESULTS with Epilogue
    threadgroup float sStore[{mt} * {nt}];
    for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
        for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
            simdgroup_store(acc[mi][ni], &sStore[(sg_base_row + mi * 16) * {nt} + (sg_base_col + ni * 16)], {nt});
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {mt} * {nt}; i += 128) {{
        uint out_row = bid.y * {mt} + i / {nt};
        uint out_col = bid.x * {nt} + i % {nt};
        if (out_row < M && out_col < N) {{
            float val = sStore[i];
            uint channel_idx = out_col;
            uint global_out_idx = out_row * N + out_col;
{epi_code}
            C[global_out_idx] = val;
        }}
    }}
}}
"#, mt=mt, nt=nt, kt=kt, primitives=primitives, m_subtiles=m_subtiles, n_subtiles=n_subtiles, epi_args=epi_args, epi_code=epi_code)
}

fn generate_gemm_tiled(ir: &UnifiedOpIR) -> String {
    let mt = ir.tiling.m_tile.max(32);
    let nt = ir.tiling.n_tile.max(32);
    let kt = ir.tiling.k_tile.max(32);
    let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();
    
    let epilogue = match &ir.op_type {
        UnifiedOpType::Gemm { epilogue, .. } => epilogue,
        _ => &vec![],
    };
    let (epi_args, epi_code) = generate_epilogue_code(epilogue, "val", "channel_idx", "global_out_idx");

    format!(r#"
{primitives}

kernel void gemm_tiled_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]]{epi_args},
    uint2 bid [[threadgroup_position_in_grid]],
    uint  t_idx [[thread_index_in_threadgroup]]
) {{
    threadgroup half sA[2][{mt} * {kt}];
    threadgroup half sB[2][{kt} * {nt}];
    float acc = 0.0f;
    
    uint thread_row = t_idx / 16;
    uint thread_col = t_idx % 16;
    uint num_k_tiles = (K + {kt} - 1) / {kt};
    uint curr_buf = 0;
    
    for(uint k_tile = 0; k_tile < num_k_tiles; ++k_tile) {{
        uint k_off = k_tile * {kt};
        if (t_idx < {mt} * {kt}) {{
            uint r = t_idx / {kt}, c = t_idx % {kt};
            uint gr = bid.y * {mt} + r, gc = k_off + c;
            sA[curr_buf][t_idx] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0h;
        }}
        if (t_idx < {kt} * {nt}) {{
            uint r = t_idx / {nt}, c = t_idx % {nt};
            uint gr = k_off + r, gc = bid.x * {nt} + c;
            sB[curr_buf][t_idx] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0h;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for(uint k = 0; k < {kt}; ++k) {{
            acc += (float)sA[curr_buf][thread_row * {kt} + k] * (float)sB[curr_buf][k * {nt} + thread_col];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        curr_buf = 1 - curr_buf;
    }}
    
    uint out_row = bid.y * {mt} + thread_row;
    uint out_col = bid.x * {nt} + thread_col;
    if (out_row < M && out_col < N) {{ 
        float val = acc;
        uint channel_idx = out_col;
        uint global_out_idx = out_row * N + out_col;
{epi_code}
        C[global_out_idx] = val; 
    }}
}}
"#, mt=mt, nt=nt, kt=kt, primitives=primitives, epi_args=epi_args, epi_code=epi_code)
}
