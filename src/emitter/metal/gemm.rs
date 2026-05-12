use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use super::utils::{generate_epilogue_code};

pub fn generate_metal_gemm(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::Gemm { .. } = ir.op_type {
        let num_warps = ir.tiling.force_num_warps.unwrap_or(4);
        let simd_width: u32 = 32; // Metal SIMD group width
        let thread_count = num_warps * simd_width;
        if ir.tiling.gemm_variant == crate::core::config::GemmVariant::Tiled {
            generate_gemm_tiled(ir, thread_count)
        } else if ir.tiling.double_buffer {
            generate_gemm_double_buffer(ir, thread_count)
        } else {
            generate_gemm_single_buffer(ir, thread_count)
        }
    } else {
        panic!("Metal gemm emitter called with invalid op");
    }
}

fn generate_gemm_single_buffer(ir: &UnifiedOpIR, thread_count: u32) -> String {
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

// Single Buffer GEMM — each simdgroup handles exactly one 8-wide column group
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

    // Each simdgroup owns one 8-wide column group (no overlap, no OOB)
    uint my_ni = simd_id % {n_subtiles};
    uint local_col_off = my_ni * 8;

    // One accumulator per M sub-tile (not N — that's handled by simdgroup id)
    simdgroup_float8x8 acc[{m_subtiles}];
    for (uint mi = 0; mi < {m_subtiles}; ++mi) {{
        acc[mi] = simdgroup_float8x8(0.0f);
    }}

    for (uint k_step = 0; k_step < K; k_step += {kt}) {{
        for (uint i = tid; i < {mt} * {kt}; i += {thread_count}) {{
            uint r = i / {kt}; uint c = i % {kt};
            uint gr = bid.y * {mt} + r;
            uint gc = k_step + c;
            sA[i] = (gr < M && gc < K) ? A[gr * K + gc] : half(0);
        }}
        for (uint i = tid; i < {kt} * {nt}; i += {thread_count}) {{
            uint r = i / {nt}; uint c = i % {nt};
            uint gr = k_step + r;
            uint gc = bid.x * {nt} + c;
            sB[i] = (gr < K && gc < N) ? B[gr * N + gc] : half(0);
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;

        for (uint mi = 0; mi < {m_subtiles}; ++mi) {{
            uint local_row = mi * 8;

            for (uint ki = 0; ki < {kt}; ki += 8) {{
                simdgroup_load(ma, &sA[local_row * {kt} + ki], {kt});
                simdgroup_load(mb, &sB[ki * {nt} + local_col_off], {nt});
                simdgroup_multiply_accumulate(acc[mi], ma, mb, acc[mi]);
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Store — each simdgroup stores its column group exclusively (no overlap)
    threadgroup float sStore[{mt} * {nt}];
    for (uint mi = 0; mi < {m_subtiles}; ++mi) {{
        uint store_row_off = mi * 8;
        simdgroup_store(acc[mi], &sStore[store_row_off * {nt} + local_col_off], {nt});
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {mt} * {nt}; i += {thread_count}) {{
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
"#, mt=mt, nt=nt, kt=kt, thread_count=thread_count, primitives=primitives, m_subtiles=m_subtiles, n_subtiles=n_subtiles, epi_args=epi_args, epi_code=epi_code)
}

fn generate_gemm_double_buffer(ir: &UnifiedOpIR, thread_count: u32) -> String {
    let mt = ir.tiling.m_tile;
    let nt = ir.tiling.n_tile;
    let kt = ir.tiling.k_tile;
    let fc = ir.tiling.fusion_count.max(1);
    let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();

    let m_subtiles = mt / 8;
    let n_subtiles = nt / 8;

    let epilogue = match &ir.op_type {
        UnifiedOpType::Gemm { epilogue, .. } => epilogue,
        _ => &vec![],
    };
    let (epi_args, epi_code) = generate_epilogue_code(epilogue, "val", "channel_idx", "global_out_idx");

    // When fusion_count > 1: bid.x indexes fused tile groups.
    // Each group processes `fc` consecutive N-tiles, sharing A data.
    let fusion_loop = if fc > 1 {
        format!(r#"
    // FUSION: processing {{fc}} N-tiles, fused N-span per threadgroup = {{fc}} * {nt}
    #define FUSION_FC {fc}
    for (uint f = 0; f < FUSION_FC; ++f) {{
        uint fc_bid_x = bid.x * FUSION_FC + f;
        if (fc_bid_x * {nt} >= N) break;
"#)
    } else { String::new() };

    let fusion_store_loop = if fc > 1 {
        r#"
        // End fusion iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}"#.to_string()
    } else { String::new() };

    let fusion_close = if fc > 1 {
        String::from("\n}")
    } else { String::new() };

    // Inner compute uses `fc_bid_x` for the N-coordinate instead of `bid.x`.
    // When fusion_count = 1: `fc_bid_x` is always equal to `bid.x`.
    let n_coord = if fc > 1 { "fc_bid_x" } else { "bid.x" };

    // Store path: each simdgroup stores its column group exclusively
    let use_sstore = !epi_code.is_empty();
    let store_section = if use_sstore {
        format!(r#"
    // STORE RESULTS with Epilogue (via sStore)
    threadgroup float sStore[{mt} * {nt}];
    for (uint mi = 0; mi < {m_subtiles}; ++mi) {{
        uint store_row_off = mi * 8;
        simdgroup_store(acc[mi], &sStore[store_row_off * {nt} + local_col_off], {nt});
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {mt} * {nt}; i += {thread_count}) {{
        uint out_row = bid.y * {mt} + i / {nt};
        uint out_col = {n_coord} * {nt} + i % {nt};
        if (out_row < M && out_col < N) {{
            float val = sStore[i];
            uint channel_idx = out_col;
            uint global_out_idx = out_row * N + out_col;
{{epi_code}}
            C[global_out_idx] = val;
        }}
    }}"#, mt=mt, nt=nt, thread_count=thread_count, n_coord=n_coord, m_subtiles=m_subtiles)
    } else {
        format!(r#"
    // STORE RESULTS directly to device memory (no epilogue, no sStore)
    for (uint mi = 0; mi < {m_subtiles}; ++mi) {{
        uint store_row = bid.y * {mt} + mi * 8;
        uint store_col = {n_coord} * {nt} + local_col_off;
        if (store_row < M && store_col < N) {{
            simdgroup_store(acc[mi], &C[store_row * N + store_col], N);
        }}
    }}"#, mt=mt, nt=nt, m_subtiles=m_subtiles, n_coord=n_coord)
    };

    format!(r#"
{primitives}

// Double Buffer GEMM — each simdgroup handles exactly one 8-wide column group
// Fusion_count = {fc} (TTG topology-optimized)
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

    // Each simdgroup owns one 8-wide column group (no overlap, no OOB)
    uint my_ni = simd_id % {n_subtiles};
    uint local_col_off = my_ni * 8;{fusion_loop}

    uint num_k_tiles = (K + {kt} - 1) / {kt};
    uint fc_bid_x = {n_coord};

    // One accumulator per M sub-tile (column group handled by simdgroup id)
    simdgroup_float8x8 acc[{m_subtiles}];
    for (uint mi = 0; mi < {m_subtiles}; ++mi) {{
        acc[mi] = simdgroup_float8x8(0.0f);
    }}

    // PROLOGUE: Load first tile into buffer 0
    for (uint i = tid; i < {mt} * {kt}; i += {thread_count}) {{
        uint r = i / {kt}; uint c = i % {kt};
        uint gr = bid.y * {mt} + r;
        sA[0][i] = (gr < M && c < K) ? A[gr * K + c] : half(0);
    }}
    for (uint i = tid; i < {kt} * {nt}; i += {thread_count}) {{
        uint r = i / {nt}; uint c = i % {nt};
        uint gc = fc_bid_x * {nt} + c;
        sB[0][i] = (r < K && gc < N) ? B[r * N + gc] : half(0);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MAIN LOOP: Load next, compute current
    uint curr_buf = 0;
    for (uint k_tile = 0; k_tile < num_k_tiles; ++k_tile) {{
        uint next_buf = 1 - curr_buf;
        uint k_off_next = (k_tile + 1) * {kt};

        if (k_tile + 1 < num_k_tiles) {{
            for (uint i = tid; i < {mt} * {kt}; i += {thread_count}) {{
                uint r = i / {kt}; uint c = i % {kt};
                uint gr = bid.y * {mt} + r;
                uint gc = k_off_next + c;
                sA[next_buf][i] = (gr < M && gc < K) ? A[gr * K + gc] : half(0);
            }}
            for (uint i = tid; i < {kt} * {nt}; i += {thread_count}) {{
                uint r = i / {nt}; uint c = i % {nt};
                uint gr = k_off_next + r;
                uint gc = fc_bid_x * {nt} + c;
                sB[next_buf][i] = (gr < K && gc < N) ? B[gr * N + gc] : half(0);
            }}
        }}

        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;

        for (uint mi = 0; mi < {m_subtiles}; ++mi) {{
            uint local_row = mi * 8;

            for (uint ki = 0; ki < {kt}; ki += 8) {{
                simdgroup_load(ma, &sA[curr_buf][local_row * {kt} + ki], {kt});
                simdgroup_load(mb, &sB[curr_buf][ki * {nt} + local_col_off], {nt});
                simdgroup_multiply_accumulate(acc[mi], ma, mb, acc[mi]);
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
        curr_buf = next_buf;
    }}

{store_section}{fusion_store_loop}{fusion_close}
}}"#, mt=mt, nt=nt, kt=kt, fc=fc, thread_count=thread_count, primitives=primitives,
    m_subtiles=m_subtiles, n_subtiles=n_subtiles, epi_args=epi_args,
    fusion_loop=fusion_loop, fusion_store_loop=fusion_store_loop, fusion_close=fusion_close,
    n_coord=n_coord, store_section=store_section)
}

fn generate_gemm_tiled(ir: &UnifiedOpIR, thread_count: u32) -> String {
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
        // Strided load to cover all elements regardless of thread count
        for (uint i = t_idx; i < {mt} * {kt}; i += {thread_count}) {{
            uint r = i / {kt}; uint c = i % {kt};
            uint gr = bid.y * {mt} + r; uint gc = k_off + c;
            sA[curr_buf][i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0h;
        }}
        for (uint i = t_idx; i < {kt} * {nt}; i += {thread_count}) {{
            uint r = i / {nt}; uint c = i % {nt};
            uint gr = k_off + r; uint gc = bid.x * {nt} + c;
            sB[curr_buf][i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0h;
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
"#, mt=mt, nt=nt, kt=kt, thread_count=thread_count, primitives=primitives, epi_args=epi_args, epi_code=epi_code)
}
