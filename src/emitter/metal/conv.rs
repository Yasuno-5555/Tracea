use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use crate::core::config::RegisterStrategy;
use super::utils::{generate_epilogue_code};

pub fn generate_metal_conv(ir: &UnifiedOpIR) -> String {
    if let UnifiedOpType::Conv2d { n: batch, h: h_in, w: w_in, c: c_in, k: k_out, r, s, stride, pad, dilation, .. } = ir.op_type {
        let h_out = (h_in + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * pad - dilation * (s - 1) - 1) / stride + 1;
        
        let m_gemm = batch * h_out * w_out;
        let k_gemm = c_in * r * s;
        let n_gemm = k_out;

        let mt = ir.tiling.m_tile.max(32);
        let nt = ir.tiling.n_tile.max(32);
        let kt = ir.tiling.k_tile.max(16);

        let m_subtiles = mt / 8;
        let n_subtiles = nt / 8;
        
        let primitives = crate::backend::metal::MetalBackend::get_primitive_defs();
        let use_double_buffer = ir.tiling.double_buffer;

        if use_double_buffer {
            generate_metal_conv_double_buffer(ir, mt, nt, kt, m_subtiles, n_subtiles, primitives, m_gemm as u32, k_gemm as u32, n_gemm as u32)
        } else {
            generate_metal_conv_single_buffer(ir, mt, nt, kt, m_subtiles, n_subtiles, primitives, m_gemm as u32, k_gemm as u32, n_gemm as u32)
        }
    } else {
        panic!("Metal conv emitter called with invalid op");
    }
}

fn generate_metal_conv_single_buffer(ir: &UnifiedOpIR, mt: u32, nt: u32, kt: u32, m_subtiles: u32, n_subtiles: u32, primitives: String, m_gemm: u32, k_gemm: u32, n_gemm: u32) -> String {
    let unroll_directive = if ir.tiling.k_unroll > 1 { format!("\n#pragma unroll({})", ir.tiling.k_unroll) } else { "".to_string() };
    let use_expanded = matches!(ir.tiling.register_strategy, RegisterStrategy::Expanded);
    
    let epilogue = match &ir.op_type {
        UnifiedOpType::Conv2d { epilogue, .. } => epilogue,
        _ => &vec![],
    };
    let (epi_args, epi_code) = generate_epilogue_code(epilogue, "val", "channel_idx", "global_out_idx");

    let acc_def_code = if use_expanded {
        let mut s = String::new();
        for mi in 0..m_subtiles/2 {
            for ni in 0..n_subtiles/2 {
                s.push_str(&format!("    simdgroup_float8x8 acc_{}_{} = simdgroup_float8x8(0.0f);\n", mi, ni));
            }
        }
        s
    } else {
        format!(r#"
    simdgroup_float8x8 acc[{}/2][{}/2];
    for (uint mi = 0; mi < {}/2; ++mi) {{
        for (uint ni = 0; ni < {}/2; ++ni) {{
            acc[mi][ni] = simdgroup_float8x8(0.0f);
        }}
    }}
        "#, m_subtiles, n_subtiles, m_subtiles, n_subtiles)
    };

    let compute_code = if use_expanded {
        let mut s = String::new();
        s.push_str("    simdgroup_half8x8 ma;\n    simdgroup_half8x8 mb;\n");
        for mi in 0..m_subtiles/2 {
            for ni in 0..n_subtiles/2 {
                s.push_str(&format!(r#"
                {{
                    uint local_row = sg_base_row + {} * 8;
                    uint local_col = sg_base_col + {} * 8;
                    {}
                    for (uint ki = 0; ki < {}; ki += 8) {{
                        simdgroup_load(ma, &sA[local_row * {} + ki], {});
                        simdgroup_load(mb, &sB[ki * {} + local_col], {});
                        simdgroup_multiply_accumulate(acc_{}_{}, ma, mb, acc_{}_{});
                    }}
                }}
                "#, mi, ni, unroll_directive, kt, kt, kt, nt, nt, mi, ni, mi, ni));
            }
        }
        s
    } else {
        format!(r#"
        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;
        for (uint mi = 0; mi < {}/2; ++mi) {{
            for (uint ni = 0; ni < {}/2; ++ni) {{
                uint local_row = sg_base_row + mi * 8;
                uint local_col = sg_base_col + ni * 8;
                {}
                for (uint ki = 0; ki < {}; ki += 8) {{
                    simdgroup_load(ma, &sA[local_row * {} + ki], {});
                    simdgroup_load(mb, &sB[ki * {} + local_col], {});
                    simdgroup_multiply_accumulate(acc[mi][ni], ma, mb, acc[mi][ni]);
                }}
            }}
        }}
        "#, m_subtiles, n_subtiles, unroll_directive, kt, kt, kt, nt, nt)
    };

    let store_code = if use_expanded {
        let mut s = String::new();
        s.push_str(&format!("    threadgroup float sStore[{} * {}];\n", mt, nt));
        for mi in 0..m_subtiles/2 {
            for ni in 0..n_subtiles/2 {
                 s.push_str(&format!(r#"
                {{
                    simdgroup_store(acc_{mi}_{ni}, &sStore[(sg_base_row + {mi} * 8) * {nt} + (sg_base_col + {ni} * 8)], {nt}, 0);
                    simdgroup_store(acc_{mi}_{ni}, &sStore[(sg_base_row + {mi} * 8 + 8) * {nt} + (sg_base_col + {ni} * 8)], {nt}, 1);
                }}
                 "#, mi=mi, ni=ni, nt=nt));
            }
        }
        s.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        s.push_str(&format!(r#"
    for (uint i = tid; i < {mt} * {nt}; i += 128) {{
        uint out_row = tile_m_idx * {mt} + i / {nt};
        uint out_col = tile_n_idx * {nt} + i % {nt};
        if (out_row < {m_gemm} && out_col < {n_gemm}) {{
            float val = sStore[i];
            uint channel_idx = out_col;
            uint global_out_idx = out_row * {n_gemm} + out_col;
{epi_code}
            Output[global_out_idx] = val;
        }}
    }}
        "#, mt=mt, nt=nt, m_gemm=m_gemm, n_gemm=n_gemm, epi_code=epi_code));
        s
    } else {
        let mut s = String::new();
        s.push_str(&format!("    threadgroup float sStore[{} * {}];\n", mt, nt));
        s.push_str(&format!(r#"
    for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
        for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
            simdgroup_store(acc[mi][ni], &sStore[(sg_base_row + mi * 16) * {nt} + (sg_base_col + ni * 16)], {nt});
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {mt} * {nt}; i += 128) {{
        uint out_row = tile_m_idx * {mt} + i / {nt};
        uint out_col = tile_n_idx * {nt} + i % {nt};
        if (out_row < {m_gemm} && out_col < {n_gemm}) {{
            float val = sStore[i];
            uint channel_idx = out_col;
            uint global_out_idx = out_row * {n_gemm} + out_col;
{epi_code}
            Output[global_out_idx] = val;
        }}
    }}
        "#, mt=mt, nt=nt, m_subtiles=m_subtiles, n_subtiles=n_subtiles, m_gemm=m_gemm, n_gemm=n_gemm, epi_code=epi_code));
        s
    };
    
    format!(r#"
#include <metal_stdlib>
using namespace metal;

{primitives}

struct ConvParams {{
    uint batch, h_in, w_in, c_in, k_out;
    uint h_out, w_out, r_sz, s_sz;
    uint stride, pad, dilation;
}};

struct TileMetadata {{
    uint region_m, region_n, k_start, k_end, role;
}};

kernel void conv2d_implicit_gemm(
    device const half* Input [[buffer(0)]],
    device const half* Weight [[buffer(1)]],
    device float* Output [[buffer(2)]],
    constant ConvParams& p [[buffer(3)]],
    device const uint* l1_map [[buffer(4)]],
    device const TileMetadata* l2_table [[buffer(5)]]{epi_args},
    uint  bid [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {{
    threadgroup half sA[{mt} * {kt}]; 
    threadgroup half sB[{nt} * {kt}]; 
    
    uint logical_id = l1_map[bid];
    TileMetadata tile = l2_table[logical_id];
    uint tile_m_idx = tile.region_m;
    uint tile_n_idx = tile.region_n;

    {acc_def_code}

    uint sg_id = simd_gid;
    uint t_idx = tid; 
    uint sg_row = (sg_id / ({n_subtiles}/2)); 
    uint sg_col = (sg_id % ({n_subtiles}/2));
    uint sg_base_row = sg_row * 16; 
    uint sg_base_col = sg_col * 16;
    uint num_k_tiles = ({k_gemm} + {kt} - 1) / {kt};

    for (uint k_tile = 0; k_tile < num_k_tiles; ++k_tile) {{
        uint k_off = k_tile * {kt};
        
        uint lk = t_idx % {kt};      
        uint lm = t_idx / {kt};      
        uint lm_step = 128 / {kt};   

        for (uint i = t_idx; i < {mt} * {kt}; i += 128) {{
            uint curr_lm = i / {kt};
            uint curr_lk = i % {kt};
            uint global_m = tile_m_idx * {mt} + curr_lm;
            uint global_k = k_off + curr_lk;
            
            half val = 0.0h;
            if (global_m < {m_gemm} && global_k < {k_gemm}) {{
                uint rs_sz = p.r_sz * p.s_sz;
                uint cin = global_k / rs_sz;
                uint rem_rs = global_k % rs_sz;
                uint r_k = rem_rs / p.s_sz;
                uint s_k = rem_rs % p.s_sz;

                uint b_curr = global_m / (p.h_out * p.w_out);
                uint rem_m = global_m % (p.h_out * p.w_out);
                uint h_curr = rem_m / p.w_out;
                uint w_curr = rem_m % p.w_out;

                int ih = (int)h_curr * (int)p.stride + (int)r_k * (int)p.dilation - (int)p.pad;
                int iw = (int)w_curr * (int)p.stride + (int)s_k * (int)p.dilation - (int)p.pad;
                if (ih >= 0 && ih < (int)p.h_in && iw >= 0 && iw < (int)p.w_in) {{
                   uint in_idx = ((b_curr * p.h_in + (uint)ih) * p.w_in + (uint)iw) * p.c_in + cin;
                   val = Input[in_idx];
                }}
            }}
            sA[i] = val;
        }}

        for (uint i = t_idx; i < {kt} * {nt}; i += 128) {{
            uint r = i / {nt}; 
            uint c = i % {nt}; 
            uint global_k_b = k_off + r;
            uint global_n_b = tile_n_idx * {nt} + c;
            half val = 0.0h;
            if (global_k_b < {k_gemm} && global_n_b < {n_gemm}) {{
                val = Weight[global_n_b * {k_gemm} + global_k_b];
            }}
            sB[i] = val;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {compute_code}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    {store_code}
}}
"#, 
    mt=mt, nt=nt, kt=kt, m_gemm=m_gemm, k_gemm=k_gemm, n_gemm=n_gemm, n_subtiles=n_subtiles, 
    primitives=primitives, acc_def_code=acc_def_code, compute_code=compute_code, store_code=store_code, epi_args=epi_args
    )
}

fn generate_metal_conv_double_buffer(ir: &UnifiedOpIR, mt: u32, nt: u32, kt: u32, m_subtiles: u32, n_subtiles: u32, primitives: String, m_gemm: u32, k_gemm: u32, n_gemm: u32) -> String {
    let unroll_directive = if ir.tiling.k_unroll > 1 { format!("\n#pragma unroll({})", ir.tiling.k_unroll) } else { "".to_string() };
    
    let epilogue = match &ir.op_type {
        UnifiedOpType::Conv2d { epilogue, .. } => epilogue,
        _ => &vec![],
    };
    let (epi_args, epi_code) = generate_epilogue_code(epilogue, "val", "channel_idx", "global_out_idx");

    format!(r#"
#include <metal_stdlib>
using namespace metal;

{primitives}

struct ConvParams {{
    uint batch, h_in, w_in, c_in, k_out;
    uint h_out, w_out, r_sz, s_sz;
    uint stride, pad, dilation;
}};

struct TileMetadata {{
    uint region_m, region_n, k_start, k_end, role;
}};

kernel void conv2d_implicit_gemm(
    device const half* Input [[buffer(0)]],
    device const half* Weight [[buffer(1)]],
    device float* Output [[buffer(2)]],
    constant ConvParams& p [[buffer(3)]],
    device const uint* l1_map [[buffer(4)]],
    device const TileMetadata* l2_table [[buffer(5)]]{epi_args},
    uint  bid [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {{
    threadgroup half sA[2][{mt} * {kt}]; 
    threadgroup half sB[2][{nt} * {kt}]; 
    
    uint logical_id = l1_map[bid];
    TileMetadata tile = l2_table[logical_id];
    uint tile_m_idx = tile.region_m;
    uint tile_n_idx = tile.region_n;

    simdgroup_float8x8 acc[{m_subtiles}/2][{n_subtiles}/2];
    for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
        for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
            acc[mi][ni] = simdgroup_float8x8(0.0f);
        }}
    }}

    uint sg_id = simd_gid;
    uint t_idx = tid; 
    uint sg_row = (sg_id / ({n_subtiles}/2)); 
    uint sg_col = (sg_id % ({n_subtiles}/2));
    uint sg_base_row = sg_row * 16; 
    uint sg_base_col = sg_col * 16;
    uint num_k_tiles = ({k_gemm} + {kt} - 1) / {kt};

    {{
        uint k_off = 0;
        uint curr_buf = 0;
        uint lk = t_idx % {kt};      
        uint lm = t_idx / {kt};      
        uint lm_step = 128 / {kt};   

        uint global_k = k_off + lk;
        bool k_valid = global_k < {k_gemm};
        
        uint cin = 0, r_k = 0, s_k = 0;
        if (k_valid) {{
             uint rs_sz = p.r_sz * p.s_sz;
             cin = global_k / rs_sz;
             uint rem_rs = global_k % rs_sz;
             r_k = rem_rs / p.s_sz;
             s_k = rem_rs % p.s_sz;
        }}

        uint global_m = tile_m_idx * {mt} + lm;
        uint b_curr = global_m / (p.h_out * p.w_out);
        uint rem_m = global_m % (p.h_out * p.w_out);
        uint h_curr = rem_m / p.w_out;
        uint w_curr = rem_m % p.w_out;

        for (uint i = t_idx; i < {mt} * {kt}; i += 128) {{
            half val = 0.0h;
            if (global_m < {m_gemm} && k_valid) {{
                int ih = (int)h_curr * (int)p.stride + (int)r_k * (int)p.dilation - (int)p.pad;
                int iw = (int)w_curr * (int)p.stride + (int)s_k * (int)p.dilation - (int)p.pad;
                if (ih >= 0 && ih < (int)p.h_in && iw >= 0 && iw < (int)p.w_in) {{
                   uint in_idx = ((b_curr * p.h_in + (uint)ih) * p.w_in + (uint)iw) * p.c_in + cin;
                   val = Input[in_idx];
                }}
            }}
            sA[curr_buf][lm * {kt} + lk] = val;
            lm += lm_step;
            global_m += lm_step;
            w_curr += lm_step;
            while (w_curr >= p.w_out) {{
                w_curr -= p.w_out;
                h_curr += 1;
                if (h_curr >= p.h_out) {{
                    h_curr -= p.h_out;
                    b_curr += 1;
                }}
            }}
        }}

        for (uint i = t_idx; i < {kt} * {nt}; i += 128) {{
            uint r = i / {nt}; 
            uint c = i % {nt}; 
            uint global_k_b = k_off + r;
            uint global_n_b = tile_n_idx * {nt} + c;
            half val = 0.0h;
            if (global_k_b < {k_gemm} && global_n_b < {n_gemm}) {{
                val = Weight[global_n_b * {k_gemm} + global_k_b];
            }}
            sB[curr_buf][r * {nt} + c] = val;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint curr_buf = 0;
    for (uint k_tile = 0; k_tile < num_k_tiles; ++k_tile) {{
        uint next_buf = 1 - curr_buf;
        uint next_k_tile = k_tile + 1;
        
        if (next_k_tile < num_k_tiles) {{
            uint k_off = next_k_tile * {kt};
            uint lk = t_idx % {kt};      
            uint lm = t_idx / {kt};      
            uint lm_step = 128 / {kt};   

            uint global_k = k_off + lk;
            bool k_valid = global_k < {k_gemm};
            
            uint cin = 0, r_k = 0, s_k = 0;
            if (k_valid) {{
                 uint rs_sz = p.r_sz * p.s_sz;
                 cin = global_k / rs_sz;
                 uint rem_rs = global_k % rs_sz;
                 r_k = rem_rs / p.s_sz;
                 s_k = rem_rs % p.s_sz;
            }}

            uint global_m = tile_m_idx * {mt} + lm;
            uint b_curr = global_m / (p.h_out * p.w_out);
            uint rem_m = global_m % (p.h_out * p.w_out);
            uint h_curr = rem_m / p.w_out;
            uint w_curr = rem_m % p.w_out;

            for (uint i = t_idx; i < {mt} * {kt}; i += 128) {{
                half val = 0.0h;
                if (global_m < {m_gemm} && k_valid) {{
                    int ih = (int)h_curr * (int)p.stride + (int)r_k * (int)p.dilation - (int)p.pad;
                    int iw = (int)w_curr * (int)p.stride + (int)s_k * (int)p.dilation - (int)p.pad;
                    if (ih >= 0 && ih < (int)p.h_in && iw >= 0 && iw < (int)p.w_in) {{
                       uint in_idx = ((b_curr * p.h_in + (uint)ih) * p.w_in + (uint)iw) * p.c_in + cin;
                       val = Input[in_idx];
                    }}
                }}
                sA[next_buf][lm * {kt} + lk] = val;
                lm += lm_step;
                global_m += lm_step;
                w_curr += lm_step;
                while (w_curr >= p.w_out) {{
                    w_curr -= p.w_out;
                    h_curr += 1;
                    if (h_curr >= p.h_out) {{
                        h_curr -= p.h_out;
                        b_curr += 1;
                    }}
                }}
            }}

            for (uint i = t_idx; i < {kt} * {nt}; i += 128) {{
                uint r = i / {nt}; 
                uint c = i % {nt}; 
                uint global_k_b = k_off + r;
                uint global_n_b = tile_n_idx * {nt} + c;
                half val = 0.0h;
                if (global_k_b < {k_gemm} && global_n_b < {n_gemm}) {{
                    val = Weight[global_n_b * {k_gemm} + global_k_b];
                }}
                sB[next_buf][r * {nt} + c] = val;
            }}
        }}

        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;
        for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
            for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
                uint local_row = sg_base_row + mi * 16; 
                uint local_col = sg_base_col + ni * 16;
                {unroll_directive}
                for (uint ki = 0; ki < {kt}; ki += 8) {{
                    simdgroup_load(ma, &sA[curr_buf][local_row * {kt} + ki], {kt});
                    simdgroup_load(mb, &sB[curr_buf][ki * {nt} + local_col], {nt});
                    simdgroup_multiply_accumulate(acc[mi][ni], ma, mb, acc[mi][ni]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        curr_buf = 1 - curr_buf;
    }}

    threadgroup float sStore[{mt} * {nt}];
    for (uint mi = 0; mi < {m_subtiles}/2; ++mi) {{
        for (uint ni = 0; ni < {n_subtiles}/2; ++ni) {{
            simdgroup_store(acc[mi][ni], &sStore[(sg_base_row + mi * 16) * {nt} + (sg_base_col + ni * 16)], {nt});
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {mt} * {nt}; i += 128) {{
        uint out_row = tile_m_idx * {mt} + i / {nt};
        uint out_col = tile_n_idx * {nt} + i % {nt};
        if (out_row < {m_gemm} && out_col < {n_gemm}) {{
            float val = sStore[i];
            uint channel_idx = out_col;
            uint global_out_idx = out_row * {n_gemm} + out_col;
{epi_code}
            Output[global_out_idx] = val;
        }}
    }}
}}
"#, mt=mt, nt=nt, kt=kt, m_gemm=m_gemm, k_gemm=k_gemm, n_gemm=n_gemm, m_subtiles=m_subtiles, n_subtiles=n_subtiles, primitives=primitives, unroll_directive=unroll_directive, epi_args=epi_args, epi_code=epi_code)
}
