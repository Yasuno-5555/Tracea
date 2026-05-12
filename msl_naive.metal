

#include <metal_stdlib>
using namespace metal;

template<typename T>
inline T tracea_simd_shuffle(T val, ushort lane) {
    return simd_shuffle(val, lane);
}

template<typename T>
inline T tracea_simd_broadcast(T val, ushort lane) {
    return simd_broadcast(val, lane);
}

inline void tracea_barrier() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


// Single Buffer GEMM — A loaded via texture2D for better cache line utilization
kernel void unified_gemm_kernel(
    texture2d<half, access::read> texA [[texture(0)]],
    device const half* B [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint  tid [[thread_index_in_threadgroup]]
) {
    // Bank conflict mitigation: padded strides
    threadgroup half sA[64 * 34];
    threadgroup half sB[32 * 68];

    // Each simdgroup owns one 8-wide column group (no overlap, no OOB)
    uint my_ni = simd_id % 8;
    uint local_col_off = my_ni * 8;

    // One accumulator per M sub-tile
    simdgroup_float8x8 acc[8];
    for (uint mi = 0; mi < 8; ++mi) {
        acc[mi] = simdgroup_float8x8(0.0f);
    }

    for (uint k_step = 0; k_step < K; k_step += 32) {
        // Load A tile via texture2D (2D cache optimized, ~128B cache line efficient)
        for (uint i = tid; i < 64 * 32; i += 128) {
            uint r = i / 32; uint c = i % 32;
            uint gr = bid.y * 64 + r;
            uint gc = k_step + c;
            sA[r * 34 + c] = (gr < M && gc < K) ? texA.read(uint2(gc, gr)).x : half(0);
        }
        for (uint i = tid; i < 32 * 64; i += 128) {
            uint r = i / 64; uint c = i % 64;
            uint gr = k_step + r;
            uint gc = bid.x * 64 + c;
            sB[r * 68 + c] = (gr < K && gc < N) ? B[gr * N + gc] : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;

        for (uint mi = 0; mi < 8; ++mi) {
            uint local_row = mi * 8;

            for (uint ki = 0; ki < 32; ki += 8) {
                simdgroup_load(ma, &sA[local_row * 34 + ki], 34);
                simdgroup_load(mb, &sB[ki * 68 + local_col_off], 68);
                simdgroup_multiply_accumulate(acc[mi], ma, mb, acc[mi]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store — each simdgroup stores its column group exclusively
    threadgroup float sStore[64 * 64];
    for (uint mi = 0; mi < 8; ++mi) {
        uint store_row_off = mi * 8;
        simdgroup_store(acc[mi], &sStore[store_row_off * 64 + local_col_off], 64);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 64 * 64; i += 128) {
        uint out_row = bid.y * 64 + i / 64;
        uint out_col = bid.x * 64 + i % 64;
        if (out_row < M && out_col < N) {
            float val = sStore[i];
            uint channel_idx = out_col;
            uint global_out_idx = out_row * N + out_col;

            C[global_out_idx] = val;
        }
    }
}
