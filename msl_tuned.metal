

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


// Double Buffer GEMM — A loaded via texture2D for better cache line utilization
// Fusion_count = 1 (TTG topology-optimized)
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
    // Double buffers with bank conflict padding
    threadgroup half sA[2][32 * 18];
    threadgroup half sB[2][16 * 36];

    // Each simdgroup owns one 8-wide column group (no overlap, no OOB)
    uint my_ni = simd_id % 4;
    uint local_col_off = my_ni * 8;

    uint num_k_tiles = (K + 16 - 1) / 16;
    uint fc_bid_x = bid.x;

    // One accumulator per M sub-tile
    simdgroup_float8x8 acc[4];
    for (uint mi = 0; mi < 4; ++mi) {
        acc[mi] = simdgroup_float8x8(0.0f);
    }

    // PROLOGUE: Load first tile (A via texture2D, B via buffer)
    for (uint i = tid; i < 32 * 16; i += 128) {
        uint r = i / 16; uint c = i % 16;
        uint gr = bid.y * 32 + r;
        sA[0][r * 18 + c] = (gr < M && c < K) ? texA.read(uint2(c, gr)).x : half(0);
    }
    for (uint i = tid; i < 16 * 32; i += 128) {
        uint r = i / 32; uint c = i % 32;
        uint gc = fc_bid_x * 32 + c;
        sB[0][r * 36 + c] = (r < K && gc < N) ? B[r * N + gc] : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MAIN LOOP
    uint curr_buf = 0;
    for (uint k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        uint next_buf = 1 - curr_buf;
        uint k_off_next = (k_tile + 1) * 16;

        if (k_tile + 1 < num_k_tiles) {
            for (uint i = tid; i < 32 * 16; i += 128) {
                uint r = i / 16; uint c = i % 16;
                uint gr = bid.y * 32 + r;
                uint gc = k_off_next + c;
                sA[next_buf][r * 18 + c] = (gr < M && gc < K) ? texA.read(uint2(gc, gr)).x : half(0);
            }
            for (uint i = tid; i < 16 * 32; i += 128) {
                uint r = i / 32; uint c = i % 32;
                uint gr = k_off_next + r;
                uint gc = fc_bid_x * 32 + c;
                sB[next_buf][r * 36 + c] = (gr < K && gc < N) ? B[gr * N + gc] : half(0);
            }
        }

        simdgroup_half8x8 ma;
        simdgroup_half8x8 mb;

        for (uint mi = 0; mi < 4; ++mi) {
            uint local_row = mi * 8;

            for (uint ki = 0; ki < 16; ki += 8) {
                simdgroup_load(ma, &sA[curr_buf][local_row * 18 + ki], 18);
                simdgroup_load(mb, &sB[curr_buf][ki * 36 + local_col_off], 36);
                simdgroup_multiply_accumulate(acc[mi], ma, mb, acc[mi]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        curr_buf = next_buf;
    }


    // STORE RESULTS directly to device memory (no epilogue, no sStore)
    for (uint mi = 0; mi < 4; ++mi) {
        uint store_row = bid.y * 32 + mi * 8;
        uint store_col = bid.x * 32 + local_col_off;
        if (store_row < M && store_col < N) {
            simdgroup_store(acc[mi], &C[store_row * N + store_col], N);
        }
    }
}