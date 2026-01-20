
#include <cuda_fp16.h>
#include <mma.h>

extern "C" __global__ void __launch_bounds__(32, 1) gemm_mma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, __float2half(0.0f));
    int row = blockIdx.y * 16;
    int col = blockIdx.x * 16;

    if (row >= M || col >= N) return;

    for (int k_idx = 0; k_idx < K; k_idx += 16) {
        nvcuda::wmma::load_matrix_sync(a_frag, A + row * K + k_idx, (unsigned)K);
        nvcuda::wmma::load_matrix_sync(b_frag, B + k_idx * N + col, (unsigned)N);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    nvcuda::wmma::store_matrix_sync(C + row * N + col, acc_frag, (unsigned)N, nvcuda::wmma::mem_row_major);
}
