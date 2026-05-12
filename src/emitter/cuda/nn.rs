use crate::emitter::traits::EmissionError;

pub fn generate_softmax(_dim_size: usize, _total_elements: usize) -> Result<String, EmissionError> {
    Ok(format!(r#"
typedef unsigned int uint;

extern "C" __global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim_size,
    int num_rows
) {{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= num_rows) return;

    int base = row_idx * dim_size;

    // Find max for numerical stability
    float max_val = -1e38f;
    for (int i = 0; i < dim_size; i++) {{
        max_val = fmaxf(max_val, input[base + i]);
    }}

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < dim_size; i++) {{
        float val = expf(input[base + i] - max_val);
        output[base + i] = val;
        sum += val;
    }}

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim_size; i++) {{
        output[base + i] *= inv_sum;
    }}
}}
"#))
}

pub fn generate_batchnorm(_n: usize, c: usize, h: usize, w: usize, _epsilon: f32) -> Result<String, EmissionError> {
    Ok(format!(r#"
#include <cuda_fp16.h>
typedef unsigned int uint;

extern "C" __global__ void batchnorm_forward(
    const half* __restrict__ Input,
    const half* __restrict__ Gamma,
    const half* __restrict__ Beta,
    const half* __restrict__ Mean,
    const half* __restrict__ Var,
    half* __restrict__ Output,
    float epsilon,
    int total_elements
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // NCHW layout: idx = n*C*H*W + c*H*W + h*W + w
    int c_idx = (idx / ({h} * {w})) % {c};

    half val = Input[idx];
    half mean = Mean[c_idx];
    half var = Var[c_idx];
    half gamma = Gamma[c_idx];
    half beta = Beta[c_idx];

    float inv_std = rsqrtf(__half2float(var) + epsilon);
    float normalized = (__half2float(val) - __half2float(mean)) * inv_std;
    float result = normalized * __half2float(gamma) + __half2float(beta);

    Output[idx] = __float2half(result);
}}
"#, c=c, h=h, w=w))
}

pub fn generate_global_avg_pool(_n: usize, _c: usize, _h: usize, _w: usize) -> Result<String, EmissionError> {
    Ok(format!(r#"
typedef unsigned int uint;

extern "C" __global__ void global_avg_pool_kernel(
    const float* __restrict__ Input,
    float* __restrict__ Output,
    int batch_size,
    int channels,
    int spatial_size
) {{
    // One thread per (batch, channel) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels;
    if (idx >= total_outputs) return;

    int b = idx / channels;
    int c = idx % channels;
    int input_offset = b * channels * spatial_size + c * spatial_size;

    float sum = 0.0f;
    for (int i = 0; i < spatial_size; i++) {{
        sum += Input[input_offset + i];
    }}

    Output[idx] = sum / (float)spatial_size;
}}
"#))
}

pub fn generate_linear(_batch: usize, _m: usize, _n: usize, _k: usize) -> Result<String, EmissionError> {
    Ok(format!(r#"
#include <cuda_fp16.h>
typedef unsigned int uint;

#define TILE_SIZE 32

extern "C" __global__ void linear_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, int Batch
) {{
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {{
        float a_val = __half2float(A[b * M * K + row * K + k]);
        float b_val = __half2float(B[b * K * N + k * N + col]);
        acc += a_val * b_val;
    }}

    C[b * M * N + row * N + col] = acc;
}}
"#))
}

pub fn generate_matrix_core(_m: u32, _n: u32, _k: u32) -> Result<String, EmissionError> {
    Ok(format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void matrix_core_kernel(const half* a, const half* b, float* c) {{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);
}}
"#))
}
