#pragma once
#include <cuda_fp16.h>

/**
 * Tracea v3.1 Epilogue Functors
 * Used for fusing activation and residual operations into GEMM/Conv kernels.
 */

namespace tracea {
namespace epilogue {

struct Identity {
  __device__ __forceinline__ float operator()(float x) const { return x; }
};

struct ReLU {
  __device__ __forceinline__ float operator()(float x) const {
    return fmaxf(0.0f, x);
  }
};

struct Gelu {
  __device__ __forceinline__ float operator()(float x) const {
    // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 *
    // x^3)))
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
  }
};

struct SiLU {
  __device__ __forceinline__ float operator()(float x) const {
    return x / (1.0f + expf(-x));
  }
};

struct BiasAdd {
  const float *bias;
  __device__ __forceinline__ float operator()(float x, int channel) const {
    return x + bias[channel];
  }
};

struct BiasAddReLU {
  const float *bias;
  __device__ __forceinline__ float operator()(float x, int channel) const {
    return fmaxf(0.0f, x + bias[channel]);
  }
};

struct BiasAddSiLU {
  const float *bias;
  __device__ __forceinline__ float operator()(float x, int channel) const {
    float val = x + bias[channel];
    return val / (1.0f + expf(-val));
  }
};

struct ResidualAdd {
  const float *residual;
  __device__ __forceinline__ float operator()(float x, int index) const {
    return x + residual[index];
  }
};

} // namespace epilogue
} // namespace tracea
