/// Tracea C++ Header
/// Header-only wrapper for tracea-ffi C ABI
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// ============================================================================
// Status Codes
// ============================================================================

typedef enum {
  TRACEA_SUCCESS = 0,
  TRACEA_INVALID_PARAMS = 1,
  TRACEA_UNSUPPORTED_CONFIG = 2,
  TRACEA_CUDA_ERROR = 3,
  TRACEA_CPU_ERROR = 4,
  TRACEA_UNKNOWN_ERROR = 99,
} TraceaStatus;

// ============================================================================
// Data Types
// ============================================================================

typedef enum {
  TRACEA_DTYPE_FLOAT32 = 0,
  TRACEA_DTYPE_FLOAT16 = 1,
  TRACEA_DTYPE_BFLOAT16 = 2,
  TRACEA_DTYPE_INT32 = 3,
  TRACEA_DTYPE_INT8 = 4,
} TraceaDType;

typedef enum {
  TRACEA_EPILOGUE_IDENTITY = 0,
  TRACEA_EPILOGUE_BIAS_ADD = 1,
  TRACEA_EPILOGUE_RELU = 2,
  TRACEA_EPILOGUE_GELU = 3,
  TRACEA_EPILOGUE_SILU = 4,
  TRACEA_EPILOGUE_BIAS_RELU = 5,
  TRACEA_EPILOGUE_BIAS_SILU = 6,
  TRACEA_EPILOGUE_RESIDUAL = 7,
  TRACEA_EPILOGUE_RESIDUAL_RELU = 8,
} TraceaEpilogueKind;

typedef enum {
  TRACEA_SOFTMAX_AUTO = 0,
  TRACEA_SOFTMAX_PER_TILE = 1,
  TRACEA_SOFTMAX_PER_TWO_TILES = 2,
} TraceaSoftmaxGranularity;

// ============================================================================
// Tensor View (Borrowed pointer, no ownership transfer)
// ============================================================================

typedef struct {
  void *ptr;             // Raw pointer (CPU or GPU)
  uint32_t rank;         // Number of dimensions
  const uint64_t *shape; // Shape array (caller-owned)
  const int64_t *stride; // Stride array (caller-owned)
  TraceaDType dtype;     // Data type
  int32_t device_id;     // -1 = CPU, 0+ = CUDA device
} TraceaTensorView;

// ============================================================================
// Conv2d Parameters
// ============================================================================

typedef struct {
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t padding_h;
  uint32_t padding_w;
  uint32_t dilation_h;
  uint32_t dilation_w;
  uint32_t groups;
  TraceaEpilogueKind epilogue;
  void *stream; // cudaStream_t (pass-through)
} TraceaConv2dParams;

typedef struct {
  TraceaEpilogueKind epilogue;
  void *stream;
} TraceaGemmParams;

typedef struct {
  uint8_t causal; // bool
  TraceaSoftmaxGranularity softmax_mode;
  float scale; // If 0, use 1/sqrt(d)
  void *stream;
} TraceaAttentionParams;

// ============================================================================
// API Functions
// ============================================================================

/// 2D Convolution
/// @param x Input tensor (NCHW)
/// @param w Weight tensor (OIHW)
/// @param b Bias tensor (can be NULL)
/// @param residual Residual tensor (can be NULL)
/// @param out Pre-allocated output tensor
/// @param params Convolution parameters
TraceaStatus tracea_conv2d(TraceaTensorView x, TraceaTensorView w,
                           const TraceaTensorView *b,
                           const TraceaTensorView *residual,
                           TraceaTensorView *out, TraceaConv2dParams params);

/// 2D Transposed Convolution (Deconvolution)
/// @param x Input tensor (NHWC)
/// @param w Weight tensor (KRSC)
/// @param b Bias tensor (can be NULL)
/// @param residual Residual tensor (can be NULL)
/// @param out Pre-allocated output tensor
/// @param params ConvTranspose2d parameters
typedef struct {
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t padding_h;
  uint32_t padding_w;
  uint32_t output_padding_h;
  uint32_t output_padding_w;
  uint32_t dilation_h;
  uint32_t dilation_w;
  uint32_t groups;
  TraceaEpilogueKind epilogue;
  void *stream;
} TraceaConvTranspose2dParams;

TraceaStatus tracea_conv_transpose2d(TraceaTensorView x, TraceaTensorView w,
                                     const TraceaTensorView *b,
                                     const TraceaTensorView *residual,
                                     TraceaTensorView *out,
                                     TraceaConvTranspose2dParams params);

/// GEMM (C = alpha * A * B + beta * C) -> Adjusted for Tracea: C =
/// Epilogue(A*B)
TraceaStatus tracea_gemm(TraceaTensorView a, TraceaTensorView b,
                         const TraceaTensorView *bias,
                         const TraceaTensorView *residual, TraceaTensorView *c,
                         TraceaGemmParams params);

/// FlashAttention-2
TraceaStatus tracea_attention(TraceaTensorView q, TraceaTensorView k,
                              TraceaTensorView v, TraceaTensorView *o,
                              TraceaAttentionParams params);

/// Get last error message
/// @param buf Buffer to write error message
/// @param len Buffer length
/// @return Number of bytes written, -1 on error
int32_t tracea_get_last_error(uint8_t *buf, size_t len);

/// Clear last error
void tracea_clear_error(void);

/// Version info
uint32_t tracea_version_major(void);
uint32_t tracea_version_minor(void);
uint32_t tracea_version_patch(void);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ Wrapper (Header-only)
// ============================================================================

#ifdef __cplusplus

#include <stdexcept>
#include <string>
#include <vector>

namespace tracea {

enum class DType {
  Float32 = TRACEA_DTYPE_FLOAT32,
  Float16 = TRACEA_DTYPE_FLOAT16,
  BFloat16 = TRACEA_DTYPE_BFLOAT16,
  Int32 = TRACEA_DTYPE_INT32,
  Int8 = TRACEA_DTYPE_INT8,
};

enum class EpilogueKind {
  Identity = TRACEA_EPILOGUE_IDENTITY,
  BiasAdd = TRACEA_EPILOGUE_BIAS_ADD,
  ReLU = TRACEA_EPILOGUE_RELU,
  Gelu = TRACEA_EPILOGUE_GELU,
  SiLU = TRACEA_EPILOGUE_SILU,
  BiasReLU = TRACEA_EPILOGUE_BIAS_RELU,
  BiasSiLU = TRACEA_EPILOGUE_BIAS_SILU,
  Residual = TRACEA_EPILOGUE_RESIDUAL,
  ResidualReLU = TRACEA_EPILOGUE_RESIDUAL_RELU,
};

enum class SoftmaxGranularity {
  Auto = TRACEA_SOFTMAX_AUTO,
  PerTile = TRACEA_SOFTMAX_PER_TILE,
  PerTwoTiles = TRACEA_SOFTMAX_PER_TWO_TILES,
};

/// TensorView - STL-friendly wrapper
class TensorView {
public:
  void *ptr = nullptr;
  std::vector<uint64_t> shape;
  std::vector<int64_t> stride;
  DType dtype = DType::Float32;
  int32_t device_id = -1; // -1 = CPU

  TensorView() = default;

  TensorView(void *ptr_, std::vector<uint64_t> shape_, DType dtype_,
             int32_t device_ = -1)
      : ptr(ptr_), shape(std::move(shape_)), dtype(dtype_), device_id(device_) {
    // Default stride: contiguous row-major
    stride.resize(shape.size());
    int64_t s = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      stride[i] = s;
      s *= shape[i];
    }
  }

  /// Convert to C ABI struct
  TraceaTensorView to_c() const {
    return TraceaTensorView{
        ptr,           static_cast<uint32_t>(shape.size()), shape.data(),
        stride.data(), static_cast<TraceaDType>(dtype),     device_id};
  }
};

/// Conv2d parameters
struct Conv2dParams {
  uint32_t stride_h = 1;
  uint32_t stride_w = 1;
  uint32_t padding_h = 0;
  uint32_t padding_w = 0;
  uint32_t dilation_h = 1;
  uint32_t dilation_w = 1;
  uint32_t groups = 1;
  EpilogueKind epilogue = EpilogueKind::Identity;
  void *stream = nullptr; // cudaStream_t

  TraceaConv2dParams to_c() const {
    return TraceaConv2dParams{
        stride_h,   stride_w,
        padding_h,  padding_w,
        dilation_h, dilation_w,
        groups,     static_cast<TraceaEpilogueKind>(epilogue),
        stream};
  }
};

struct GemmParams {
  EpilogueKind epilogue = EpilogueKind::Identity;
  void *stream = nullptr;

  TraceaGemmParams to_c() const {
    return TraceaGemmParams{static_cast<TraceaEpilogueKind>(epilogue), stream};
  }
};

struct AttentionParams {
  bool causal = true;
  SoftmaxGranularity softmax_mode = SoftmaxGranularity::Auto;
  float scale = 0.0f;
  void *stream = nullptr;

  TraceaAttentionParams to_c() const {
    return TraceaAttentionParams{
        (uint8_t)causal, static_cast<TraceaSoftmaxGranularity>(softmax_mode),
        scale, stream};
  }
};

/// Get last error as string
inline std::string get_last_error() {
  char buf[1024];
  int32_t len =
      tracea_get_last_error(reinterpret_cast<uint8_t *>(buf), sizeof(buf));
  return (len > 0) ? std::string(buf, len) : "";
}

/// 2D Convolution
/// @throws std::runtime_error on failure
inline void conv2d(const TensorView &x, const TensorView &w,
                   const TensorView *b, const TensorView *residual,
                   TensorView &out, const Conv2dParams &params) {
  TraceaTensorView c_b_storage;
  const TraceaTensorView *c_b_ptr = nullptr;
  if (b != nullptr) {
    c_b_storage = b->to_c();
    c_b_ptr = &c_b_storage;
  }

  TraceaTensorView c_res_storage;
  const TraceaTensorView *c_res_ptr = nullptr;
  if (residual != nullptr) {
    c_res_storage = residual->to_c();
    c_res_ptr = &c_res_storage;
  }

  TraceaTensorView c_out = out.to_c();
  TraceaStatus status = tracea_conv2d(x.to_c(), w.to_c(), c_b_ptr, c_res_ptr,
                                      &c_out, params.to_c());

  if (status != TRACEA_SUCCESS) {
    throw std::runtime_error("tracea::conv2d failed: " + get_last_error());
  }
}

/// ConvTranspose2d parameters (Deconvolution)
struct ConvTranspose2dParams {
  uint32_t stride_h = 1;
  uint32_t stride_w = 1;
  uint32_t padding_h = 0;
  uint32_t padding_w = 0;
  uint32_t output_padding_h = 0;
  uint32_t output_padding_w = 0;
  uint32_t dilation_h = 1;
  uint32_t dilation_w = 1;
  uint32_t groups = 1;
  EpilogueKind epilogue = EpilogueKind::Identity;
  void *stream = nullptr;

  TraceaConvTranspose2dParams to_c() const {
    return TraceaConvTranspose2dParams{
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        dilation_h,
        dilation_w,
        groups,
        static_cast<TraceaEpilogueKind>(epilogue),
        stream};
  }
};

/// ConvTranspose2d (Deconvolution)
/// @throws std::runtime_error on failure
inline void conv_transpose2d(const TensorView &x, const TensorView &w,
                             const TensorView *b, const TensorView *residual,
                             TensorView &out,
                             const ConvTranspose2dParams &params) {
  TraceaTensorView c_b_storage;
  const TraceaTensorView *c_b_ptr = nullptr;
  if (b != nullptr) {
    c_b_storage = b->to_c();
    c_b_ptr = &c_b_storage;
  }

  TraceaTensorView c_res_storage;
  const TraceaTensorView *c_res_ptr = nullptr;
  if (residual != nullptr) {
    c_res_storage = residual->to_c();
    c_res_ptr = &c_res_storage;
  }

  TraceaTensorView c_out = out.to_c();
  TraceaStatus status = tracea_conv_transpose2d(
      x.to_c(), w.to_c(), c_b_ptr, c_res_ptr, &c_out, params.to_c());

  if (status != TRACEA_SUCCESS) {
    throw std::runtime_error("tracea::conv_transpose2d failed: " +
                             get_last_error());
  }
}

/// GEMM wrapper
inline void gemm(const TensorView &a, const TensorView &b,
                 const TensorView *bias, const TensorView *residual,
                 TensorView &c, const GemmParams &params) {
  TraceaTensorView c_bias, c_res;
  const TraceaTensorView *p_bias = nullptr, *p_res = nullptr;

  if (bias) {
    c_bias = bias->to_c();
    p_bias = &c_bias;
  }
  if (residual) {
    c_res = residual->to_c();
    p_res = &c_res;
  }
  TraceaTensorView c_out = c.to_c();

  if (tracea_gemm(a.to_c(), b.to_c(), p_bias, p_res, &c_out, params.to_c()) !=
      TRACEA_SUCCESS) {
    throw std::runtime_error("tracea::gemm failed: " + get_last_error());
  }
}

/// Attention wrapper
inline void attention(const TensorView &q, const TensorView &k,
                      const TensorView &v, TensorView &o,
                      const AttentionParams &params) {
  TraceaTensorView c_out = o.to_c();
  if (tracea_attention(q.to_c(), k.to_c(), v.to_c(), &c_out, params.to_c()) !=
      TRACEA_SUCCESS) {
    throw std::runtime_error("tracea::attention failed: " + get_last_error());
  }
}

/// Version string
inline std::string version() {
  return std::to_string(tracea_version_major()) + "." +
         std::to_string(tracea_version_minor()) + "." +
         std::to_string(tracea_version_patch());
}

} // namespace tracea

#endif // __cplusplus
