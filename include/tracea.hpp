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
  void *stream; // cudaStream_t (pass-through)
} TraceaConv2dParams;

// ============================================================================
// API Functions
// ============================================================================

/// 2D Convolution
/// @param x Input tensor (NCHW)
/// @param w Weight tensor (OIHW)
/// @param b Bias tensor (can be NULL)
/// @param out Pre-allocated output tensor
/// @param params Convolution parameters
TraceaStatus tracea_conv2d(TraceaTensorView x, TraceaTensorView w,
                           const TraceaTensorView *b, TraceaTensorView *out,
                           TraceaConv2dParams params);

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
  void *stream = nullptr; // cudaStream_t

  TraceaConv2dParams to_c() const {
    return TraceaConv2dParams{stride_h,   stride_w,   padding_h, padding_w,
                              dilation_h, dilation_w, groups,    stream};
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
                   const TensorView *b, TensorView &out,
                   const Conv2dParams &params) {
  TraceaTensorView c_b_storage;
  const TraceaTensorView *c_b_ptr = nullptr;

  if (b != nullptr) {
    c_b_storage = b->to_c();
    c_b_ptr = &c_b_storage;
  }

  TraceaTensorView c_out = out.to_c();
  TraceaStatus status =
      tracea_conv2d(x.to_c(), w.to_c(), c_b_ptr, &c_out, params.to_c());

  if (status != TRACEA_SUCCESS) {
    throw std::runtime_error("tracea::conv2d failed: " + get_last_error());
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
