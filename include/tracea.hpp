#include <cstdint>
#include <string>
#include <vector>

extern "C" {
struct TraceaContext;
TraceaContext *tracea_context_create(const char *device_name);
void tracea_context_destroy(TraceaContext *ctx);
void tracea_execute_fused(const float *a, const float *b, float *c, int m,
                          int n, int k, const uint32_t *ops, uint32_t num_ops);
}

namespace tracea {

struct Shape {
  int m, n, k;
};

enum class EpilogueType { None = 0, ReLU = 1, Gelu = 2, BiasAdd = 3 };

struct EpilogueOp {
  EpilogueType type;
  uintptr_t bias_ptr = 0;
};

// Factory helpers
inline EpilogueOp relu() { return {EpilogueType::ReLU}; }
inline EpilogueOp gelu() { return {EpilogueType::Gelu}; }
inline EpilogueOp bias_add(const float *bias) {
  return {EpilogueType::BiasAdd, reinterpret_cast<uintptr_t>(bias)};
}

class Context {
public:
  explicit Context(const std::string &device) {
    ctx_ = tracea_context_create(device.c_str());
  }
  ~Context() {
    if (ctx_)
      tracea_context_destroy(ctx_);
  }

  void execute(const float *a, const float *b, float *c, Shape shape,
               const std::vector<EpilogueOp> &ops) {
    std::vector<uint32_t> raw_ops;
    for (auto &op : ops)
      raw_ops.push_back(static_cast<uint32_t>(op.type));
    // Note: Real impl would pass bias_ptr as well
    tracea_execute_fused(a, b, c, shape.m, shape.n, shape.k, raw_ops.data(),
                         raw_ops.size());
  }

  // Modern overload for C++20 users (conceptual)
  // void execute(std::span<const float> a, ...)

private:
  TraceaContext *ctx_;
};

} // namespace tracea
