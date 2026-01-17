#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Opaque context handle
typedef struct TraceaContextOpaque TraceaContextOpaque;
typedef TraceaContextOpaque *TraceaContextHandle;

// Core API
TraceaContextHandle tracea_create_context(const char *device_name);
void tracea_destroy_context(TraceaContextHandle ctx);

// Benchmark Utils (Phase 1)
int tracea_compile_empty(TraceaContextHandle ctx);
int tracea_launch_empty(TraceaContextHandle ctx);

// Placeholder for future optimization API
// int tracea_optimize_graph(TraceaContextHandle ctx, ...);

#ifdef __cplusplus
}
#endif

// C++ Wrapper for convenience
#ifdef __cplusplus
#include <stdexcept>
#include <string>

namespace tracea {

class Context {
public:
  explicit Context(const std::string &device_name) {
    handle_ = tracea_create_context(device_name.c_str());
    if (!handle_) {
      throw std::runtime_error("Failed to create Tracea context");
    }
  }

  ~Context() {
    if (handle_) {
      tracea_destroy_context(handle_);
      handle_ = nullptr;
    }
  }

  // Prevent copying
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  // Allow moving
  Context(Context &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  Context &operator=(Context &&other) noexcept {
    if (this != &other) {
      if (handle_)
        tracea_destroy_context(handle_);
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  TraceaContextHandle native_handle() const { return handle_; }

  void compile_empty() {
    int res = tracea_compile_empty(handle_);
    if (res != 0) {
      throw std::runtime_error("Tracea compilation failed: " +
                               std::to_string(res));
    }
  }

  void launch_empty() {
    int res = tracea_launch_empty(handle_);
    if (res != 0) {
      throw std::runtime_error("Tracea launch failed: " + std::to_string(res));
    }
  }

private:
  TraceaContextHandle handle_;
};

} // namespace tracea
#endif
