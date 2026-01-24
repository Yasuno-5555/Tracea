# Tracea: Mathematically Unified Universal GPU Optimization 🏛️✨

**Tracea** is a production-grade universal GPU kernel optimization framework. It transforms high-level **Semantic IR** into mathematically verified, high-performance kernels for **NVIDIA (CUDA)**, **AMD (ROCm)**, **Apple (Metal)**, and **CPU**. By leveraging asynchronous pipelining, Tensor Core acceleration, and Bayesian auto-tuning, Tracea achieves "Zero-Configuration" peak performance across platforms.

---

## 🚀 Latest Benchmark Results (v3.1)

| Operation | Hardware | Performance | Notes |
|-----------|----------|-------------|-------|
| **GEMM** | RTX 3070 | >20 TFLOPS | Tensor Core MMA, 2-stage pipeline |
| **GEMM** | RTX 3070 | >20 TFLOPS | Tensor Core MMA, 2-stage pipeline |
| **Conv2d** | RTX 3070 | 22.73 TFLOPS | **Implicit GEMM**, 3-stage Pipeline, 78% Peak |
| **FA2** | RTX 3070 | 11.09 TFLOPS | S=2048, causal masking |
| **CPU GEMM** | Ryzen 5600X | 0.37 TFLOPS | 3.67x vs naive (SIMD packing) |

---

## 🆕 v3.2 Features

### Implicit GEMM Convolution Engine
- **Zero-im2col Architecture**: Coordinate mapping via fast-divmod primitives.
- **Hybrid Hoisting**: Adaptive SMEM management for coordinate tables.
- **Alignment-Safe Stores**: Native support for odd channel counts without performance drop.
- **Fused Epilogue Pipeline**: One-shot BiasAdd + SiLU/ReLU + Residual integration.

### mbarrier Integration
- Asynchronous GPU pipelining with producer-consumer warp roles
- Rule A: Single Init Responsibility
- Rule B: Static Warp-Role Assignment

### Python Integration (tracea-python)
```python
import tracea
# Zero-copy PyTorch backend with automatic fallback
tracea.patch_conv2d()
```

### C++ FFI (tracea-ffi)
```cpp
#include "tracea.hpp"
tracea::conv2d(x, w, nullptr, out, params);
```

---

## 🚀 Key Features

- **Multi-Backend Excellence**: Native support for **CUDA**, **ROCm (HIP)**, **Metal**, and **CPU (SIMD)**.
- **Tracea Doctor**: Intelligent environment diagnostic tool that profiles hardware capabilities and selects the optimal kernel variant.
- **Persistent Bayesian Auto-Tuning**: Hardware-aware search using Gaussian Processes with intelligent caching.
- **Zero-Copy Integration**: Python (PyTorch) and C++ bindings with borrowed pointer semantics.
- **Graph-Level Intelligence**: Optimize entire sequences of operations using priority-based scheduling.

---

## 📦 Installation

### Rust Library
```bash
cargo build --lib --release
```

### Python Module
```bash
cd tracea-python && maturin develop
```

### C++ FFI
```bash
cd tracea-ffi && cargo build --release
# Link against target/release/libtracea_ffi.a
```

---

## 🐍 Python Usage

```python
import tracea

# Create context (auto-detects hardware via Doctor)
ctx = tracea.Context()

# Build and optimize a graph
graph = tracea.Graph()
graph.add_gemm(m=2048, n=2048, k=2048)
config = ctx.optimize_graph(graph)
```

### Monkey Patch for PyTorch
```python
from tracea import patch_conv2d
patch_conv2d()  # Replaces torch.nn.Conv2d
```

---

## 🔧 C++ Usage

```cpp
#include "tracea.hpp"

tracea::TensorView x(x_ptr, {1, 64, 224, 224}, tracea::DType::Float32, 0);
tracea::TensorView w(w_ptr, {128, 64, 3, 3}, tracea::DType::Float32, 0);
tracea::TensorView out(out_ptr, {1, 128, 222, 222}, tracea::DType::Float32, 0);

tracea::Conv2dParams params;
params.stride_h = params.stride_w = 1;
params.stream = my_cuda_stream;

tracea::conv2d(x, w, nullptr, out, params);
```

---

## 📂 Project Structure

```
src/
├── core/           # IR definitions, graph logic
├── kernels/        # CUDA/ROCm/Metal/CPU implementations
├── interface/      # Python and C++ bindings
├── optimizer/      # Bayesian Tuner, Profiler
├── emitter/        # Universal Code Generators
└── doctor/         # Multi-backend diagnostics

tracea-python/      # PyO3 Python extension
tracea-ffi/         # C ABI for C++ integration
docs/               # Design documents
examples/           # Demonstration code
```

---

## 📄 Documentation

- [API Usage](docs/API_USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Optimization Guide](docs/OPTIMIZATION.md)
- [Benchmark Report](docs/BENCHMARK_REPORT.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)

---

**Tracea IS a "meaning-understanding optimization engine."** 🏛️🚀
