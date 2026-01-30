# Tracea: Mathematically Unified Universal GPU Optimization 🏛️✨

**Tracea** is a production-grade universal GPU kernel optimization framework. It transforms high-level **Semantic IR** into mathematically verified, high-performance kernels for **NVIDIA (CUDA)**, **AMD (ROCm)**, **Apple (Metal)**, and **CPU**. By leveraging asynchronous pipelining, Tensor Core acceleration, and Bayesian auto-tuning, Tracea achieves "Zero-Configuration" peak performance across platforms.

---

## 🚀 Latest Benchmark Results (Jan 2026 - v0.1.0)

| Operation | Hardware | Performance | Vs. PyTorch | Notes |
|-----------|----------|-------------|-------------|-------|
| **GEMM (4096³)** | RTX 3070 | **65.43 TFLOPS** | **1.9x Faster** 🏆 | High-Occupancy Polyhedral Tiling |
| **Fused Conv2d** | RTX 3070 | 8.62 TFLOPS | 0.6x | Implicit GEMM (Winograd planned for v0.2) |
| **ResNet Block** | Apple M1 | 17.65 ms | N/A | Fused (Conv+BN+ReLU), Metal Backend |

---

## 🆕 v3.2 Features

### Topological Tile Graph (TTG) & Policy Engine
- **Dynamic Execution Planning**: Policy Engine generates optimized execution strategies based on hardware profiles and model topology.
- **Topological Tile Graph (TTG)**: A unified representation for sparse and dense tiling, enabling zero-copy sparse execution.
- **Architecture-Aware Dispatch**: Automated kernel selection (GEMM, Low-Rank MLP, Attention) via Policy decisions.

### Implicit GEMM Convolution Engine
- **Zero-im2col Architecture**: Coordinate mapping via fast-divmod primitives.
- **Hybrid Hoisting**: Adaptive SMEM management for coordinate tables.
- **Fused Epilogue Pipeline**: One-shot BiasAdd + SiLU/ReLU + Residual integration.

### mbarrier Integration
- Asynchronous GPU pipelining with producer-consumer warp roles.

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
├── core/           # IR definitions, TTG types
├── semantic/       # Semantic IR, tiling abstractions
├── backend/        # Multi-backend hardware abstractions
├── runtime/        # TTG Builder, Runtime Manager, Device buffers
├── kernels/        # CUDA/ROCm/Metal/CPU implementations
├── policy/         # Policy Engine, Tiling/Exec policies
├── optimizer/      # Bayesian Tuner, Hero configs
├── emitter/        # Universal Code Generators
├── doctor/         # Multi-backend diagnostics
└── optimized/      # Pre-compiled high-performance variants

tracea-python/      # PyO3 Python extension
tracea-ffi/         # C ABI for C++ integration
docs/               # Architecture and usage guides
examples/           # TTG sparse/low-rank demonstration code
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
