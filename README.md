# Tracea: Mathematically Unified GPU Optimization 🏛️✨

**Tracea** is a production-grade universal GPU kernel optimization framework. It transforms high-level **Semantic IR** into mathematically verified, high-performance kernels for **NVIDIA (CUDA)**. By leveraging asynchronous pipelining, Tensor Core acceleration, and Bayesian auto-tuning, Tracea achieves "Zero-Configuration" peak performance.

---

## 🚀 Key Features

- **Persistent Bayesian Auto-Tuning**: Hardware-aware search using Gaussian Processes. Optimal configurations are stored in an intelligent cache (`.tracea/tuning_cache.json`) for sub-millisecond reuse.
- **Environment-Aware Isolation**: Automatic cache invalidation based on CUDA version, Driver version, and precise SM architecture (stepping). Your optimizations are always "environment-safe."
- **Safe & Robust FFI**: Protected GPU memory handling via `PyDeviceBuffer`, preventing segmentation faults and undefined behavior.
- **Graph-Level Intelligence**: Optimize entire sequences of operations using priority-based scheduling and cross-node configuration reuse.
- **Multi-Objective Optimization**: Tailor your "Masterpiece." Optimize for raw **TFLOPS**, minimal **Latency**, or a balanced utility score.
- **Fused Epilogue Mastery**: High-performance fusion of BiasAdd, ReLU, and Gelu directly into Tensor Core GEMM kernels.

---

## 📦 Installation

Tracea requires **Rust (Cargo)**, **Python 3.8+**, and **CUDA Toolkit (11.x or 12.x)**.

1.  **Build the Library**:
    ```bash
    cargo build --lib --release
    ```
2.  **Install Python Module**:
    Copy the generated artifact to your project or install:
    ```bash
    # Windows
    copy target\release\tracea.dll tracea.pyd
    
    # Linux/Mac
    cp target/release/libtracea.so tracea.so
    ```

---

## 🐍 Python Usage

### 1. Initialize Context
```python
import tracea
import torch

# Create context (auto-detects GPU)
ctx = tracea.Context("GeForce RTX 3070")
```

### 2. **NEW: Safe Memory Management**
To ensure safety, Tracea now requires explicit device buffers. You can create them zero-copy from PyTorch pointers.

```python
# Create PyTorch tensors
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16) # FP16 for Tensor Cores
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# Wrap safely for Tracea (Zero-Copy)
# Note: Tensor Cores require U16/Half buffers for inputs
a_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(a.data_ptr(), a.numel(), ctx)
b_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(b.data_ptr(), b.numel(), ctx)
c_buf = ctx.scratch_c # Use internal scratchpad or wrap your own F32 output
```

### 3. Graph Optimization (The Masterpiece)
```python
# Define Computation Graph
graph = tracea.Graph()
graph.add_gemm(4096, 4096, 4096)
graph.add_gemm(1024, 1024, 1024)

# Optimize (Bayesian Search + Persistence)
ctx.optimize_graph(graph, iterations=15, goal=tracea.OptimizationGoal.MaximizeTFLOPS)
```

### 4. Execute Kernel
```python
# Execute using optimized config from cache
# Matmul now takes safe buffers, NOT integer pointers
ctx.matmul(a_buf, b_buf, c_buf, 1024, 1024, 1024, tracea.Epilogue.empty())
ctx.synchronize()
```

---

## ⚡ C++ Usage (Zero-Latency)

Tracea provides a header-only C++17 RAII wrapper for maximum performance.

```cpp
#include "tracea.hpp"

int main() {
    try {
        // 1. Create Context (RAII)
        tracea::Context ctx("GeForce RTX 4090");

        // 2. Compile/Load optimized kernels (from cache if available)
        ctx.compile_empty(); // Example API
        
        // 3. Launch with < 5µs latency
        for (int i = 0; i < 1000; ++i) {
            ctx.launch_empty();
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

---

## 🏗️ Architecture

Tracea's design separates logical meaning from physical execution:

1. **L1: User Interface**: Premium Python (PyO3) and Rust APIs.
2. **L2: Core IR**: Pipelined Op and Graph definitions.
3. **L3: Semantic IR**: Phase Cyclicity, Swizzle Modes, and Lane Mapping.
4. **L4: Optimized IR**: Bayesian-searched Hardware-aware configurations.
5. **L5: Universal Emitters**: High-performance CUDA code generation (PTX/SASS-ready).
6. **L6: Optimizer**: Persistent Bayesian tuner with Environment-aware profiling.

---

## 📂 Project Structure

- `src/`: Rust source code (Core, Optimizer, Emitter, Interface).
- `benchmarks/`: Python scripts for performance testing.
- `scripts/`: Utility and test scripts (`verify_fix.py`).
- `logs/`: Build logs and benchmark results.
- `.tracea/`: Persistent tuning cache.

---

**The Revolution is Disciplined. Peak Performance is Persistent.** 🏛️🚀
