# Tracea: Mathematically Unified Universal GPU Optimization 🏛️✨

**Tracea** is a production-grade universal GPU kernel optimization framework. It transforms high-level **Semantic IR** into mathematically verified, high-performance kernels for **NVIDIA (CUDA)**, **AMD (ROCm)**, **Apple (Metal)**, and **CPU**. By leveraging asynchronous pipelining, Tensor Core acceleration, and Bayesian auto-tuning, Tracea achieves "Zero-Configuration" peak performance across platforms.

---

## 🚀 Key Features

- **Multi-Backend Excellence**: Native support for **CUDA**, **ROCm (HIP)**, **Metal**, and **CPU (SIMD)**.
- **Tracea Doctor**: Intelligent environment diagnostic tool that profiles hardware capabilities and selects the optimal kernel variant (e.g., Tensor Core vs. Matrix Core vs. SIMD).
- **Persistent Bayesian Auto-Tuning**: Hardware-aware search using Gaussian Processes. Optimal configurations are stored in an intelligent cache (`.tracea/tuning_cache.json`) for sub-millisecond reuse.
- **Environment-Aware Isolation**: Automatic cache invalidation based on backend, architecture, and driver versions.
- **Safe & Robust FFI**: Protected GPU memory handling via `PyDeviceBuffer`.
- **Graph-Level Intelligence**: Optimize entire sequences of operations using priority-based scheduling.

---

## 👨‍⚕️ Tracea Doctor (Multi-Backend Engine)

The **Tracea Doctor** automatically detects your hardware and selects the best implementation:

- **NVIDIA**: Leverages Ampere/Hopper Tensor Cores (sm_80+).
- **AMD**: Utilizes CDNA/RDNA Matrix Cores (gfx900+).
- **Apple**: Optimized for Apple Silicon M1/M2/M3 via Metal simdgroups.
- **CPU**: High-performance SIMD fallbacks (AVX512/AVX2/Neon).

---

## 📦 Installation

Tracea requires **Rust (Cargo)** and optionally **CUDA Toolkit**, **ROCm**, or **Metal** SDKs depending on your target.

1.  **Build the Library**:
    ```bash
    cargo build --lib --release
    ```
2.  **Install Python Module**:
    ```bash
    # Windows
    copy target\release\tracea.dll tracea.pyd
    ```

---

## 🐍 Python Usage

### 1. Initialize Context
```python
import tracea

# Create context (auto-detects hardware via Doctor)
ctx = tracea.Context()
```

### 2. Planning with Doctor
```python
# The Doctor selects the best variant for your current backend
decision = ctx.plan_kernel("flash_attention_2", precision="BF16")
print(f"Selected Variant: {decision.variant_id} on {decision.backend}")
```

---

## ⚡ C++ Usage (Zero-Latency)

```cpp
#include "tracea.hpp"

int main() {
    // Doctor identifies the backend and loads optimized variants automatically
    tracea::Context ctx();
    ctx.launch_gemm(...);
    return 0;
}
```

---

## 🏗️ Architecture

Tracea's design separates logical meaning from physical execution:

1. **L1: User Interface**: Premium Python (PyO3) and Rust APIs.
2. **L2: Core IR**: Pipelined Op and Graph definitions.
3. **L3: Tracea Doctor**: Multi-backend capability profiling and variant selection.
4. **L4: Semantic IR**: Phase Cyclicity, Swizzle Modes, and Lane Mapping.
5. **L5: Universal Emitters**: Multi-target code generation (PTX, HIP, MSL, SIMD).
6. **L6: Optimizer**: Persistent Bayesian tuner with backend-aware profiling.

---

## 📂 Project Structure

- `src/core/`: IR definitions and graph logic.
- `src/kernels/`: Optimized kernel implementations (CUDA/ROCm/Metal/CPU).
- `src/bindings/`: Python and C++ bindings.
- `src/optimizer/`: Bayesian Tuner and Profiler.
- `examples/`: Demonstration code.

---

## 📊 Backend Support Matrix

| Feature | CUDA | ROCm | Metal | CPU |
| :--- | :---: | :---: | :---: | :---: |
| Tensor/Matrix Cores | ◎ | ◎ | ◎ | × |
| Async Copy | ◎ | ○ | ○ | × |
| Pipelined GEMM | ◎ | ◎ | ◎ | ○ |
| SIMD Optimization | ◎ | ◎ | ◎ | ◎ |
| Auto-Tuning | ◎ | ○ | ○ | ○ |

---

**Tracea IS a "meaning-understanding optimization engine."** 🏛️🚀
