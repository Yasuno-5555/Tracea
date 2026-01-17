# Tracea: Mathematically Unified GPU Optimization 🏛️✨

**Tracea** is a world-first universal GPU kernel optimization framework designed to bridge the gap between architectural diversity and peak performance. By leveraging a high-level **Semantic IR**, Tracea generates mathematically verified, high-performance kernels for **NVIDIA (CUDA)**, **AMD (HIP)**, and **Intel (SYCL)** from a single logical definition.

---

## 🚀 Key Features

- **Multi-Vendor Pipelining**: Generate 4+ stage asynchronous pipelined GEMM kernels for Tensor Cores, Matrix Cores (v_mfma), and XMX (sub-group matrix).
- **Bayesian Auto-Tuning**: Hardware-aware search using Gaussian Processes to find the optimal tile size and swizzle patterns in seconds.
- **TorchDynamo Fusion**: A `torch.compile` backend that automatically fuses Matmul with activations (ReLU, Gelu, BiasAdd) into single-kernel epilogues.
- **Zero-Copy Interop**: Direct integration with PyTorch via PyO3, sharing data pointers without overhead.
- **Dual Interface**: Premium Python bindings (PyO3) and high-performance C++ RAII headers (`tracea.hpp`).

---

## 🏗️ Architecture: The 6-Layer Tracea Model

Tracea's design is strictly layered to separate logical meaning from physical execution:

1. **L1: User Interface** (Python/C++/Core API)
2. **L2: Core IR** (Pipelined Op Definitions)
3. **L3: Semantic IR** (Phase Cyclicity, Swizzle Modes, Lane Mapping)
4. **L4: Optimized IR** (Auto-tuned Hardware-aware configurations)
5. **L5: Universal Emitters** (CUDA, HIP, SYCL Code Generation)
6. **L6: Optimizer** (Bayesian Auto-tuner, Micro-benchmarks)

---

## 🐍 Python Usage

```python
import torch
import tracea
from tracea.backend import compile

# 1. Define your model
class Model(torch.nn.Module):
    def forward(self, x, w):
        return torch.relu(torch.mm(x, w))

# 2. Compile with Tracea backend
model = Model()
optimized_model = torch.compile(model, backend="tracea")

# 3. Execute (Tracea captures and fuses mm + relu)
y = optimized_model(x_gpu, w_gpu)
```

## 🏛️ C++ Usage

```cpp
#include <tracea.hpp>

int main() {
    auto ctx = tracea::Context("A100");
    
    // Execute Matmul + Bias + ReLU fusion
    ctx.execute(a_ptr, b_ptr, c_ptr, M, N, K, 
                {tracea::Epilogue::BiasAdd, tracea::Epilogue::ReLU});
    
    return 0;
}
```

## 📚 Documentation

For more in-depth information, please refer to the following guides:

- [**Architecture Deep Dive**](docs/ARCHITECTURE.md): Understanding the 6-layer model.
- [**API Usage Guide**](docs/API_USAGE.md): Detailed Python, C++, and Rust examples.
- [**Developer Guide**](docs/DEVELOPER_GUIDE.md): Extending backends and epilogues.
- [**Optimization & Tuning**](docs/OPTIMIZATION.md): How Tracea achieves peak performance.

---

## 🛠️ Build & Development

Requires Rust 2024 and the respective vendor SDKs (CUDA, ROCm, or oneAPI).

```powershell
# Build the Rust core and Python bindings
cargo build --release

# Run internal benchmarks and tests
cargo run --bin tracea
```

---

## 🏛️ Project Philosophy

Tracea is built on the belief that **mathematics is the ultimate bridge**. We don't write kernels; we define the **group-theoretic properties** of hardware lanes and the **phasic transitions** of asynchronous pipes. The code is not "written"—it is "solved."

**The Revolution is Accomplished.** 🏛️🚀
