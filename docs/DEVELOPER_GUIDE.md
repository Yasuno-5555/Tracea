# Tracea Developer Guide

## Quick Start

### Prerequisites
- Rust 1.70+
- CUDA Toolkit 12.x (for GPU)
- Python 3.8+ (for Python bindings)
- maturin (for Python build)

### Build Commands

```bash
# Core library
cargo build --lib --release

# Python extension
cd tracea-python && maturin develop

# C++ FFI
cd tracea-ffi && cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo run --release --example gemm_bench
```

---

## Project Structure

```
Tracea/
├── src/
│   ├── core/          # IR, config, graph
│   ├── kernels/       # CUDA/ROCm/Metal/CPU implementations
│   ├── emitter/       # Code generators
│   ├── optimizer/     # Bayesian tuner, policies
│   ├── doctor/        # Multi-backend diagnostics
│   └── interface/     # Python/C bindings
├── tracea-python/     # PyO3 extension crate
├── tracea-ffi/        # C ABI crate
├── examples/          # Demo code
├── docs/              # Documentation
└── include/           # C++ headers
```

---

## Key APIs

### Rust
```rust
use tracea::{AutoTuner, GPUInfo, ProblemDescriptor};

let gpu = GPUInfo::rtx3070();
let tuner = AutoTuner::new(gpu);
let config = tuner.optimize(&problem);
```

### Python
```python
import tracea
ctx = tracea.Context()
graph = tracea.Graph()
graph.add_gemm(m=2048, n=2048, k=2048)
```

### C++
```cpp
#include "tracea.hpp"
tracea::conv2d(x, w, nullptr, out, params);
```

---

## Adding a New Kernel

1. Define operation in `src/core/op.rs`
2. Implement emitter in `src/emitter/`
3. Add policy in `src/optimizer/policy.rs`
4. Create benchmark in `examples/`
5. Update docs

---

## Safety Rules

1. **FFI Boundary**: All `extern "C"` use `catch_unwind`
2. **Memory**: Never take ownership from Python/C++
3. **Panics**: Contained within Rust boundary
