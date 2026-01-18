# Tracea API Usage Guide 📘

Tracea provides three primary interfaces: **Python (High-level)**, **C++ (System-level)**, and **Rust (Core)**.

---

## 🐍 Python API (via PyO3)

The Python API combines stateful context management with a declarative fusion builder.

### 1. `tracea.Context` & Hardware Auto-Detection
```python
import tracea

# Initialize context (Doctor automatically detects CUDA/ROCm/Metal/CPU)
ctx = tracea.Context()

# Plan a specific kernel variant (e.g., FlashAttention-2)
decision = ctx.plan_kernel("flash_attention_2", precision="BF16")
print(f"Executing {decision.variant_id} on {decision.backend}")

# Execute using safe buffers
with ctx.profiling():
    ctx.matmul(a_buf, b_buf, c_buf, 4096, 4096, 4096)
```

---

## 🏛️ C++ API (via tracea.hpp)

Modern RAII-based interface with type-safety for dimensions and pointers.

### 1. `tracea::Context` (Hardware Agnostic)
```cpp
#include <tracea.hpp>

// Doctor detects the best platform automatically
tracea::Context ctx(); 

// Explicit shape structure
tracea::Shape shape{.m=4096, .n=4096, .k=4096};

// Launch - The system chooses the best backend-specific implementation
ctx.execute_gemm(a, b, c, shape);
```

---

## 🦀 Rust Core API (Internal)

Advanced extension points for backend developers.

### 1. `tracea_emitter!` Macro
A DSL-like macro to eliminate boilerplate when adding new hardware.

```rust
tracea_emitter!(AmdEmitter {
    sync: |req| { /* emit s_waitcnt etc */ },
    epilogue: |ops, acc| { /* emit v_mfma epilogue */ }
});
```

### 2. LaneMapping Invariance Check
```rust
let mapping = LaneMapping::new(MatrixLayout::Amd16x16F32);
mapping.verify_injectivity().expect("Mapping collision!");
```
