# Tracea API Usage Guide 📘

Tracea provides three primary interfaces: **Python (High-level)**, **C++ (System-level)**, and **Rust (Core)**.

---

## 🐍 Python API (via PyO3)

The Python API combines stateful context management with a declarative fusion builder.

### 1. `tracea.Context` & Pipe-style Fusion
```python
import tracea

# Initialize context
ctx = tracea.Context("A100")

# Create a fusion pipe using the builder or >> operator
pipeline = tracea.Epilogue().bias_add(b_ptr).relu()
# OR
# pipeline = tracea.BiasAdd(b_ptr) >> tracea.ReLU()

with ctx.profiling():
    ctx.matmul(a_ptr, b_ptr, c_ptr, 1024, 1024, 1024, pipeline)
```

---

## 🏛️ C++ API (via tracea.hpp)

Modern RAII-based interface with type-safety for dimensions and pointers.

### 1. `tracea::Context` & `tracea::Shape`
```cpp
#include <tracea.hpp>

tracea::Context ctx("A100");

// Explicit shape structure to avoid parameter swap errors
tracea::Shape shape{.m=1024, .n=1024, .k=1024};

// Type-safe descriptors with structured parameters
auto ops = {
    tracea::bias_add(bias_ptr),
    tracea::relu()
};

ctx.execute(a, b, c, shape, ops);
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
