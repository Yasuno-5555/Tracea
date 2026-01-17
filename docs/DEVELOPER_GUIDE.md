# Tracea Developer Guide: Extending the Framework 🛠️

Tracea is designed to be easily extensible. This guide covers how to add new fusion operations and hardware backends.

## 1. Adding a New Epilogue Fusion (e.g., Sigmoid)

### Step A: Update Semantic IR
In `src/semantic/fusion.rs`, add the new variant:
```rust
pub enum EpilogueOp {
    // ...
    Sigmoid,
}
```

### Step B: Update Emitters
In `src/emitter/traits.rs`, the `emit_epilogue` method will now receive the new op. Implement it in the vendor emitters (e.g., `cuda.rs`):
```rust
EpilogueOp::Sigmoid => {
    code.push_str(&format!("  {acc} = 1.0f / (1.0f + expf(-{acc}));\n", acc = acc_name));
}
```

### Step C: Update Python Bridge
In `src/interface/python.rs`, add a static method to `PyEpilogueOp` to expose it to TorchDynamo.

---

## 2. Adding a New Backend (e.g., Metal or WebGPU)

1.  **Trait Implementation**: Create a new file in `src/emitter/` and implement the `Emitter` trait.
2.  **Addressing**: Implement swizzle logic that matches the new target's bank structure.
3.  **Intrinsics**: Map Tracea's `LaneMapping` concepts to the target's matrix intrinsics (e.g., Metal's `simdgroup_multiply_accumulate`).

---

## 3. Coding Standards
- **Mathematical Correctness**: Changes to L3 (Semantic) must be verified against the Phasic Transition model.
- **Zero Overhead**: Avoid any runtime branching or allocations inside the kernel generation loop.
- **FFI Stability**: Maintain C-FFI compatibility for C++ users.
