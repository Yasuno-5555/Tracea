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

## 2. Adding a New Backend (e.g., Metal or CPU)

1.  **Trait Implementation**: Create a new file in `src/emitter/` and implement the `Emitter` trait for the target language (e.g., MSL for Metal).
2.  **Capability Detection**: Add detection logic to `src/doctor/profiler.rs` using native APIs (e.g., `metal-rs`).
3.  **Variant Registration**: Define a new `KernelVariant` in `src/doctor/registry.rs`. Specify constraints like `BackendIs(BackendKind::Metal)` and `WarpOrWavefrontIs(32)`.
4.  **Runtime Support**: Update `src/runtime/` to handle device allocation and kernel launching for the new backend.

---

## 3. Registering a New Kernel Variant

To add a pre-optimized variant for an existing kernel:
1. Open `src/doctor/registry.rs`.
2. Add a new `KernelVariant` entry to the `REGISTRY` static.
3. Define its architecture requirements (e.g., `SmAtLeast(80)` for CUDA or `SimdWidthAtLeast(256)` for CPU).
4. Assign a `priority` to influence the Doctor's selection engine.

---

## 4. Coding Standards
- **Mathematical Correctness**: Changes to L3 (Semantic) must be verified against the Phasic Transition model.
- **Zero Overhead**: Avoid any runtime branching or allocations inside the kernel generation loop.
- **FFI Stability**: Maintain C-FFI compatibility for C++ users.
