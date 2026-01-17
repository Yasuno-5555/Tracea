# Tracea Architecture: The 6-Layer Semantic Model 🏛️

Tracea is built on a strictly layered architecture that separates "Meaning" (Semantic IR) from "Physics" (Emitter/Hardware).

## Layer Overview

### L1: User Interface
- **Python**: PyO3 bindings + TorchDynamo backend. Captures FX Graphs and translates them to L2.
- **C++**: RAII header (`tracea.hpp`) wrapping a stable C-FFI.
- **Core API**: Rust entry points for manual kernel dispatch.

### L2: Core IR
- Defines the logical operation: `GemmOp(m, n, k)`, `FusedGemmOp`.
- Represents the goal without specifying the implementation details.

### L3: Semantic IR (The Heart)
- **Phase Transition**: Models $Z/NZ$ cyclicity in asynchronous pipes.
- **Lane Mapping**: Matrix Core / Tensor Core / XMX register layout abstractions.
- **Swizzle Mode**: Algebraic bank conflict resolution (e.g., XOR swizzle).

### L4: Optimized IR
- A specific configuration of tiles, stages, and swizzles that has been validated for a specific hardware target.

### L5: Universal Emitters
- **CUDA Emitter**: PTX-level pipelining, Register Double Buffering.
- **HIP Emitter**: AMD GCN/CDNA intrinsics ($v\_mfma$).
- **SYCL Emitter**: Intel ESIMD and sub-group matrix ($XMX$).

### L6: Optimizer
- **Bayesian Auto-tuner**: Uses Gaussian Processes to explore the L4 space.
- **Micro-benchmarks**: Hardware-aware data collection.

---

## Data Flow: From Python to Kernel

1.  **Capture**: `torch.compile` passes an FX Graph to `tracea.backend`.
2.  **Lowering**: Python parses `aten.mm` and `aten.relu` into `PyEpilogueOp` descriptors.
3.  **Optimization**: `AutoTuner` searches for the best `PipelineConfig`.
4.  **Emission**: A backend-specific `Emitter` generates the kernel source.
5.  **JIT**: The generated code is compiled via `NVRTC`, `HIPRTC`, or `oneAPI` and launched.
